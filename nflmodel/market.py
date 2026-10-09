"""Market reference from contemporaneous, two-sided bookmaker quotes.

Input: offers that already passed odds.validate_offers (fresh, timestamped, pre-kickoff).

1. Pairs. For each book, the latest quote of each side is paired with the other
   side of the SAME market at the SAME line (spread +x / -x, total o/u at one
   number, moneyline home/away). Sides quoted more than `pair_max_gap_minutes`
   apart are not paired, because they are not contemporaneous.
2. Vig removal. Proportional: p_side = implied_side / (implied_home + implied_away).
   Simple and standard for two-way markets; with ~4-5% margins its difference from
   power/Shin methods is small for spreads and totals but slightly overstates
   long shots on lopsided moneylines.
3. Different lines. A spread or total pair gives P(outcome over the line | no push).
   That is converted to an implied mean margin/total `mu_b` by inverting the
   key-number outcome distribution (the same one the model uses), so +3 at one
   book and +3.5 at another become comparable. Moneylines are compared in logit space.
4. Aggregation. The reference is the MEDIAN across books (of mu_b, or of
   logit p). The reference line is the median quoted line, snapped to 0.5.
5. Self-reference. When an offer from book B is evaluated, B is left out of its
   reference (leave-one-book-out), and at least `min_reference_books` OTHER books
   are required. With fewer, there is no live reference and the offer cannot be
   executable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .odds import implied_probability

PAIR_MAX_GAP_MINUTES = 5.0


@dataclass
class Reference:
    market: str
    n_books: int
    books: list[str]
    line: float | None            # spread (home margin, nflverse convention) or total; None for ml
    mu: float | None              # implied mean home margin / total
    p: float                      # P(home covers | no push) at `line`, P(over | no push), or P(home wins)
    spread_of_books: float        # max - min of mu_b (pts) or of p (ml), a dispersion check
    oldest_utc: pd.Timestamp | None = None
    newest_utc: pd.Timestamp | None = None
    excluded_book: str | None = None
    kind: str = "live"
    per_book: dict = field(default_factory=dict)
    quotes: list = field(default_factory=list)    # the contributing two-sided pairs (provenance)

    def provenance(self) -> dict:
        """JSON-safe record of exactly which quotes built this reference."""
        def t(x):
            return None if x is None else pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"market": self.market, "excluded_book": self.excluded_book, "books": self.books,
                "line": self.line, "p": self.p, "oldest_utc": t(self.oldest_utc), "newest_utc": t(self.newest_utc),
                "quotes": [{"book": q["book"], "line": q["line"],
                            "sides": [{"side": sd["side"], "point": sd["point"], "price": sd["price"],
                                       "time_utc": t(sd["time"])} for sd in q["sides"]]}
                           for q in self.quotes]}


def _latest(offers: pd.DataFrame) -> pd.DataFrame:
    """Latest quote per (book, market, side, point)."""
    o = offers.sort_values("odds_time_utc")
    return o.groupby(["book", "market", "side", "point"], dropna=False, as_index=False).tail(1)


def book_pairs(offers: pd.DataFrame, market: str, gap_minutes: float = PAIR_MAX_GAP_MINUTES) -> list[dict]:
    """Two-sided no-vig quotes per book: [{book, line, p, time}], line in home/over terms."""
    o = _latest(offers[offers["market"] == market])
    out = []
    gap = pd.Timedelta(minutes=gap_minutes)
    for book, b in o.groupby("book"):
        if market == "ml":
            h, a = b[b["side"] == "home"], b[b["side"] == "away"]
            cands = [(h.iloc[-1], a.iloc[-1], None)] if len(h) and len(a) else []
        elif market == "spread":
            cands = []
            for r in b[b["side"] == "home"].itertuples(index=False):
                a = b[(b["side"] == "away") & (b["point"] == -r.point)]
                if len(a):
                    # nflverse convention: home covers when margin > line, line = -home_point
                    cands.append((r, a.iloc[-1], -r.point))
        else:
            cands = []
            for r in b[b["side"] == "over"].itertuples(index=False):
                u = b[(b["side"] == "under") & (b["point"] == r.point)]
                if len(u):
                    cands.append((r, u.iloc[-1], r.point))
        for first, second, line in cands:
            t1, t2 = first.odds_time_utc, second.odds_time_utc
            if abs(t1 - t2) > gap:
                continue
            i1, i2 = implied_probability(first.price), implied_probability(second.price)
            sides = [{"side": x.side, "point": None if pd.isna(x.point) else float(x.point),
                      "price": float(x.price), "time": x.odds_time_utc} for x in (first, second)]
            out.append({"book": book, "line": line, "p": i1 / (i1 + i2),
                        "overround": i1 + i2 - 1, "time": max(t1, t2), "oldest": min(t1, t2), "sides": sides})
    return out


def _invert(dist, line: float, p: float, lo: float, hi: float) -> float:
    """mu such that P(X > line | no push; mu) = p (monotone increasing in mu)."""
    def f(mu):
        over, push, under = dist.prob_over(mu, line)
        return over / (over + under)
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_reference(offers: pd.DataFrame, market: str, model, min_books: int,
                    exclude_book: str | None = None,
                    gap_minutes: float = PAIR_MAX_GAP_MINUTES) -> Reference | None:
    pairs = [q for q in book_pairs(offers, market, gap_minutes) if q["book"] != exclude_book]
    # one pair per book: the pair closest to that book's median line (books may post alt lines)
    by_book: dict[str, list] = {}
    for q in pairs:
        by_book.setdefault(q["book"], []).append(q)
    if len(by_book) < min_books:
        return None
    if market == "ml":
        chosen = [qs[-1] for qs in by_book.values()]
        logits = np.array([np.log(q["p"] / (1 - q["p"])) for q in chosen])
        p = float(1 / (1 + np.exp(-np.median(logits))))
        return Reference("ml", len(chosen), sorted(by_book), None, None, p,
                         float(max(q["p"] for q in chosen) - min(q["p"] for q in chosen)),
                         min(q["oldest"] for q in chosen), max(q["time"] for q in chosen), exclude_book,
                         per_book={q["book"]: q["p"] for q in chosen}, quotes=chosen)
    dist = model.margin_dist if market == "spread" else model.total_dist
    lo, hi = (-60.0, 60.0) if market == "spread" else (5.0, 125.0)
    all_lines = np.median([q["line"] for q in pairs])
    chosen = [min(qs, key=lambda q: abs(q["line"] - all_lines)) for qs in by_book.values()]
    mus = np.array([_invert(dist, q["line"], q["p"], lo, hi) for q in chosen])
    mu = float(np.median(mus))
    line = float(np.round(np.median([q["line"] for q in chosen]) * 2) / 2)
    over, push, under = dist.prob_over(mu, line)
    return Reference(market, len(chosen), sorted(by_book), line, mu, over / (over + under),
                     float(mus.max() - mus.min()),
                     min(q["oldest"] for q in chosen), max(q["time"] for q in chosen), exclude_book,
                     per_book={q["book"]: float(m) for q, m in zip(chosen, mus)}, quotes=chosen)


def references_frame(refs_by_game: dict) -> pd.DataFrame:
    """Snapshot table: one row per (game, market, excluded book) with full provenance as JSON."""
    import json
    rows = []
    for gid, refs in refs_by_game.items():
        for (mk, ex), r in refs.items():
            if r is None:
                continue
            rows.append({"game_id": gid, "market": mk, "excluded_book": ex, "n_books": r.n_books,
                         "books": ",".join(r.books), "line": r.line, "mu": r.mu, "p": r.p,
                         "oldest_utc": r.provenance()["oldest_utc"], "newest_utc": r.provenance()["newest_utc"],
                         "provenance": json.dumps(r.provenance(), sort_keys=True)})
    return pd.DataFrame(rows)


def references_for_game(offers: pd.DataFrame, model, min_books: int,
                        gap_minutes: float = PAIR_MAX_GAP_MINUTES) -> dict:
    """{(market, excluded_book or None): Reference or None} for one game's validated offers."""
    out = {}
    for market in ("spread", "ml", "total"):
        out[(market, None)] = build_reference(offers, market, model, min_books, None, gap_minutes)
        for book in offers.loc[offers["market"] == market, "book"].unique():
            out[(market, book)] = build_reference(offers, market, model, min_books, book, gap_minutes)
    return out
