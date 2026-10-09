"""Closing-line value against the LAST VALID PRE-KICKOFF market reference in the snapshots.

Every `collect` / `predict` snapshot keeps its validated bookmaker quotes (offers_valid.csv,
offers_valid_final.csv, with odds_time_utc). For each game and market the closing reference
is rebuilt, model-free, from the latest snapshot whose quotes all precede kickoff and that
has two-sided quotes from at least `min_books` books:
  spread / total  line = median of the books' lines (snapped to 0.5); p = median no-vig
                  P(home covers | over) among books quoting exactly that line
  moneyline       p = median no-vig P(home wins) (median taken in logit space)

CLV of a side priced earlier (forecast or wager):
  points       spread: bet line minus closing line from that side's view (home -3 vs close -4 = +1)
               total:  over: close - bet total; under: bet total - close
  probability  closing no-vig probability of the side minus the market probability recorded
               when the side was priced. Exact when the bet line equals the closing line;
               otherwise the closing distribution is shifted with a normal approximation
               (SD 13.5 points, no key numbers) and flagged.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .market import book_pairs
from .timeutil import parse_utc

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "snapshots"
SD_APPROX = 13.5
QUOTE_FILES = ("offers_valid_final.csv", "offers_valid.csv")


def snapshot_quotes(root: Path = SNAPSHOTS) -> pd.DataFrame:
    """Validated quotes from every snapshot (one quote file per snapshot, the final one if present)."""
    frames = []
    if not root.exists():
        return pd.DataFrame()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = next((d / n for n in QUOTE_FILES if (d / n).exists()), None)
        if f is None:
            continue
        q = pd.read_csv(f)
        if q.empty or "odds_time_utc" not in q:
            continue
        q = q[q["source"] != "consensus"] if "source" in q else q
        frames.append(q.assign(snapshot=d.name))
    if not frames:
        return pd.DataFrame()
    q = pd.concat(frames, ignore_index=True)
    q["odds_time_utc"] = pd.to_datetime(q["odds_time_utc"], utc=True)
    return q


def _reference(q: pd.DataFrame, market: str, min_books: int) -> dict | None:
    pairs = book_pairs(q, market)
    books = {p["book"] for p in pairs}
    if len(books) < min_books:
        return None
    t = max(p["time"] for p in pairs)
    if market == "ml":
        lg = np.median([math.log(p["p"] / (1 - p["p"])) for p in pairs])
        return {"line": np.nan, "p": float(1 / (1 + math.exp(-lg))), "n_books": len(books), "as_of_utc": t}
    line = float(np.round(np.median([p["line"] for p in pairs]) * 2) / 2)
    at = [p["p"] for p in pairs if p["line"] == line]
    if not at:   # no book exactly at the median line: nearest line's books
        best = min(pairs, key=lambda p: abs(p["line"] - line))["line"]
        line, at = best, [p["p"] for p in pairs if p["line"] == best]
    return {"line": line, "p": float(np.median(at)), "n_books": len(books), "as_of_utc": t}


def last_references(quotes: pd.DataFrame, kickoffs: dict, min_books: int = 2) -> pd.DataFrame:
    """One row per (game, market): the reference from the latest snapshot before kickoff that
    has enough books. kickoffs: {game_id: UTC Timestamp}."""
    cols = ["game_id", "market", "line", "p", "n_books", "as_of_utc", "snapshot"]
    if quotes.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (gid, mk), g in quotes.groupby(["game_id", "market"]):
        kick = kickoffs.get(gid)
        if kick is None or pd.isna(kick):
            continue
        # latest snapshot first; a snapshot counts only if all its quotes for this game precede kickoff
        order = g.groupby("snapshot")["odds_time_utc"].max().sort_values(ascending=False)
        for snap, newest in order.items():
            if newest >= kick:
                continue
            ref = _reference(g[g["snapshot"] == snap], mk, min_books)
            if ref is not None:
                out.append({"game_id": gid, "market": mk, **ref, "snapshot": snap})
                break
    return pd.DataFrame(out, columns=cols)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _shift_prob(p_line: float, line: float, target: float) -> float:
    """P(X > target | no push) given P(X > line) = p_line under a normal with SD_APPROX."""
    p_line = min(max(p_line, 1e-6), 1 - 1e-6)
    z = _ppf(1 - p_line)
    mu = line - z * SD_APPROX
    return 1 - _phi((target - mu) / SD_APPROX)


def _ppf(q: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _phi(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clv(rows: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    """Add clv_points, clv_prob, clv_method and closing-reference columns to priced sides
    (columns: game_id, market, side, point, market_prob)."""
    if rows.empty:
        return rows.assign(clv_points=np.nan, clv_prob=np.nan, clv_method="")
    r = rows.merge(refs.rename(columns={"line": "close_line", "p": "close_p", "n_books": "close_books",
                                        "as_of_utc": "close_as_of", "snapshot": "close_snapshot"}),
                   on=["game_id", "market"], how="left")
    pts, prob, how = [], [], []
    for x in r.itertuples(index=False):
        if pd.isna(x.close_p):
            pts.append(np.nan); prob.append(np.nan); how.append("no pre-kickoff reference in snapshots")
            continue
        if x.market == "ml":
            p_side = x.close_p if x.side == "home" else 1 - x.close_p
            pts.append(np.nan); how.append("exact")
        else:
            bet_line = (-x.point if x.side == "home" else x.point) if x.market == "spread" else x.point
            if x.market == "spread":
                pts.append((x.close_line - bet_line) if x.side == "home" else (bet_line - x.close_line))
            else:
                pts.append((x.close_line - bet_line) if x.side == "over" else (bet_line - x.close_line))
            if bet_line == x.close_line:
                p_over, how_ = x.close_p, "exact"
            else:
                p_over, how_ = _shift_prob(x.close_p, x.close_line, bet_line), "normal approximation (line moved)"
            p_side = p_over if x.side in ("home", "over") else 1 - p_over
            how.append(how_)
        prob.append(p_side - x.market_prob if pd.notna(x.market_prob) else np.nan)
    return r.assign(clv_points=pts, clv_prob=prob, clv_method=how)


def summary(rows: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Mean CLV by market with game-clustered bootstrap 95% CIs."""
    out = []
    for mk, g in rows.groupby("market"):
        r = {"market": mk, "sides": len(g), "with reference": int(g["clv_prob"].notna().sum())}
        for col, name in (("clv_points", "points"), ("clv_prob", "probability")):
            x = g.dropna(subset=[col])
            if len(x) >= 2:
                est, lo, hi = metrics.cluster_bootstrap(x, "game_id", col, n=n)
                r[f"mean CLV {name}"], r[f"95% CI ({name})"] = est, f"{lo:+.4f} to {hi:+.4f}"
            else:
                r[f"mean CLV {name}"], r[f"95% CI ({name})"] = (x[col].mean() if len(x) else np.nan), "n/a"
        out.append(r)
    return pd.DataFrame(out)


def kickoff_map(games: pd.DataFrame) -> dict:
    from .timeutil import kickoff_utc
    out = {}
    for g in games.itertuples(index=False):
        try:
            out[g.game_id] = kickoff_utc(g.gameday, g.gametime)
        except (ValueError, TypeError):
            out[g.game_id] = pd.NaT
    return out


__all__ = ["snapshot_quotes", "last_references", "clv", "summary", "kickoff_map", "parse_utc"]
