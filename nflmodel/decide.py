"""Bet / no-bet decisions with the evidence behind them.

For each game this module assembles:
  * verified facts      - things read directly from a dated source (schedule, injury
                          report, sportsbook prices), each tagged with its source
  * assumptions         - things the estimate relies on that are not confirmed
  * missing information - inputs that matter but are unavailable or stale
and for each market/side:
  * model probability, the price's implied probability, the market's no-vig
    probability, EV at the actual price, and how EV moves if the model is off
  * a decision, with the reasons:
      BET                    executable: validated, timestamped book price, live
                             reference without that book, nothing left to confirm
      BET IF CONFIRMED       executable price, but something must be confirmed first
                             (e.g. a starter whose availability is unknown)
      BET IF PRICE AVAILABLE conditional: price unverified (untimed consensus, or no
                             live reference without that book)
      NO BET

No-bet rules (any one is enough):
  1. EV at the available price is not above the threshold (default 2%).
  2. The edge is fragile: EV is not positive if the true line is 0.5 point worse.
  3. The projected starting QB is listed Out/Doubtful/Questionable, or not listed.
     (Unknown availability, e.g. no injury report yet, only makes a bet conditional.)
  4. The model's own line differs from the market by 4+ points (unexplained).
  5. Totals in outdoor or unknown-roof stadiums without a weather forecast.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Settings
from .market import _invert
from .model import FittedModel, novig_first
from .odds import ev, implied_probability, kelly

UNSURE = {"Out", "Doubtful", "Questionable"}
BOOK_SOURCES_WITH_TIME = {"odds_api", "manual"}


@dataclass
class GameContext:
    facts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    block: dict[str, list[str]] = field(default_factory=lambda: {"spread": [], "ml": [], "total": []})
    # Conditions that must be confirmed before a bet becomes executable (e.g. a starter
    # whose availability is unknown). They downgrade a bet to "BET IF ...", never up.
    conditions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- context

def _injury_lines(inj: pd.DataFrame, team: str, week: int) -> tuple[list[str], int | None, bool]:
    """(lines, report week, game statuses issued?) for a team's latest report up to `week`."""
    t = inj[(inj["team"] == team) & (inj["week"] <= week)] if len(inj) else inj
    if t is None or t.empty:
        return [], None, False
    wk = int(t["week"].max())
    cur = t[t["week"] == wk]
    issued = cur["report_status"].notna().any()
    if issued:
        sel = cur[cur["report_status"].isin(UNSURE)]
        lines = [f"{r.full_name} ({r.position}) {r.report_status}"
                 + (f" - {r.report_primary_injury}" if pd.notna(r.report_primary_injury) else "")
                 for r in sel.itertuples(index=False)]
    else:
        sel = cur[cur["practice_status"].fillna("").str.startswith("Did Not")]
        lines = [f"{r.full_name} ({r.position}) did not practice"
                 + (f" - {r.practice_primary_injury}" if pd.notna(r.practice_primary_injury) else "")
                 for r in sel.itertuples(index=False)]
    return lines, wk, bool(issued)


def starter_status(inj: pd.DataFrame, feed_ok: bool, qb_id, team: str, week: int) -> tuple[str, str]:
    """Availability of a projected starter from the injury feed.

    Returns (state, detail) with state one of:
      available     - final game statuses for this team/week are issued and the QB
                      has no Out/Doubtful/Questionable designation
      ruled_out     - listed Out or Doubtful
      questionable  - listed Questionable
      unknown       - feed unavailable, no report for this team/week, or game
                      statuses not issued yet. Missing data is NOT treated as healthy.
    """
    if not isinstance(qb_id, str):
        return "unknown", "projected starter not listed"
    if not feed_ok or inj is None or inj.empty:
        return "unknown", "injury feed unavailable"
    team_wk = inj[(inj["team"] == team) & (inj["week"] == week)]
    if team_wk.empty:
        return "unknown", f"no week {week} injury report for {team} in the feed yet"
    me = team_wk[team_wk["gsis_id"] == qb_id]
    status = me["report_status"].dropna().iloc[-1] if len(me) and me["report_status"].notna().any() else None
    practice = me["practice_status"].dropna().iloc[-1] if len(me) and me["practice_status"].notna().any() else None
    if status in ("Out", "Doubtful"):
        return "ruled_out", f"listed {status}"
    if status == "Questionable":
        return "questionable", f"listed Questionable (practice: {practice or 'n/a'})"
    if team_wk["report_status"].notna().any():
        return "available", ("no game-status designation on the final report"
                             + (f" (practice: {practice})" if practice else ""))
    return "unknown", ("game statuses not issued yet"
                       + (f"; practice: {practice}" if practice else "; not on the practice report"))


def build_context(g, inj: pd.DataFrame, sources: dict, coefs: dict, k_spread: float,
                  overrides: dict, weather: dict, settings: Settings,
                  injury_feed_ok: bool | None = None) -> GameContext:
    """g: one prediction row (namedtuple) for an upcoming game."""
    c = GameContext()
    if injury_feed_ok is None:
        injury_feed_ok = inj is not None and not inj.empty and "error" not in sources.get("injury_reports", {})
    if not injury_feed_ok:
        c.missing.append("Injury feed unavailable: availability of every player is unknown")
    sched = sources.get("schedule_scores_lines", {})
    sched_tag = f"[nflverse schedule, retrieved {sched.get('retrieved_utc', '?')}]"
    inj_src = sources.get("injury_reports", {})
    inj_tag = (f"[NFL injury report via nflverse, file updated {inj_src.get('last_modified', 'unknown')}, "
               f"retrieved {inj_src.get('retrieved_utc', '?')}]")

    kick = f"{pd.Timestamp(g.gameday).strftime('%a %b %d %Y')} {g.gametime} ET"
    c.facts.append(f"Kickoff {kick}, {g.stadium if isinstance(g.stadium, str) else 'stadium not listed'}"
                   f"{' (neutral site)' if str(g.location) == 'Neutral' else ''} {sched_tag}")
    roof = g.roof if isinstance(g.roof, str) else None
    c.facts.append(f"Roof: {roof or 'not listed'}; rest days {g.away_team} {g.away_rest:g}, "
                   f"{g.home_team} {g.home_rest:g} {sched_tag}")

    week = int(g.week)
    for side, team, qb_id, qb_name, delta, rating in (
            ("away", g.away_team, g.away_qb_id, g.away_qb_name, g.away_qb_delta, g.away_qb_rating),
            ("home", g.home_team, g.home_qb_id, g.home_qb_name, g.home_qb_delta, g.home_qb_rating)):
        ov = overrides.get((g.game_id, team))
        if ov:
            c.assumptions.append(f"{team} starter set to {ov['qb_name']} by qb_overrides.csv "
                                 f"(source: {ov.get('source') or 'not given'}, confirmed {ov['confirmed_utc']})")
        if not isinstance(qb_id, str) and not ov:
            c.missing.append(f"{team} projected starting QB not listed")
            for m in c.block:
                c.block[m].append(f"{team} starting QB unknown")
            continue
        if not ov:
            c.facts.append(f"{team} projected starter: {qb_name} {sched_tag} (projection, not official until inactives)")
        state, detail = starter_status(inj, injury_feed_ok, qb_id, team, week)
        if state == "unknown":
            c.missing.append(f"{qb_name} ({team}) availability unknown: {detail}")
        else:
            c.facts.append(f"{qb_name} availability: {detail} {inj_tag}")
        if ov:
            pass  # a confirmed starter from qb_overrides.csv settles the starter question
        elif state in ("ruled_out", "questionable"):
            for m in c.block:
                c.block[m].append(f"{team} projected starter {qb_name} is {detail.replace('listed ', '')}")
        elif state == "unknown":
            c.conditions.append(f"{qb_name} confirmed as {team}'s starter")
        if abs(delta) >= 0.02:
            pts = delta * coefs.get("f_qb", 0.0)
            c.assumptions.append(
                f"{qb_name} starts for {team}; his rating differs from the QB play {team}'s team ratings "
                f"reflect, worth {pts:+.1f} pts to {team} in the model's own line")
        else:
            c.assumptions.append(f"{qb_name} starts for {team} (same QB the team ratings reflect)")

    for team in (g.away_team, g.home_team):
        lines, wk, issued = _injury_lines(inj, team, week)
        if wk is None:
            c.missing.append(f"No injury report found for {team}")
        else:
            stale = "" if wk == week else f" (STALE: latest report is week {wk})"
            if issued:
                body = "; ".join(lines) if lines else "no players listed Out/Doubtful/Questionable"
            else:
                body = ("practice report only, game statuses not yet issued"
                        + (". Did not practice: " + "; ".join(lines) if lines else ""))
            c.facts.append(f"{team} injury report week {wk}{stale}: {body} {inj_tag}")
            if wk != week:
                c.missing.append(f"{team} week {week} injury report not yet in data")
            elif not issued:
                c.missing.append(f"{team} week {week} game statuses (Out/Doubtful/Questionable) not yet issued")
    c.assumptions.append("Non-QB injuries are not modeled; the market line is assumed to price them "
                         "(the fair line is mostly the market line)")

    outdoor = roof not in ("dome", "closed")
    if g.game_id in weather:
        w = weather[g.game_id]
        c.facts.append(f"Forecast wind {w['wind_mph']} mph, temp {w['temp_f']} F "
                       f"[{w.get('source') or 'weather_manual.csv'}, forecast issued {w['forecast_utc']}]")
    elif outdoor:
        c.missing.append("Weather forecast (wind matters for totals) - "
                         + ("roof status not listed" if roof is None else "outdoor stadium"))
        c.block["total"].append("no weather forecast for an outdoor/unknown-roof game")

    gap = abs(g.model_margin - g.spread_line) if pd.notna(g.spread_line) else 0.0
    if gap >= settings.gap_points:
        msg = (f"model's own line differs from market by {gap:.1f} pts "
               f"(>= gap_points {settings.gap_points:g}; unexplained news?)")
        c.block["spread"].append(msg); c.block["ml"].append(msg)
    tgap = abs(g.model_total - g.total_line) if pd.notna(g.total_line) else 0.0
    if tgap >= settings.gap_points:
        c.block["total"].append(f"model's own total differs from market by {tgap:.1f} pts "
                                f"(>= gap_points {settings.gap_points:g})")

    if week <= 4:
        c.risks.append(f"Week {week}: current-season ratings rest on few games")
    c.assumptions.append(f"Home field worth {coefs.get('f_hfa', 0):.1f} pts in the model's own line; "
                         f"only {k_spread:.0%} of the model's disagreement with the spread is kept "
                         "(share that held up out of sample)")
    return c


# ---------------------------------------------------------------- probabilities

class GamePricer:
    """Probabilities for any line/price in one game.

    The estimate starts from the market: the consensus no-vig probability at the
    consensus line, moved by the model only as much as out-of-sample calibration
    says the model's disagreement is worth (model.spread_cal / total_cal / ml_cal).
    For a different line at another book (e.g. +3.5 vs +3) the key-number outcome
    distribution supplies the difference in probability, including pushes.
    """

    def __init__(self, model: FittedModel, g, key_numbers: bool = True, refs: dict | None = None):
        """refs: {market: market.Reference} built from live quotes (leave-one-book-out).
        Markets without a live reference fall back to the untimed nflverse consensus in g."""
        self.m, self.g = model, g
        self.mdist, self.tdist = model.margin_dist, model.total_dist
        if not key_numbers:
            self.mdist = copy.copy(self.mdist); self.mdist.w = np.ones_like(self.mdist.w)
            self.tdist = copy.copy(self.tdist); self.tdist.w = np.ones_like(self.tdist.w)
        refs = refs or {}
        self.refs = refs
        self.edge_m = (g.blend_margin - g.spread_line) if pd.notna(g.blend_margin) else 0.0
        self.edge_t = (g.blend_total - g.total_line) if pd.notna(g.blend_total) else 0.0
        self.has_spread_mkt = pd.notna(g.spread_line)
        self.has_total_mkt = pd.notna(g.total_line)
        if refs.get("spread") is not None and self.has_spread_mkt:
            self.p_spread_mkt = self._cond_over(model.margin_dist, refs["spread"].mu, g.spread_line)[0]
        else:
            self.p_spread_mkt = float(novig_first(g.home_spread_odds, g.away_spread_odds)) if self.has_spread_mkt else np.nan
        if refs.get("total") is not None and self.has_total_mkt:
            self.p_total_mkt = self._cond_over(model.total_dist, refs["total"].mu, g.total_line)[0]
        else:
            self.p_total_mkt = float(novig_first(g.over_odds, g.under_odds)) if self.has_total_mkt else np.nan
        if refs.get("ml") is not None:
            self.has_ml_mkt, self.p_ml_mkt = True, refs["ml"].p
        else:
            self.has_ml_mkt = pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline)
            self.p_ml_mkt = float(novig_first(g.home_moneyline, g.away_moneyline)) if self.has_ml_mkt else np.nan

    def reference_prob(self, market: str, side: str, point) -> float:
        """The market's own no-vig probability for this exact bet (no model input), or NaN."""
        if market == "ml":
            p = self.p_ml_mkt
            return p if side == "home" else 1 - p
        dist = self.m.margin_dist if market == "spread" else self.m.total_dist
        ref = self.refs.get(market)
        if ref is not None:
            mu = ref.mu
        else:
            line = self.g.spread_line if market == "spread" else self.g.total_line
            p_line = self.p_spread_mkt if market == "spread" else self.p_total_mkt
            if pd.isna(line):
                return np.nan
            mu = _invert(dist, line, p_line, *((-60, 60) if market == "spread" else (5, 125)))
        t = (-point if side == "home" else point) if market == "spread" else point
        cond, _ = self._cond_over(dist, mu, t)
        return cond if side in ("home", "over") else 1 - cond

    @staticmethod
    def _cond_over(dist, mu, t):
        over, push, under = dist.prob_over(mu, t)
        return over / (over + under), push

    def probs(self, market: str, side: str, point, shift: float = 0.0, edge_mult: float = 1.0):
        """(p_win, p_push). shift moves the true margin/total toward the home/over side."""
        g, m = self.g, self.m
        if market == "ml":
            if self.has_ml_mkt:
                base = m.anchored(m.ml_cal, self.p_ml_mkt, self.edge_m, edge_mult)
                p = base + (m.win_prob(g.fair_margin + shift) - m.win_prob(g.fair_margin))
            else:
                p = m.win_prob(g.fair_margin + shift)
            p = float(np.clip(p, 0.001, 0.999))
            return (p, 0.0) if side == "home" else (1 - p, 0.0)
        if market == "spread":
            dist, mu, ref, t = self.mdist, g.fair_margin, g.spread_line, (-point if side == "home" else point)
            cal, p_mkt, edge, has = m.spread_cal, self.p_spread_mkt, self.edge_m, self.has_spread_mkt
            home_or_over = side == "home"
        else:
            dist, mu, ref, t = self.tdist, g.fair_total, g.total_line, point
            cal, p_mkt, edge, has = m.total_cal, self.p_total_mkt, self.edge_t, self.has_total_mkt
            home_or_over = side == "over"
        cond_t, push_t = self._cond_over(dist, mu + shift, t)
        if has:
            cond_ref, _ = self._cond_over(dist, mu, ref)
            cond_t = m.anchored(cal, p_mkt, edge, edge_mult) + (cond_t - cond_ref)
        cond_t = float(np.clip(cond_t, 0.001, 0.999))
        p = cond_t if home_or_over else 1 - cond_t
        return p * (1 - push_t), push_t


def _against(market: str, side: str) -> float:
    """Direction that makes the bet worse: -1 lowers margin/total, +1 raises it."""
    return -1.0 if side in ("home", "over") else 1.0


def price_for_edge(p_win: float, p_push: float, edge: float) -> float | None:
    """Worst American price at which EV still equals `edge`."""
    p_lose = 1 - p_win - p_push
    if p_win <= 0:
        return None
    b = (edge + p_lose) / p_win          # needed decimal - 1
    if b <= 0:
        return None
    return round(b * 100) if b >= 1 else round(-100 / b)


# ---------------------------------------------------------------- decisions

def decide_game(model: FittedModel, g, offers: pd.DataFrame, ctx: GameContext, settings: Settings,
                refs: dict | None = None) -> list[dict]:
    """One record per market and side, best available price for that side.

    refs: {(market, excluded_book): Reference|None} from market.references_for_game.
    Each offer is priced against the reference built WITHOUT its own book. Without
    refs (historical replay) the untimed consensus in `g` is the reference.
    """
    rows = []
    pricers: dict = {}
    min_edge, min_ref_books = settings.min_edge, settings.min_reference_books

    def pricer_for(book, plain=False):
        key = (book, plain)
        if key not in pricers:
            r = {m: refs.get((m, book)) for m in ("spread", "ml", "total")} if refs else None
            pricers[key] = GamePricer(model, g, key_numbers=not plain, refs=r)
        return pricers[key]

    for (market, side), o in offers.groupby(["market", "side"]):
        o = o.dropna(subset=["price"])
        if market != "ml":
            o = o.dropna(subset=["point"])
        if refs is not None and (o["source"] != "consensus").any():
            o = o[o["source"] != "consensus"]   # live quotes exist: untimed consensus is not an offer
        if o.empty:
            continue
        scored = []
        for r in o.itertuples(index=False):
            pw, pp = pricer_for(r.book).probs(market, side, r.point)
            scored.append((ev(pw, pp, r.price), r, pw, pp))
        e, best, pw, pp = max(scored, key=lambda x: x[0])
        point = best.point
        d = _against(market, side)
        pricer, plain = pricer_for(best.book), pricer_for(best.book, plain=True)
        ref = refs.get((market, best.book)) if refs else None

        def ev_at(**kw):
            a, b = (plain if kw.pop("plain", False) else pricer).probs(market, side, point, **kw)
            return ev(a, b, best.price)

        sens = {
            "fair 1.0 worse": ev_at(shift=d * 1.0),
            "fair 0.5 worse": ev_at(shift=d * 0.5),
            "fair 0.5 better": ev_at(shift=-d * 0.5),
            "market only": ev_at(edge_mult=0.0),
            "double model weight": ev_at(edge_mult=2.0),
            "no key-number shape": ev_at(plain=True),
        }
        mkt_p = pricer.reference_prob(market, side, point)
        if ref is not None:
            ref_desc = (f"live: {ref.n_books} other books ({', '.join(ref.books)}), quotes "
                        f"{ref.oldest_utc:%H:%M}-{ref.newest_utc:%H:%M}Z, spread across books "
                        f"{ref.spread_of_books:.2f}{' pts' if market != 'ml' else ''}")
        elif refs is not None:
            ref_desc = f"no live reference (fewer than {min_ref_books} other books with two-sided quotes); untimed consensus used"
        else:
            ref_desc = "nflverse consensus (untimed)"
        decided = 1 - pp
        rec = {
            "game_id": g.game_id, "market": market, "side": side,
            "team": {"home": g.home_team, "away": g.away_team}.get(side, side.capitalize()),
            "book": best.book, "point": point, "price": best.price,
            "price_source": getattr(best, "source", "consensus"),
            "odds_time": getattr(best, "odds_time", None),
            "n_books": o["book"].nunique(),
            "all_prices": "; ".join(sorted(
                f"{r.book} {'' if pd.isna(r.point) else f'{r.point:+g} ' if market == 'spread' else f'{r.point:g} '}{int(r.price):+d}"
                for r in o.itertuples(index=False))),
            "p_win": pw, "p_push": pp, "model_prob": pw / decided if decided > 0 else np.nan,
            "implied": implied_probability(best.price), "market_prob": mkt_p, "reference": ref_desc,
            "reference_live": ref is not None,
            "ev": e, "kelly": kelly(pw, pp, best.price),
            "min_price": price_for_edge(pw, pp, min_edge),
            **{f"ev[{k}]": v for k, v in sens.items()},
        }
        reasons = []
        if e <= min_edge:
            reasons.append(f"EV {e:+.1%} not above {min_edge:.0%}")
        elif sens["fair 0.5 worse"] <= 0:
            reasons.append(f"too sensitive: EV {sens['fair 0.5 worse']:+.1%} if fair line 0.5 pt worse")
        reasons += ctx.block.get(market, [])
        if reasons:
            rec["decision"] = "NO BET"
        elif rec["price_source"] in BOOK_SOURCES_WITH_TIME and ref is not None and not ctx.conditions:
            rec["decision"] = "BET"
        elif ctx.conditions and rec["price_source"] in BOOK_SOURCES_WITH_TIME and ref is not None:
            rec["decision"] = "BET IF CONFIRMED"
            reasons.append("only if: " + "; ".join(ctx.conditions))
        else:
            rec["decision"] = "BET IF PRICE AVAILABLE"
            why = ("only an untimed consensus price was seen" if rec["price_source"] not in BOOK_SOURCES_WITH_TIME
                   else "no live market reference without this book")
            reasons.append(f"{why}; confirm at your book "
                           f"(still +{min_edge:.0%} EV at {_fmt_price(rec['min_price'])} or better at this line)")
            if ctx.conditions:
                reasons.append("and only if: " + "; ".join(ctx.conditions))
        rec["reasons"] = " | ".join(reasons)
        rec["stake_units"] = stake_units(rec["kelly"], settings) if rec["decision"] != "NO BET" else 0.0

        risks = list(ctx.risks)
        if market == "spread" and (pp > 0.04 or abs(point) in (2.5, 3.5, 6.5, 7.5)):
            risks.append("EV leans on how often games land exactly on 3/7 (key-number model)")
        if abs(sens["market only"] - e) < 0.005:
            risks.append("the edge is from price/line shopping, not from the model disagreeing with the market")
        if sens["market only"] <= 0:
            risks.append("using the market's probability alone the bet is not +EV")
        if sens["no key-number shape"] <= 0:
            risks.append("not +EV without the key-number outcome model")
        rec["risks"] = " | ".join(risks)
        rows.append(rec)
    return rows


def stake_units(full_kelly: float, settings: Settings) -> float:
    """Fractional-Kelly stake in units (1u = 1% of bankroll), capped at max_stake_units."""
    return round(min(full_kelly * settings.kelly_fraction * 100, settings.max_stake_units), 2)


def _fmt_price(p) -> str:
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{int(p):+d}"


def best_per_market(rows: pd.DataFrame) -> pd.DataFrame:
    """The higher-EV side of each game/market (the side that would be bet)."""
    if rows.empty:
        return rows
    return (rows.sort_values("ev", ascending=False)
                .groupby(["game_id", "market"]).head(1).reset_index(drop=True))
