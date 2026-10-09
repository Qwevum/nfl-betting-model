"""Uncertainty of the EV of each priced side (betting assessment, separate from forecasts).

For every bootstrap replicate the side is re-priced exactly as decide.decide_game priced it:
the same book, line and FIXED offered price, the same leave-one-book-out market reference
(held fixed: intervals are conditional on the observed market snapshot), with the replicate's
model. EV is computed per replicate from that replicate's JOINT (win, push) probabilities, so
win/push dependence is kept and no marginal bounds are combined. The interval is
EV_hat +/- z * SD(EV replicates), centered on the point model's EV (the decision's EV).

Moneyline rows keep the existing convention: P(win | game not tied), push 0. A tie refunds a
two-way moneyline at US books, so the exact EV is (1 - P(tie)) times this value; P(tie) is
shown in the game forecast (about 0.3-0.6% in the regular season, 0 in the postseason).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import uncertainty as unc
from .decide import make_pricer
from .odds import ev

EV_COLS = ["p_lose", "p_win_lo", "p_win_hi", "ev_lo", "ev_hi", "ev_sd", "ev_replicates", "unc_level",
           "unc_status"]


def _refs_for(refs_by_game: dict | None, game_id, book):
    """Same reference selection as decide.decide_game.pricer_for(book)."""
    if refs_by_game is None:
        return None
    rg = refs_by_game.get(game_id) or {}
    return {m: rg.get((m, book)) for m in ("spread", "ml", "total")} if rg else None


def ev_uncertainty(rows: pd.DataFrame, day: pd.DataFrame, replicates: list, refs_by_game: dict | None,
                   settings, level: float) -> pd.DataFrame:
    """rows: decide_game records (one per market/side). day: the slate rows the point model priced.
    refs_by_game: {game_id: {(market, excluded_book): Reference}} or None (consensus-only replay)."""
    out = rows.copy()
    out["p_lose"] = 1 - out["p_win"] - out["p_push"]
    if out.empty:
        for c in EV_COLS:
            out[c] = pd.Series(dtype=float)
        return out
    out["unc_level"] = level
    out["ev_replicates"] = len(replicates)
    if len(replicates) < 2:
        for c in ("p_win_lo", "p_win_hi", "ev_lo", "ev_hi", "ev_sd"):
            out[c] = np.nan
        out["unc_status"] = "not computed (uncertainty off)"
        return out
    by_game = [{g.game_id: g for g in m.predict(day).itertuples(index=False)} for m in replicates]
    evs = np.full((len(replicates), len(out)), np.nan)
    pws = np.full_like(evs, np.nan)
    pricers: dict = {}
    for j, r in enumerate(out.itertuples(index=False)):
        refs = _refs_for(refs_by_game, r.game_id, r.book)
        for b, (m, rows_b) in enumerate(zip(replicates, by_game)):
            g = rows_b.get(r.game_id)
            if g is None:
                continue
            key = (b, r.game_id, r.book)
            if key not in pricers:
                pricers[key] = make_pricer(m, g, settings, refs=refs)
            pw, pp = pricers[key].probs(r.market, r.side, r.point)
            pws[b, j] = pw
            evs[b, j] = ev(pw, pp, r.price)
    lo, hi, sd, plo, phi, status = [], [], [], [], [], []
    for j, r in enumerate(out.itertuples(index=False)):
        a, b_, s = unc.linear_interval(r.ev, evs[:, j], level)
        p_lo, p_hi, _ = unc.prob_interval(r.p_win, pws[:, j], level)
        lo.append(a); hi.append(b_); sd.append(s); plo.append(p_lo); phi.append(p_hi)
        status.append("ok" if np.isfinite(s) else "unavailable: replicates could not price this side")
    out["ev_lo"], out["ev_hi"], out["ev_sd"] = lo, hi, sd
    out["p_win_lo"], out["p_win_hi"] = plo, phi
    out["unc_status"] = status
    return out


def apply_lower_bound_rule(rows: pd.DataFrame, settings) -> pd.DataFrame:
    """EXPERIMENTAL (settings.experimental_ev_lower_bound_rule, off by default): a side that passed
    every existing rule also needs an EV interval lower bound above 0. It can only turn a
    candidate into NO BET; it never relaxes a threshold."""
    if not settings.experimental_ev_lower_bound_rule or rows.empty:
        return rows
    out = rows.copy()
    cand = out["decision"] != "NO BET"
    fail = cand & ~(out["ev_lo"] > 0)
    for i in out.index[fail]:
        lo = out.at[i, "ev_lo"]
        why = ("experimental rule: EV lower bound unavailable" if pd.isna(lo)
               else f"experimental rule: EV lower bound {lo:+.1%} not above 0")
        out.at[i, "reasons"] = " | ".join([x for x in [out.at[i, "reasons"], why] if x])
        out.at[i, "decision"] = "NO BET"
        out.at[i, "stake_units"] = 0.0
    return out


__all__ = ["ev_uncertainty", "apply_lower_bound_rule", "EV_COLS"]
