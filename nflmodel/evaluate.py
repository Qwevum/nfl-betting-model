"""Turn predictions + offers into win/push/lose probabilities, EV and picks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FittedModel
from .odds import ev, kelly, no_vig

# Minimum expected value (profit per unit staked) to recommend a bet.
MIN_EV = {"spread": 0.02, "ml": 0.03, "total": 0.02}
# Disagreements this large with the market almost always mean news the model
# can't see (QB out, injuries, weather). Flag instead of recommending.
CHECK_NEWS_POINTS = 4.0
KELLY_FRACTION = 0.25     # quarter Kelly
MAX_STAKE_UNITS = 2.0     # 1 unit = 1% of bankroll


def outcome_probs(model: FittedModel, row, market: str, side: str, point: float):
    """(p_win, p_push) for one offer."""
    if market == "spread":
        mu = row.fair_margin
        if side == "home":  # home covers when margin + point > 0  ->  margin > -point
            w, p, _ = model.margin_dist.prob_over(mu, -point)
        else:               # away covers when margin < point
            _, p, w = model.margin_dist.prob_over(mu, point)
        return w, p
    if market == "ml":
        # Calibrated on past seasons; ties (~0.3% of games) are ignored.
        p_home = model.win_prob(row.fair_margin)
        return (p_home, 0.0) if side == "home" else (1 - p_home, 0.0)
    mu = row.fair_total
    over, push, under = model.total_dist.prob_over(mu, point)
    return (over, push) if side == "over" else (under, push)


def evaluate_offers(model: FittedModel, preds: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Score every offer, then keep the best-EV book for each game/market/side."""
    preds = preds.set_index("game_id")
    out = []
    for o in offers.itertuples(index=False):
        if o.game_id not in preds.index or pd.isna(o.price):
            continue
        row = preds.loc[o.game_id]
        if o.market != "ml" and pd.isna(o.point):
            continue
        p_win, p_push = outcome_probs(model, row, o.market, o.side, o.point)
        out.append({**o._asdict(), "p_win": p_win, "p_push": p_push,
                    "ev": ev(p_win, p_push, o.price), "kelly": kelly(p_win, p_push, o.price)})
    scored = pd.DataFrame(out)
    if scored.empty:
        return scored
    best = (scored.sort_values("ev", ascending=False)
                  .groupby(["game_id", "market", "side"]).head(1))
    books = scored.groupby(["game_id", "market", "side"])["book"].nunique().rename("n_books")
    return best.merge(books.reset_index(), on=["game_id", "market", "side"])


def pick_bets(best: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One recommended side per game and market (the higher-EV side)."""
    if best.empty:
        return best
    p = preds.set_index("game_id")
    top = (best.sort_values("ev", ascending=False)
               .groupby(["game_id", "market"]).head(1).reset_index(drop=True))
    top["min_ev"] = top["market"].map(MIN_EV)
    top["stake_units"] = np.minimum(top["kelly"] * KELLY_FRACTION * 100, MAX_STAKE_UNITS).round(2)

    def disagreement(r):
        g = p.loc[r.game_id]
        if r.market == "total":
            return abs(g.model_total - g.total_line) if pd.notna(g.total_line) else 0.0
        return abs(g.model_margin - g.spread_line) if pd.notna(g.spread_line) else 0.0

    top["model_vs_market"] = top.apply(disagreement, axis=1)
    top["status"] = np.where(top["ev"] < top["min_ev"], "pass",
                     np.where(top["model_vs_market"] >= CHECK_NEWS_POINTS, "check news", "BET"))
    return top


def market_no_vig(preds: pd.DataFrame) -> pd.Series:
    """No-vig home win probability from the consensus moneylines."""
    vals = []
    for r in preds.itertuples(index=False):
        if pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
            vals.append(no_vig(r.home_moneyline, r.away_moneyline)[0])
        else:
            vals.append(np.nan)
    return pd.Series(vals, index=preds.index)
