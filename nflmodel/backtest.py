"""Walk-forward backtest against closing lines.

For each test season the model is fit only on earlier seasons, then every game
of the test season is priced with pre-game ratings and bet against the
nflverse closing line at the posted price (-110 when the price is missing).
Closing lines are the hardest benchmark there is; a model that is roughly
break-even here can still win if you bet earlier numbers and shop books.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate
from .model import fit
from .odds import consensus_offers, decimal


def grade(market: str, side: str, point: float, home: float, away: float) -> str:
    margin, total = home - away, home + away
    if market == "spread":
        v = (margin if side == "home" else -margin) + point
    elif market == "ml":
        v = margin if side == "home" else -margin
    else:
        v = (total - point) if side == "over" else (point - total)
    return "W" if v > 0 else ("P" if v == 0 else "L")


def profit(result: str, price: float) -> float:
    return {"W": decimal(price) - 1, "P": 0.0, "L": -1.0}[result]


def run(feat: pd.DataFrame, test_seasons: list[int],
        min_train_seasons: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    all_bets, fits = [], []
    for season in test_seasons:
        train = played[(played["season"] < season) & (played["season"] > first)]  # skip warm-up year
        if train["season"].nunique() < min_train_seasons:
            continue
        model = fit(train)
        test = model.predict(played[played["season"] == season])
        fits.append({
            "season": season, "games": len(test),
            "mae_market": np.nanmean(np.abs(test["result"] - test["spread_line"])),
            "mae_model": np.nanmean(np.abs(test["result"] - test["model_margin"])),
            "mae_blend": np.nanmean(np.abs(test["result"] - test["fair_margin"])),
            "k_spread": model.k_spread, "k_total": model.k_total,
        })
        best = evaluate.evaluate_offers(model, test, consensus_offers(test))
        picks = evaluate.pick_bets(best, test)
        picks = picks.merge(test[["game_id", "season", "week", "home_score", "away_score"]], on="game_id")
        picks["result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                           for r in picks.itertuples(index=False)]
        picks["profit"] = [profit(r, p) for r, p in zip(picks["result"], picks["price"])]
        all_bets.append(picks)
        print(f"  {season}: fit on {train['season'].nunique()} seasons, {len(test)} games")

    fits = pd.DataFrame(fits)
    bets = pd.concat(all_bets, ignore_index=True)
    return fits, bets


def summarize(bets: pd.DataFrame, thresholds=(0.0, 0.02, 0.04, 0.06)) -> pd.DataFrame:
    rows = []
    for market in ("spread", "ml", "total"):
        m = bets[(bets["market"] == market) & (bets["status"] != "check news")]
        for t in thresholds:
            b = m[m["ev"] >= t]
            w, l, p = (b["result"] == "W").sum(), (b["result"] == "L").sum(), (b["result"] == "P").sum()
            n = w + l
            win = w / n if n else np.nan
            ci = 1.96 * np.sqrt(win * (1 - win) / n) if n else np.nan
            rows.append({"market": market, "min_ev": t, "bets": len(b), "W": w, "L": l, "P": p,
                         "win_pct": win, "win_pct_95ci": f"{win - ci:.3f}-{win + ci:.3f}" if n else "",
                         "units": b["profit"].sum(), "roi": b["profit"].sum() / len(b) if len(b) else np.nan})
    return pd.DataFrame(rows)
