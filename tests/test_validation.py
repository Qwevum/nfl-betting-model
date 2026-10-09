"""Evaluation helpers: identical rows for model and market, re-pricing, clustering."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import validate  # noqa: E402
from nflmodel.model import novig_first  # noqa: E402


def preds(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)], "season": 2020,
        "result": rng.choice([-7, -3, 3, 7, 10], n).astype(float), "spread_line": 2.5,
        "total": rng.choice([37, 41, 47, 51], n).astype(float), "total_line": 44.5,
        "p_model": rng.uniform(0.3, 0.7, n), "p_market": rng.uniform(0.3, 0.7, n),
        "p_ratings_only": 0.5, "p_home_rate": 0.57,
        "p_cover": 0.5, "p_cover_mkt": 0.5, "p_over": 0.5, "p_over_mkt": 0.5})
    return d


class SameRows(unittest.TestCase):
    def test_market_missing_rows_are_dropped_for_every_predictor(self):
        p = preds()
        p.loc[:4, "p_market"] = np.nan              # market unavailable for 5 games
        t = validate.prob_table(p, n_boot=50)
        w = t[t["target"] == "winner"]
        self.assertEqual(set(w["n"]), {35})          # model, market and baselines on the same 35 games

    def test_difference_row_matches_brier_gap(self):
        t = validate.prob_table(preds(), n_boot=50)
        w = t[t["target"] == "winner"].set_index("predictor")
        gap = w.loc["model", "brier"] - w.loc["market no-vig (closing)", "brier"]
        self.assertAlmostEqual(w.loc["model minus market (Brier)", "brier"], gap, places=10)


class Revig(unittest.TestCase):
    def test_reprices_to_target_overround_and_keeps_no_vig(self):
        t = pd.DataFrame({"home_spread_odds": [-105.0, -125.0], "away_spread_odds": [-105.0, 105.0],
                          "over_odds": [-108.0, np.nan], "under_odds": [-102.0, np.nan],
                          "home_moneyline": [-150.0, np.nan], "away_moneyline": [130.0, np.nan]})
        r = validate.revig(t, 0.0476)
        def imp(x):
            x = np.asarray(x, float)
            return np.where(x < 0, -x / (-x + 100), 100 / (x + 100))
        over = imp(r["home_spread_odds"]) + imp(r["away_spread_odds"]) - 1
        self.assertTrue(np.allclose(over, 0.0476, atol=0.003))
        self.assertTrue(np.allclose(novig_first(t["home_spread_odds"], t["away_spread_odds"]),
                                    novig_first(r["home_spread_odds"], r["away_spread_odds"]), atol=0.003))
        self.assertTrue(r.loc[1, ["over_odds", "under_odds"]].isna().all())   # missing pairs stay missing


if __name__ == "__main__":
    unittest.main()
