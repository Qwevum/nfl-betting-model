"""OL1 machinery (nflmodel/openline.py) on SYNTHETIC data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import openline as ol  # noqa: E402
from test_forecast_uncertainty import FEAT, POINT  # noqa: E402


def synth_lines(feat, seed=0):
    rng = np.random.default_rng(seed)
    close = feat["spread_line"].to_numpy(float)
    open_ = np.round((close - 0.2 * feat["f_mrtg"].to_numpy(float) * 3 + rng.normal(0, 0.5, len(feat))) * 2) / 2
    return pd.DataFrame({"game_id": feat["game_id"], "season": feat["season"], "open_line": open_, "close_line": close,
                         "home_line_odds_open": -108.0, "away_line_odds_open": -112.0, "notes": None})


class Pieces(unittest.TestCase):
    def test_crosses_key(self):
        self.assertTrue(ol.crosses_key(2.5, 3.0))       # reaches 3
        self.assertFalse(ol.crosses_key(3.0, 3.5))      # leaves 3: does not cross it
        self.assertTrue(ol.crosses_key(-6.5, -7.5))     # through -7
        self.assertFalse(ol.crosses_key(4.0, 6.5))
        self.assertTrue(ol.crosses_key(3.5, 2.5))       # downward through 3

    def test_fit_move_recovers_slope(self):
        rng = np.random.default_rng(1)
        g = rng.normal(0, 3, 500)
        rows = pd.DataFrame({"g": g, "open": 0.0, "close": 0.2 * g + 0.1 + rng.normal(0, 0.3, 500)})
        a, b = ol.fit_move(rows)
        self.assertAlmostEqual(b, 0.2, delta=0.02)
        self.assertAlmostEqual(a, 0.1, delta=0.05)

    def test_move_params_use_earlier_seasons_only(self):
        rows = pd.DataFrame({"season": np.repeat([2014, 2015, 2016], 100), "g": np.tile(np.linspace(-5, 5, 100), 3),
                             "open": 0.0})
        rows["close"] = rows["g"] * np.where(rows["season"] == 2016, 1.0, 0.2)
        m = ol.move_params(rows, [2015, 2016, 2017])
        self.assertAlmostEqual(m[2015][1], 0.2)
        self.assertAlmostEqual(m[2016][1], 0.2)         # 2016's own (different) moves are not used for 2016
        self.assertGreater(m[2017][1], 0.2)
        self.assertNotIn(2014, ol.move_params(rows, [2014]))

    def test_choose_threshold(self):
        by = {2.0: {"bets": 300, "roi_110": 0.01}, 3.0: {"bets": 150, "roi_110": 0.03},
              4.0: {"bets": 60, "roi_110": 0.20}}
        self.assertEqual(ol.choose_threshold(by), 3.0)                 # 4 has too few bets
        by[2.0]["roi_110"] = 0.03
        self.assertEqual(ol.choose_threshold(by), 3.0)                 # tie -> larger T
        self.assertIsNone(ol.choose_threshold({2.0: {"bets": 10, "roi_110": 0.5}}))


class Bets(unittest.TestCase):
    MODELS = {2018: POINT}

    def rows(self):
        return pd.DataFrame([
            {"game_id": "a", "season": 2018, "result": 7.0, "open": 3.0, "close": 4.0, "own_line": 8.0,
             "open_price_home": -105.0, "open_price_away": -115.0, "home_qb_delta": 0.0, "away_qb_delta": 0.0},
            {"game_id": "b", "season": 2018, "result": 3.0, "open": 3.0, "close": 2.0, "own_line": -1.0,
             "open_price_home": np.nan, "open_price_away": np.nan, "home_qb_delta": 0.1, "away_qb_delta": 0.0},
            {"game_id": "c", "season": 2018, "result": -10.0, "open": -2.5, "close": -2.5, "own_line": -3.5,
             "open_price_home": -110.0, "open_price_away": -110.0, "home_qb_delta": 0.0, "away_qb_delta": 0.0},
        ]).assign(g=lambda d: d["own_line"] - d["open"])

    def test_grading_prices_and_clv(self):
        b = ol.bets(self.rows(), 3.0, {2018: (0.0, 0.0)}, self.MODELS).set_index("game_id")
        self.assertEqual(set(b.index), {"a", "b"})                     # c: |g| = 1 < 3
        self.assertEqual((b.loc["a", "side"], b.loc["a", "result"]), ("home", "W"))
        self.assertAlmostEqual(b.loc["a", "profit_110"], 100 / 110)
        self.assertAlmostEqual(b.loc["a", "profit_105"], 100 / 105)
        self.assertAlmostEqual(b.loc["a", "profit_open"], 100 / 105)
        self.assertEqual(b.loc["a", "clv_points"], 1.0)
        self.assertEqual((b.loc["b", "side"], b.loc["b", "result"]), ("away", "P"))   # 3 = open: push
        self.assertEqual(b.loc["b", "profit_110"], 0.0)
        self.assertTrue(np.isnan(b.loc["b", "profit_open"]))           # no file price
        self.assertTrue(bool(b.loc["a", "usual_qbs"]) and not bool(b.loc["b", "usual_qbs"]))

    def test_predicted_cover_follows_expected_move(self):
        flat = ol.bets(self.rows(), 3.0, {2018: (0.0, 0.0)}, self.MODELS).set_index("game_id")
        self.assertAlmostEqual(flat.loc["a", "p_cover_pred"], 0.5, places=3)          # no expected move
        moving = ol.bets(self.rows(), 3.0, {2018: (0.0, 0.3)}, self.MODELS).set_index("game_id")
        self.assertGreater(moving.loc["a", "p_cover_pred"], 0.5)        # close expected 1.5 pts its way
        self.assertGreater(moving.loc["b", "p_cover_pred"], 0.5)
        self.assertFalse(moving.loc["a", "crosses_key"])               # 3 -> 4.5 leaves 3; crosses nothing
        self.assertFalse(moving.loc["b", "crosses_key"])               # 3 -> 1.8: starts on 3, crosses nothing
        self.assertGreater(moving.loc["a", "p_push_pred"], 0.0)        # whole-number open line can push

    def test_no_move_parameters_still_graded(self):
        b = ol.bets(self.rows(), 3.0, {}, self.MODELS).set_index("game_id")
        self.assertEqual(set(b.index), {"a", "b"})
        self.assertTrue(b["p_cover_pred"].isna().all())
        self.assertTrue(b["crosses_key"].isna().all())

    def test_summary(self):
        b = ol.bets(self.rows(), 0.5, {2018: (0.0, 0.2)}, self.MODELS)
        s = ol.summarize(b, n=200)
        self.assertEqual((s["bets"], s["W"], s["L"], s["P"]), (3, 2, 0, 1))
        self.assertEqual(s["cover"], 1.0)
        self.assertEqual(ol.summarize(b.iloc[0:0]), {"bets": 0})


class Rows(unittest.TestCase):
    def test_walk_forward_rows_and_exclusions(self):
        lines = synth_lines(FEAT)
        lines.loc[lines["season"] == 2017, "notes"] = "Data supply issue this week"
        rows, models = ol.season_rows(FEAT, lines, [2015, 2016, 2017, 2018])
        self.assertNotIn(2017, set(rows["season"]))
        self.assertEqual(set(models), {2016, 2018})     # 2015: only 3 training seasons (needs 4); 2017 excluded
        changed = FEAT.copy()
        changed.loc[changed["season"] == 2018, "result"] += 20
        r2, _ = ol.season_rows(changed, lines, [2018])
        np.testing.assert_allclose(rows[rows["season"] == 2018]["own_line"].to_numpy(), r2["own_line"].to_numpy())


if __name__ == "__main__":
    unittest.main()
