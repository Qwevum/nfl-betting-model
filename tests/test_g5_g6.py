"""G5 (preseason prior) and G6 (ratings hyperparameter search) machinery, on SYNTHETIC data."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import experiment, ratings, store  # noqa: E402
from test_forecast_uncertainty import FEAT  # noqa: E402


def mini_history():
    """Two seasons, teams A and B. A changes QB between seasons (qa1 -> qa2); B keeps qb1."""
    games, tg, qg = [], [], []
    plan = [(2020, "2020-09-13", "A", "B"), (2020, "2020-09-20", "B", "A"), (2020, "2020-09-27", "A", "B"),
            (2021, "2021-09-12", "A", "B"), (2021, "2021-09-19", "B", "A")]
    for i, (s, d, h, a) in enumerate(plan):
        gid = f"{s}_{i}_{a}_{h}"
        qa = "qa1" if s == 2020 else "qa2"
        hq, aq = (qa, "qb1") if h == "A" else ("qb1", qa)
        games.append({"game_id": gid, "season": s, "game_type": "REG", "week": i + 1, "gameday": d, "gametime": "13:00",
                      "home_team": h, "away_team": a, "home_score": 24, "away_score": 17, "result": 7.0, "total": 41.0,
                      "location": "Home", "home_rest": 7, "away_rest": 7, "roof": "outdoors", "wind": 5.0,
                      "div_game": 1, "home_qb_id": hq, "away_qb_id": aq})
        for team, epa, qb in ((h, 0.1, hq), (a, -0.05, aq)):
            tg.append({"game_id": gid, "team": team, "season": s, "epa": epa, "sr": 0.45, "pass_epa": epa, "plays": 62})
            qg.append({"game_id": gid, "team": team, "qb_id": qb, "dropbacks": 35, "epa": 0.3 if qb == "qa1" else -0.1})
    return pd.DataFrame(games), pd.DataFrame(tg), pd.DataFrame(qg)


G, TG, QG = mini_history()


class G5b(unittest.TestCase):
    def build(self, pts):
        f, _, qbr = ratings.build_features(G, TG, QG, params={"preseason_qb_pts": pts})
        return f.set_index("game_id"), qbr

    def test_off_is_unchanged(self):
        a, _ = self.build(0.0)
        b, _, _ = ratings.build_features(G, TG, QG)
        cols = ratings.FEATURES + ratings.TOTAL_FEATURES
        np.testing.assert_array_equal(a[cols].to_numpy(), b.set_index("game_id")[cols].to_numpy())

    def test_new_season_qb_change_moves_rating_once(self):
        off, _ = self.build(0.0)
        on, qbr = self.build(35.0)
        first21, second21 = "2021_3_B_A", "2021_4_A_B"
        prev = off.loc[[i for i in off.index if i.startswith("2020")]]
        np.testing.assert_array_equal(prev["f_mrtg"], on.loc[prev.index, "f_mrtg"])   # 2020 untouched
        # first 2021 game: A (home) gets the QB-change shift; f_qb carries it no longer
        shift = on.loc[first21, "f_mrtg"] - off.loc[first21, "f_mrtg"]
        self.assertLess(shift, 0)                          # qa2 is rated below qa1
        self.assertNotEqual(off.loc[first21, "home_qb_delta"], 0.0)      # A's change, counted in f_qb before
        self.assertAlmostEqual(on.loc[first21, "home_qb_delta"], 0.0, places=12)   # now in the rating instead
        self.assertAlmostEqual(on.loc[first21, "away_qb_delta"], off.loc[first21, "away_qb_delta"])
        self.assertNotEqual(on.loc[second21, "f_mrtg"], off.loc[second21, "f_mrtg"])   # carried, then updated


class WinTotals(unittest.TestCase):
    def test_priors_reject_late_rows_and_center(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wt.csv"
            pd.DataFrame([{"season": 2021, "team": "A", "win_total": 11.5, "odds": -110, "retrieved_utc": "2021-08-01T00:00:00Z"},
                          {"season": 2021, "team": "B", "win_total": 5.5, "odds": -110, "retrieved_utc": "2021-08-01T00:00:00Z"},
                          {"season": 2021, "team": "C", "win_total": 9.0, "odds": -110, "retrieved_utc": "2021-09-20T00:00:00Z"}]
                         ).to_csv(p, index=False)
            pri, notes = ratings.win_total_priors(p, G)
        self.assertEqual(set(pri), {(2021, "A"), (2021, "B")})
        self.assertAlmostEqual(pri[(2021, "A")] + pri[(2021, "B")], 0.0)
        self.assertGreater(pri[(2021, "A")], 0)
        self.assertEqual(len(notes), 1)

    def test_prior_enters_at_new_season(self):
        f0, _, _ = ratings.build_features(G, TG, QG)
        f1, _, _ = ratings.build_features(G, TG, QG, params={"win_total_weight": 0.5,
                                                             "win_totals": {(2021, "A"): 6.0, (2021, "B"): -6.0}})
        a, b = f0.set_index("game_id"), f1.set_index("game_id")
        self.assertAlmostEqual(b.loc["2020_0_B_A", "f_mrtg"], a.loc["2020_0_B_A", "f_mrtg"])
        self.assertGreater(b.loc["2021_3_B_A", "f_mrtg"], a.loc["2021_3_B_A", "f_mrtg"])


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (experiment.LOG, experiment.DEV, experiment.HOLDOUT_FIRST)
        experiment.LOG = Path(self.tmp.name) / "experiments.jsonl"
        experiment.DEV, experiment.HOLDOUT_FIRST = [2016, 2017], 2018

    def tearDown(self):
        experiment.LOG, experiment.DEV, experiment.HOLDOUT_FIRST = self.old
        self.tmp.cleanup()

    @staticmethod
    def fake_build(params):
        """Feature table whose ratings get noisier as k_margin grows (0.05 is clearly best)."""
        rng = np.random.default_rng(1)
        noise = max(0.0, params.get("k_margin", 0.07) - 0.04) * 30
        return FEAT.assign(f_mrtg=FEAT["f_mrtg"] + rng.normal(0, 1, len(FEAT)) * noise)

    def test_identical_arms_fail_and_are_recorded(self):
        f = self.fake_build({})
        rec = experiment.run_alt(f, f, "G5", False, 2018, "test", {"variant": "G5b"})
        self.assertFalse(rec["passed"])
        self.assertEqual(rec["results"]["home covers"]["d_logloss"], 0.0)
        self.assertTrue(store.verify(experiment.LOG)[0])
        with self.assertRaises(experiment.ProtocolError):     # no development pass -> no holdout
            experiment.run_alt(f, f, "G5", True, 2018, "test", {})
        with self.assertRaises(experiment.ProtocolError):
            experiment.run_alt(f, f, "G9", False, 2018, "test", {})

    def test_score_config_on_real_fits(self):
        s = experiment.score_config(FEAT, seasons=[2017, 2018])
        self.assertEqual(set(s), {"own_line_mae_margin", "own_line_mae_total", "logloss_winner",
                                  "logloss_home covers", "logloss_over hits"})
        self.assertTrue(all(np.isfinite(v) and v > 0 for v in s.values()))

    def test_g6_search_records_every_configuration(self):
        """Search logic with a known score surface: best at k_margin=0.05 and carry=0.70."""
        def score(params):
            mae = 10 + 100 * (params["k_margin"] - 0.05) ** 2 + 10 * (params["carry"] - 0.70) ** 2
            return {"own_line_mae_margin": mae, "own_line_mae_total": 10.0, "logloss_winner": 0.6,
                    "logloss_home covers": 0.69, "logloss_over hits": 0.69}
        old = experiment.score_config
        experiment.score_config = score
        try:
            best, tried = experiment.g6_search(lambda p: dict(p), dict(ratings.PARAMS), "test")
        finally:
            experiment.score_config = old
        self.assertEqual(tried[0]["config"], "current")
        self.assertEqual(len(tried), 1 + 2 * len(experiment.G6_GRID) + 1)     # + the combined configuration
        self.assertTrue(tried[-1]["config"].startswith("combined"))
        self.assertEqual({k: v for k, v in best.items() if ratings.PARAMS.get(k) != v},
                         {"k_margin": 0.05, "carry": 0.70})
        rec = store.read(experiment.LOG)[-1]["data"]
        self.assertEqual(rec["group"], "G6-search")
        self.assertEqual(len(rec["configurations"]), len(tried))


if __name__ == "__main__":
    unittest.main()
