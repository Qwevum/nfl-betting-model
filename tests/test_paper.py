"""OL1-paper tracking (paper.py) and the Odds API quota helper (quota.py). SYNTHETIC data only."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import paper, quota, store  # noqa: E402
from nflmodel.config import Settings  # noqa: E402

S = Settings()
NOW = pd.Timestamp("2026-10-05T03:00:00Z")          # Sunday night
KICK_EARLY = NOW + pd.Timedelta(hours=130)          # inside 84-168 h
KICK_LATE = NOW + pd.Timedelta(hours=40)            # outside


def q(game, book, home_point, hp=-110, ap=-110, t=NOW - pd.Timedelta(minutes=5)):
    return [dict(game_id=game, book=book, market="spread", side="home", point=home_point, price=hp,
                 source="odds_api", odds_time_utc=t),
            dict(game_id=game, book=book, market="spread", side="away", point=-home_point, price=ap,
                 source="odds_api", odds_time_utc=t)]


def preds(own_lines: dict, kicks: dict):
    return pd.DataFrame([{"game_id": g, "season": 2026, "week": 6, "away_team": "A" + g, "home_team": "H" + g,
                          "kickoff_utc": kicks[g], "model_margin": m} for g, m in own_lines.items()])


VALID = pd.DataFrame(q("G1", "a", -3.0) + q("G1", "b", -3.0, -105, -115) + q("G1", "c", -2.5, -112, -108)
                     + q("G2", "a", -3.0) + q("G2", "b", -3.0)
                     + q("G3", "a", 7.0) + q("G3", "b", 7.5, -110, -110)
                     + q("G4", "a", -3.0))


class Candidates(unittest.TestCase):
    def setUp(self):
        p = preds({"G1": 8.0, "G2": 4.0, "G3": -12.0, "G4": 10.0, "G5": 9.0},
                  {"G1": KICK_EARLY, "G2": KICK_EARLY, "G3": KICK_EARLY, "G4": KICK_EARLY, "G5": KICK_LATE})
        self.c, self.notes = paper.candidates(p, VALID, NOW, S)
        self.c = self.c.set_index("game_id")

    def test_window_gap_and_books(self):
        # G1: ref 3, own 8 -> gap 5 (home). G2: gap 1 (no). G3: ref -7, own -12 -> gap -5 (away).
        # G4: one book only -> no reference. G5: outside the early window.
        self.assertEqual(set(self.c.index), {"G1", "G3"})
        self.assertTrue(any("G4" in n and "reference" in n for n in self.notes))
        self.assertEqual(self.c.loc["G1", "side"], "home")
        self.assertEqual(self.c.loc["G3", "side"], "away")
        self.assertAlmostEqual(self.c.loc["G1", "gap"], 5.0)

    def test_best_price_across_books(self):
        g1 = self.c.loc["G1"]
        self.assertEqual((g1["book"], g1["point"], g1["price"]), ("c", -2.5, -112.0))   # best line beats price
        g3 = self.c.loc["G3"]                     # away is the favorite: laying 7 (book a) beats laying 7.5
        self.assertEqual((g3["book"], g3["point"]), ("a", -7.0))
        self.assertEqual(g1["market_prob_method"], "normal approximation (line differs)")
        self.assertGreater(g1["market_prob"], 0.5)        # -2.5 is better for home than the -3 reference

    def test_independent_of_live_rules(self):
        """No GameContext / gap_points input: a 5-point gap (>= gap_points 4) is still recorded."""
        self.assertGreaterEqual(abs(self.c.loc["G1", "gap"]), S.gap_points)


class Grading(unittest.TestCase):
    def test_earliest_per_game_clv_and_result(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paper.jsonl"
            base = {"game_id": "G1", "kickoff_utc": KICK_EARLY.strftime("%Y-%m-%dT%H:%M:%SZ"), "market": "spread",
                    "side": "home", "point": -3.0, "price": -110.0, "market_prob": 0.5}
            store.append(path, paper.KIND, [base], recorded_utc="2026-10-05T03:00:00Z")
            store.append(path, paper.KIND, [{**base, "point": -2.0}], recorded_utc="2026-10-06T03:00:00Z")
            store.append(path, "other", [{"game_id": "X"}])
            fc = paper.frame(path)
        self.assertEqual(len(fc), 2)                       # other kinds ignored
        refs = pd.DataFrame([{"game_id": "G1", "market": "spread", "line": 4.0, "p": 0.5, "n_books": 3,
                              "as_of_utc": KICK_EARLY, "snapshot": "s"}])
        games = pd.DataFrame([{"game_id": "G1", "home_score": 24, "away_score": 20}])
        g = paper.graded(fc, refs, games)
        self.assertEqual(len(g), 1)
        self.assertEqual(g.iloc[0]["point"], -3.0)         # the EARLIEST record, not the better -2
        self.assertEqual(g.iloc[0]["clv_points"], 1.0)     # -3 vs close -4
        self.assertEqual(g.iloc[0]["result"], "W")         # won by 4, laid 3
        self.assertAlmostEqual(g.iloc[0]["profit"], 100 / 110)

    def test_store_path_resolved_at_call_time(self):
        """Redirecting paper.STORE (as tests and smoke runs do) must never write the real log."""
        old = paper.STORE
        with tempfile.TemporaryDirectory() as d:
            paper.STORE = Path(d) / "p.jsonl"
            try:
                paper.record(pd.DataFrame([{"game_id": "G"}]), {"run_id": "t"})
                self.assertTrue(paper.STORE.exists())
                self.assertEqual(len(paper.frame()), 1)
            finally:
                paper.STORE = old

    def test_decision_rule(self):
        self.assertTrue(paper.decision(49, 0.01, 0.05).startswith("collecting"))
        self.assertIn("-105 or better", paper.decision(50, 0.001, 0.03))
        self.assertIn("continue to n=100", paper.decision(50, -0.01, 0.03))
        self.assertIn("retire", paper.decision(50, -0.05, -0.01))
        self.assertIn("retire", paper.decision(100, -0.01, 0.03))
        self.assertIn("-105 or better", paper.decision(120, 0.001, 0.03))

    def test_summary_uses_the_look_sample(self):
        rng = np.random.default_rng(0)
        n = 60
        g = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "kickoff_utc": [f"2026-10-{i % 28 + 1:02d}" for i in range(n)],
                          "clv_prob": rng.normal(0.03, 0.01, n), "clv_points": rng.normal(1, 0.5, n),
                          "result": "W", "profit": 0.9})
        s = paper.summary(g)
        self.assertEqual(s["graded_with_clv"], 60)
        self.assertIn("look at n=50", s["decision"])
        self.assertIn("-105 or better", s["decision"])
        self.assertTrue(paper.summary(g.head(10))["decision"].startswith("collecting"))


class Quota(unittest.TestCase):
    def test_usage_estimate(self):
        self.assertAlmostEqual(quota.weekly_runs(4, False), 42)
        self.assertAlmostEqual(quota.weekly_runs(6, False), 28)
        self.assertAlmostEqual(quota.weekly_runs(4, True), 42 + 3 * 9)
        self.assertAlmostEqual(quota.monthly_credits(3, 4, False), 42 * 3 * 52 / 12)
        txt = quota.report({"used": 10, "remaining": 490}, 3, 4, False)
        self.assertIn("monthly allowance about 500", txt)
        self.assertIn("EXCEEDS", txt)                       # 4-hourly ~546 > 500
        self.assertIn("<- requested", txt)

    def test_errors_never_show_the_key(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("bad url ...apiKey=secret-xyz")), \
                mock.patch("nflmodel.apikey.get", return_value="secret-xyz"):
            r = quota.remaining("secret-xyz")
        self.assertIn("error", r)
        self.assertNotIn("secret-xyz", r["error"])


if __name__ == "__main__":
    unittest.main()
