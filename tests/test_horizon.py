"""Horizon evaluation: eligibility window, issue time, one run per game, coverage, scoring."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import store, track  # noqa: E402
from nflmodel.config import Settings  # noqa: E402

KICK = "2026-10-11T17:00:00Z"            # window with defaults: 15:45 - 16:00
S = Settings()


def fc(run, completed, recorded=None, game="G1", kick=KICK, markets=("spread", "ml"), version="v1",
       mp=0.55, kp=0.52, legacy=False):
    rows = []
    for m in markets:
        for side in ("home", "away"):
            d = dict(game_id=game, kickoff_utc=kick, market=m, side=side, team=side, book="bk",
                     point=-3.0 if m == "spread" else None, price=-110, decision="NO BET", tier="pass",
                     model_prob=mp if side == "home" else 1 - mp, market_prob=kp if side == "home" else 1 - kp,
                     ev=-0.01, model_version=version, is_best_side=side == "home")
            if not legacy:
                d.update(run_id=run, prediction_completed_utc=completed)
            rows.append((d, recorded or completed))
    return rows


def frame(*batches):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.jsonl"
        for batch in batches:
            for data, rec in batch:
                store.append(p, "forecast", [data], recorded_utc=rec)
        return store.forecasts_frame(p)


class Window(unittest.TestCase):
    def test_week_old_forecast_fails_the_window(self):
        sel, cov = track.at_horizon(frame(fc("old", "2026-10-04T17:00:00Z")), S)
        self.assertTrue(sel.empty)
        self.assertEqual(cov["status"].iloc[0], "excluded")
        self.assertIn("window", cov["reason"].iloc[0])

    def test_inside_window_eligible_after_cutoff_excluded(self):
        sel, cov = track.at_horizon(frame(fc("r70", "2026-10-11T15:50:00Z")), S)
        self.assertEqual(cov["status"].iloc[0], "eligible")
        self.assertAlmostEqual(cov["lead_minutes"].iloc[0], 70)
        self.assertTrue((sel["lead_minutes"] == 70).all())
        sel, cov = track.at_horizon(frame(fc("r55", "2026-10-11T16:05:00Z")), S)
        self.assertTrue(sel.empty)
        self.assertIn("after the cutoff", cov["reason"].iloc[0])

    def test_tolerance_is_configurable(self):
        f = frame(fc("r90", "2026-10-11T15:30:00Z"))
        self.assertTrue(track.at_horizon(f, S)[0].empty)
        self.assertFalse(track.at_horizon(f, Settings(horizon_tolerance_minutes=40))[0].empty)

    def test_late_completing_run_cannot_count_as_earlier_forecast(self):
        # run started early, but completed (and was recorded) after the cutoff
        rows = fc("late", "2026-10-11T16:04:00Z", recorded="2026-10-11T16:04:30Z")
        for d, _ in rows:
            d["run_started_utc"] = "2026-10-11T15:40:00Z"
        sel, cov = track.at_horizon(frame(rows), S)
        self.assertTrue(sel.empty)
        # and a record whose recorded time is later than its claimed completion uses the later time
        rows = fc("lied", "2026-10-11T15:50:00Z", recorded="2026-10-11T16:10:00Z")
        self.assertTrue(track.at_horizon(frame(rows), S)[0].empty)

    def test_legacy_rows_without_completion_time_are_excluded_with_reason(self):
        sel, cov = track.at_horizon(frame(fc(None, None, recorded="2026-10-11T15:50:00Z", legacy=True)), S)
        self.assertTrue(sel.empty)
        self.assertIn("legacy", cov["reason"].iloc[0])


class OneRunPerGame(unittest.TestCase):
    def test_latest_eligible_run_used_for_all_sides(self):
        a = fc("A", "2026-10-11T15:48:00Z", markets=("spread", "ml"))
        b = fc("B", "2026-10-11T15:55:00Z", markets=("spread",), mp=0.60)
        sel, cov = track.at_horizon(frame(a, b), S)
        self.assertEqual(set(sel["run_id"]), {"B"})               # sides never mixed across runs
        self.assertEqual(set(sel["market"]), {"spread"})
        self.assertEqual(cov["selected_run"].iloc[0], "B")
        self.assertEqual(cov["runs_considered"].iloc[0], 2)

    def test_games_selected_independently(self):
        g1 = fc("A", "2026-10-11T15:50:00Z", game="G1")
        g2 = fc("A", "2026-10-11T15:50:00Z", game="G2", kick="2026-10-11T20:25:00Z")   # too early for G2
        sel, cov = track.at_horizon(frame(g1, g2), S)
        self.assertEqual(set(sel["game_id"]), {"G1"})
        self.assertEqual(dict(zip(cov["game_id"], cov["status"])), {"G1": "eligible", "G2": "excluded"})


class Scoring(unittest.TestCase):
    def test_identical_rows_by_market_and_version(self):
        rows = fc("A", "2026-10-11T15:50:00Z", version="v1") + \
            fc("B", "2026-10-11T15:50:00Z", game="G2", version="v2")
        rows[0][0]["market_prob"] = None                          # one row lacks a market reference
        sel, _ = track.at_horizon(frame(rows), S)
        games = pd.DataFrame([dict(game_id=g, home_score=24, away_score=17, spread_line=3, total_line=44,
                                   home_moneyline=-150, away_moneyline=130, result=7) for g in ("G1", "G2")])
        h = track._results(sel, games)
        t = track.score_table(h)
        self.assertEqual(set(zip(t["market"], t["model_version"])),
                         {("spread", "v1"), ("ml", "v1"), ("spread", "v2"), ("ml", "v2")})
        v1_spread = t[(t["market"] == "spread") & (t["model_version"] == "v1")].iloc[0]
        self.assertEqual(v1_spread["sides"], 1)                    # the row without market_prob is dropped for both


if __name__ == "__main__":
    unittest.main()
