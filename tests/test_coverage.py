"""Coverage against an explicit slate: every scheduled game gets exactly one status."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import store, track  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402
from test_horizon import fc  # noqa: E402

S = Settings()
NOW = parse_utc("2026-10-12T00:00:00Z")
# kickoffs (Eastern): G1/G2/G3 Sun 13:00 ET = 17:00Z (past); G4 Mon 20:15 ET (future); G5 no kickoff time
GAMES = pd.DataFrame([
    dict(game_id="G1", season=2026, week=5, away_team="A", home_team="B", gameday="2026-10-11", gametime="13:00", result=3),
    dict(game_id="G2", season=2026, week=5, away_team="C", home_team="D", gameday="2026-10-11", gametime="13:00", result=-7),
    dict(game_id="G3", season=2026, week=5, away_team="E", home_team="F", gameday="2026-10-11", gametime="13:00", result=1),
    dict(game_id="G4", season=2026, week=5, away_team="G", home_team="H", gameday="2026-10-12", gametime="20:15", result=None),
    dict(game_id="G5", season=2026, week=5, away_team="I", home_team="J", gameday="2026-10-11", gametime=None, result=None),
    dict(game_id="X9", season=2026, week=6, away_team="K", home_team="L", gameday="2026-10-18", gametime="13:00", result=None),
])
KICK = "2026-10-11T17:00:00Z"


def history(*batches):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.jsonl"
        for batch in batches:
            for data, rec in batch:
                store.append(p, "forecast", [data], recorded_utc=rec)
        return store.forecasts_frame(p)


def status(cov):
    return dict(zip(cov["game_id"], cov["status"]))


class Slate(unittest.TestCase):
    def test_slate_selection_by_week_and_dates(self):
        self.assertEqual(list(track.define_slate(GAMES, season=2026, weeks=(5, 5))["game_id"]),
                         ["G1", "G2", "G3", "G4", "G5"])
        self.assertEqual(list(track.define_slate(GAMES, date_from="2026-10-12", date_to="2026-10-18")["game_id"]),
                         ["G4", "X9"])


class Coverage(unittest.TestCase):
    slate = track.define_slate(GAMES, season=2026, weeks=(5, 5))

    def test_empty_history(self):
        cov, mk, sel = track.slate_coverage(self.slate, pd.DataFrame(), S, NOW)
        self.assertEqual(status(cov), {"G1": "no forecast recorded", "G2": "no forecast recorded",
                                       "G3": "no forecast recorded", "G4": "window not closed",
                                       "G5": "unknown kickoff"})
        summ = track.coverage_summary(cov).set_index("status")
        self.assertAlmostEqual(summ.loc["no forecast recorded", "pct_of_closed_windows"], 1.0)
        self.assertEqual(int(summ.loc[track.STATUS_ORDER[:3], "games"].sum()), 3)   # denominator
        self.assertTrue(sel.empty)
        self.assertEqual(list(mk["eligible_games"]), [0, 0, 0])

    def test_out_of_window_forecasts_and_partial_market_coverage(self):
        eligible_spread_only = fc("R1", "2026-10-11T15:50:00Z", game="G1", markets=("spread",))
        too_early = fc("R0", "2026-10-10T15:50:00Z", game="G2")
        cov, mk, sel = track.slate_coverage(self.slate, history(eligible_spread_only, too_early), S, NOW)
        st = status(cov)
        self.assertEqual(st["G1"], "eligible")
        self.assertEqual(st["G2"], "forecasts exist, none qualify")
        self.assertIn("window", cov.set_index("game_id").loc["G2", "reason"])
        self.assertEqual(st["G3"], "no forecast recorded")
        mk = mk.set_index("market")
        self.assertEqual((mk.loc["spread", "with_forecast"], mk.loc["spread", "missing"]), (1, 0))
        self.assertEqual((mk.loc["ml", "with_forecast"], mk.loc["ml", "missing_games"]), (0, "G1"))
        self.assertEqual(set(sel["game_id"]), {"G1"})                # only eligible games are scored
        self.assertEqual(set(sel["run_id"]), {"R1"})                 # one run per game
        summ = track.coverage_summary(cov).set_index("status")
        self.assertAlmostEqual(summ.loc["eligible", "pct_of_closed_windows"], 1 / 3)

    def test_future_game_not_counted_as_missed(self):
        early_for_future = fc("R0", "2026-10-11T12:00:00Z", game="G4", kick="2026-10-13T00:15:00Z")
        cov, _, _ = track.slate_coverage(self.slate, history(early_for_future), S, NOW)
        self.assertEqual(status(cov)["G4"], "window not closed")
        summ = track.coverage_summary(cov).set_index("status")
        self.assertTrue(pd.isna(summ.loc["window not closed", "pct_of_closed_windows"]))

    def test_markets_filter_and_forecasts_outside_slate_ignored(self):
        outside = fc("R9", "2026-10-11T15:50:00Z", game="ZZ")
        g1 = fc("R1", "2026-10-11T15:50:00Z", game="G1", markets=("ml",))
        cov, mk, sel = track.slate_coverage(self.slate, history(outside, g1), S, NOW, markets=("spread",))
        self.assertEqual(status(cov)["G1"], "no forecast recorded")   # no forecast for the requested market
        self.assertNotIn("ZZ", set(cov["game_id"]))
        self.assertTrue(sel.empty)


if __name__ == "__main__":
    unittest.main()
