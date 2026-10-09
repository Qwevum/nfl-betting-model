"""Run timestamps: stage clock, recheck at completion, and recorded-time guarantees."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import runtime, store  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.odds import OFFER_COLS, validate_offers  # noqa: E402
from nflmodel.timeutil import fmt, now_utc, parse_utc  # noqa: E402

KICK = parse_utc("2026-10-11T17:00:00Z")
S = Settings(max_odds_age_minutes=30)


def rows(decision="BET", odds_time="2026-10-11T15:40:00Z", game="G"):
    return pd.DataFrame([{"game_id": game, "market": "spread", "side": "home", "decision": decision,
                          "reasons": "", "odds_time": odds_time, "stake_units": 1.2}])


class Recheck(unittest.TestCase):
    def test_quote_fresh_at_collection_but_stale_at_completion_is_downgraded(self):
        out = runtime.recheck_before_issue(rows(), {"G": KICK}, parse_utc("2026-10-11T16:20:00Z"), S)
        self.assertEqual(out["decision"].iloc[0], "BET IF PRICE AVAILABLE")
        self.assertIn("stale by completion", out["reasons"].iloc[0])
        self.assertFalse(out["post_kickoff"].iloc[0])

    def test_fresh_at_completion_stays_executable(self):
        out = runtime.recheck_before_issue(rows(), {"G": KICK}, parse_utc("2026-10-11T16:00:00Z"), S)
        self.assertEqual(out["decision"].iloc[0], "BET")

    def test_kickoff_passed_before_completion_is_no_bet_and_flagged(self):
        out = runtime.recheck_before_issue(rows(), {"G": KICK}, parse_utc("2026-10-11T17:00:30Z"), S)
        self.assertEqual(out["decision"].iloc[0], "NO BET")
        self.assertTrue(out["post_kickoff"].iloc[0])
        self.assertEqual(out["stake_units"].iloc[0], 0.0)

    def test_conditional_and_untimed_rows_untouched_unless_kickoff(self):
        r = rows(decision="BET IF PRICE AVAILABLE", odds_time=None)
        out = runtime.recheck_before_issue(r, {"G": KICK}, parse_utc("2026-10-11T16:59:00Z"), S)
        self.assertEqual(out["decision"].iloc[0], "BET IF PRICE AVAILABLE")


class Clock(unittest.TestCase):
    def test_simulated_clock_is_fixed_and_labelled(self):
        c = runtime.Clock(parse_utc("2026-10-11T12:00:00Z"))
        self.assertTrue(c.simulated)
        c.stamp("run_started"); c.stamp("prediction_completed")
        self.assertEqual(c.as_dict(), {"run_started_utc": "2026-10-11T12:00:00Z",
                                       "prediction_completed_utc": "2026-10-11T12:00:00Z"})

    def test_real_clock_stamps_in_order(self):
        c = runtime.Clock()
        self.assertFalse(c.simulated)
        a, b = c.stamp("run_started"), c.stamp("prediction_completed")
        self.assertLessEqual(a, b)


class ValidationUsesCollectionTime(unittest.TestCase):
    def test_quote_after_run_start_is_valid_against_its_collection_time(self):
        start, collected = parse_utc("2026-10-11T15:00:00Z"), parse_utc("2026-10-11T15:06:00Z")
        o = pd.DataFrame([dict(game_id="G", book="bk", market="spread", side="home", point=-3.0, price=-110,
                               source="odds_api", odds_time="2026-10-11T15:05:00Z")], columns=OFFER_COLS)
        self.assertEqual(len(validate_offers(o, {"G": KICK}, start, S)[0]), 0)       # wrongly "future" at start
        self.assertEqual(len(validate_offers(o, {"G": KICK}, collected, S)[0]), 1)


class RecordedTime(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{"game_id": "G", "kickoff_utc": fmt(KICK), "market": "spread", "side": "home",
                              "decision": "BET", "price": -110, "point": 3.0}])

    def test_recorded_time_is_write_time_not_run_start(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.jsonl"
            completed = now_utc() - pd.Timedelta(minutes=5)
            run = {"run_started_utc": fmt(completed - pd.Timedelta(minutes=20)),
                   "prediction_completed_utc": fmt(completed)}
            rec = store.record_forecasts(self.frame(), run, p)[0]
            self.assertGreaterEqual(parse_utc(rec["recorded_utc"]), completed)
            self.assertEqual(rec["data"]["run_started_utc"], run["run_started_utc"])

    def test_refuses_completion_time_in_the_future(self):
        with tempfile.TemporaryDirectory() as d:
            run = {"prediction_completed_utc": fmt(now_utc() + pd.Timedelta(hours=1))}
            with self.assertRaises(store.StoreError):
                store.record_forecasts(self.frame(), run, Path(d) / "f.jsonl")


if __name__ == "__main__":
    unittest.main()
