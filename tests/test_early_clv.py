"""Early decision horizon (track.window / horizon_tag / at_horizon) and CLV against the last valid
pre-kickoff snapshot reference (closing.py). All data SYNTHETIC."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import closing, store, track  # noqa: E402
from nflmodel.config import Settings, SettingsError  # noqa: E402

S = Settings()
KICK = pd.Timestamp("2026-10-11T17:00:00Z")


def quotes(book, t, line, hp=-110, ap=-110, ml=(-150, 130), total=44.5):
    t = pd.Timestamp(t)
    return [dict(game_id="G", book=book, market="spread", side="home", point=-line, price=hp, source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="spread", side="away", point=line, price=ap, source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="ml", side="home", point=np.nan, price=ml[0], source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="ml", side="away", point=np.nan, price=ml[1], source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="total", side="over", point=total, price=-110, source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="total", side="under", point=total, price=-110, source="odds_api", odds_time_utc=t)]


def write_snapshots(root: Path):
    snaps = {
        "collect_early": quotes("a", "2026-10-08T12:00Z", 3.0) + quotes("b", "2026-10-08T12:00Z", 3.0),
        "collect_close": quotes("a", "2026-10-11T16:30Z", 4.0, -115, -105) + quotes("b", "2026-10-11T16:31Z", 4.0)
                         + quotes("c", "2026-10-11T16:31Z", 3.5),
        "collect_late": quotes("a", "2026-10-11T17:05Z", 6.0) + quotes("b", "2026-10-11T17:05Z", 6.0),   # after kickoff
        "collect_thin": quotes("a", "2026-10-11T16:50Z", 5.0),                                          # one book only
    }
    for name, rows in snaps.items():
        (root / name).mkdir()
        pd.DataFrame(rows).to_csv(root / name / "offers_valid.csv", index=False)


class References(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        write_snapshots(root)
        self.q = closing.snapshot_quotes(root)
        self.refs = closing.last_references(self.q, {"G": KICK}, min_books=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_last_valid_pre_kickoff_snapshot_with_enough_books(self):
        r = self.refs.set_index("market")
        self.assertEqual(set(r["snapshot"]), {"collect_close"})   # late = after kickoff, thin = one book
        self.assertEqual(r.loc["spread", "line"], 4.0)             # median of 4, 4, 3.5
        self.assertEqual(r.loc["spread", "n_books"], 3)
        self.assertLess(r.loc["spread", "as_of_utc"], KICK)

    def test_spread_probability_uses_books_at_the_reference_line(self):
        r = self.refs.set_index("market")
        a = 115 / 215 / (115 / 215 + 105 / 205)   # book a at 4: -115 / -105
        self.assertAlmostEqual(r.loc["spread", "p"], float(np.median([a, 0.5])))

    def test_no_reference_without_snapshots(self):
        self.assertTrue(closing.last_references(pd.DataFrame(), {"G": KICK}).empty)
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(closing.snapshot_quotes(Path(d)).empty)


class CLV(unittest.TestCase):
    REFS = pd.DataFrame([{"game_id": "G", "market": "spread", "line": 4.0, "p": 0.5, "n_books": 3,
                          "as_of_utc": KICK, "snapshot": "s"},
                         {"game_id": "G", "market": "total", "line": 45.5, "p": 0.5, "n_books": 3,
                          "as_of_utc": KICK, "snapshot": "s"},
                         {"game_id": "G", "market": "ml", "line": np.nan, "p": 0.62, "n_books": 3,
                          "as_of_utc": KICK, "snapshot": "s"}])

    def test_signs_and_exactness(self):
        rows = pd.DataFrame([
            {"game_id": "G", "market": "spread", "side": "home", "point": -3.0, "market_prob": 0.48},   # close -4: +1
            {"game_id": "G", "market": "spread", "side": "away", "point": 3.0, "market_prob": 0.52},    # close +4: -1
            {"game_id": "G", "market": "spread", "side": "home", "point": -4.0, "market_prob": 0.47},   # same line
            {"game_id": "G", "market": "total", "side": "over", "point": 44.5, "market_prob": 0.5},     # close 45.5: +1
            {"game_id": "G", "market": "total", "side": "under", "point": 44.5, "market_prob": 0.5},    # -1
            {"game_id": "G", "market": "ml", "side": "away", "point": np.nan, "market_prob": 0.40},
            {"game_id": "X", "market": "ml", "side": "away", "point": np.nan, "market_prob": 0.40}])
        c = closing.clv(rows, self.REFS)
        np.testing.assert_allclose(c["clv_points"].iloc[:5], [1.0, -1.0, 0.0, 1.0, -1.0])
        self.assertGreater(c["clv_prob"].iloc[0], 0)          # better number -> higher closing probability
        self.assertLess(c["clv_prob"].iloc[1], 0)
        self.assertEqual(c["clv_method"].iloc[2], "exact")
        self.assertAlmostEqual(c["clv_prob"].iloc[2], 0.5 - 0.47)
        self.assertIn("normal approximation", c["clv_method"].iloc[0])
        self.assertAlmostEqual(c["clv_prob"].iloc[5], 0.38 - 0.40)
        self.assertTrue(np.isnan(c["clv_prob"].iloc[6]))
        self.assertEqual(c["clv_method"].iloc[6], "no pre-kickoff reference in snapshots")

    def test_normal_shift_is_monotone_and_consistent(self):
        p = closing._shift_prob(0.5, 4.0, 4.0)
        self.assertAlmostEqual(p, 0.5, places=6)
        self.assertGreater(closing._shift_prob(0.5, 4.0, 3.0), 0.5)
        self.assertLess(closing._shift_prob(0.5, 4.0, 5.0), 0.5)

    def test_summary_cis(self):
        rng = np.random.default_rng(0)
        rows = pd.DataFrame({"game_id": [f"g{i}" for i in range(40)], "market": "spread",
                             "clv_points": rng.normal(0.5, 1, 40), "clv_prob": rng.normal(0.01, 0.02, 40)})
        s = closing.summary(rows, n=200).iloc[0]
        lo, hi = (float(x) for x in s["95% CI (points)"].split(" to "))
        self.assertTrue(lo < s["mean CLV points"] < hi)


class EarlyHorizon(unittest.TestCase):
    def test_windows_and_tags(self):
        self.assertEqual(track.window(S, "kickoff"), (75.0, 60.0))
        self.assertEqual(track.window(S, "early"), (168 * 60.0, 84 * 60.0))
        self.assertEqual(track.horizon_tag(65, S), "kickoff-60")
        self.assertEqual(track.horizon_tag(100 * 60, S), "early")
        self.assertEqual(track.horizon_tag(30 * 60, S), "other")
        self.assertEqual(track.horizon_tag(np.nan, S), "unknown")

    def test_settings_validation(self):
        with self.assertRaises(SettingsError):
            replace(S, early_horizon_min_hours=170.0)
        with self.assertRaises(SettingsError):
            replace(S, early_horizon_min_hours=1.0)        # overlaps the kickoff window

    def _fc(self, completed: list[str]):
        rows = []
        for i, t in enumerate(completed):
            rows.append({"game_id": "G", "kickoff_utc": "2026-10-11T17:00:00Z", "run_id": f"r{i}",
                         "recorded_utc": t, "prediction_completed_utc": t, "market": "spread", "side": "home"})
        return pd.DataFrame(rows)

    def test_early_selection_independent_of_kickoff_selection(self):
        fc = self._fc(["2026-10-07T12:00:00Z", "2026-10-07T20:00:00Z", "2026-10-11T15:50:00Z", "2026-10-10T12:00:00Z"])
        e, cov_e = track.at_horizon(fc, S, "early")
        k, cov_k = track.at_horizon(fc, S, "kickoff")
        self.assertEqual(set(e["run_id"]), {"r1"})        # latest run inside [kick-168h, kick-84h]
        self.assertEqual(set(k["run_id"]), {"r2"})        # 70 min before kickoff
        none, cov = track.at_horizon(self._fc(["2026-10-10T12:00:00Z"]), S, "early")
        self.assertTrue(none.empty)
        self.assertEqual(cov.iloc[0]["status"], "excluded")

    def test_new_fields_recorded_and_old_records_readable(self):
        self.assertIn("horizon_tag", store.FORECAST_FIELDS)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.jsonl"
            store.append(p, "forecast", [{"game_id": "OLD"}])
            store.append(p, "forecast", [{"game_id": "NEW", "horizon_tag": "early", "lead_minutes_at_issue": 6000.0}])
            df = store.forecasts_frame(p)
            self.assertTrue(pd.isna(df.loc[0, "horizon_tag"]))
            self.assertEqual(df.loc[1, "horizon_tag"], "early")


if __name__ == "__main__":
    unittest.main()
