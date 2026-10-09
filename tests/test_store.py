"""Forecast history integrity and placed-bet accounting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import metrics, store, track  # noqa: E402

KICK = "2026-10-11T17:00:00Z"


def forecast(**kw):
    d = dict(game_id="2026_05_CIN_MIA", kickoff_utc=KICK, season=2026, week=5, away_team="CIN",
             home_team="MIA", market="spread", side="home", team="MIA", book="bk", point=7.0, price=-110,
             price_source="odds_api", decision="BET", model_prob=0.56, market_prob=0.52,
             is_best_side=True, stake_units=1.0)
    d.update(kw)
    return d


class Chain(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "f.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def test_append_and_verify(self):
        store.append(self.path, "forecast", [forecast(), forecast(side="away")], recorded_utc="2026-10-11T15:00:00Z")
        store.append(self.path, "forecast", [forecast()], recorded_utc="2026-10-11T15:30:00Z")
        ok, msg = store.verify(self.path)
        self.assertTrue(ok, msg)
        self.assertEqual(len(store.read(self.path)), 3)

    def test_edit_is_detected_and_blocks_append(self):
        store.append(self.path, "forecast", [forecast(), forecast(side="away")], recorded_utc="2026-10-11T15:00:00Z")
        lines = self.path.read_text().splitlines()
        rec = json.loads(lines[0]); rec["data"]["price"] = 150
        lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        self.path.write_text("\n".join(lines) + "\n")
        ok, msg = store.verify(self.path)
        self.assertFalse(ok)
        with self.assertRaises(store.StoreError):
            store.append(self.path, "forecast", [forecast()])

    def test_deleted_or_reordered_line_is_detected(self):
        store.append(self.path, "forecast", [forecast(), forecast(side="away"), forecast(market="ml")])
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
        self.assertFalse(store.verify(self.path)[0])
        self.path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
        self.assertFalse(store.verify(self.path)[0])

    def test_nan_and_timestamps_serialize(self):
        store.append(self.path, "forecast", [forecast(point=np.nan, odds_time=pd.Timestamp("2026-10-11T15:00Z"))])
        d = store.read(self.path)[0]["data"]
        self.assertIsNone(d["point"])
        self.assertEqual(d["odds_time"], "2026-10-11T15:00:00Z")


class Ledger(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.f = Path(self.dir.name) / "f.jsonl"
        self.l = Path(self.dir.name) / "l.jsonl"
        recs = store.append(self.f, "forecast", [forecast(), forecast(game_id="2026_05_PHI_JAX", team="JAX",
                            home_team="JAX", away_team="PHI", point=-7.0)], recorded_utc="2026-10-11T15:00:00Z")
        self.h1, self.h2 = recs[0]["hash"], recs[1]["hash"]

    def tearDown(self):
        self.dir.cleanup()

    def place(self, h, **kw):
        kw.setdefault("placed_utc", "2026-10-11T15:10:00Z")
        return store.place_bet(h[:12], kw.pop("stake", 1.0), forecasts_path=self.f, ledger_path=self.l, **kw)

    def test_place_records_actual_price_and_links_forecast(self):
        rec = self.place(self.h1, stake=2.0, price=-115, book="other")
        d = rec["data"]
        self.assertEqual((d["price"], d["forecast_price"], d["book"]), (-115, -110, "other"))
        self.assertEqual(d["forecast_hash"], self.h1)
        self.assertTrue(store.verify(self.l)[0])

    def test_rejects_after_kickoff_before_forecast_bad_stake_duplicates(self):
        with self.assertRaises(store.StoreError):
            self.place(self.h1, placed_utc="2026-10-11T17:00:00Z")
        with self.assertRaises(store.StoreError):
            self.place(self.h1, placed_utc="2026-10-11T14:00:00Z")
        for bad in (0, -1, float("nan")):
            with self.assertRaises(store.StoreError):
                self.place(self.h1, stake=bad)
        with self.assertRaises(store.StoreError):
            self.place(self.h1, price=-50)
        self.place(self.h1)
        with self.assertRaises(store.StoreError):
            self.place(self.h1)
        with self.assertRaises(store.StoreError):
            store.place_bet("", 1.0, placed_utc="2026-10-11T15:10:00Z", forecasts_path=self.f, ledger_path=self.l)

    def test_settlement_void_roi_and_drawdown(self):
        b1 = self.place(self.h1, stake=2.0)["data"]["bet_id"]          # MIA +7 -110, 2u
        self.place(self.h2, stake=1.0)                                 # JAX -7 -110, 1u
        b3 = self.place(self.h1, stake=5.0, placed_utc="2026-10-11T15:20:00Z")["data"]["bet_id"]
        store.void_bet(b3, "book cancelled", ledger_path=self.l)
        with self.assertRaises(store.StoreError):
            store.void_bet(b3, "again", ledger_path=self.l)
        games = pd.DataFrame([
            dict(game_id="2026_05_CIN_MIA", home_score=17, away_score=20, spread_line=-7, total_line=44,
                 home_moneyline=250, away_moneyline=-300),   # MIA lost by 3: +7 wins
            dict(game_id="2026_05_PHI_JAX", home_score=24, away_score=17, spread_line=7, total_line=44,
                 home_moneyline=-300, away_moneyline=250)])  # JAX by exactly 7: push
        g = track.grade_all(games, 60, forecasts_path=self.f, ledger_path=self.l)
        led = g["ledger"].set_index("bet_id")
        self.assertNotIn(b3, led.index)                                # voided bet excluded
        self.assertAlmostEqual(led.loc[b1, "units"], 2.0 * 100 / 110)
        self.assertEqual(sorted(led["result"]), ["P", "W"])
        self.assertAlmostEqual(led["units"].sum(), 2.0 * 100 / 110)

    def test_forecast_horizon_selection(self):
        store.append(self.f, "forecast", [forecast(price=-120)], recorded_utc="2026-10-11T16:30:00Z")  # inside 60 min
        h = track.at_horizon(store.forecasts_frame(self.f), 60)
        row = h[(h["game_id"] == "2026_05_CIN_MIA") & (h["side"] == "home")]
        self.assertEqual(row["price"].iloc[0], -110)                    # the 16:30 forecast is after the horizon


class Metrics(unittest.TestCase):
    def test_max_drawdown(self):
        self.assertAlmostEqual(metrics.max_drawdown([1, -2, 1, -3, 4]), 4.0)
        self.assertEqual(metrics.max_drawdown([1, 1, 1]), 0.0)
        self.assertAlmostEqual(metrics.max_drawdown([-1, -1]), 2.0)

    def test_cluster_bootstrap_widens_for_correlated_bets(self):
        rng = np.random.default_rng(1)
        outcome = rng.choice([-1.0, 1.0], 300)
        iid = pd.DataFrame({"g": np.arange(300), "x": outcome})
        dup = pd.DataFrame({"g": np.repeat(np.arange(300), 3), "x": np.repeat(outcome, 3)})  # 3 identical bets/game
        _, lo1, hi1 = metrics.cluster_bootstrap(iid, "g", lambda d: d["x"].mean(), n=400)
        _, lo2, hi2 = metrics.cluster_bootstrap(dup, "g", lambda d: d["x"].mean(), n=400)
        naive_se = dup["x"].std() / np.sqrt(len(dup))
        self.assertGreater(hi2 - lo2, 1.5 * 2 * 1.96 * naive_se)     # clustering widens vs naive iid
        self.assertAlmostEqual(hi2 - lo2, hi1 - lo1, delta=0.05)


if __name__ == "__main__":
    unittest.main()


class LegacyRows(unittest.TestCase):
    def test_report_handles_rows_without_best_side_flag(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "f.jsonl"
            rows = [forecast(decision="NO BET", ev=-0.01), forecast(side="away", decision="NO BET", ev=-0.03)]
            for r in rows:
                r.pop("is_best_side")
            store.append(f, "forecast", rows, recorded_utc="2026-10-11T15:00:00Z")
            games = pd.DataFrame([dict(game_id="2026_05_CIN_MIA", home_score=17, away_score=20, spread_line=-7,
                                       total_line=44, home_moneyline=250, away_moneyline=-300)])
            g = track.grade_all(games, 60, forecasts_path=f, ledger_path=Path(d) / "l.jsonl")
            for r in g["forecasts"].to_dict("records"):
                r.setdefault("tier", "pass")
            track.report({"forecasts": g["forecasts"].assign(tier="pass")}, 60)   # must not raise
