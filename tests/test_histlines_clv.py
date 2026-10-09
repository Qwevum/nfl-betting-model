"""Historical open/close loader (histlines.py) and opening-line CLV backtest (clv.py).
Fixtures in tests/fixtures/ are SYNTHETIC (AusSportsBetting-like layout)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import clv, histlines as hl  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from test_forecast_uncertainty import FEAT  # noqa: E402

FX = Path(__file__).resolve().parent / "fixtures"
SCHED = pd.read_csv(FX / "historical_lines_schedule.csv")
LINES, REP = hl.load(FX / "historical_lines_sample.csv", SCHED)


class Columns(unittest.TestCase):
    def test_infers_roles_loosely(self):
        m = hl.infer_columns(["Date", "Home Team", "Away Team", "Home Line Open", "Home Line Close", "Kick-off (ET)"])
        self.assertEqual(m["home_line_open"], "Home Line Open")
        self.assertEqual(m["home_team"], "Home Team")
        m = hl.infer_columns(["game_date", "HOME", "away", "home_spread_opening", "home-spread-closing"])
        self.assertEqual(m["home_line_close"], "home-spread-closing")

    def test_missing_and_ambiguous_are_errors(self):
        with self.assertRaises(hl.HistLinesError) as e:
            hl.infer_columns(["Date", "Home Team", "Away Team", "Home Line Close"])
        self.assertIn("home_line_open: not found", str(e.exception))
        with self.assertRaises(hl.HistLinesError) as e:
            hl.infer_columns(["Date", "Home Team", "home_team", "Away Team", "Home Line Open", "Home Line Close"])
        self.assertIn("home_team: ambiguous", str(e.exception))
        self.assertIn("Columns in the file", str(e.exception))

    def test_explicit_mapping_wins_and_is_checked(self):
        cols = ["d", "h", "a", "o", "c"]
        m = hl.infer_columns(cols, {"date": "d", "home_team": "h", "away_team": "a", "home_line_open": "o",
                                    "home_line_close": "c"})
        self.assertEqual(m["home_line_close"], "c")
        with self.assertRaises(hl.HistLinesError):
            hl.infer_columns(cols, {"date": "nope"})


class Teams(unittest.TestCase):
    def test_relocations_and_names(self):
        self.assertEqual(hl.team_code("Oakland Raiders", 2019), "OAK")
        self.assertEqual(hl.team_code("Las Vegas Raiders", 2020), "LV")
        self.assertEqual(hl.team_code("Raiders", 2020), "LV")
        self.assertEqual(hl.team_code("San Diego Chargers", 2016), "SD")
        self.assertEqual(hl.team_code("Los Angeles Chargers", 2017), "LAC")
        self.assertEqual(hl.team_code("St. Louis Rams", 2015), "STL")
        self.assertEqual(hl.team_code("Los Angeles Rams", 2016), "LA")
        self.assertEqual(hl.team_code("Washington Football Team", 2021), "WAS")
        self.assertEqual(hl.team_code("KC", 2019), "KC")
        self.assertIsNone(hl.team_code("London Monarchs", 2019))

    def test_season_of(self):
        self.assertEqual(hl.season_of(pd.Timestamp("2020-01-05")), 2019)
        self.assertEqual(hl.season_of(pd.Timestamp("2019-09-08")), 2019)

    def test_odds_format(self):
        self.assertEqual(hl.odds_format(np.array([1.91, 2.1, np.nan])), "decimal")
        self.assertEqual(hl.odds_format(np.array([-110, 105])), "american")
        self.assertEqual(hl.odds_format(np.array([np.nan])), "none")
        with self.assertRaises(hl.HistLinesError):
            hl.odds_format(np.array([1.91, -110]))
        self.assertEqual(hl.decimal_to_american(1.91), -110)
        self.assertEqual(hl.decimal_to_american(2.5), 150)


class Load(unittest.TestCase):
    def test_matching_report(self):
        self.assertEqual(REP["rows"], 28)
        self.assertEqual(REP["matched"], 27)
        self.assertEqual(len(REP["unmatched"]), 1)
        self.assertEqual(REP["unmatched"][0]["reason"], "unknown team name")
        self.assertEqual(REP["swapped"], 1)
        self.assertEqual(REP["odds_format"], "decimal")
        self.assertIn("handicap form (negated)", REP["sign_convention"])

    def test_lines_in_nflverse_convention(self):
        np.testing.assert_allclose(LINES["close_line"], LINES["nfl_close"])     # incl. the swapped neutral game
        self.assertIn("2019_19_BUF_NE", set(LINES["game_id"]))                  # January game -> season 2019
        self.assertIn("2020_01_KC_LV", set(LINES["game_id"]))                   # Raiders moved
        sw = LINES[LINES["swapped"]].iloc[0]
        self.assertEqual(sw["open_line"], 2.5)
        self.assertEqual(sw["home_line_odds_open"], hl.decimal_to_american(1.92))   # odds swapped with the teams

    def test_missing_file_is_a_clear_error(self):
        old = hl.CANDIDATES
        hl.CANDIDATES = [Path(tempfile.gettempdir()) / "no_such_historical_lines.csv"]
        try:
            self.assertIsNone(hl.find_file())
            with self.assertRaises(hl.HistLinesError):
                hl.load(None, SCHED)
            with self.assertRaises(hl.HistLinesError):
                hl.load(hl.CANDIDATES[0], SCHED)
        finally:
            hl.CANDIDATES = old


def _synthetic_lines(feat, seasons, move_toward_model: float, seed=0):
    """Open = close - move; the close moves toward the TRUE margin signal by move_toward_model."""
    rng = np.random.default_rng(seed)
    x = feat[feat["season"].isin(seasons)]
    close = x["spread_line"].to_numpy(float)
    open_ = close - move_toward_model * (x["f_mrtg"].to_numpy(float) * 3.0) + rng.normal(0, 0.25, len(x))
    return pd.DataFrame({"game_id": x["game_id"], "season": x["season"], "open_line": np.round(open_ * 2) / 2,
                         "close_line": close, "home_line_odds_open": -110.0, "away_line_odds_open": -110.0,
                         "home_line_odds_close": -110.0, "away_line_odds_close": -110.0})


class CLV(unittest.TestCase):
    ROWS = clv.game_rows(FEAT, _synthetic_lines(FEAT, [2017, 2018], 0.5), [2017, 2018], Settings())

    def test_rows_use_earlier_seasons_only(self):
        r = self.ROWS
        self.assertEqual(set(r["season"]), {2017, 2018})
        changed = FEAT.copy()
        changed.loc[changed["season"] == 2018, "result"] += 15      # 2018 outcomes cannot affect 2018 forecasts
        r2 = clv.game_rows(changed, _synthetic_lines(FEAT, [2018], 0.5), [2018], Settings())
        np.testing.assert_allclose(r[r["season"] == 2018]["own_line"].to_numpy(), r2["own_line"].to_numpy())

    def test_line_move_detects_known_signal(self):
        lm = clv.line_move(self.ROWS, n=300)
        own = lm[lm["model"] == "ratings-only own line"].iloc[0]
        self.assertGreater(own["slope"], 0)
        self.assertFalse(own["CI includes 0"])
        none = clv.line_move(clv.game_rows(FEAT, _synthetic_lines(FEAT, [2018], 0.0), [2018], Settings()), n=300)
        self.assertLess(abs(none.iloc[0]["slope"]), 0.1)

    def test_bets_clv_signs_and_push(self):
        rows = pd.DataFrame([
            {"game_id": "a", "season": 2018, "result": 7, "open": 3.0, "close": 4.0, "ev_home": 0.05, "ev_away": -0.1,
             "price_home": -110.0, "price_away": -110.0, "p_open_home": 0.5, "p_close_home_at_open": 0.55},
            {"game_id": "b", "season": 2018, "result": 3, "open": 3.0, "close": 2.0, "ev_home": -0.1, "ev_away": 0.04,
             "price_home": -110.0, "price_away": +100.0, "p_open_home": 0.5, "p_close_home_at_open": 0.46},
            {"game_id": "c", "season": 2018, "result": 1, "open": 3.0, "close": 3.0, "ev_home": 0.01, "ev_away": 0.0,
             "price_home": -110.0, "price_away": -110.0, "p_open_home": 0.5, "p_close_home_at_open": 0.5}])
        b = clv.bets(rows, 0.02)
        self.assertEqual(list(b["game_id"]), ["a", "b"])               # c is below the threshold
        a, bb = b.iloc[0], b.iloc[1]
        self.assertEqual((a["side"], a["result"], a["clv_points"]), ("home", "W", 1.0))
        self.assertAlmostEqual(a["clv_prob"], 0.05)
        self.assertEqual((bb["side"], bb["result"], bb["clv_points"]), ("away", "P", 1.0))   # 3 = open line: push
        self.assertAlmostEqual(bb["clv_prob"], 0.04)
        self.assertEqual(bb["profit_open"], 0.0)
        s = clv.bet_summary(b, n=100)
        self.assertEqual(s.loc[s["metric"] == "bets", "value"].iloc[0], 2)
        self.assertEqual(clv.bet_summary(clv.bets(rows, 0.5)).iloc[0]["value"], 0)


if __name__ == "__main__":
    unittest.main()
