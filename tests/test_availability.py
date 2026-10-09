"""Starter availability: missing injury information is unknown, never healthy."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel.decide import starter_status  # noqa: E402

QB = "00-0000001"


def report(rows):
    cols = ["team", "week", "gsis_id", "report_status", "practice_status"]
    return pd.DataFrame(rows, columns=cols)


class StarterStatus(unittest.TestCase):
    def test_feed_unavailable_is_unknown(self):
        self.assertEqual(starter_status(pd.DataFrame(), False, QB, "MIA", 5)[0], "unknown")
        self.assertEqual(starter_status(None, True, QB, "MIA", 5)[0], "unknown")

    def test_no_report_for_team_this_week_is_unknown(self):
        inj = report([("MIA", 4, "x", "Out", None), ("CIN", 5, "y", "Out", None)])
        state, detail = starter_status(inj, True, QB, "MIA", 5)
        self.assertEqual(state, "unknown")
        self.assertIn("no week 5", detail)

    def test_practice_report_only_is_unknown(self):
        inj = report([("MIA", 5, "other", None, "Did Not Participate In Practice")])
        self.assertEqual(starter_status(inj, True, QB, "MIA", 5)[0], "unknown")
        inj = report([("MIA", 5, QB, None, "Full Participation in Practice")])
        self.assertEqual(starter_status(inj, True, QB, "MIA", 5)[0], "unknown")

    def test_final_statuses_issued_and_not_listed_is_available(self):
        inj = report([("MIA", 5, "other", "Out", "Did Not Participate In Practice")])
        self.assertEqual(starter_status(inj, True, QB, "MIA", 5)[0], "available")

    def test_designations(self):
        for status, want in (("Out", "ruled_out"), ("Doubtful", "ruled_out"), ("Questionable", "questionable")):
            inj = report([("MIA", 5, QB, status, "Limited Participation in Practice")])
            self.assertEqual(starter_status(inj, True, QB, "MIA", 5)[0], want)

    def test_unlisted_starter_id_is_unknown(self):
        inj = report([("MIA", 5, "other", "Out", None)])
        self.assertEqual(starter_status(inj, True, None, "MIA", 5)[0], "unknown")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- decision level

import numpy as np  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from nflmodel import market  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.decide import build_context, decide_game  # noqa: E402
from nflmodel.model import MARGINS, TOTALS, FittedModel, OutcomeDist  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402

T0 = parse_utc("2026-10-11T15:50:00Z")


def fake_model():
    """Market-only model: anchored probability = market probability."""
    ident = np.array([0.0, 1.0, 0.0])
    m = SimpleNamespace(
        margin_dist=OutcomeDist(13.0, np.ones(len(MARGINS)), MARGINS),
        total_dist=OutcomeDist(13.5, np.ones(len(TOTALS)), TOTALS),
        spread_cal=ident, total_cal=ident, ml_cal=ident, anchored=FittedModel.anchored,
        offset_cal={"spread": ident, "total": ident, "ml": ident},
        # blend = the market line it is given, so the model never disagrees with the market
        core=SimpleNamespace(blend=SimpleNamespace(predict=lambda d: d["spread_line"].to_numpy(float)),
                             total_blend=SimpleNamespace(predict=lambda d: d["total_line"].to_numpy(float))),
        win_prob=lambda m: 1 / (1 + np.exp(-m / 6.0)), k_spread=0.0)
    m.cal_for = lambda market, mode: FittedModel.cal_for(m, market, mode)
    return m


def game_row():
    g = dict(game_id="G", gameday="2026-10-11", gametime="13:00", stadium="X", location="Home",
             roof="dome", away_rest=7, home_rest=7, week=5, away_team="CIN", home_team="MIA",
             away_qb_id="QA", home_qb_id=QB, away_qb_name="Away QB", home_qb_name="Home QB",
             away_qb_delta=0.0, home_qb_delta=0.0, away_qb_rating=0.0, home_qb_rating=0.0,
             model_margin=-3.0, spread_line=-3.0, model_total=44.0, total_line=44.0,
             blend_margin=-3.0, blend_total=44.0, fair_margin=-3.0, fair_total=44.0,
             home_spread_odds=-110, away_spread_odds=-110, over_odds=-110, under_odds=-110,
             home_moneyline=np.nan, away_moneyline=np.nan)
    return next(pd.DataFrame([g]).itertuples(index=False))


def offers():
    rows = []
    for book, hp, ap in (("a", -110, -110), ("b", -110, -110), ("c", +125, -150)):
        rows += [dict(game_id="G", book=book, market="spread", side="home", point=3.0, price=hp,
                      source="odds_api", odds_time=None, odds_time_utc=T0),
                 dict(game_id="G", book=book, market="spread", side="away", point=-3.0, price=ap,
                      source="odds_api", odds_time=None, odds_time_utc=T0)]
    return pd.DataFrame(rows)


class DecisionTiers(unittest.TestCase):
    def decide(self, inj, feed_ok):
        m, g, o = fake_model(), game_row(), offers()
        ctx = build_context(g, inj, {}, {"f_qb": 0.0, "f_hfa": 2.0}, 0.0, {}, {}, Settings(), injury_feed_ok=feed_ok)
        refs = market.references_for_game(o, m, min_books=2)
        rows = decide_game(m, g, o, ctx, Settings(), refs=refs)
        return next(r for r in rows if r["market"] == "spread" and r["side"] == "home")

    def test_missing_injury_feed_blocks_executable_bet(self):
        r = self.decide(pd.DataFrame(), feed_ok=False)
        self.assertGreater(r["ev"], 0.05)              # clearly +EV price at book c
        self.assertEqual(r["book"], "c")
        self.assertEqual(r["decision"], "BET IF CONFIRMED")
        self.assertIn("Home QB confirmed", r["reasons"])

    def test_final_report_without_designation_allows_bet(self):
        inj = pd.DataFrame([("MIA", 5, "other", "Out", None), ("CIN", 5, "x", "Questionable", None)],
                           columns=["team", "week", "gsis_id", "report_status", "practice_status"])
        for col in ("full_name", "position", "report_primary_injury", "practice_primary_injury"):
            inj[col] = None
        r = self.decide(inj, feed_ok=True)
        self.assertEqual(r["decision"], "BET")

    def test_questionable_starter_is_no_bet(self):
        inj = pd.DataFrame([("MIA", 5, QB, "Questionable", None), ("CIN", 5, "x", "Out", None)],
                           columns=["team", "week", "gsis_id", "report_status", "practice_status"])
        for col in ("full_name", "position", "report_primary_injury", "practice_primary_injury"):
            inj[col] = None
        r = self.decide(inj, feed_ok=True)
        self.assertEqual(r["decision"], "NO BET")
