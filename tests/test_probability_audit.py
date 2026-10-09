"""Regression tests for the 2026-10 probability audit.

1. Calibration must not create edges the market does not support: with the model
   adding nothing (edge 0), a book priced exactly at the leave-one-book-out no-vig
   reference has EV ~ 0 under "model" and "market", at the reference line AND at other
   lines. The legacy calibration (slope < 1 on market log-odds) fails this; that is the
   defect being fixed.
2. Leave-one-book-out holds for shared model inputs: the evaluated book's own quotes
   (including its line) cannot change its probability, through the anchor or through
   the blend (which takes the market line as an input).
3. Raw market, calibrated market-only and final probabilities are reported separately.
4. The report headline win probability equals the moneyline rows' final probability.
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import decide, market, report  # noqa: E402
from nflmodel.config import Settings, SettingsError  # noqa: E402
from nflmodel.model import FittedModel, OutcomeDist, MARGINS, TOTALS, logit  # noqa: E402
from nflmodel.odds import ev  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402
from test_availability import game_row  # noqa: E402

T0 = parse_utc("2026-10-11T15:00:00Z")
S = Settings()


def model(legacy=(-0.04, 0.36, 0.0), c=0.0, blend_shift=0.0):
    """Synthetic fitted model. legacy: the old [a, b, c] calibration (b < 1 shrinks the market);
    c: offset-calibration value of one point of model edge; blend_shift: the blend's opinion in
    points relative to the line it is given."""
    m = SimpleNamespace(
        margin_dist=OutcomeDist(13.0, np.ones(len(MARGINS)), MARGINS),
        total_dist=OutcomeDist(13.5, np.ones(len(TOTALS)), TOTALS),
        spread_cal=np.array(legacy), total_cal=np.array(legacy), ml_cal=np.array(legacy),
        offset_cal={k: np.array([0.0, 1.0, c]) for k in ("spread", "total", "ml")},
        anchored=FittedModel.anchored, win_prob=lambda x: 1 / (1 + np.exp(-x / 6.0)), k_spread=0.0,
        core=SimpleNamespace(
            blend=SimpleNamespace(predict=lambda d: d["spread_line"].to_numpy(float) + blend_shift),
            total_blend=SimpleNamespace(predict=lambda d: d["total_line"].to_numpy(float) + blend_shift)))
    m.cal_for = lambda mk, mode: FittedModel.cal_for(m, mk, mode)
    return m


def quotes(rows):
    """rows: (book, home_point, home_price, away_price)."""
    out = []
    for book, hp, hpr, apr in rows:
        out += [dict(game_id="G", book=book, market="spread", side="home", point=hp, price=hpr, source="odds_api",
                     odds_time="2026-10-11T14:58:00Z", odds_time_utc=T0),
                dict(game_id="G", book=book, market="spread", side="away", point=-hp, price=apr, source="odds_api",
                     odds_time="2026-10-11T14:58:00Z", odds_time_utc=T0)]
    return pd.DataFrame(out)


def refs_for(offers, m):
    return market.references_for_game(offers, m, S.min_reference_books)


class NoArtificialEdge(unittest.TestCase):
    """Books a, b define the market; book c is priced exactly at a/b's no-vig fair odds."""

    def setUp(self):
        self.m = model()
        g = game_row()._asdict()
        # market moved: consensus says home -3 at -110, live books say home -4.5 with skewed juice
        g.update(spread_line=3.0, blend_margin=3.0, fair_margin=3.0)
        self.g = next(pd.DataFrame([g]).itertuples(index=False))
        o = quotes([("a", -4.5, -125, +105), ("b", -4.5, -125, +105)])
        self.refs = refs_for(o, self.m)
        # c does not quote, so its leave-one-book-out reference is the a+b reference
        self.r_c = {mk: self.refs.get((mk, None)) for mk in ("spread", "ml", "total")}
        assert self.r_c["spread"] is not None and self.r_c["spread"].line == 4.5

    def fair_ev(self, mode, point):
        pr = decide.make_pricer(self.m, self.g, replace(S, probability_model=mode), refs=self.r_c)
        p_ref = pr.reference_prob("spread", "home", point)            # raw ex-book market, no vig
        fair_price = (100 * (1 - p_ref) / p_ref) if p_ref < 0.5 else (-100 * p_ref / (1 - p_ref))
        pw, pp = pr.probs("spread", "home", point)
        return ev(pw, pp, fair_price), p_ref, pw / (1 - pp)

    def test_model_and_market_modes_have_no_edge_at_fair_price(self):
        for mode in ("model", "market"):
            for point in (-4.5, -3.5, -3.0, -6.5):
                e, p_ref, p_final = self.fair_ev(mode, point)
                self.assertAlmostEqual(p_final, p_ref, places=9, msg=(mode, point))
                self.assertLess(abs(e), 0.003, msg=(mode, point))

    def test_legacy_calibration_invents_an_edge(self):
        # the defect: at the market's own line the legacy calibration moves the no-vig probability by
        # more than a point, and EV at a fair price looks like an edge of several percent
        e, p_ref, p_final = self.fair_ev("legacy", -4.5)
        self.assertGreater(abs(p_final - p_ref), 0.01)
        self.assertGreater(abs(e), 0.02)


class LeaveOneBookOut(unittest.TestCase):
    def test_own_quotes_never_move_own_probability(self):
        m = model(c=0.05, blend_shift=1.0)   # the model has an opinion, and the blend reads the line
        g = game_row()
        base = [("a", -3.0, -110, -110), ("b", -3.5, -105, -115), ("d", -3.0, -115, -105)]
        results = []
        for c_line, c_home, c_away in ((-3.0, +100, -120), (-7.0, -110, -110), (+2.0, -200, +170)):
            o = quotes(base + [("c", c_line, c_home, c_away)])
            refs = refs_for(o, m)
            pr = decide.make_pricer(m, g, S, refs={mk: refs.get((mk, "c")) for mk in ("spread", "ml", "total")})
            results.append((pr.probs("spread", "home", -3.0), pr.edge["spread"], pr.anchor_line["spread"]))
        for r in results[1:]:
            self.assertEqual(r, results[0])

    def test_blend_recomputed_at_the_ex_book_line(self):
        m = model(c=0.05, blend_shift=1.0)
        g = game_row()   # consensus spread_line -3.0 with its own blend
        o = quotes([("a", -6.0, -110, -110), ("b", -6.0, -110, -110), ("c", -3.0, -110, -110)])
        refs = refs_for(o, m)
        pr = decide.make_pricer(m, g, S, refs={mk: refs.get((mk, "c")) for mk in ("spread", "ml", "total")})
        self.assertEqual(pr.anchor_line["spread"], 6.0)          # home -6 in nflverse margin terms
        self.assertAlmostEqual(pr.edge["spread"], 1.0)           # blend at 6.0 is 7.0: edge vs 6.0

    def test_slate_lines_not_overwritten_by_all_books_reference(self):
        from nflmodel import pricing
        m = model()
        m.predict = lambda d: d
        g = game_row()
        day = pd.DataFrame([g._asdict()]).assign(kickoff_utc=T0 + pd.Timedelta(hours=2))
        o = quotes([("a", -6.0, -110, -110), ("b", -6.0, -110, -110), ("c", -6.0, -110, -110)])
        o = o.assign(odds_time_utc=T0)
        priced = pricing.price_slate(m, day, o, pd.DataFrame(columns=o.columns), S, lambda gg: decide.GameContext())
        self.assertEqual(priced.preds["spread_line"].iloc[0], -3.0)        # consensus kept for shared inputs
        self.assertEqual(priced.preds["live_spread_line"].iloc[0], 6.0)    # all-books line only for display


class ExplicitProbabilities(unittest.TestCase):
    def test_rows_carry_raw_calibrated_and_final(self):
        m = model(c=0.05, blend_shift=2.0)
        g = game_row()
        o = quotes([("a", -3.0, -110, -110), ("b", -3.0, -110, -110), ("c", -3.0, +105, -125)])
        rows = pd.DataFrame(decide.decide_game(m, g, o, decide.GameContext(), S, refs=refs_for(o, m)))
        r = rows[rows["side"] == "home"].iloc[0]
        self.assertAlmostEqual(r["market_prob"], r["market_cal_prob"], places=9)   # offset: calibration = raw
        self.assertNotAlmostEqual(r["model_prob"], r["market_prob"], places=4)    # model edge moves it
        self.assertEqual(r["probability_model"], "model")
        self.assertAlmostEqual(r["ev_raw_market"], r["ev[market only]"], places=3)

    def test_headline_matches_moneyline_rows(self):
        rows = pd.DataFrame([dict(market="ml", side="home", model_prob=0.6123, market_prob=0.6001),
                             dict(market="ml", side="away", model_prob=0.3877, market_prob=0.3999)])
        g = SimpleNamespace(home_team="MIA", fair_margin=10.0)   # margin-based estimate would differ a lot
        line = report.headline_win_prob(g, rows)
        self.assertIn("61.2%", line)
        self.assertIn("raw market 60.0%", line)


class ActionableBooks(unittest.TestCase):
    def test_reference_books_are_not_actionable(self):
        m = model()
        g = game_row()
        o = quotes([("a", -3.0, -110, -110), ("b", -3.0, -110, -110), ("c", -3.0, +125, -150),
                    ("fanduel", -3.0, -108, -112)])
        refs = refs_for(o, m)
        mine = replace(S, actionable_books="fanduel")
        rows = pd.DataFrame(decide.decide_game(m, g, o, decide.GameContext(), mine, refs=refs))
        home = rows[rows["side"] == "home"].iloc[0]
        self.assertEqual(home["book"], "fanduel")         # the +125 at c is reference-only
        self.assertTrue(home["venue_ok"])
        everyone = pd.DataFrame(decide.decide_game(m, g, o, decide.GameContext(), S, refs=refs))
        self.assertEqual(everyone[everyone["side"] == "home"].iloc[0]["book"], "c")

    def test_no_actionable_quote_is_never_a_bet(self):
        m = model()
        g = game_row()
        o = quotes([("a", -3.0, -110, -110), ("b", -3.0, -110, -110), ("c", -3.0, +125, -150)])
        rows = pd.DataFrame(decide.decide_game(m, g, o, decide.GameContext(),
                                               replace(S, actionable_books="draftkings"), refs=refs_for(o, m)))
        home = rows[rows["side"] == "home"].iloc[0]
        self.assertFalse(home["venue_ok"])
        self.assertEqual(home["decision"], "NO BET")
        self.assertIn("books you don't use", home["reasons"])

    def test_settings_validation(self):
        Settings(actionable_books="draftkings, fanduel")
        for bad in ("Draft Kings", "fan-duel"):
            with self.assertRaises(SettingsError):
                Settings(actionable_books=bad)
        with self.assertRaises(SettingsError):
            Settings(probability_model="aggressive")


class OffsetCalibration(unittest.TestCase):
    def test_offset_fit_recovers_c_and_keeps_market_weight(self):
        from nflmodel.model import _logistic_offset
        rng = np.random.default_rng(0)
        p_mkt = rng.uniform(0.3, 0.7, 20000)
        x = rng.normal(0, 3, 20000)
        y = (rng.uniform(size=20000) < 1 / (1 + np.exp(-(logit(p_mkt) + 0.08 * x)))).astype(float)
        self.assertAlmostEqual(_logistic_offset(x, logit(p_mkt), y), 0.08, delta=0.015)
        y0 = (rng.uniform(size=20000) < p_mkt).astype(float)
        self.assertAlmostEqual(_logistic_offset(x, logit(p_mkt), y0), 0.0, delta=0.012)


if __name__ == "__main__":
    unittest.main()
