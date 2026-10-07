"""Checks for the betting math. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel.backtest import grade, profit  # noqa: E402
from nflmodel.decide import price_for_edge  # noqa: E402
from nflmodel.model import MARGINS, OutcomeDist, logit, novig_first, sigmoid  # noqa: E402
from nflmodel.odds import decimal, ev, implied_probability, kelly, no_vig  # noqa: E402


class OddsMath(unittest.TestCase):
    def test_implied_probability(self):
        self.assertAlmostEqual(implied_probability(-110), 110 / 210)
        self.assertAlmostEqual(implied_probability(150), 0.4)
        self.assertAlmostEqual(implied_probability(100), 0.5)
        self.assertAlmostEqual(implied_probability(-200), 2 / 3)
        with self.assertRaises(ValueError):
            implied_probability(50)

    def test_implied_is_inverse_of_decimal(self):
        for a in (-300, -110, -105, 100, 120, 450):
            self.assertAlmostEqual(implied_probability(a), 1 / decimal(a))

    def test_no_vig_sums_to_one_and_is_symmetric(self):
        h, a = no_vig(-110, -110)
        self.assertAlmostEqual(h, 0.5)
        h, a = no_vig(-150, 130)
        self.assertAlmostEqual(h + a, 1.0)
        self.assertAlmostEqual(float(novig_first(-150, 130)), h)

    def test_ev_and_kelly(self):
        # 55% at -110: EV = 0.55 * 100/110 - 0.45 = +5.0%; Kelly = (b p - q) / b = 5.5%
        self.assertAlmostEqual(ev(0.55, 0.0, -110), 0.55 * (100 / 110) - 0.45)
        self.assertAlmostEqual(kelly(0.55, 0.0, -110), (100 / 110 * 0.55 - 0.45) / (100 / 110))
        # break-even price has zero EV and zero Kelly
        self.assertAlmostEqual(ev(110 / 210, 0.0, -110), 0.0)
        self.assertEqual(kelly(0.40, 0.0, -110), 0.0)
        # a push returns the stake: 50/10/40 at +100 -> +10%
        self.assertAlmostEqual(ev(0.5, 0.1, 100), 0.1)

    def test_price_for_edge_round_trip(self):
        for p_win, p_push, edge in ((0.55, 0.0, 0.02), (0.30, 0.0, 0.03), (0.50, 0.05, 0.02)):
            price = price_for_edge(p_win, p_push, edge)
            self.assertAlmostEqual(ev(p_win, p_push, price), edge, delta=0.01)


class Grading(unittest.TestCase):
    def test_spread(self):
        self.assertEqual(grade("spread", "home", -3, 24, 21), "P")
        self.assertEqual(grade("spread", "home", -3.5, 24, 21), "L")
        self.assertEqual(grade("spread", "away", 3.5, 24, 21), "W")

    def test_ml_and_total(self):
        self.assertEqual(grade("ml", "away", np.nan, 20, 23), "W")
        self.assertEqual(grade("total", "over", 44.5, 24, 21), "W")
        self.assertEqual(grade("total", "under", 45, 24, 21), "P")

    def test_profit(self):
        self.assertAlmostEqual(profit("W", -110), 100 / 110)
        self.assertEqual(profit("L", 250), -1.0)
        self.assertEqual(profit("P", -110), 0.0)


class Distribution(unittest.TestCase):
    def test_pmf_sums_to_one_and_symmetric(self):
        d = OutcomeDist(13.0, np.ones(len(MARGINS)), MARGINS)
        self.assertAlmostEqual(d.pmf(3.0).sum(), 1.0)
        over, push, under = d.prob_over(0.0, 0)
        self.assertAlmostEqual(over, under, places=6)

    def test_logit_sigmoid(self):
        for p in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(float(sigmoid(logit(p))), p)


if __name__ == "__main__":
    unittest.main()
