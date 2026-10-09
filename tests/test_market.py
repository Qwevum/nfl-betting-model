"""Market reference from two-sided quotes. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import market  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.model import MARGINS, TOTALS, OutcomeDist  # noqa: E402
from nflmodel.odds import validate_offers  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402

MODEL = SimpleNamespace(margin_dist=OutcomeDist(13.0, np.ones(len(MARGINS)), MARGINS),
                        total_dist=OutcomeDist(13.5, np.ones(len(TOTALS)), TOTALS))
T0 = parse_utc("2026-10-11T15:50:00Z")


def fair_price(p: float) -> int:
    """American price with no vig for probability p."""
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def spread_quotes(book, home_point, p_home, t=T0, vig=0.0):
    """Two-sided spread quote; vig added proportionally to both sides."""
    ph, pa = p_home * (1 + vig), (1 - p_home) * (1 + vig)
    return [dict(game_id="G", book=book, market="spread", side="home", point=home_point,
                 price=fair_price(ph) if vig == 0 else _vig_price(ph), source="odds_api", odds_time_utc=t),
            dict(game_id="G", book=book, market="spread", side="away", point=-home_point,
                 price=fair_price(pa) if vig == 0 else _vig_price(pa), source="odds_api", odds_time_utc=t)]


def _vig_price(implied: float) -> int:
    return int(round(-100 * implied / (1 - implied))) if implied >= 0.5 else int(round(100 * (1 - implied) / implied))


def cond_over(mu, line):
    over, _, under = MODEL.margin_dist.prob_over(mu, line)
    return over / (over + under)


class Devig(unittest.TestCase):
    def test_minus110_both_sides_is_fifty_fifty(self):
        o = pd.DataFrame([dict(game_id="G", book="a", market="spread", side="home", point=-3, price=-110,
                               source="odds_api", odds_time_utc=T0),
                          dict(game_id="G", book="a", market="spread", side="away", point=3, price=-110,
                               source="odds_api", odds_time_utc=T0)])
        pairs = market.book_pairs(o, "spread")
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["p"], 0.5)
        self.assertEqual(pairs[0]["line"], 3)           # home -3 -> home covers if margin > 3
        self.assertAlmostEqual(pairs[0]["overround"], 2 * 110 / 210 - 1)


class DifferentLines(unittest.TestCase):
    def test_books_at_different_lines_recover_same_mean(self):
        mu = 3.2
        rows = (spread_quotes("a", -3.0, cond_over(mu, 3.0)) + spread_quotes("b", -3.5, cond_over(mu, 3.5))
                + spread_quotes("c", -2.5, cond_over(mu, 2.5)))
        ref = market.build_reference(pd.DataFrame(rows), "spread", MODEL, min_books=2)
        self.assertEqual(ref.n_books, 3)
        self.assertAlmostEqual(ref.mu, mu, delta=0.15)   # integer prices add rounding noise
        self.assertEqual(ref.line, 3.0)


class LeaveOneBookOut(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(spread_quotes("a", -3.0, 0.50) + spread_quotes("b", -3.0, 0.51)
                            + spread_quotes("outlier", -3.0, 0.70))

    def test_offer_book_is_excluded_from_its_reference(self):
        ref = market.build_reference(self.rows(), "spread", MODEL, min_books=2, exclude_book="outlier")
        self.assertNotIn("outlier", ref.books)
        self.assertLess(ref.p, 0.52)

    def test_median_resists_single_outlier(self):
        ref = market.build_reference(self.rows(), "spread", MODEL, min_books=2)
        self.assertLess(abs(ref.p - 0.51), 0.01)

    def test_too_few_other_books_gives_no_reference(self):
        two = pd.DataFrame(spread_quotes("a", -3.0, 0.5) + spread_quotes("b", -3.0, 0.5))
        self.assertIsNotNone(market.build_reference(two, "spread", MODEL, min_books=2))
        self.assertIsNone(market.build_reference(two, "spread", MODEL, min_books=2, exclude_book="a"))
        refs = market.references_for_game(two, MODEL, min_books=2)
        self.assertIsNone(refs[("spread", "a")])
        self.assertIsNotNone(refs[("spread", None)])


class Contemporaneity(unittest.TestCase):
    def test_sides_quoted_far_apart_are_not_paired(self):
        rows = spread_quotes("a", -3.0, 0.5)
        rows[1]["odds_time_utc"] = T0 + pd.Timedelta(minutes=10)
        self.assertEqual(market.book_pairs(pd.DataFrame(rows), "spread"), [])

    def test_reference_uses_latest_quote_per_book(self):
        old = spread_quotes("a", -3.0, 0.50, t=T0 - pd.Timedelta(minutes=20))
        new = spread_quotes("a", -3.0, 0.60, t=T0)
        pairs = market.book_pairs(pd.DataFrame(old + new), "spread")
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["p"], 0.60, delta=0.005)

    def test_stale_quotes_never_reach_the_reference(self):
        rows = (spread_quotes("fresh1", -3.0, 0.5) + spread_quotes("fresh2", -3.0, 0.5)
                + spread_quotes("stale", -3.0, 0.8, t=T0 - pd.Timedelta(hours=2)))
        raw = pd.DataFrame(rows).rename(columns={"odds_time_utc": "odds_time"})
        raw["odds_time"] = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in raw["odds_time"]]
        valid, rejected = validate_offers(raw, {"G": parse_utc("2026-10-11T17:00:00Z")},
                                          parse_utc("2026-10-11T16:00:00Z"), Settings(max_odds_age_minutes=30))
        ref = market.build_reference(valid, "spread", MODEL, min_books=2)
        self.assertEqual(ref.books, ["fresh1", "fresh2"])
        self.assertEqual(set(rejected["book"]), {"stale"})


class Moneyline(unittest.TestCase):
    def test_ml_reference_is_median_logit(self):
        rows = []
        for book, (h, a) in {"a": (-150, 130), "b": (-155, 135), "c": (-300, 250)}.items():
            rows += [dict(game_id="G", book=book, market="ml", side="home", point=np.nan, price=h,
                          source="odds_api", odds_time_utc=T0),
                     dict(game_id="G", book=book, market="ml", side="away", point=np.nan, price=a,
                          source="odds_api", odds_time_utc=T0)]
        ref = market.build_reference(pd.DataFrame(rows), "ml", MODEL, min_books=2)
        self.assertGreater(ref.p, 0.58)
        self.assertLess(ref.p, 0.62)     # the -300 outlier does not drag the median
        loo = market.build_reference(pd.DataFrame(rows), "ml", MODEL, min_books=2, exclude_book="c")
        self.assertEqual(loo.books, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
