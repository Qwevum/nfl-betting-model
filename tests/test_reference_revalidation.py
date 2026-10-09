"""Completion check revalidates the WHOLE market reference, not just the selected quote.

Synthetic market-only model: the model probability equals the market reference, so
any +EV comes from a book's price beating the leave-one-book-out reference."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import decide, pricing, runtime  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.odds import OFFER_COLS, validate_offers  # noqa: E402
from nflmodel.timeutil import fmt, parse_utc  # noqa: E402
from test_availability import fake_model, game_row  # noqa: E402

T = parse_utc("2026-10-11T15:00:00Z")          # collection time
KICK = T + pd.Timedelta(hours=2)
S = Settings(max_odds_age_minutes=30)


class StepClock(runtime.Clock):
    """Returns the given times in order, then repeats the last one."""

    def __init__(self, *times):
        super().__init__()
        self._times = list(times)

    def now(self):
        return self._times.pop(0) if len(self._times) > 1 else self._times[0]


def m(minutes):
    return T + pd.Timedelta(minutes=minutes)


def pair(book, home_price, away_price, home_age, away_age=None):
    away_age = home_age if away_age is None else away_age
    return [dict(game_id="G", book=book, market="spread", side="home", point=3.0, price=home_price,
                 source="odds_api", odds_time=fmt(m(-home_age))),
            dict(game_id="G", book=book, market="spread", side="away", point=-3.0, price=away_price,
                 source="odds_api", odds_time=fmt(m(-away_age)))]


def setup(books_rows, with_consensus=False):
    model = fake_model()
    model.predict = lambda d: d
    g = game_row()
    day = pd.DataFrame([g._asdict()]).assign(kickoff_utc=KICK)
    books = pd.DataFrame(books_rows, columns=OFFER_COLS)
    consensus = pd.DataFrame(columns=OFFER_COLS)
    if with_consensus:   # the live pipeline always carries untimed consensus lines as a fallback
        consensus = pd.DataFrame([dict(game_id="G", book="consensus", market="spread", side=sd, point=pt,
                                       price=-110, source="consensus", odds_time=None)
                                  for sd, pt in (("home", 3.0), ("away", -3.0))], columns=OFFER_COLS)

    def price(valid):
        return pricing.price_slate(model, day, valid, consensus, S, lambda gg: decide.GameContext())

    valid0, _ = validate_offers(books, {"G": KICK}, T, S)
    return price, books, valid0


def home_spread(priced):
    return priced.rows[(priced.rows["market"] == "spread") & (priced.rows["side"] == "home")].iloc[0]


class StaleReference(unittest.TestCase):
    def test_reproduction_fresh_offer_expired_reference_is_not_bet(self):
        # comparison books a, b quoted 29 min before collection; selected book c 1 min before
        price, books, valid0 = setup(pair("a", -110, -110, 29) + pair("b", -110, -110, 29)
                                     + pair("c", +125, -150, 1))
        at_collection = home_spread(price(valid0))
        self.assertEqual((at_collection["decision"], at_collection["book"]), ("BET", "c"))
        priced, info = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0)
        r = home_spread(priced)
        self.assertNotEqual(r["decision"], "BET")
        self.assertEqual(r["decision"], "BET IF PRICE AVAILABLE")
        self.assertIn("no live market reference", r["reasons"])
        self.assertFalse(r["reference_live"])
        self.assertEqual(info["repriced"], 1)
        self.assertEqual(info["expired_quotes"], 4)

    def test_one_side_of_a_pair_expiring_removes_that_book(self):
        # book a's home side is 29 min old, its away side 26 min; at +2 min the home side expires
        price, books, valid0 = setup(pair("a", -125, +105, 29, 26) + pair("b", -120, +100, 5)
                                     + pair("d", -110, -110, 5) + pair("c", +125, -150, 1))
        before = home_spread(price(valid0))
        self.assertIn("a", before["reference_books"].split(","))
        priced, _ = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0)
        after = home_spread(priced)
        self.assertEqual(after["book"], "c")
        self.assertEqual(after["reference_books"].split(","), ["b", "d"])

    def test_rebuild_keeps_exclusion_and_recomputes_dependent_values(self):
        price, books, valid0 = setup(pair("a", -125, +105, 29) + pair("b", -120, +100, 5)
                                     + pair("d", -110, -110, 5) + pair("c", +125, -150, 1))
        before = home_spread(price(valid0))
        priced, _ = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0)
        after = home_spread(priced)
        import json
        prov = json.loads(after["reference_provenance"])
        self.assertEqual(prov["excluded_book"], "c")
        self.assertNotIn("c", prov["books"])
        self.assertNotIn("a", prov["books"])
        for col in ("market_prob", "p_win", "ev", "kelly", "ev[fair 0.5 worse]", "ev[market only]"):
            self.assertNotAlmostEqual(before[col], after[col], places=6, msg=col)
        self.assertEqual(after["stake_units"], decide.stake_units(after["kelly"], S))   # stake from new Kelly
        self.assertEqual(after["decision"], "BET")       # still +EV against the rebuilt reference
        self.assertGreater(after["ev"], S.min_edge)
        self.assertGreater(parse_utc(after["reference_oldest_utc"]), m(2) - pd.Timedelta(minutes=30))
        self.assertEqual(set(priced.references["excluded_book"].dropna()), {"b", "c", "d"})

    def test_insufficient_remaining_books_is_explicit_downgrade(self):
        price, books, valid0 = setup(pair("a", -110, -110, 29) + pair("b", -110, -110, 5)
                                     + pair("c", +125, -150, 1))
        priced, _ = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0)
        r = home_spread(priced)
        self.assertEqual(r["decision"], "BET IF PRICE AVAILABLE")
        self.assertIn("fewer than 2 other books", r["reference"])

    def test_safeguard_when_repricing_is_not_allowed(self):
        price, books, valid0 = setup(pair("a", -110, -110, 29) + pair("b", -110, -110, 29)
                                     + pair("c", +125, -150, 1))
        priced, info = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0, max_reprice=0)
        r = home_spread(priced)
        self.assertFalse(info["stable"])
        self.assertEqual(r["decision"], "BET IF PRICE AVAILABLE")
        self.assertIn("market reference quotes expired", r["reasons"])

    def test_stable_quotes_keep_bet_and_do_not_reprice(self):
        price, books, valid0 = setup(pair("a", -110, -110, 5) + pair("b", -110, -110, 5)
                                     + pair("c", +125, -150, 1))
        priced, info = runtime.finalize(price, books, {"G": KICK}, StepClock(m(2)), S, valid0)
        self.assertEqual(home_spread(priced)["decision"], "BET")
        self.assertEqual((info["repriced"], info["stable"]), (0, True))


class KickoffDuringProcessing(unittest.TestCase):
    def test_kickoff_crossing_drops_the_game(self):
        price, books, valid0 = setup(pair("a", -110, -110, 5) + pair("b", -110, -110, 5)
                                     + pair("c", +125, -150, 1), with_consensus=True)
        priced, _ = runtime.finalize(price, books, {"G": KICK}, StepClock(KICK + pd.Timedelta(seconds=30)), S,
                                     valid0)
        r = home_spread(priced)
        self.assertEqual(r["decision"], "NO BET")
        self.assertTrue(r["post_kickoff"])
        self.assertEqual(r["stake_units"], 0.0)
        self.assertTrue(priced.rows["post_kickoff"].all())


if __name__ == "__main__":
    unittest.main()


class EmptyAfterRevalidation(unittest.TestCase):
    def test_no_offers_left_gives_no_rows_and_no_error(self):
        price, books, valid0 = setup(pair("a", -110, -110, 5) + pair("c", +125, -150, 1))
        priced, _ = runtime.finalize(price, books, {"G": KICK}, StepClock(KICK + pd.Timedelta(minutes=1)), S,
                                     valid0)
        self.assertTrue(priced.rows.empty)
        self.assertIn("post_kickoff", priced.rows.columns)
