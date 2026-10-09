"""Offer validation, kickoff cutoffs and game-keyed inputs. Run: python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import inputs  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.odds import OFFER_COLS, check_game_keys, odds_api_offers, validate_offers  # noqa: E402
from nflmodel.timeutil import kickoff_utc, parse_utc  # noqa: E402

KICK = parse_utc("2026-10-11T17:00:00Z")
NOW = parse_utc("2026-10-11T16:00:00Z")
KICKOFFS = {"2026_05_CIN_MIA": KICK}
S = Settings(max_odds_age_minutes=30)


def offer(**kw):
    base = dict(game_id="2026_05_CIN_MIA", book="bk", market="spread", side="away", point=7.0,
                price=-110, source="odds_api", odds_time="2026-10-11T15:50:00Z")
    base.update(kw)
    return pd.DataFrame([base], columns=OFFER_COLS)


class TimeParsing(unittest.TestCase):
    def test_parse_requires_zone(self):
        self.assertEqual(parse_utc("2026-10-11T12:00:00-04:00"), parse_utc("2026-10-11T16:00:00Z"))
        for bad in ("2026-10-11T12:00:00", "", None, float("nan"), "yesterday"):
            with self.assertRaises(ValueError):
                parse_utc(bad)

    def test_kickoff_eastern_to_utc(self):
        self.assertEqual(kickoff_utc("2026-10-11", "13:00"), parse_utc("2026-10-11T17:00:00Z"))   # EDT
        self.assertEqual(kickoff_utc("2026-11-15", "13:00"), parse_utc("2026-11-15T18:00:00Z"))   # EST
        with self.assertRaises(ValueError):
            kickoff_utc("2026-10-11", None)


class OfferValidation(unittest.TestCase):
    def check(self, now=NOW, **kw):
        valid, rejected = validate_offers(offer(**kw), KICKOFFS, now, S)
        return valid, rejected

    def test_fresh_offer_passes(self):
        valid, rejected = self.check()
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(rejected), 0)

    def test_stale_offer_rejected(self):
        valid, rejected = self.check(odds_time="2026-10-11T15:29:00Z")   # 31 min old
        self.assertEqual(len(valid), 0)
        self.assertIn("stale", rejected["reject_reason"].iloc[0])

    def test_freshness_limit_is_configurable(self):
        valid, _ = validate_offers(offer(odds_time="2026-10-11T15:29:00Z"), KICKOFFS, NOW,
                                   Settings(max_odds_age_minutes=45))
        self.assertEqual(len(valid), 1)

    def test_naive_or_missing_timestamp_rejected(self):
        for t in ("2026-10-11T15:50:00", None, "soon"):
            valid, rejected = self.check(odds_time=t)
            self.assertEqual(len(valid), 0, t)
            self.assertIn("timestamp", rejected["reject_reason"].iloc[0])

    def test_future_timestamp_rejected(self):
        _, rejected = self.check(odds_time="2026-10-11T16:10:00Z")
        self.assertIn("future", rejected["reject_reason"].iloc[0])

    def test_quote_at_or_after_kickoff_rejected(self):
        now = parse_utc("2026-10-11T17:05:00Z")
        _, rejected = self.check(now=now, odds_time="2026-10-11T17:01:00Z")
        self.assertIn("kickoff", rejected["reject_reason"].iloc[0])

    def test_now_after_kickoff_rejects_even_old_quote(self):
        now = parse_utc("2026-10-11T17:00:00Z")
        _, rejected = self.check(now=now, odds_time="2026-10-11T16:55:00Z")
        self.assertIn("kickoff", rejected["reject_reason"].iloc[0])

    def test_bad_price_point_side_game(self):
        self.assertEqual(len(self.check(price=-50)[0]), 0)
        self.assertEqual(len(self.check(point=7.25)[0]), 0)
        self.assertEqual(len(self.check(market="total", side="over", point=-3)[0]), 0)
        self.assertEqual(len(self.check(side="over")[0]), 0)
        self.assertEqual(len(self.check(game_id="2026_05_XXX_YYY")[0]), 0)

    def test_stale_best_price_cannot_win_selection(self):
        both = pd.concat([offer(book="stale", price=+120, odds_time="2026-10-11T14:00:00Z"),
                          offer(book="fresh", price=-105)], ignore_index=True)
        valid, _ = validate_offers(both, KICKOFFS, NOW, S)
        self.assertEqual(valid["book"].tolist(), ["fresh"])


class GameKeys(unittest.TestCase):
    def test_check_game_keys(self):
        rows = pd.DataFrame([
            {"game_id": "2026_05_CIN_MIA", "kickoff_utc": "2026-10-11T17:00:00Z"},
            {"game_id": "2026_05_CIN_MIA", "kickoff_utc": "2026-10-12T17:00:00Z"},   # rescheduled/mismatch
            {"game_id": "2026_05_CIN_MIA", "kickoff_utc": "2026-10-11T13:00:00"},    # naive
            {"game_id": "2026_06_CIN_MIA", "kickoff_utc": "2026-10-11T17:00:00Z"},   # not in slate
        ])
        ok, bad = check_game_keys(rows, KICKOFFS, 10)
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(bad), 3)

    def test_qb_override_applies_only_to_named_game_and_team(self):
        day = pd.DataFrame([{"game_id": "2026_05_CIN_MIA", "kickoff_utc": KICK, "home_team": "MIA", "away_team": "CIN"}])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "qb.csv"
            p.write_text("game_id,kickoff_utc,team,qb_name,source,confirmed_utc\n"
                         "2026_05_CIN_MIA,2026-10-11T17:00:00Z,MIA,Some QB,team,2026-10-11T15:30:00Z\n"
                         "2026_05_CIN_MIA,2026-10-11T17:00:00Z,BUF,Other QB,team,2026-10-11T15:30:00Z\n"
                         "2026_05_CIN_MIA,2026-10-11T17:00:00Z,CIN,No Time QB,team,\n")
            got, notes = inputs.load_qb_overrides(day, 10, p)
        self.assertEqual(list(got), [("2026_05_CIN_MIA", "MIA")])
        self.assertEqual(len(notes), 2)

    def test_odds_api_matches_on_kickoff_not_just_teams(self):
        games = pd.DataFrame([{"game_id": "A", "home_team": "MIA", "away_team": "CIN"},
                              {"game_id": "B", "home_team": "MIA", "away_team": "CIN"}])
        kick = {"A": parse_utc("2026-10-11T17:00:00Z"), "B": parse_utc("2026-12-20T18:00:00Z")}
        ev = [{"home_team": "Miami Dolphins", "away_team": "Cincinnati Bengals",
               "commence_time": "2026-12-20T18:00:00Z",
               "bookmakers": [{"key": "bk", "markets": [{"key": "h2h", "last_update": "2026-12-20T16:00:00Z",
                                                         "outcomes": [{"name": "Miami Dolphins", "price": 150},
                                                                      {"name": "Cincinnati Bengals", "price": -170}]}]}]}]
        out = odds_api_offers(ev, games, kick)
        self.assertEqual(set(out["game_id"]), {"B"})


if __name__ == "__main__":
    unittest.main()
