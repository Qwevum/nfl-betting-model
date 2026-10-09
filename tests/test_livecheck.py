"""Live-feed check with synthetic Odds API fixtures (no network, no real key)."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import livecheck  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.odds import odds_api_offers, validate_offers  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402

GAMES = pd.DataFrame([{"game_id": "G1", "home_team": "MIA", "away_team": "CIN"},
                      {"game_id": "G2", "home_team": "JAX", "away_team": "PHI"}])
KICK = {"G1": parse_utc("2026-10-11T17:00:00Z"), "G2": parse_utc("2026-10-11T13:30:00Z")}
COLLECTED = parse_utc("2026-10-11T12:00:00Z")


def book(key, updated, spread_home=-7.0, sp=(-110, -110), ml=(-300, 250)):
    return {"key": key, "last_update": updated, "markets": [
        {"key": "spreads", "last_update": updated, "outcomes": [
            {"name": "Miami Dolphins", "price": sp[0], "point": -spread_home},
            {"name": "Cincinnati Bengals", "price": sp[1], "point": spread_home}]},
        {"key": "h2h", "last_update": updated, "outcomes": [
            {"name": "Cincinnati Bengals", "price": ml[0]}, {"name": "Miami Dolphins", "price": ml[1]}]}]}


EVENTS = [
    {"home_team": "Miami Dolphins", "away_team": "Cincinnati Bengals", "commence_time": "2026-10-11T17:00:00Z",
     "bookmakers": [book("a", "2026-10-11T11:55:00Z"), book("b", "2026-10-11T11:50:00Z"),
                    book("c", "2026-10-11T11:58:00Z", sp=(-105, -115)),
                    book("stale", "2026-10-11T09:00:00Z")]},
    {"home_team": "Jacksonville Jaguars", "away_team": "Philadelphia Eagles", "commence_time": "2026-10-11T13:30:00Z",
     "bookmakers": [{"key": "a", "markets": [{"key": "h2h", "last_update": "2026-10-11T11:59:00Z", "outcomes": [
         {"name": "Jacksonville Jaguars", "price": -350}, {"name": "Philadelphia Eagles", "price": 280}]}]}]},
]


class Summary(unittest.TestCase):
    def setUp(self):
        offers = odds_api_offers(EVENTS, GAMES, KICK)
        self.valid, self.rejected = validate_offers(offers, KICK, COLLECTED, Settings(max_odds_age_minutes=30))
        self.s = livecheck.summarize(self.valid, self.rejected, KICK, COLLECTED, min_reference_books=2)

    def test_counts_books_and_rejections(self):
        self.assertEqual(self.s["books"], 3)                         # a, b, c (stale book rejected)
        self.assertEqual(self.s["rejected_quotes"], 4)               # stale: 2 spread + 2 ml sides
        self.assertEqual(self.s["reject_reasons"], {"stale": 4})
        self.assertEqual(self.s["games_with_quotes"], 2)

    def test_freshness_and_reference_readiness(self):
        a = self.s["age_minutes"]
        self.assertAlmostEqual(a["min"], 1.0)
        self.assertAlmostEqual(a["max"], 10.0)
        # G1 has 3 books two-sided -> each book has 2 others; G2 has only one book
        self.assertEqual(self.s["reference_ready"], {"spread": 1, "ml": 1, "total": 0})
        text = livecheck.format_summary(self.s, 2)
        self.assertIn("3 distinct books", text)


class KeyHandling(unittest.TestCase):
    def test_status_never_contains_value(self):
        with mock.patch.dict(os.environ, {"ODDS_API_KEY": "secret-xyz"}):
            ok, desc = livecheck.key_status()
            self.assertTrue(ok)
            self.assertNotIn("secret-xyz", desc)
            self.assertNotIn("secret-xyz", livecheck.redact("GET ...?apiKey=secret-xyz failed"))
            self.assertNotIn("secret-xyz", livecheck.describe_failure(RuntimeError("bad url apiKey=secret-xyz")))
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(livecheck.apikey, "ENV_FILE", Path("/nonexistent/.env")):
            self.assertFalse(livecheck.key_status()[0])
        with mock.patch.dict(os.environ, {"ODDS_API_KEY": "  "}):
            self.assertFalse(livecheck.key_status()[0])

    def test_failure_messages(self):
        self.assertIn("not reachable", livecheck.describe_failure(OSError("Tunnel connection failed: 403 Forbidden")))
        self.assertIn("401", livecheck.describe_failure(OSError("HTTP Error 401: Unauthorized")))
        self.assertIn("quota", livecheck.describe_failure(OSError("HTTP Error 429")))


if __name__ == "__main__":
    unittest.main()
