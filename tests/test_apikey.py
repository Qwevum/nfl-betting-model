"""Odds API key lookup (env var, then .env), redaction, regions setting and request URL."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import apikey, livecheck, odds  # noqa: E402
from nflmodel.config import Settings, SettingsError  # noqa: E402

FAKE = "0123456789abcdef0123456789abcdef"   # synthetic, not a real key


def env_file(text):
    d = tempfile.mkdtemp()
    p = Path(d) / ".env"
    p.write_text(text)
    return p


class Lookup(unittest.TestCase):
    def test_env_var_wins_over_file(self):
        with mock.patch.dict(os.environ, {"ODDS_API_KEY": "fromenv"}):
            self.assertEqual(apikey.lookup(env_file(f"ODDS_API_KEY={FAKE}\n")), ("fromenv", "environment variable"))

    def test_env_file_parsing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            for text in (f"ODDS_API_KEY={FAKE}", f"# comment\nOTHER=1\nexport ODDS_API_KEY=\"{FAKE}\"\n",
                         f"ODDS_API_KEY = '{FAKE}'"):
                key, src = apikey.lookup(env_file(text))
                self.assertEqual((key, src), (FAKE, ".env file"))

    def test_missing_and_empty(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(apikey.lookup(Path(tempfile.mkdtemp()) / ".env"), (None, "not set"))
            self.assertEqual(apikey.lookup(env_file("ODDS_API_KEY=\n")), (None, ".env file"))

    def test_status_never_contains_value_and_redaction(self):
        with mock.patch.dict(os.environ, {"ODDS_API_KEY": FAKE}):
            ok, desc = livecheck.key_status()
            self.assertTrue(ok)
            self.assertNotIn(FAKE, desc)
            msg = livecheck.redact(f"failed: https://x/?apiKey={FAKE}&regions=us")
            self.assertNotIn(FAKE, msg)
            self.assertIn("***", msg)

    def test_env_file_is_gitignored(self):
        lines = (Path(__file__).resolve().parent.parent / ".gitignore").read_text().split()
        self.assertIn(".env", lines)


class Regions(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(Settings().odds_api_regions, "us")
        Settings(odds_api_regions="us,us2")
        for bad in ("", "mars", "us,,", 3):
            with self.assertRaises(SettingsError):
                Settings(odds_api_regions=bad)

    def test_request_url_uses_regions_and_key_is_not_recorded(self):
        seen = {}

        def fake_fetch(url, label=None):
            seen["url"] = url
            return json.dumps([]).encode()

        with mock.patch.object(odds, "_fetch", fake_fetch):
            out = odds.fetch_odds_api(pd.DataFrame(columns=["game_id", "home_team", "away_team"]), {}, FAKE,
                                      "us, us2")
        self.assertTrue(out.empty)
        self.assertIn("regions=us,us2&", seen["url"])
        self.assertIn("markets=h2h,spreads,totals", seen["url"])

    def test_quota_line(self):
        self.assertIsNone(livecheck.quota_line({}))
        line = livecheck.quota_line({"odds_api": {"quota": {"x-requests-used": "3", "x-requests-remaining": "497"}}})
        self.assertEqual(line, "The Odds API credits: used 3, remaining 497")


if __name__ == "__main__":
    unittest.main()
