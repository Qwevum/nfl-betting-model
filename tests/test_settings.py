"""Settings are validated and actually drive decisions (stake cap, Kelly fraction, gap rule)."""
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import market  # noqa: E402
from nflmodel.config import Settings, SettingsError, load_settings  # noqa: E402
from nflmodel.decide import build_context, decide_game, stake_units  # noqa: E402
from test_availability import QB, fake_model, game_row, offers  # noqa: E402

FINAL_REPORT = pd.DataFrame([("MIA", 5, "other", "Out", None), ("CIN", 5, "x", "Questionable", None)],
                            columns=["team", "week", "gsis_id", "report_status", "practice_status"])
for col in ("full_name", "position", "report_primary_injury", "practice_primary_injury"):
    FINAL_REPORT[col] = None


def decide(settings: Settings, g=None):
    m, g, o = fake_model(), g or game_row(), offers()
    ctx = build_context(g, FINAL_REPORT, {}, {"f_qb": 0.0, "f_hfa": 2.0}, 0.0, {}, {}, settings, injury_feed_ok=True)
    refs = market.references_for_game(o, m, min_books=settings.min_reference_books)
    rows = decide_game(m, g, o, ctx, settings, refs=refs)
    return next(r for r in rows if r["market"] == "spread" and r["side"] == "home")


class Validation(unittest.TestCase):
    def test_defaults_are_valid(self):
        Settings()

    def test_rejects_nonsense(self):
        bad = [dict(kelly_fraction=0), dict(kelly_fraction=1.5), dict(kelly_fraction=-0.1),
               dict(max_stake_units=0), dict(max_stake_units=-1), dict(max_stake_units=math.nan),
               dict(max_stake_units=math.inf), dict(gap_points=0), dict(min_edge=1.0), dict(min_edge=-0.01),
               dict(min_reference_books=0), dict(min_reference_books=2.5), dict(max_odds_age_minutes=0),
               dict(horizon_minutes=-5), dict(kelly_fraction=True), dict(min_edge="0.02"),
               dict(clock_skew_minutes=30, max_odds_age_minutes=30)]
        for kw in bad:
            with self.assertRaises(SettingsError, msg=str(kw)):
                Settings(**kw)

    def test_toml_unknown_key_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.toml"
            p.write_text("max_stake_units = 1.0\nkelly_fraction = 0.5\n")
            s = load_settings(p, min_edge=0.03)
            self.assertEqual((s.max_stake_units, s.kelly_fraction, s.min_edge), (1.0, 0.5, 0.03))
            p.write_text("max_stake = 1.0\n")
            with self.assertRaises(SettingsError):
                load_settings(p)
            p.write_text("kelly_fraction = 0\n")
            with self.assertRaises(SettingsError):
                load_settings(p)


class DrivesDecisions(unittest.TestCase):
    def test_baseline_bet_exists(self):
        r = decide(Settings())
        self.assertEqual(r["decision"], "BET")
        self.assertGreater(r["kelly"], 0.05)

    def test_lowering_stake_cap_caps_stake(self):
        full = decide(Settings(kelly_fraction=1.0, max_stake_units=100))
        capped = decide(Settings(kelly_fraction=1.0, max_stake_units=0.5))
        self.assertGreater(full["stake_units"], 0.5)
        self.assertEqual(capped["stake_units"], 0.5)

    def test_kelly_fraction_scales_stake(self):
        q = decide(Settings(kelly_fraction=0.25, max_stake_units=100))["stake_units"]
        h = decide(Settings(kelly_fraction=0.5, max_stake_units=100))["stake_units"]
        self.assertAlmostEqual(h, 2 * q, delta=0.011)
        self.assertEqual(stake_units(0.08, Settings(kelly_fraction=0.25, max_stake_units=100)), 2.0)

    def test_gap_threshold_changes_decision(self):
        g = game_row()._replace(model_margin=3.0)                      # 6 pts from the line of -3
        self.assertEqual(decide(Settings(gap_points=10), g)["decision"], "BET")
        r = decide(Settings(gap_points=4), g)
        self.assertEqual(r["decision"], "NO BET")
        self.assertIn("gap_points 4", r["reasons"])

    def test_min_edge_changes_decision(self):
        r = decide(Settings())
        self.assertEqual(decide(replace(Settings(), min_edge=round(r["ev"] + 0.01, 3)))["decision"], "NO BET")


if __name__ == "__main__":
    unittest.main()
