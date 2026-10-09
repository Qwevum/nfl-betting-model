"""QB confirmations and weather forecasts: timing, freshness and value validation,
and proof that rejected rows cannot unlock recommendations."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nflmodel import inputs, market  # noqa: E402
from nflmodel.config import Settings  # noqa: E402
from nflmodel.decide import build_context, decide_game  # noqa: E402
from nflmodel.timeutil import parse_utc  # noqa: E402
from test_availability import fake_model, game_row, offers  # noqa: E402

GID = "2026_05_CIN_MIA"
KICK = "2026-10-11T17:00:00Z"
DECISION = parse_utc("2026-10-11T15:55:00Z")
S = Settings()
DAY = pd.DataFrame([{"game_id": GID, "kickoff_utc": parse_utc(KICK), "home_team": "MIA", "away_team": "CIN"}])


def qb_file(rows):
    lines = ["game_id,kickoff_utc,team,qb_name,source,confirmed_utc"]
    lines += [",".join(str(x) for x in r) for r in rows]
    return "\n".join(lines) + "\n"


def wx_file(rows):
    lines = ["game_id,kickoff_utc,wind_mph,temp_f,source,forecast_utc,retrieved_utc,valid_for_utc"]
    lines += [",".join(str(x) for x in r) for r in rows]
    return "\n".join(lines) + "\n"


def load(kind, text, decision=DECISION, settings=S):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{kind}.csv"
        p.write_text(text)
        f = inputs.load_qb_overrides if kind == "qb" else inputs.load_weather
        return f(DAY, settings, decision, p)


GOOD_QB = (GID, KICK, "MIA", "Some QB", "team account", "2026-10-11T15:30:00Z")
GOOD_WX = (GID, KICK, 12, 71, "nws", "2026-10-11T12:00:00Z", "2026-10-11T12:30:00Z", KICK)


class QBConfirmations(unittest.TestCase):
    def reject(self, row, needle):
        ok, rej = load("qb", qb_file([row]))
        self.assertEqual(ok, {}, row)
        self.assertEqual(len(rej), 1)
        self.assertIn(needle, rej["reason"].iloc[0])

    def test_valid_row_accepted(self):
        ok, rej = load("qb", qb_file([GOOD_QB]))
        self.assertEqual(list(ok), [(GID, "MIA")])
        self.assertTrue(rej.empty)

    def test_future_confirmation_rejected(self):
        self.reject(GOOD_QB[:5] + ("2026-10-11T16:10:00Z",), "after the decision time")

    def test_post_kickoff_confirmation_rejected(self):
        ok, rej = load("qb", qb_file([GOOD_QB[:5] + ("2026-10-11T17:05:00Z",)]),
                       decision=parse_utc("2026-10-11T17:10:00Z"))
        self.assertEqual(ok, {})
        self.assertIn("kickoff", rej["reason"].iloc[0])

    def test_stale_confirmation_rejected_and_limit_configurable(self):
        old = GOOD_QB[:5] + ("2026-10-08T15:00:00Z",)        # ~73 h before the decision
        self.reject(old, "stale")
        ok, _ = load("qb", qb_file([old]), settings=Settings(qb_confirm_max_age_minutes=96 * 60))
        self.assertEqual(len(ok), 1)

    def test_missing_source_name_or_naive_time_rejected(self):
        self.reject(GOOD_QB[:4] + ("", GOOD_QB[5]), "source")
        self.reject(GOOD_QB[:3] + ("", "team", GOOD_QB[5]), "qb_name")
        self.reject(GOOD_QB[:5] + ("2026-10-11T15:30:00",), "timezone")
        self.reject(GOOD_QB[:5] + ("",), "confirmed_utc")


class WeatherForecasts(unittest.TestCase):
    def reject(self, row, needle, decision=DECISION):
        ok, rej = load("wx", wx_file([row]), decision=decision)
        self.assertEqual(ok, {}, row)
        self.assertEqual(len(rej), 1)
        self.assertIn(needle, rej["reason"].iloc[0])

    def test_valid_row_accepted(self):
        ok, rej = load("wx", wx_file([GOOD_WX]))
        self.assertEqual(ok[GID]["wind_mph"], 12.0)
        self.assertTrue(rej.empty)

    def test_future_and_post_kickoff_forecasts_rejected(self):
        self.reject(GOOD_WX[:5] + ("2026-10-11T16:30:00Z", "2026-10-11T16:31:00Z", KICK), "after the decision time")
        self.reject(GOOD_WX[:5] + ("2026-10-11T17:00:00Z", "2026-10-11T17:01:00Z", KICK), "kickoff",
                    decision=parse_utc("2026-10-11T17:30:00Z"))

    def test_stale_forecast_rejected(self):
        self.reject(GOOD_WX[:5] + ("2026-10-10T12:00:00Z", "2026-10-10T12:05:00Z", KICK), "stale")

    def test_invalid_values_rejected(self):
        for wind, temp, needle in ((-3, 70, "wind_mph"), ("nan", 70, "wind_mph"), ("inf", 70, "wind_mph"),
                                   (250, 70, "wind_mph"), ("calm", 70, "not numeric"), (10, "warm", "not numeric"),
                                   (10, 400, "temp_f"), (10, "nan", "temp_f")):
            self.reject((GID, KICK, wind, temp) + GOOD_WX[4:], needle)

    def test_source_retrieval_order_and_valid_for_window(self):
        self.reject(GOOD_WX[:4] + ("",) + GOOD_WX[5:], "source")
        self.reject(GOOD_WX[:6] + ("2026-10-11T11:00:00Z", KICK), "before forecast_utc")
        self.reject(GOOD_WX[:7] + ("2026-10-12T17:00:00Z",), "valid_for_utc")


class RejectedInputsCannotUnlock(unittest.TestCase):
    """Decision-level: a rejected override/forecast leaves the restriction in place."""

    def ctx_and_decision(self, overrides, weather, roof="outdoors"):
        m, o = fake_model(), offers()
        g = game_row()._replace(roof=roof, game_id=GID)
        o = o.assign(game_id=GID)
        ctx = build_context(g, pd.DataFrame(), {}, {"f_qb": 0.0, "f_hfa": 2.0}, 0.0, overrides, weather, S,
                            injury_feed_ok=False)
        refs = market.references_for_game(o, m, min_books=2)
        r = next(x for x in decide_game(m, g, o, ctx, S, refs=refs) if x["market"] == "spread" and x["side"] == "home")
        return ctx, r

    def test_rejected_qb_confirmation_keeps_bet_conditional(self):
        future = GOOD_QB[:5] + ("2026-10-11T16:30:00Z",)
        ok, _ = load("qb", qb_file([future]))
        _, r = self.ctx_and_decision(ok, {})
        self.assertEqual(r["decision"], "BET IF CONFIRMED")
        ok, _ = load("qb", qb_file([GOOD_QB, GOOD_QB[:2] + ("CIN",) + GOOD_QB[3:]]))
        _, r = self.ctx_and_decision(ok, {})
        self.assertEqual(r["decision"], "BET")

    def test_rejected_weather_keeps_outdoor_total_blocked(self):
        bad, _ = load("wx", wx_file([(GID, KICK, -5, 70) + GOOD_WX[4:]]))
        ctx, _ = self.ctx_and_decision({}, bad)
        self.assertTrue(any("weather" in b for b in ctx.block["total"]))
        good, _ = load("wx", wx_file([GOOD_WX]))
        ctx, _ = self.ctx_and_decision({}, good)
        self.assertFalse(any("weather" in b for b in ctx.block["total"]))


if __name__ == "__main__":
    unittest.main()


class WeatherIsNotAModelInput(unittest.TestCase):
    def test_wind_not_in_features_and_not_applied(self):
        from nflmodel.ratings import EXPERIMENT_GROUPS, FEATURES, TOTAL_FEATURES
        self.assertNotIn("t_wind", TOTAL_FEATURES)
        self.assertNotIn("t_wind", FEATURES)
        self.assertFalse(hasattr(inputs, "apply_weather"))
        self.assertFalse(any("t_wind" in m + t for m, t in EXPERIMENT_GROUPS.values()))

    def test_report_states_forecast_only_unlocks_total(self):
        good, _ = load("wx", wx_file([GOOD_WX]))
        g = game_row()._replace(roof="outdoors", game_id=GID)
        ctx = build_context(g, pd.DataFrame(), {}, {"f_qb": 0.0, "f_hfa": 2.0}, 0.0, {}, good, S, injury_feed_ok=False)
        self.assertTrue(any("not a model input" in a for a in ctx.assumptions))
        self.assertTrue(any("issued 2026-10-11T12:00:00Z" in f and "retrieved" in f for f in ctx.facts))


class WeatherArchive(unittest.TestCase):
    def test_archives_valid_forecasts_once_with_all_times(self):
        from nflmodel import store
        good, _ = load("wx", wx_file([GOOD_WX]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.jsonl"
            self.assertEqual(store.archive_weather(good, "run1", p), 1)
            self.assertEqual(store.archive_weather(good, "run2", p), 0)      # identical forecast not duplicated
            rec = store.read(p)[0]["data"]
            for k in ("forecast_utc", "retrieved_utc", "valid_for_utc", "kickoff_utc", "source", "wind_mph"):
                self.assertIsNotNone(rec[k], k)
            self.assertEqual(rec["kickoff_utc"], KICK)
            self.assertTrue(store.verify(p)[0])
