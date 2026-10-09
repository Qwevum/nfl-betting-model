"""NO BET diagnostics: categories, funnel, watchlist; book comparison; prospective arms."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import diagnose, livecheck, report, track  # noqa: E402
from nflmodel.config import Settings  # noqa: E402

S = Settings()


def row(ev, decision="NO BET", sens=None, source="odds_api", live=True, blocks="", conditions="", venue=True,
        reasons="", min_price=-105, game="G1", market="spread", side="home", best=True):
    return dict(game_id=game, market=market, side=side, team="MIA", point=-3.0, price=-110, book="b",
                ev=ev, decision=decision, **{"ev[fair 0.5 worse]": ev - 0.03 if sens is None else sens},
                price_source=source, reference_live=live, blocks=blocks, conditions=conditions, venue_ok=venue,
                reasons=reasons, min_price=min_price, is_best_side=best, model_prob=0.5, market_prob=0.5)


ROWS = pd.DataFrame([
    row(0.05, decision="BET", game="A"),                                         # actionable
    row(-0.03, game="B"),                                                        # negative EV
    row(0.01, game="C"),                                                         # small positive
    row(0.04, sens=-0.01, game="D"),                                             # too sensitive
    row(0.05, source="consensus", live=False, decision="BET IF PRICE AVAILABLE", game="E"),   # no live data
    row(0.05, conditions="QB X confirmed", decision="BET IF CONFIRMED", game="F"),            # starter
    row(0.05, blocks="no valid weather forecast for an outdoor game", game="G", market="total", side="under"),
    row(0.05, blocks="model's own line differs from market by 5.0 pts", game="H"),            # gap rule
    row(0.05, venue=False, game="I"),                                            # not your book
    row(0.05, reasons="quote from x was stale by completion", decision="BET IF PRICE AVAILABLE", game="J"),
])


class Categories(unittest.TestCase):
    def test_each_case(self):
        c = dict(zip(ROWS["game_id"], diagnose.categorize(ROWS, S)))
        self.assertEqual(c, {"A": "actionable", "B": "negative estimated value",
                             "C": "small positive value below threshold", "D": "small positive value below threshold",
                             "E": "insufficient information", "F": "insufficient information",
                             "G": "insufficient information", "H": "blocked by rule", "I": "blocked by rule",
                             "J": "insufficient information"})

    def test_funnel_ends_at_actionable_count(self):
        f = diagnose.funnel(ROWS, S)
        self.assertEqual(f.iloc[0]["remaining"], len(ROWS))
        self.assertEqual(f.iloc[-1]["remaining"], int((ROWS["decision"] == "BET").sum()))
        self.assertTrue((f["remaining"].diff().dropna() <= 0).all())
        self.assertEqual(int(f["removed"].sum()), len(ROWS) - 1)

    def test_overlap_counts_multiple_reasons(self):
        r = pd.DataFrame([row(-0.01, source="consensus", live=False, conditions="QB", blocks="no valid weather x")])
        o = diagnose.overlap(r, S).set_index("reason")["sides"]
        self.assertEqual(int(o.sum()), 5)   # EV<=0, weather, no quote, no reference, starter

    def test_ev_distribution(self):
        d = diagnose.ev_distribution(ROWS, S).set_index("market")
        self.assertEqual(d.loc["all", "sides"], len(ROWS))
        self.assertEqual(d.loc["all", "> 0"], 9)


class Watchlist(unittest.TestCase):
    def test_never_contains_a_bet_and_is_sorted_and_labelled(self):
        w = diagnose.watchlist(ROWS, replace(S, watchlist_size=5))
        self.assertEqual(len(w), 5)
        self.assertNotIn("BET", set(w["decision"]))
        self.assertNotIn("actionable", set(w["category"]))
        self.assertTrue(w["ev"].is_monotonic_decreasing)
        self.assertTrue(all(isinstance(x, str) and x for x in w["why"]))
        self.assertEqual(w.iloc[0]["needs_price"], "-105")

    def test_status_labels_only_bet_is_a_recommendation(self):
        r = ROWS.assign(category=diagnose.categorize(ROWS, S))
        labels = [report.status_label(x) for _, x in r.iterrows()]
        self.assertEqual(labels[0], "BET (actionable)")
        self.assertTrue(all(lbl.startswith("not a bet") for lbl in labels[1:]))
        md = report.watchlist_md(diagnose.watchlist(ROWS, S), S, pd.DataFrame(
            [{"game_id": g, "away_team": "X", "home_team": "Y"} for g in ROWS["game_id"]]))
        self.assertIn("NOT recommendations", md)
        self.assertNotIn("BET (actionable)", md)

    def test_size_zero(self):
        self.assertTrue(diagnose.watchlist(ROWS, replace(S, watchlist_size=0)).empty)


class BookComparison(unittest.TestCase):
    def test_lines_prices_and_actionable_flags(self):
        v = pd.DataFrame([
            dict(game_id="G", book="draftkings", market="spread", side="home", point=-3.0, price=-110),
            dict(game_id="G", book="fanduel", market="spread", side="home", point=-3.5, price=+100),
            dict(game_id="G", book="betmgm", market="spread", side="home", point=-3.0, price=-105)])
        t = livecheck.compare_books(v, {"draftkings"}).iloc[0]
        self.assertEqual((t["books"], t["lines"]), (3, "-3.5,-3"))
        self.assertEqual(t["best"], "draftkings -110")       # best among YOUR books
        self.assertEqual(t["worst"], "draftkings -110")
        self.assertIn("fanduel (ref only) -3.5 +100", t["quotes"])
        self.assertEqual(livecheck.compare_books(v).iloc[0]["best"], "fanduel +100")


class Arms(unittest.TestCase):
    def test_market_only_and_model_arms_on_same_rows(self):
        fc = pd.DataFrame([
            dict(ev_raw_market=0.04, reference_live=True, price_source="odds_api", decision="NO BET",
                 result="W", flat_units=1.0, clv=0.1),
            dict(ev_raw_market=0.04, reference_live=True, price_source="odds_api", decision="BET",
                 result="L", flat_units=-1.0, clv=-0.1),
            dict(ev_raw_market=0.04, reference_live=False, price_source="consensus", decision="NO BET",
                 result="W", flat_units=1.0, clv=0.0),                    # not executable: not counted
            dict(ev_raw_market=np.nan, reference_live=True, price_source="odds_api", decision="BET",
                 result="W", flat_units=1.0, clv=0.0)])                    # older record: not counted
        t = track.arm_table(fc, S).set_index("arm")
        self.assertEqual(t.loc["price shopping (market only)", "sides flagged"], 2)
        self.assertEqual(t.loc["model-assisted (BET)", "sides flagged"], 1)
        self.assertEqual(t.loc["model-assisted (BET)", "W-L-P"], "0-1-0")


if __name__ == "__main__":
    unittest.main()
