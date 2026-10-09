"""Forecasts with uncertainty (docs/UNCERTAINTY.md): leakage, reproducibility, coherence,
interval ordering, push-aware EV, reports and compatibility with historical logs.

All data here is SYNTHETIC (generated below); nothing reads or writes the real logs."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflmodel import betrange, forecast, report, store, uncertainty as unc  # noqa: E402
from nflmodel.config import Settings, SettingsError, load_settings  # noqa: E402
from nflmodel.decide import AnchoredPricer, decide_game, GameContext, best_per_market  # noqa: E402
from nflmodel.model import MARGINS, TOTALS, fit  # noqa: E402
from nflmodel.odds import consensus_offers, ev  # noqa: E402
from nflmodel.ratings import FEATURES, TOTAL_FEATURES  # noqa: E402

S = Settings()
TEAMS = [f"T{i:02d}" for i in range(16)]


def synthetic(seasons=range(2011, 2019), games_per_season=64, seed=0) -> pd.DataFrame:
    """Feature table shaped like ratings.build_features output, with a known linear truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for i in range(games_per_season):
            week = 1 + i // 4
            h, a = rng.choice(TEAMS, 2, replace=False)
            f = {k: rng.normal(0, 1) for k in FEATURES + TOTAL_FEATURES}
            f["f_hfa"], f["f_div"] = 1.0, float(rng.random() < 0.3)
            true_m = 2.0 + 3.0 * f["f_mrtg"] + 1.5 * f["f_epa"] + 0.8 * f["f_qb"]
            true_t = 44 + 3.0 * f["t_off"] - 2.0 * f["t_def"]
            line = np.round((true_m + rng.normal(0, 1.5)) * 2) / 2
            tline = np.round((true_t + rng.normal(0, 1.5)) * 2) / 2
            res = float(np.round(true_m + rng.normal(0, 13)))
            tot = float(max(3, np.round(true_t + rng.normal(0, 13))))
            pm = 1 / (1 + np.exp(-line / 6.5))
            hm = -round(100 * pm / (1 - pm)) if pm >= 0.5 else round(100 * (1 - pm) / pm)
            am = -round(100 * (1 - pm) / pm) if pm < 0.5 else round(100 * pm / (1 - pm))
            rows.append({**f, "game_id": f"{s}_{week:02d}_{a}_{h}_{i}", "season": s, "week": week,
                         "gameday": pd.Timestamp(f"{s}-09-07") + pd.Timedelta(days=7 * (week - 1)),
                         "home_team": h, "away_team": a, "game_type": "REG", "result": res, "total": tot,
                         "spread_line": line, "total_line": tline, "home_spread_odds": -110.0,
                         "away_spread_odds": -110.0, "over_odds": -110.0, "under_odds": -110.0,
                         "home_moneyline": float(hm), "away_moneyline": float(am),
                         "kickoff_utc": pd.Timestamp(f"{s}-09-07 17:00", tz="UTC") + pd.Timedelta(days=7 * (week - 1))})
    return pd.DataFrame(rows)


FEAT = synthetic()
TRAIN = FEAT[FEAT["season"] < 2018].reset_index(drop=True)
TEST = FEAT[FEAT["season"] == 2018].reset_index(drop=True)
POINT = fit(TRAIN)
REPS = unc.fit_replicates(TRAIN, 12, seed=7, scheme="season", use_cache=False)


class WeightedFit(unittest.TestCase):
    def test_unweighted_path_unchanged_and_ones_equivalent(self):
        a = fit(TRAIN)
        b = fit(TRAIN, weights=np.ones(len(TRAIN)))
        np.testing.assert_allclose(a.core.blend.coef_, b.core.blend.coef_, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a.offset_cal["spread"], b.offset_cal["spread"], rtol=1e-7, atol=1e-10)
        self.assertAlmostEqual(a.margin_dist.sd, b.margin_dist.sd, places=9)
        np.testing.assert_allclose(a.margin_dist.w, b.margin_dist.w, rtol=1e-9)
        self.assertNotIn(unc.model_mod.WEIGHT, TRAIN.columns)   # the caller's frame is not modified

    def test_bad_weights_rejected(self):
        for w in (np.ones(3), -np.ones(len(TRAIN)), np.full(len(TRAIN), np.nan)):
            with self.assertRaises(ValueError):
                fit(TRAIN, weights=w)

    def test_weights_change_the_fit(self):
        w = unc.replicate_weights(TRAIN, "season", 1, 0)
        self.assertFalse(np.allclose(fit(TRAIN, weights=w).core.blend.coef_, POINT.core.blend.coef_))


class Chronology(unittest.TestCase):
    def test_future_games_cannot_change_replicates_or_forecasts(self):
        """Only rows before the cutoff are fit; scrambling every later outcome changes nothing."""
        feat2 = FEAT.copy()
        later = feat2["season"] >= 2018
        feat2.loc[later, "result"] = -feat2.loc[later, "result"] + 30
        feat2.loc[later, "total"] = feat2.loc[later, "total"] + 25
        train2 = feat2[feat2["season"] < 2018].reset_index(drop=True)
        reps2 = unc.fit_replicates(train2, 12, seed=7, scheme="season", use_cache=False)
        for a, b in zip(REPS, reps2):
            np.testing.assert_array_equal(a.core.blend.coef_, b.core.blend.coef_)
        f1 = forecast.forecast_slate(POINT, TEST, None, S, REPS, 0.9)
        f2 = forecast.forecast_slate(fit(train2), feat2[feat2["season"] == 2018].reset_index(drop=True), None, S,
                                     reps2, 0.9)
        np.testing.assert_allclose(f1["p_home"], f2["p_home"])
        np.testing.assert_allclose(f1["p_home_lo"], f2["p_home_lo"])

    def test_weights_keep_every_row_in_its_season(self):
        """Blocks never span seasons, so a replicate cannot move information across a season boundary;
        Exp(1) weights are positive, so no season disappears from the inner walk-forward."""
        for scheme in unc.SCHEMES:
            a, b = unc.block_ids(TRAIN, scheme)
            lab = pd.Series(a if scheme != "season*week4" else b)
            self.assertTrue((lab.groupby(lab).apply(lambda x: TRAIN.loc[x.index, "season"].nunique()) == 1).all()
                            if scheme != "game" else True)
            w = unc.replicate_weights(TRAIN, scheme, 3, 0)
            self.assertTrue(np.all(w > 0))
            self.assertAlmostEqual(w.mean(), 1.0)

    def test_season_weights_are_constant_within_season(self):
        w = pd.Series(unc.replicate_weights(TRAIN, "season", 3, 5))
        self.assertTrue((w.groupby(TRAIN["season"]).nunique() == 1).all())


class Reproducibility(unittest.TestCase):
    def test_same_seed_same_replicates(self):
        again = unc.fit_replicates(TRAIN, 12, seed=7, scheme="season", use_cache=False)
        for a, b in zip(REPS, again):
            np.testing.assert_array_equal(a.core.blend.coef_, b.core.blend.coef_)

    def test_replicate_b_independent_of_count_and_start(self):
        five = unc.fit_replicates(TRAIN, 5, seed=7, scheme="season", use_cache=False)
        tail = unc.fit_replicates(TRAIN, 3, seed=7, scheme="season", use_cache=False, start=2)
        for a, b in zip(REPS[:5], five):
            np.testing.assert_array_equal(a.core.blend.coef_, b.core.blend.coef_)
        for a, b in zip(REPS[2:5], tail):
            np.testing.assert_array_equal(a.core.blend.coef_, b.core.blend.coef_)

    def test_different_seed_differs(self):
        other = unc.fit_replicates(TRAIN, 2, seed=8, scheme="season", use_cache=False)
        self.assertFalse(np.allclose(other[0].core.blend.coef_, REPS[0].core.blend.coef_))

    def test_cache_key_tracks_inputs(self):
        k = unc.cache_key(TRAIN, 10, 1, "season")
        self.assertEqual(k, unc.cache_key(TRAIN.copy(), 10, 1, "season"))
        changed = TRAIN.copy()
        changed.loc[0, "result"] += 1
        for other in (unc.cache_key(changed, 10, 1, "season"), unc.cache_key(TRAIN, 11, 1, "season"),
                      unc.cache_key(TRAIN, 10, 2, "season"), unc.cache_key(TRAIN, 10, 1, "game"),
                      unc.cache_key(TRAIN.iloc[:-1], 10, 1, "season")):
            self.assertNotEqual(k, other)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            old = unc.CACHE_DIR
            unc.CACHE_DIR = Path(d)
            try:
                a = unc.fit_replicates(TRAIN, 2, seed=3, scheme="week4")
                self.assertTrue(unc.is_cached(TRAIN, 2, 3, "week4"))
                b = unc.fit_replicates(TRAIN, 2, seed=3, scheme="week4")
                np.testing.assert_array_equal(a[1].core.blend.coef_, b[1].core.blend.coef_)
            finally:
                unc.CACHE_DIR = old


class Coherence(unittest.TestCase):
    FC = forecast.forecast_slate(POINT, TEST, None, S, REPS, 0.9, pi_levels=(0.5, 0.8, 0.95))

    def test_outcome_probabilities_sum_to_one(self):
        f = self.FC
        np.testing.assert_allclose(f["p_home"] + f["p_away"] + f["p_tie"], 1.0, atol=1e-12)
        self.assertTrue((f["p_tie"] > 0).all())
        self.assertTrue(((f["winner"] == f["home_team"]) == (f["p_home"] >= f["p_away"])).all())

    def test_postseason_has_no_ties(self):
        post = TEST.head(3).assign(game_type="WC")
        f = forecast.forecast_slate(POINT, post, None, S, REPS, 0.9)
        np.testing.assert_allclose(f["p_tie"], 0.0)
        np.testing.assert_allclose(f["p_home"] + f["p_away"], 1.0)

    def test_matches_betting_probability_at_anchor_line(self):
        """The forecast distribution reproduces the final P(home covers) the betting side uses."""
        pred = POINT.predict(TEST)
        b = forecast.batch(POINT, pred)
        for i, g in enumerate(pred.head(20).itertuples(index=False)):
            pr = AnchoredPricer(POINT, g)
            pw, pp = pr.probs("spread", "home", -g.spread_line)
            over, _, under = POINT.margin_dist.prob_over(b.mu_margin[i], g.spread_line)
            self.assertAlmostEqual(over / (over + under), pw / (1 - pp), places=10)

    def test_ratings_only_when_no_market_and_none_when_no_data(self):
        nomkt = TEST.head(2).assign(spread_line=np.nan, total_line=np.nan, home_moneyline=np.nan,
                                    away_moneyline=np.nan)
        f = forecast.forecast_slate(POINT, nomkt, None, S, REPS, 0.9)
        self.assertTrue((f["margin_basis"] == "team ratings only").all())
        self.assertTrue((f["total_basis"] == "team ratings only").all())
        nodata = nomkt.assign(f_mrtg=np.nan, t_off=np.nan)
        f = forecast.forecast_slate(POINT, nodata, None, S, REPS, 0.9)
        self.assertTrue(f["p_home"].isna().all())
        self.assertTrue(f["p_home_lo"].isna().all())          # no invented bounds
        self.assertTrue(f["margin_lo_90"].isna().all())
        self.assertTrue(f["margin_source"].str.startswith("insufficient data").all())

    def test_ratings_only_is_wider(self):
        nomkt = TEST.head(5).assign(spread_line=np.nan, home_moneyline=np.nan, away_moneyline=np.nan)
        a = forecast.forecast_slate(POINT, TEST.head(5), None, S, None, 0.9)
        b = forecast.forecast_slate(POINT, nomkt, None, S, None, 0.9)
        self.assertGreaterEqual(POINT.pure_margin_sd, POINT.margin_dist.sd * 0.9)
        self.assertTrue(((b["margin_hi_90"] - b["margin_lo_90"]) >= (a["margin_hi_90"] - a["margin_lo_90"]) - 1).all())


class IntervalOrdering(unittest.TestCase):
    FC = Coherence.FC

    def test_probability_intervals_contain_estimate(self):
        f = self.FC
        self.assertTrue((f["p_home_lo"] <= f["p_home"]).all() and (f["p_home"] <= f["p_home_hi"]).all())
        self.assertTrue((f["p_away_lo"] <= f["p_away"]).all() and (f["p_away"] <= f["p_away_hi"]).all())
        self.assertTrue(((f["p_home_lo"] > 0) & (f["p_home_hi"] < 1)).all())

    def test_prediction_intervals_nested_by_level(self):
        f = self.FC
        for name in ("margin", "total"):
            for a, b in (("50", "80"), ("80", "90"), ("90", "95")):
                self.assertTrue((f[f"{name}_lo_{b}"] <= f[f"{name}_lo_{a}"]).all())
                self.assertTrue((f[f"{name}_hi_{a}"] <= f[f"{name}_hi_{b}"]).all())
            self.assertTrue((f[f"{name}_lo_90"] <= f[f"{name}_median"]).all())
            self.assertTrue((f[f"{name}_median"] <= f[f"{name}_hi_90"]).all())

    def test_outcome_interval_is_not_the_spread_of_means(self):
        """The PI includes game-to-game variability: far wider than the replicates' means vary."""
        f = self.FC
        pred = [forecast.batch(m, m.predict(TEST)).mu_margin for m in REPS]
        spread_of_means = np.ptp(np.array(pred), axis=0)
        width = f["margin_hi_90"] - f["margin_lo_90"]
        self.assertTrue((width > 3 * spread_of_means).all())
        self.assertTrue((width >= 2 * 1.6 * POINT.margin_dist.sd * 0.9).all())   # game-to-game variability

    def test_pmf_interval_coverage_and_levels(self):
        p = POINT.margin_dist.pmf(3.0)
        for lv in (0.5, 0.8, 0.9, 0.95):
            lo, hi = unc.pmf_interval(p, MARGINS, lv)
            self.assertGreaterEqual(p[(MARGINS >= lo) & (MARGINS <= hi)].sum(), lv - 1e-9)

    def test_interval_helpers(self):
        lo, hi, sd = unc.prob_interval(0.6, [0.58, 0.61, 0.62, 0.59], 0.9)
        self.assertTrue(lo < 0.6 < hi and sd > 0)
        self.assertTrue(all(np.isnan(unc.prob_interval(0.6, [0.6], 0.9))))   # < 2 replicates: no bounds
        lo2, hi2, _ = unc.prob_interval(0.6, [0.58, 0.61, 0.62, 0.59], 0.5)
        self.assertTrue(lo < lo2 and hi2 < hi)
        with self.assertRaises(ValueError):
            unc.z_for(1.0)


def _rows(model, day):
    pred = model.predict(day)
    offers = consensus_offers(day)
    out = []
    for g in pred.itertuples(index=False):
        out += decide_game(model, g, offers[offers["game_id"] == g.game_id], GameContext(), S, refs=None)
    rows = pd.DataFrame(out)
    best = best_per_market(rows)
    rows["is_best_side"] = rows.set_index(["game_id", "market", "side"]).index.isin(
        best.set_index(["game_id", "market", "side"]).index)
    return pred, rows


class PushAwareEV(unittest.TestCase):
    DAY = TEST.head(8).assign(spread_line=[-3.0, 3.0, 7.0, -7.0, 2.5, -1.0, 6.0, 10.0])
    PRED, ROWS = _rows(POINT, DAY)
    OUT = betrange.ev_uncertainty(ROWS, PRED, REPS, None, S, 0.9)

    def test_probabilities_sum_to_one_with_pushes(self):
        o = self.OUT
        np.testing.assert_allclose(o["p_win"] + o["p_push"] + o["p_lose"], 1.0, atol=1e-12)
        sp = o[(o["market"] == "spread") & (o["point"] % 1 == 0)]
        self.assertTrue((sp["p_push"] > 0.01).all())          # whole-number spreads can push

    def test_central_ev_is_push_aware(self):
        from nflmodel.odds import decimal
        for r in self.OUT.itertuples(index=False):
            self.assertAlmostEqual(r.ev, r.p_win * (decimal(r.price) - 1) - r.p_lose, places=12)

    def test_ev_interval_from_joint_replicates(self):
        """Recompute one row's EV replicates by hand from each replicate's joint (win, push)."""
        o = self.OUT
        r = o[(o["market"] == "spread") & (o["p_push"] > 0)].iloc[0]
        evs = []
        for m in REPS:
            g = next(x for x in m.predict(self.DAY).itertuples(index=False) if x.game_id == r["game_id"])
            pw, pp = AnchoredPricer(m, g).probs("spread", r["side"], r["point"])
            evs.append(ev(pw, pp, r["price"]))
        lo, hi, sd = unc.linear_interval(r["ev"], evs, 0.9)
        self.assertAlmostEqual(r["ev_lo"], lo, places=12)
        self.assertAlmostEqual(r["ev_hi"], hi, places=12)
        self.assertTrue(r["ev_lo"] <= r["ev"] <= r["ev_hi"])
        self.assertTrue((o["ev_lo"] <= o["ev"]).all() and (o["ev"] <= o["ev_hi"]).all())

    def test_no_replicates_no_bounds(self):
        o = betrange.ev_uncertainty(self.ROWS, self.PRED, [], None, S, 0.9)
        self.assertTrue(o["ev_lo"].isna().all())
        self.assertTrue((o["unc_status"] == "not computed (uncertainty off)").all())


class ExperimentalRule(unittest.TestCase):
    def test_off_by_default_and_only_downgrades(self):
        self.assertFalse(S.experimental_ev_lower_bound_rule)
        rows = pd.DataFrame({"decision": ["BET", "BET IF PRICE AVAILABLE", "NO BET", "BET"],
                             "ev_lo": [0.01, -0.02, 0.05, np.nan], "reasons": ["", "x", "y", ""],
                             "stake_units": [1.0, 0.5, 0.0, 1.0]})
        self.assertTrue(betrange.apply_lower_bound_rule(rows, S).equals(rows))
        on = betrange.apply_lower_bound_rule(rows, replace(S, experimental_ev_lower_bound_rule=True))
        self.assertEqual(list(on["decision"]), ["BET", "NO BET", "NO BET", "NO BET"])
        self.assertEqual(on.loc[1, "stake_units"], 0.0)
        self.assertIn("experimental rule", on.loc[3, "reasons"])

    def test_settings_validation(self):
        for bad in (dict(uncertainty_level=1.0), dict(uncertainty_scheme="weekly"),
                    dict(experimental_ev_lower_bound_rule="yes"), dict(uncertainty_replicates=-1)):
            with self.assertRaises(SettingsError):
                replace(S, **bad)


META = {"method": unc.METHOD, "method_version": unc.METHOD_VERSION, "scheme": "season", "seed": 7,
        "level": 0.9, "replicates": len(REPS), "status": "ok (fast)", "train_hash": "abc", "seconds": 1.0,
        "cached": False, "model_version": "test", "data_time": "synthetic"}


class Reports(unittest.TestCase):
    FC = Coherence.FC

    def test_forecast_section_matches_table(self):
        md = report.forecasts_md(self.FC, META, {})
        self.assertIn("## GAME FORECASTS", md)
        self.assertIn("not a 90% chance", md)
        self.assertIn("conditional on the market", md)
        r = self.FC.iloc[0]
        self.assertIn(f"{r['p_home']:.1%} ({r['p_home_lo']:.1%}-{r['p_home_hi']:.1%})", md)
        self.assertIn(f"{r['margin_mean']:+.1f} ({r['margin_lo_90']:+.0f} to {r['margin_hi_90']:+.0f})", md)
        self.assertEqual(md.count(" @ "), len(self.FC))

    def test_off_and_failed_status_are_explicit(self):
        off = forecast.forecast_slate(POINT, TEST.head(3), None, S, None, 0.9)
        md = report.forecasts_md(off, {**META, "replicates": 0, "status": "off (--uncertainty off)"}, {})
        self.assertIn("off (--uncertainty off)", md)
        self.assertNotIn("%-", md.split("| Missing inputs |")[1])          # no probability bounds printed
        md = report.forecasts_md(off, {**META, "replicates": 0, "status": "FAILED (x); no uncertainty bounds shown"}, {})
        self.assertIn("FAILED", md)

    def test_betting_section_separate_and_consistent(self):
        o = PushAwareEV.OUT.assign(category="", decision=PushAwareEV.OUT["decision"])
        md = report.betting_md(o, PushAwareEV.PRED, S)
        self.assertIn("## BETTING OPPORTUNITIES", md)
        best = o[o["is_best_side"]]
        r = best.iloc[0]
        self.assertIn(f"{r['ev']:+.1%} | {r['ev_lo']:+.1%} to {r['ev_hi']:+.1%}", md)
        body = md.split("|---|")[-1]
        self.assertEqual(sum(1 for line in md.splitlines() if line.startswith("| ") and " @ " in line), len(best))
        self.assertNotIn("||", body)

    def test_forecast_line_and_ml_note(self):
        r = self.FC.iloc[0]
        line = report.forecast_line(r)
        self.assertIn(f"{r['home_team']} {r['p_home']:.1%}", line)
        rows = PushAwareEV.ROWS
        g = rows[(rows["market"] == "ml") & (rows["side"] == "home")].iloc[0]
        fc = Coherence.FC.set_index("game_id").loc[g["game_id"]]
        note = report.forecast_vs_ml_note(fc, rows[rows["game_id"] == g["game_id"]])
        self.assertIn("excluding ties", note)


class HistoricalLogs(unittest.TestCase):
    def test_old_and_new_records_read_together(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "forecasts.jsonl"
            old_fields = [f for f in store.FORECAST_FIELDS if f not in store.UNCERTAINTY_FIELDS]
            store.append(path, "forecast", [{k: None for k in old_fields} | {"game_id": "OLD", "ev": 0.01}])
            new = {k: None for k in store.FORECAST_FIELDS} | {"game_id": "NEW", "ev": 0.02, "ev_lo": -0.01,
                                                             "ev_hi": 0.05, "unc_method": unc.METHOD_VERSION}
            store.append(path, "forecast", [new])
            self.assertEqual(store.verify(path)[0], True)
            df = store.forecasts_frame(path)
            self.assertEqual(list(df["game_id"]), ["OLD", "NEW"])
            self.assertTrue(pd.isna(df.loc[0, "ev_lo"]) if "ev_lo" in df else True)
            self.assertAlmostEqual(df.loc[1, "ev_lo"], -0.01)

    def test_game_forecast_store(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "game_forecasts.jsonl"
            fc = Coherence.FC.head(2).assign(kickoff_utc="2018-09-09T17:00:00Z")
            out = store.record_game_forecasts(fc, {"prediction_completed_utc": "2018-09-01T00:00:00Z", "run_id": "t",
                                                   "method_version": unc.METHOD_VERSION, "seed": 7}, path)
            self.assertEqual(len(out), 2)
            self.assertTrue(store.verify(path)[0])
            rec = json.loads(path.read_text().splitlines()[0])["data"]
            for k in ("p_home", "p_home_lo", "p_home_hi", "margin_lo_90", "method_version", "seed", "level"):
                self.assertIn(k, rec)

    def test_existing_logs_still_verify_and_load(self):
        """Read-only check of the real logs: unchanged chain, readable without the new fields."""
        if store.FORECASTS.exists():
            self.assertTrue(store.verify(store.FORECASTS)[0])
            df = store.forecasts_frame()
            self.assertIn("ev", df.columns)


if __name__ == "__main__":
    unittest.main()
