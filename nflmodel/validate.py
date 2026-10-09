"""Out-of-sample validation against baselines and the market.

Chronology: every test season is predicted by a model whose ratings use only
earlier games, whose regressions are fit only on earlier seasons, and whose
calibration (k shrinkage, market-anchored logistic) is fit by an inner
walk-forward that is also restricted to earlier seasons.

Prediction horizon: live forecasts are meant for kickoff - 60 minutes
(settings.horizon_minutes). What each historical input knew relative to that
horizon:

  input                    | historical source                 | available by horizon?
  -------------------------|-----------------------------------|-----------------------------
  team/QB ratings          | play-by-play of earlier games     | yes
  rest, division, site     | schedule                          | yes
  starting QB              | actual starter (nflverse)         | yes in practice: inactives are
                           |                                   | published ~90 min before kickoff
  roof (dome/closed)       | schedule, observed roof state     | assumed: retractable-roof calls are
                           |                                   | usually announced before the horizon
  weather                  | not used (observed wind removed)  | n/a, no historical forecasts exist
  injury reports           | not used historically             | n/a, the feed has no publish times
  market prices            | nflverse CLOSING consensus        | NO: the close is after the horizon

Market prices are the binding limitation: no timestamped historical multi-book
snapshots exist in this repository, so model and market are both evaluated at the
CLOSING information set (the market's best case), on exactly the same games.
Horizon-matched data is collected from now on by `run.py collect` / `predict`
snapshots; nothing is backfilled.

Decision rules replayed historically: EV > min_edge at the consensus closing price,
robustness to a 0.5-point error, and the gap_points model-vs-market rule, all from
the Settings passed in (the report prints the values used). NOT
replayed (no data): quote timestamps/freshness, the multi-book leave-one-book-out
reference, starter-availability conditions, the outdoor-total weather rule, and the
executable/conditional distinction. Historical "bets" are therefore closer to
"conditional recommendations at the closing consensus price" than to live
executable bets.

Model design choices (features, hyperparameters, calibration method) were made
after looking at 2015-2025 results, so those seasons are not a pristine holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import decide, metrics
from .config import Settings
from .backtest import grade, profit
from .model import _logistic, fit, novig_first
from .odds import consensus_offers, no_vig
from .ratings import FEATURES


def revig(test: pd.DataFrame, overround: float) -> pd.DataFrame:
    """Re-price every consensus two-way pair at a fixed overround (proportional), keeping
    each pair's no-vig probabilities. Used to replay history at standard retail juice."""
    def price(implied):
        implied = np.clip(implied, 1e-6, 1 - 1e-6)
        return np.where(implied >= 0.5, -100 * implied / (1 - implied), 100 * (1 - implied) / implied)
    out = test.copy()
    for a, b in (("home_spread_odds", "away_spread_odds"), ("over_odds", "under_odds"),
                 ("home_moneyline", "away_moneyline")):
        ok = out[a].notna() & out[b].notna()
        if not ok.any():
            continue
        pa = np.asarray(novig_first(out.loc[ok, a], out.loc[ok, b]), float)
        out.loc[ok, a] = np.round(price(pa * (1 + overround)))
        out.loc[ok, b] = np.round(price((1 - pa) * (1 + overround)))
    return out


def overround_by_season(feat: pd.DataFrame) -> pd.DataFrame:
    g = feat[feat["result"].notna()]
    def imp(x):
        x = np.asarray(x, float)
        return np.where(x < 0, -x / (-x + 100), 100 / (x + 100))
    g = g.assign(spread_or=imp(g["home_spread_odds"]) + imp(g["away_spread_odds"]) - 1,
                 ml_or=imp(g["home_moneyline"]) + imp(g["away_moneyline"]) - 1,
                 total_or=imp(g["over_odds"]) + imp(g["under_odds"]) - 1)
    return g.groupby("season")[["spread_or", "ml_or", "total_or"]].median().reset_index()


def _decisions(model, test: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    offers = consensus_offers(test)
    rows = []
    for g in test.itertuples(index=False):
        ctx = decide.GameContext()
        if pd.notna(g.spread_line) and abs(g.model_margin - g.spread_line) >= settings.gap_points:
            ctx.block["spread"].append("gap"); ctx.block["ml"].append("gap")
        if pd.notna(g.total_line) and abs(g.model_total - g.total_line) >= settings.gap_points:
            ctx.block["total"].append("gap")
        rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, settings)
    return pd.DataFrame(rows)


def run(feat: pd.DataFrame, seasons: list[int], settings: Settings | None = None,
        features: list[str] | None = None, with_bets: bool = True, bet_overround: float | None = None,
        total_features: list[str] | None = None, verbose: bool = True):
    """bet_overround: if set, the betting replay prices every bet at this fixed overround
    (e.g. 0.0476 = -110/-110) instead of the recorded consensus prices."""
    settings = settings or Settings()
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    preds, bets = [], []
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        if train["season"].nunique() < 3:
            continue
        model = fit(train, features=features, total_features=total_features)
        test = model.predict(played[played["season"] == s]).copy()

        # Win probabilities: model and baselines (all fit on training seasons only)
        pricers = [decide.make_pricer(model, g, settings) for g in test.itertuples(index=False)]
        test["p_model"] = [pr.probs("ml", "home", np.nan)[0] for pr in pricers]
        tr = model.core.raw(train)
        dec = tr["result"] != 0
        a, b = _logistic(tr.loc[dec, "model_margin"].to_numpy(), (tr.loc[dec, "result"] > 0).to_numpy(float))
        test["p_ratings_only"] = 1 / (1 + np.exp(-(a + b * test["model_margin"])))
        test["p_home_rate"] = float((train.loc[train["result"] != 0, "result"] > 0).mean())
        test["p_market"] = [no_vig(h, aw)[0] if pd.notna(h) and pd.notna(aw) else np.nan
                            for h, aw in zip(test["home_moneyline"], test["away_moneyline"])]
        # Spread: P(home covers | no push) at the closing line; totals: P(over | no push)
        cov, ovr, cov_m, ovr_m = [], [], [], []
        for g, pr in zip(test.itertuples(index=False), pricers):
            if pd.notna(g.spread_line):
                w, p = pr.probs("spread", "home", -g.spread_line)
                cov.append(w / (1 - p)); cov_m.append(pr.reference_prob("spread", "home", -g.spread_line))
            else:
                cov.append(np.nan); cov_m.append(np.nan)
            if pd.notna(g.total_line):
                w, p = pr.probs("total", "over", g.total_line)
                ovr.append(w / (1 - p)); ovr_m.append(pr.reference_prob("total", "over", g.total_line))
            else:
                ovr.append(np.nan); ovr_m.append(np.nan)
        test["p_cover"], test["p_cover_mkt"] = cov, cov_m
        test["p_over"], test["p_over_mkt"] = ovr, ovr_m
        preds.append(test)

        if with_bets:
            d = _decisions(model, revig(test, bet_overround) if bet_overround else test, settings)
            d = decide.best_per_market(d)
            d = d.merge(test[["game_id", "season", "gameday", "home_score", "away_score"]], on="game_id")
            d["result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                           for r in d.itertuples(index=False)]
            d["flat"] = [profit(r, p) for r, p in zip(d["result"], d["price"])]
            d["kelly_stake"] = [decide.stake_units(k, settings) for k in d["kelly"]]
            d["kelly_units"] = [profit(r, p) * st for r, p, st in zip(d["result"], d["price"], d["kelly_stake"])]
            bets.append(d)
        if verbose:
            print(f"  {s}: {len(test)} games", flush=True)
    return pd.concat(preds, ignore_index=True), (pd.concat(bets, ignore_index=True) if bets else None)


def _targets(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per target: rows where the model AND the market both have a probability (same games)."""
    ml = p[(p["result"] != 0)].dropna(subset=["p_model", "p_market"]).assign(y=lambda d: (d["result"] > 0) * 1.0)
    ml = ml.assign(pm=ml["p_model"], pk=ml["p_market"])
    sp = p.dropna(subset=["p_cover", "p_cover_mkt", "spread_line"])
    sp = sp[sp["result"] != sp["spread_line"]]
    sp = sp.assign(y=(sp["result"] > sp["spread_line"]) * 1.0, pm=sp["p_cover"], pk=sp["p_cover_mkt"])
    tt = p.dropna(subset=["p_over", "p_over_mkt", "total_line"])
    tt = tt[tt["total"] != tt["total_line"]]
    tt = tt.assign(y=(tt["total"] > tt["total_line"]) * 1.0, pm=tt["p_over"], pk=tt["p_over_mkt"])
    return {"winner": ml, "home covers": sp, "over hits": tt}


def prob_table(p: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Brier / log loss / ECE for every predictor on the SAME rows, plus a game-level
    bootstrap CI of the model-minus-market Brier difference (negative = model better)."""
    rows = []
    for target, d in _targets(p).items():
        preds = {"model": d["pm"], "market no-vig (closing)": d["pk"], "coin flip": pd.Series(0.5, index=d.index)}
        if target == "winner":
            preds["ratings only (no market)"] = d["p_ratings_only"]
            preds["home-team base rate"] = d["p_home_rate"]
        for name, pp in preds.items():
            b, l = metrics.brier_logloss(pp, d["y"])
            rows.append({"target": target, "predictor": name, "n": len(d), "brier": b, "log_loss": l,
                         "ece": metrics.ece(pp, d["y"]) if name != "coin flip" else np.nan})
        diff = d.assign(sq=(d["pm"] - d["y"]) ** 2 - (d["pk"] - d["y"]) ** 2)
        est, lo, hi = metrics.cluster_bootstrap(diff, "game_id", "sq", n=n_boot)
        rows.append({"target": target, "predictor": "model minus market (Brier)", "n": len(d),
                     "brier": est, "log_loss": np.nan, "ece": np.nan, "ci95": f"{lo:+.5f} to {hi:+.5f}"})
    return pd.DataFrame(rows)


def season_diffs(p: pd.DataFrame) -> pd.DataFrame:
    """Model-minus-market Brier by season and target (negative = model better)."""
    out = []
    for target, d in _targets(p).items():
        for season, g in d.groupby("season"):
            out.append({"target": target, "season": season, "n": len(g),
                        "model - market": float(((g["pm"] - g["y"]) ** 2).mean() - ((g["pk"] - g["y"]) ** 2).mean())})
    return pd.DataFrame(out).pivot(index="season", columns="target", values="model - market").reset_index()


def calibration(p: pd.DataFrame) -> pd.DataFrame:
    d = _targets(p)["winner"]
    m = metrics.calibration_table(d["pm"], d["y"]).rename(columns={"predicted": "model predicted", "actual": "actual"})
    k = metrics.calibration_table(d["pk"], d["y"]).rename(columns={"n": "n (market)", "predicted": "market predicted",
                                                                   "actual": "actual (market bins)"})
    return m.merge(k, on="bin", how="outer")


def bet_table(bets: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Flat 1u and quarter-Kelly results with game-clustered bootstrap CIs and drawdown."""
    rows = []
    b = bets[bets["decision"] != "NO BET"].sort_values(["gameday", "game_id"])
    for market, g in list(b.groupby("market")) + [("all", b)]:
        w, l, p = (g["result"] == "W").sum(), (g["result"] == "L").sum(), (g["result"] == "P").sum()
        roi, lo, hi = metrics.cluster_bootstrap(g, "game_id", "flat", n=n_boot)
        k_roi = g["kelly_units"].sum() / g["kelly_stake"].sum() if g["kelly_stake"].sum() else np.nan
        rows.append({"market": market, "bets": len(g), "games": g["game_id"].nunique(), "W-L-P": f"{w}-{l}-{p}",
                     "flat ROI": roi, "flat ROI 95% (game-clustered)": f"{lo:+.1%} to {hi:+.1%}",
                     "flat units": g["flat"].sum(), "flat max drawdown": metrics.max_drawdown(g["flat"]),
                     "qtr-Kelly ROI": k_roi, "qtr-Kelly max drawdown": metrics.max_drawdown(g["kelly_units"])})
    return pd.DataFrame(rows)


def bet_seasons(bets: pd.DataFrame) -> pd.DataFrame:
    b = bets[bets["decision"] != "NO BET"]
    return (b.groupby("season").agg(bets=("flat", "size"), flat_units=("flat", "sum"), flat_roi=("flat", "mean"))
             .reset_index())


def mae_table(p: pd.DataFrame) -> pd.DataFrame:
    return (p.groupby("season")
             .apply(lambda g: pd.Series({
                 "games": len(g),
                 "MAE market": np.nanmean(np.abs(g["result"] - g["spread_line"])),
                 "MAE model own line": np.nanmean(np.abs(g["result"] - g["model_margin"])),
                 "MAE fair line": np.nanmean(np.abs(g["result"] - g["fair_margin"]))}), include_groups=False)
             .reset_index())


def to_markdown(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and pd.notna(v):
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


__all__ = ["run", "prob_table", "season_diffs", "calibration", "bet_table", "bet_seasons", "mae_table",
           "to_markdown", "FEATURES"]
