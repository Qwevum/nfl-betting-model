"""Out-of-sample validation against baselines and the market.

Walk-forward: every test season is predicted by a model fit (and calibrated) only
on earlier seasons. Ratings feeding each game use only games before it.

Information-timing notes (what a historical prediction "knew"):
  * Prices are nflverse consensus closing lines: available just before kickoff.
  * Starting QBs are the actual starters. Those are normally known ~90 minutes
    before kickoff (inactives); live predictions use projected starters instead.
  * Game-time wind/temperature are observed values; live predictions would need a
    forecast. This makes historical totals slightly easier than live ones.
  * Model design choices (features, hyperparameters, calibration method) were made
    after looking at 2015-2025 results, so those seasons are not a pristine
    holdout. The pristine test is the prediction log written from now on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import decide
from .backtest import grade, profit
from .model import _logistic, fit
from .odds import consensus_offers, no_vig
from .ratings import FEATURES


def _bl(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _decisions(model, test: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    offers = consensus_offers(test)
    rows = []
    for g in test.itertuples(index=False):
        ctx = decide.GameContext()
        if pd.notna(g.spread_line) and abs(g.model_margin - g.spread_line) >= decide.GAP_POINTS:
            ctx.block["spread"].append("gap"); ctx.block["ml"].append("gap")
        if pd.notna(g.total_line) and abs(g.model_total - g.total_line) >= decide.GAP_POINTS:
            ctx.block["total"].append("gap")
        rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, min_edge)
    return pd.DataFrame(rows)


def run(feat: pd.DataFrame, seasons: list[int], min_edge: float = 0.02,
        features: list[str] | None = None, with_bets: bool = True):
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    preds, bets = [], []
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        if train["season"].nunique() < 3:
            continue
        model = fit(train, features=features)
        test = model.predict(played[played["season"] == s]).copy()

        # Win probabilities: model and baselines (all fit on training seasons only)
        pricers = [decide.GamePricer(model, g) for g in test.itertuples(index=False)]
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
                cov.append(w / (1 - p)); cov_m.append(pr.p_spread_mkt)
            else:
                cov.append(np.nan); cov_m.append(np.nan)
            if pd.notna(g.total_line):
                w, p = pr.probs("total", "over", g.total_line)
                ovr.append(w / (1 - p)); ovr_m.append(pr.p_total_mkt)
            else:
                ovr.append(np.nan); ovr_m.append(np.nan)
        test["p_cover"], test["p_cover_mkt"] = cov, cov_m
        test["p_over"], test["p_over_mkt"] = ovr, ovr_m
        preds.append(test)

        if with_bets:
            d = _decisions(model, test, min_edge)
            d = decide.best_per_market(d)
            d = d.merge(test[["game_id", "season", "home_score", "away_score"]], on="game_id")
            d["result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                           for r in d.itertuples(index=False)]
            d["flat"] = [profit(r, p) for r, p in zip(d["result"], d["price"])]
            d["kelly_units"] = [profit(r, p) * min(k * decide.KELLY_FRACTION * 100, decide.MAX_STAKE_UNITS)
                                for r, p, k in zip(d["result"], d["price"], d["kelly"])]
            d["kelly_stake"] = [min(k * decide.KELLY_FRACTION * 100, decide.MAX_STAKE_UNITS) for k in d["kelly"]]
            bets.append(d)
        print(f"  {s}: {len(test)} games", flush=True)
    return pd.concat(preds, ignore_index=True), (pd.concat(bets, ignore_index=True) if bets else None)


def prob_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ml = p[p["result"] != 0]
    y = (ml["result"] > 0).to_numpy(float)
    for name, col in (("model", "p_model"), ("market no-vig (closing)", "p_market"),
                      ("ratings only (no market)", "p_ratings_only"),
                      ("home-team base rate", "p_home_rate"), ("coin flip", None)):
        m = ml if col is None else ml.dropna(subset=[col])
        yy = (m["result"] > 0).to_numpy(float)
        pp = np.full(len(m), 0.5) if col is None else m[col].to_numpy(float)
        b, l = _bl(pp, yy)
        rows.append({"target": "winner", "predictor": name, "n": len(m), "brier": b, "log_loss": l})
    sp = p.dropna(subset=["p_cover"])
    sp = sp[sp["result"] != sp["spread_line"]]
    y = (sp["result"] > sp["spread_line"]).to_numpy(float)
    for name, pp in (("model", sp["p_cover"]), ("market no-vig (closing)", sp["p_cover_mkt"]),
                     ("coin flip", np.full(len(sp), 0.5))):
        b, l = _bl(pp, y)
        rows.append({"target": "home covers", "predictor": name, "n": len(sp), "brier": b, "log_loss": l})
    tt = p.dropna(subset=["p_over"])
    tt = tt[tt["total"] != tt["total_line"]]
    y = (tt["total"] > tt["total_line"]).to_numpy(float)
    for name, pp in (("model", tt["p_over"]), ("market no-vig (closing)", tt["p_over_mkt"]),
                     ("coin flip", np.full(len(tt), 0.5))):
        b, l = _bl(pp, y)
        rows.append({"target": "over hits", "predictor": name, "n": len(tt), "brier": b, "log_loss": l})
    return pd.DataFrame(rows)


def calibration(p: pd.DataFrame) -> pd.DataFrame:
    ml = p[p["result"] != 0].copy()
    ml["bin"] = pd.cut(ml["p_model"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    return (ml.groupby("bin", observed=True)
              .agg(n=("p_model", "size"), predicted=("p_model", "mean"),
                   actual=("result", lambda r: (r > 0).mean()))
              .reset_index())


def bet_table(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    b = bets[bets["decision"] != "NO BET"]
    for market, g in list(b.groupby("market")) + [("all", b)]:
        w, l, p = (g["result"] == "W").sum(), (g["result"] == "L").sum(), (g["result"] == "P").sum()
        n = w + l
        win = w / n if n else np.nan
        se_roi = g["flat"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        rows.append({"market": market, "bets": len(g), "W-L-P": f"{w}-{l}-{p}",
                     "win%": win, "flat ROI": g["flat"].mean(),
                     "flat ROI 95%": f"{g['flat'].mean() - 1.96 * se_roi:+.1%} to {g['flat'].mean() + 1.96 * se_roi:+.1%}",
                     "qtr-Kelly units": g["kelly_units"].sum(),
                     "qtr-Kelly ROI": g["kelly_units"].sum() / g["kelly_stake"].sum() if g["kelly_stake"].sum() else np.nan})
    return pd.DataFrame(rows)


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


__all__ = ["run", "prob_table", "calibration", "bet_table", "mae_table", "to_markdown", "FEATURES"]
