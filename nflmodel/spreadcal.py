"""Diagnostics of the spread calibration (read-only: nothing here changes predictions).

Two calibrations of P(home covers | no push), both fit on the inner walk-forward rows
(model.inner_oos), i.e. on out-of-sample blend predictions:
  legacy  sigmoid(a + b * logit(p_novig) + c * edge)       (model.spread_cal)
  offset  sigmoid(logit(p_novig) + c * edge)  (a = 0, b = 1) (model.offset_cal["spread"], live default)
with p_novig the proportional no-vig probability of the consensus spread pair and
edge = blend model margin - spread line (points).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .model import _logistic_n, _logistic_offset, inner_oos, logit, novig_first, sigmoid

PARAMS = ("a", "b", "c")


def spread_rows(oos: pd.DataFrame) -> pd.DataFrame:
    """One row per game with a line, a blend prediction and no push (as model.fit uses)."""
    m = oos.dropna(subset=["blend_margin", "result", "spread_line"])
    m = m[m["result"] != m["spread_line"]]
    return pd.DataFrame({
        "game_id": m["game_id"].to_numpy(), "season": m["season"].to_numpy(),
        "x": logit(novig_first(m["home_spread_odds"], m["away_spread_odds"])),
        "edge": (m["blend_margin"] - m["spread_line"]).to_numpy(float),
        "y": (m["result"] > m["spread_line"]).to_numpy(float)})


def market_rows(oos: pd.DataFrame, market: str) -> pd.DataFrame:
    """Rows the offset calibration of `market` is fit on in model.fit (spread / total / ml)."""
    if market == "spread":
        return spread_rows(oos)
    if market == "total":
        t = oos.dropna(subset=["blend_total", "total", "total_line"])
        t = t[t["total"] != t["total_line"]]
        return pd.DataFrame({"game_id": t["game_id"].to_numpy(), "season": t["season"].to_numpy(),
                             "x": logit(novig_first(t["over_odds"], t["under_odds"])),
                             "edge": (t["blend_total"] - t["total_line"]).to_numpy(float),
                             "y": (t["total"] > t["total_line"]).to_numpy(float)})
    if market == "ml":
        m = oos.dropna(subset=["blend_margin", "result", "spread_line"])
        m = m[(m["result"] != 0) & m["home_moneyline"].notna() & m["away_moneyline"].notna()]
        return pd.DataFrame({"game_id": m["game_id"].to_numpy(), "season": m["season"].to_numpy(),
                             "x": logit(novig_first(m["home_moneyline"], m["away_moneyline"])),
                             "edge": (m["blend_margin"] - m["spread_line"]).to_numpy(float),
                             "y": (m["result"] > 0).to_numpy(float)})
    raise ValueError(f"unknown market {market!r}")


def c_interval(rows: pd.DataFrame, n: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Offset-calibration c (log-odds per point of model edge) with a game-bootstrap CI."""
    rng = np.random.default_rng(seed)
    k = len(rows)
    est = fit_offset(rows)
    reps = [fit_offset(rows, rng.multinomial(k, np.full(k, 1 / k)).astype(float)) for _ in range(n)]
    lo, hi = np.quantile(reps, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"games": k, "c": est, "lo": float(lo), "hi": float(hi), "includes 0": bool(lo <= 0 <= hi)}


def c_table(feat: pd.DataFrame, seasons: list[int], n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per walk-forward test season (model fit on earlier seasons, as validate.run does): the
    offset c of each market with its game-bootstrap 95% CI over that model's inner
    out-of-sample rows."""
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    out = []
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        if train["season"].nunique() < 4:
            continue
        oos = inner_oos(train)
        for mk in ("spread", "total", "ml"):
            out.append({"test season": s, "market": mk, **c_interval(market_rows(oos, mk), n, seed)})
    return pd.DataFrame(out)


def fit_legacy(rows: pd.DataFrame, w: np.ndarray | None = None) -> np.ndarray:
    """[a, b, c] exactly as model.fit fits spread_cal (light ridge on the slopes)."""
    return _logistic_n(np.column_stack([rows["x"], rows["edge"]]), rows["y"].to_numpy(float), sw=w)


def fit_offset(rows: pd.DataFrame, w: np.ndarray | None = None) -> float:
    """c with a = 0, b = 1 fixed, exactly as model.fit fits offset_cal['spread']."""
    return _logistic_offset(rows["edge"].to_numpy(float), rows["x"].to_numpy(float), rows["y"].to_numpy(float),
                            sw=w)


def bootstrap(rows: pd.DataFrame, n: int = 1000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Game-level bootstrap (resample whole games, as multinomial counts) of both calibrations."""
    rng = np.random.default_rng(seed)
    k = len(rows)
    leg, off = [], []
    for _ in range(n):
        w = rng.multinomial(k, np.full(k, 1 / k)).astype(float)
        leg.append(fit_legacy(rows, w))
        off.append(fit_offset(rows, w))
    leg, off = np.array(leg), np.array(off)
    a = (1 - level) / 2
    est_l, est_o = fit_legacy(rows), fit_offset(rows)
    out = [{"calibration": "legacy", "param": p, "estimate": est_l[i], "lo": np.quantile(leg[:, i], a),
            "hi": np.quantile(leg[:, i], 1 - a)} for i, p in enumerate(PARAMS)]
    out.append({"calibration": "offset (a=0, b=1)", "param": "c", "estimate": est_o,
                "lo": np.quantile(off, a), "hi": np.quantile(off, 1 - a)})
    df = pd.DataFrame(out)
    df["games"] = k
    # the null for a is 0, for b is 1 (the market unchanged), for c is 0 (no model value)
    null = df["param"].map({"a": 0.0, "b": 1.0, "c": 0.0})
    df["CI includes null"] = (df["lo"] <= null) & (null <= df["hi"])
    return df


def by_era(rows: pd.DataFrame, eras: dict[str, tuple[int, int]], n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Refit both calibrations on each era's rows separately (with game-bootstrap CIs)."""
    out = []
    for name, (lo, hi) in eras.items():
        r = rows[rows["season"].between(lo, hi)]
        if len(r) < 50:
            continue
        out.append(bootstrap(r.reset_index(drop=True), n, seed).assign(era=name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def decompose(model, pred: pd.DataFrame) -> pd.DataFrame:
    """Per game and spread side, at the consensus line: model probability minus proportional no-vig,
    split into the market remapping (a, b with edge 0) and the model part (c * edge)."""
    a, b, c = model.spread_cal
    c_off = model.offset_cal["spread"][2]
    out = []
    for g in pred.itertuples(index=False):
        if pd.isna(g.spread_line):
            continue
        p = float(novig_first(g.home_spread_odds, g.away_spread_odds))
        edge = float(g.blend_margin - g.spread_line) if pd.notna(g.blend_margin) else 0.0
        remap = float(sigmoid(a + b * logit(p)))
        legacy = float(sigmoid(a + b * logit(p) + c * edge))
        offset = float(sigmoid(logit(p) + c_off * edge))
        for side, sign, team, pt in (("home", 1, g.home_team, -g.spread_line), ("away", -1, g.away_team, g.spread_line)):
            out.append({"game": f"{g.away_team} @ {g.home_team}", "side": f"{team} {pt:+g}", "edge (pts)": sign * edge,
                        "no-vig": p if sign > 0 else 1 - p,
                        "legacy: remap (a,b)": sign * (remap - p), "legacy: model (c*edge)": sign * (legacy - remap),
                        "legacy: total": sign * (legacy - p),
                        "offset (live): model (c*edge)": sign * (offset - p)})
    return pd.DataFrame(out)


def compare_cover(p_legacy: pd.DataFrame, p_offset: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Home-covers log loss / Brier of two arms on identical games, with paired game-bootstrap CIs.
    Inputs: validate.run predictions (game_id, result, spread_line, p_cover) for each arm."""
    j = p_legacy[["game_id", "season", "result", "spread_line", "p_cover"]].merge(
        p_offset[["game_id", "p_cover"]], on="game_id", suffixes=("_l", "_o"))
    j = j.dropna(subset=["p_cover_l", "p_cover_o", "spread_line"])
    j = j[j["result"] != j["spread_line"]]
    y = (j["result"] > j["spread_line"]).to_numpy(float)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = j.assign(ll_l=ll(j["p_cover_l"]), ll_o=ll(j["p_cover_o"]), br_l=(j["p_cover_l"] - y) ** 2,
                 br_o=(j["p_cover_o"] - y) ** 2)
    d = d.assign(d_ll=d["ll_o"] - d["ll_l"], d_br=d["br_o"] - d["br_l"])
    rows = []
    for metric, col_l, col_o, diff in (("log loss", "ll_l", "ll_o", "d_ll"), ("Brier", "br_l", "br_o", "d_br")):
        est, lo, hi = metrics.cluster_bootstrap(d, "game_id", diff, n=n_boot)
        rows.append({"metric": metric, "games": len(d), "legacy": d[col_l].mean(), "offset (a=0,b=1)": d[col_o].mean(),
                     "offset - legacy": est, "95% CI": f"{lo:+.5f} to {hi:+.5f}", "CI includes 0": lo <= 0 <= hi})
    return pd.DataFrame(rows)


__all__ = ["market_rows", "c_interval", "c_table", "spread_rows", "fit_legacy", "fit_offset", "bootstrap", "by_era", "decompose", "compare_cover", "inner_oos"]
