"""Scoring rules, drawdown and game-clustered uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier_logloss(p, y) -> tuple[float, float]:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def max_drawdown(profits) -> float:
    """Largest peak-to-trough fall of cumulative profit (units), profits in time order."""
    c = np.cumsum(np.asarray(profits, float))
    if len(c) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float(np.max(peak - c)) if len(c) else 0.0


def cluster_bootstrap(df: pd.DataFrame, cluster: str, stat, n: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate, lo, hi) resampling whole clusters (games), so correlated bets on the
    same game (spread + moneyline + total) are not treated as independent."""
    if df.empty:
        return np.nan, np.nan, np.nan
    est = float(stat(df))
    groups = [g for _, g in df.groupby(cluster)]
    k = len(groups)
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        sims.append(stat(pd.concat([groups[i] for i in idx], ignore_index=True)))
    lo, hi = np.nanquantile(sims, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def calibration_table(p, y, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)) -> pd.DataFrame:
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    d["bin"] = pd.cut(d["p"], list(bins), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(n=("p", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |predicted - actual|."""
    t = calibration_table(p, y, np.linspace(0, 1, bins + 1))
    return float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / t["n"].sum()) if len(t) else np.nan
