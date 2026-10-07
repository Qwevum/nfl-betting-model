"""Regression models and outcome distributions.

Two margin models are fit on past games (home margin = home - away points):
  * "pure"  - team ratings only. This is our own line, like a power-ratings sheet.
  * "blend" - team ratings plus the market spread. The fitted weight on the market
              says how much our ratings add beyond what the line already knows.
              Edges are taken from this model, which keeps them honest: if the
              ratings add nothing, the blend collapses onto the market line.

Final scores are not normally distributed: margins of 3, 7, 10, 6 and 14 occur far
more often than a bell curve says. The distribution here starts from a normal
curve around the predicted margin and re-weights each integer margin by how much
more or less often it has actually happened, which gets push and key-number
probabilities right. Totals are handled the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from math import erf, sqrt

from .ratings import FEATURES, TOTAL_FEATURES

MARGINS = np.arange(-80, 81)
TOTALS = np.arange(0, 131)


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))


@dataclass
class Ridge:
    cols: list[str]
    lam: float = 5.0
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    coef_: np.ndarray = field(default=None, repr=False)
    intercept_: float = 0.0

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Ridge":
        X = df[self.cols].to_numpy(float)
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        yv = y.to_numpy(float)
        self.intercept_ = yv.mean()
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (yv - self.intercept_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        Z = (df[self.cols].to_numpy(float) - self.mean_) / self.std_
        return self.intercept_ + Z @ self.coef_

    def raw_coefs(self) -> dict[str, float]:
        """Coefficients in original units (points per unit of feature)."""
        out = dict(zip(self.cols, self.coef_ / self.std_))
        out["intercept"] = self.intercept_ - float(np.sum(self.coef_ * self.mean_ / self.std_))
        return out


def _key_weights(mu: np.ndarray, actual: np.ndarray, sd: float, support: np.ndarray, symmetric: bool):
    """Ratio of observed to normal-expected frequency for each integer outcome."""
    lo, hi = support[0] - 0.5, support[-1] + 0.5
    edges = np.arange(lo, hi + 1)
    expected = np.zeros(len(support))
    for m in mu:
        c = _norm_cdf((edges - m) / sd)
        expected += np.diff(c)
    observed = np.bincount((actual - support[0]).astype(int).clip(0, len(support) - 1),
                           minlength=len(support)).astype(float)
    if symmetric:  # margins: a 3-point win and a 3-point loss behave alike
        expected = expected + expected[::-1]
        observed = observed + observed[::-1]
    prior = 20.0   # shrink thin bins toward 1
    return (observed + prior) / (expected + prior)


class OutcomeDist:
    def __init__(self, sd: float, weights: np.ndarray, support: np.ndarray):
        self.sd, self.w, self.support = sd, weights, support

    def pmf(self, mu: float) -> np.ndarray:
        edges = np.append(self.support - 0.5, self.support[-1] + 0.5)
        p = np.diff(_norm_cdf((edges - mu) / self.sd)) * self.w
        return p / p.sum()

    def prob_over(self, mu: float, line: float) -> tuple[float, float, float]:
        """P(outcome > line), P(outcome == line), P(outcome < line)."""
        p = self.pmf(mu)
        s = self.support
        return float(p[s > line].sum()), float(p[s == line].sum()), float(p[s < line].sum())


@dataclass
class _Core:
    pure: Ridge
    blend: Ridge
    total_pure: Ridge
    total_blend: Ridge

    def raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uncalibrated predictions: pure model, and blend wherever a line exists."""
        out = df.copy()
        out["model_margin"] = self.pure.predict(out)
        out["model_total"] = self.total_pure.predict(out)
        out["blend_margin"] = np.nan
        out["blend_total"] = np.nan
        has = out["spread_line"].notna()
        if has.any():
            out.loc[has, "blend_margin"] = self.blend.predict(out[has])
        has = out["total_line"].notna()
        if has.any():
            out.loc[has, "blend_total"] = self.total_blend.predict(out[has])
        return out


def _fit_core(train: pd.DataFrame, lam: float) -> _Core:
    tr = train.dropna(subset=FEATURES + ["result"])
    trb = tr.dropna(subset=["spread_line"])
    tt = train.dropna(subset=TOTAL_FEATURES + ["total"])
    ttb = tt.dropna(subset=["total_line"])
    return _Core(
        Ridge(FEATURES, lam).fit(tr, tr["result"]),
        Ridge(FEATURES + ["spread_line"], lam).fit(trb, trb["result"]),
        Ridge(TOTAL_FEATURES, lam).fit(tt, tt["total"]),
        Ridge(TOTAL_FEATURES + ["total_line"], lam).fit(ttb, ttb["total"]),
    )


def _logistic(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, X.T @ (y - p))
    return float(w[0]), float(w[1])


def _shrink(edge: np.ndarray, outcome_vs_line: np.ndarray) -> float:
    """Slope of (actual - line) on (model - line), forced into [0, 1].
    1 = the model's disagreements with the line are fully real; 0 = pure noise."""
    if len(edge) < 200 or np.var(edge) == 0:
        return 0.0
    e = edge - edge.mean()
    return float(np.clip(np.sum(e * (outcome_vs_line - outcome_vs_line.mean())) / np.sum(e * e), 0, 1))


@dataclass
class FittedModel:
    core: _Core
    k_spread: float     # share of the blend's disagreement with the spread that is kept
    k_total: float
    ml_a: float         # P(home win) = logistic(ml_a + ml_b * fair_margin)
    ml_b: float
    margin_dist: OutcomeDist
    total_dist: OutcomeDist

    @property
    def pure(self) -> Ridge:
        return self.core.pure

    @property
    def blend(self) -> Ridge:
        return self.core.blend

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.core.raw(df)
        out["fair_margin"] = np.where(
            out["spread_line"].notna(),
            out["spread_line"] + self.k_spread * (out["blend_margin"] - out["spread_line"]),
            out["model_margin"])
        out["fair_total"] = np.where(
            out["total_line"].notna(),
            out["total_line"] + self.k_total * (out["blend_total"] - out["total_line"]),
            out["model_total"])
        return out

    def win_prob(self, margin: float) -> float:
        return float(1 / (1 + np.exp(-(self.ml_a + self.ml_b * margin))))


def fit(train: pd.DataFrame, lam: float = 5.0, inner_start: int = 3) -> FittedModel:
    """Fit on all of `train`, calibrated on out-of-sample predictions.

    Calibration uses an inner walk-forward: each training season (after the first
    few) is predicted by a model fit only on the seasons before it. Those honest
    predictions decide how much to trust the model's disagreements with the
    market, and map predicted margins to moneyline win probabilities.
    """
    core = _fit_core(train, lam)
    seasons = sorted(train["season"].unique())
    oos = []
    for s in seasons[inner_start:]:
        inner = _fit_core(train[train["season"] < s], lam)
        oos.append(inner.raw(train[train["season"] == s]))
    oos = pd.concat(oos, ignore_index=True)

    m = oos.dropna(subset=["blend_margin", "result"])
    k_spread = _shrink((m["blend_margin"] - m["spread_line"]).to_numpy(),
                       (m["result"] - m["spread_line"]).to_numpy())
    t = oos.dropna(subset=["blend_total", "total"])
    k_total = _shrink((t["blend_total"] - t["total_line"]).to_numpy(),
                      (t["total"] - t["total_line"]).to_numpy())

    fair = (m["spread_line"] + k_spread * (m["blend_margin"] - m["spread_line"])).to_numpy()
    decided = m["result"].to_numpy() != 0
    ml_a, ml_b = _logistic(fair[decided], (m["result"].to_numpy()[decided] > 0).astype(float))

    tsd = float(np.std(t["total"] - (t["total_line"] + k_total * (t["blend_total"] - t["total_line"]))))
    sd = float(np.std(m["result"].to_numpy() - fair))
    mw = _key_weights(fair, m["result"].to_numpy(), sd, MARGINS, symmetric=True)
    tfair = (t["total_line"] + k_total * (t["blend_total"] - t["total_line"])).to_numpy()
    tw = _key_weights(tfair, t["total"].to_numpy(), tsd, TOTALS, symmetric=False)

    return FittedModel(core, k_spread, k_total, ml_a, ml_b,
                       OutcomeDist(sd, mw, MARGINS), OutcomeDist(tsd, tw, TOTALS))
