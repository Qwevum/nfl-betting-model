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
class FittedModel:
    pure: Ridge
    blend: Ridge
    total_pure: Ridge
    total_blend: Ridge
    margin_dist: OutcomeDist
    total_dist: OutcomeDist

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["model_margin"] = self.pure.predict(out)
        out["model_total"] = self.total_pure.predict(out)
        has_line = out["spread_line"].notna()
        out["fair_margin"] = out["model_margin"]
        if has_line.any():
            out.loc[has_line, "fair_margin"] = self.blend.predict(out[has_line])
        has_tot = out["total_line"].notna()
        out["fair_total"] = out["model_total"]
        if has_tot.any():
            out.loc[has_tot, "fair_total"] = self.total_blend.predict(out[has_tot])
        return out


def fit(train: pd.DataFrame, lam: float = 5.0) -> FittedModel:
    tr = train.dropna(subset=FEATURES + ["result"])
    pure = Ridge(FEATURES, lam).fit(tr, tr["result"])
    trb = tr.dropna(subset=["spread_line"])
    blend = Ridge(FEATURES + ["spread_line"], lam).fit(trb, trb["result"])

    tt = train.dropna(subset=TOTAL_FEATURES + ["total"])
    total_pure = Ridge(TOTAL_FEATURES, lam).fit(tt, tt["total"])
    ttb = tt.dropna(subset=["total_line"])
    total_blend = Ridge(TOTAL_FEATURES + ["total_line"], lam).fit(ttb, ttb["total"])

    mu = blend.predict(trb)
    sd = float(np.std(trb["result"].to_numpy() - mu))
    mw = _key_weights(mu, trb["result"].to_numpy(), sd, MARGINS, symmetric=True)

    tmu = total_blend.predict(ttb)
    tsd = float(np.std(ttb["total"].to_numpy() - tmu))
    tw = _key_weights(tmu, ttb["total"].to_numpy(), tsd, TOTALS, symmetric=False)

    return FittedModel(pure, blend, total_pure, total_blend,
                       OutcomeDist(sd, mw, MARGINS), OutcomeDist(tsd, tw, TOTALS))
