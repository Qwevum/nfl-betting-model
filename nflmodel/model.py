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
from math import sqrt

from .ratings import FEATURES, TOTAL_FEATURES

MARGINS = np.arange(-80, 81)
TOTALS = np.arange(0, 131)
# Column holding bootstrap sample weights (mean 1). Absent = ordinary unweighted fit.
WEIGHT = "_boot_w"


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorized erf (Abramowitz & Stegun 7.1.26, max abs error 1.5e-7)."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
             + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1 + _erf(x / sqrt(2)))


@dataclass
class Ridge:
    cols: list[str]
    lam: float = 5.0
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    coef_: np.ndarray = field(default=None, repr=False)
    intercept_: float = 0.0

    def fit(self, df: pd.DataFrame, y: pd.Series, w: np.ndarray | None = None) -> "Ridge":
        """w: optional sample weights with mean 1 (bootstrap replicates); None = unweighted."""
        X = df[self.cols].to_numpy(float)
        yv = y.to_numpy(float)
        if w is None:
            self.mean_, self.std_ = X.mean(0), X.std(0)
            self.std_[self.std_ == 0] = 1.0
            Z = (X - self.mean_) / self.std_
            self.intercept_ = yv.mean()
            A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
            self.coef_ = np.linalg.solve(A, Z.T @ (yv - self.intercept_))
            return self
        w = np.asarray(w, float)
        self.mean_ = np.average(X, axis=0, weights=w)
        self.std_ = np.sqrt(np.average((X - self.mean_) ** 2, axis=0, weights=w))
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_
        self.intercept_ = float(np.average(yv, weights=w))
        A = Z.T @ (Z * w[:, None]) + self.lam * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (w * (yv - self.intercept_)))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        Z = (df[self.cols].to_numpy(float) - self.mean_) / self.std_
        return self.intercept_ + Z @ self.coef_

    def raw_coefs(self) -> dict[str, float]:
        """Coefficients in original units (points per unit of feature)."""
        out = dict(zip(self.cols, self.coef_ / self.std_))
        out["intercept"] = self.intercept_ - float(np.sum(self.coef_ * self.mean_ / self.std_))
        return out


def _key_weights(mu: np.ndarray, actual: np.ndarray, sd: float, support: np.ndarray, symmetric: bool,
                 w: np.ndarray | None = None):
    """Ratio of observed to normal-expected frequency for each integer outcome.
    w: optional sample weights (mean 1); None = every game counts once."""
    lo, hi = support[0] - 0.5, support[-1] + 0.5
    edges = np.arange(lo, hi + 1)
    expected = np.zeros(len(support))
    if w is None:
        for m in mu:
            c = _norm_cdf((edges - m) / sd)
            expected += np.diff(c)
    else:
        for m, wi in zip(mu, w):
            expected += wi * np.diff(_norm_cdf((edges - m) / sd))
    observed = np.bincount((actual - support[0]).astype(int).clip(0, len(support) - 1),
                           weights=None if w is None else np.asarray(w, float),
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


def _w(df: pd.DataFrame) -> np.ndarray | None:
    """Bootstrap sample weights carried in the WEIGHT column, if present."""
    return df[WEIGHT].to_numpy(float) if WEIGHT in df.columns else None


def _fit_core(train: pd.DataFrame, lam: float, features: list[str] | None = None,
              total_features: list[str] | None = None) -> _Core:
    features = features or FEATURES
    total_features = total_features or TOTAL_FEATURES
    tr = train.dropna(subset=features + ["result"])
    trb = tr.dropna(subset=["spread_line"])
    tt = train.dropna(subset=total_features + ["total"])
    ttb = tt.dropna(subset=["total_line"])
    return _Core(
        Ridge(features, lam).fit(tr, tr["result"], _w(tr)),
        Ridge(features + ["spread_line"], lam).fit(trb, trb["result"], _w(trb)),
        Ridge(total_features, lam).fit(tt, tt["total"], _w(tt)),
        Ridge(total_features + ["total_line"], lam).fit(ttb, ttb["total"], _w(ttb)),
    )


def _logistic(x: np.ndarray, y: np.ndarray, iters: int = 50, sw: np.ndarray | None = None) -> tuple[float, float]:
    """sw: optional sample weights (mean 1); None = unweighted."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    s = None if sw is None else np.asarray(sw, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        if s is None:
            H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
            w += np.linalg.solve(H, X.T @ (y - p))
        else:
            H = X.T @ (X * (s * p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
            w += np.linalg.solve(H, X.T @ (s * (y - p)))
    return float(w[0]), float(w[1])


def _logistic_n(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 50,
                sw: np.ndarray | None = None) -> np.ndarray:
    """Logistic regression with intercept; light ridge on the slopes for stability.
    sw: optional sample weights (mean 1); None = unweighted."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 1e-6
    s = None if sw is None else np.asarray(sw, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        if s is None:
            H = X.T @ (X * (p * (1 - p))[:, None]) + R
            w += np.linalg.solve(H, X.T @ (y - p) - R @ w)
        else:
            H = X.T @ (X * (s * p * (1 - p))[:, None]) + R
            w += np.linalg.solve(H, X.T @ (s * (y - p)) - R @ w)
    return w


def _logistic_offset(x: np.ndarray, offset: np.ndarray, y: np.ndarray, ridge: float = 1.0,
                     iters: int = 50, sw: np.ndarray | None = None) -> float:
    """c in P(y) = sigmoid(offset + c * x): the market's log-odds enter with weight 1 and
    no intercept, so only the model's disagreement is estimated (light ridge toward 0).
    sw: optional sample weights (mean 1); None = unweighted."""
    c = 0.0
    s = None if sw is None else np.asarray(sw, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + c * x)))
        if s is None:
            g = np.sum(x * (y - p)) - ridge * c
            h = np.sum(x * x * p * (1 - p)) + ridge
        else:
            g = np.sum(s * x * (y - p)) - ridge * c
            h = np.sum(s * x * x * p * (1 - p)) + ridge
        c += g / h
    return float(c)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def novig_first(a, b):
    """No-vig probability of the first of two American prices (missing -> -110)."""
    a = np.where(pd.isna(a), -110.0, a).astype(float)
    b = np.where(pd.isna(b), -110.0, b).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ia = np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
        ib = np.where(b < 0, -b / (-b + 100), 100 / (b + 100))
    return ia / (ia + ib)


def _shrink(edge: np.ndarray, outcome_vs_line: np.ndarray, w: np.ndarray | None = None) -> float:
    """Slope of (actual - line) on (model - line), forced into [0, 1].
    1 = the model's disagreements with the line are fully real; 0 = pure noise.
    w: optional sample weights (mean 1); None = unweighted."""
    if len(edge) < 200 or np.var(edge) == 0:
        return 0.0
    if w is None:
        e = edge - edge.mean()
        return float(np.clip(np.sum(e * (outcome_vs_line - outcome_vs_line.mean())) / np.sum(e * e), 0, 1))
    w = np.asarray(w, float)
    e = edge - np.average(edge, weights=w)
    o = outcome_vs_line - np.average(outcome_vs_line, weights=w)
    return float(np.clip(np.sum(w * e * o) / np.sum(w * e * e), 0, 1))


def _wstd(x: np.ndarray, w: np.ndarray | None) -> float:
    """Population SD (np.std) or its weighted version."""
    if w is None:
        return float(np.std(x))
    return float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))


@dataclass
class FittedModel:
    core: _Core
    k_spread: float     # share of the blend's disagreement with the spread that is kept
    k_total: float
    ml_a: float         # fallback P(home win) = logistic(ml_a + ml_b * fair_margin), no ML odds
    ml_b: float
    margin_dist: OutcomeDist
    total_dist: OutcomeDist
    # Market-anchored calibration, fit out of sample: P = sigmoid(a + b*logit(p_market) + c*edge),
    # edge = blend model minus the line. c ~ 0 means the model adds nothing to the market.
    spread_cal: np.ndarray = None
    total_cal: np.ndarray = None
    ml_cal: np.ndarray = None
    # Offset calibration [0, 1, c] per market: the market probability is kept as is (weight 1,
    # no intercept) and only c, the value of the model's edge, is fit out of sample.
    offset_cal: dict = None
    # Out-of-sample residual SD of the ratings-only (pure) margin and total: the outcome scale
    # for forecasts made without any market line. None for models pickled before 2026-10.
    pure_margin_sd: float = None
    pure_total_sd: float = None

    def cal_for(self, market: str, mode: str) -> np.ndarray:
        """Calibration vector [a, b, c] for `market` under probability_model `mode`."""
        if mode == "legacy":
            return {"spread": self.spread_cal, "total": self.total_cal, "ml": self.ml_cal}[market]
        if mode == "market":
            return np.array([0.0, 1.0, 0.0])
        return self.offset_cal[market]

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

    @staticmethod
    def anchored(cal: np.ndarray, p_market: float, edge: float, edge_mult: float = 1.0) -> float:
        return float(sigmoid(cal[0] + cal[1] * logit(p_market) + cal[2] * edge * edge_mult))


def fit(train: pd.DataFrame, lam: float = 5.0, inner_start: int = 3,
        features: list[str] | None = None, total_features: list[str] | None = None,
        weights: np.ndarray | pd.Series | None = None) -> FittedModel:
    """Fit on all of `train`, calibrated on out-of-sample predictions.

    Calibration uses an inner walk-forward: each training season (after the first
    few) is predicted by a model fit only on the seasons before it. Those honest
    predictions decide how much to trust the model's disagreements with the
    market, and map predicted margins to moneyline win probabilities.

    weights: optional per-row sample weights aligned with `train` (bootstrap replicates,
    see uncertainty.py). They are normalized to mean 1 and used by EVERY estimate below,
    including the inner walk-forward, so a replicate re-weights history without moving
    any game across a chronological boundary. None = the ordinary fit (unchanged).
    """
    if weights is not None:
        wv = np.asarray(weights, float)
        if wv.shape != (len(train),) or not np.all(np.isfinite(wv)) or np.any(wv < 0) or wv.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, one per training row")
        train = train.assign(**{WEIGHT: wv / wv.mean()})
    core = _fit_core(train, lam, features, total_features)
    seasons = sorted(train["season"].unique())
    oos = []
    for s in seasons[inner_start:]:
        inner = _fit_core(train[train["season"] < s], lam, features, total_features)
        oos.append(inner.raw(train[train["season"] == s]))
    oos = pd.concat(oos, ignore_index=True)

    m = oos.dropna(subset=["blend_margin", "result"])
    k_spread = _shrink((m["blend_margin"] - m["spread_line"]).to_numpy(),
                       (m["result"] - m["spread_line"]).to_numpy(), _w(m))
    t = oos.dropna(subset=["blend_total", "total"])
    k_total = _shrink((t["blend_total"] - t["total_line"]).to_numpy(),
                      (t["total"] - t["total_line"]).to_numpy(), _w(t))

    fair = (m["spread_line"] + k_spread * (m["blend_margin"] - m["spread_line"])).to_numpy()
    decided = m["result"].to_numpy() != 0
    wm = _w(m)
    ml_a, ml_b = _logistic(fair[decided], (m["result"].to_numpy()[decided] > 0).astype(float),
                           sw=None if wm is None else wm[decided])

    tsd = _wstd((t["total"] - (t["total_line"] + k_total * (t["blend_total"] - t["total_line"]))).to_numpy(), _w(t))
    sd = _wstd(m["result"].to_numpy() - fair, wm)
    mw = _key_weights(fair, m["result"].to_numpy(), sd, MARGINS, symmetric=True, w=wm)
    tfair = (t["total_line"] + k_total * (t["blend_total"] - t["total_line"])).to_numpy()
    tw = _key_weights(tfair, t["total"].to_numpy(), tsd, TOTALS, symmetric=False, w=_w(t))
    # ratings-only scales: out-of-sample residuals of the pure models (no market input)
    pm = oos.dropna(subset=["model_margin", "result"])
    pt = oos.dropna(subset=["model_total", "total"])
    pure_sd = _wstd((pm["result"] - pm["model_margin"]).to_numpy(), _w(pm))
    pure_tsd = _wstd((pt["total"] - pt["model_total"]).to_numpy(), _w(pt))

    # Market-anchored probabilities (spread cover, over, home win), out of sample.
    ms = m[m["result"] != m["spread_line"]]
    spread_cal = _logistic_n(
        np.column_stack([logit(novig_first(ms["home_spread_odds"], ms["away_spread_odds"])),
                         ms["blend_margin"] - ms["spread_line"]]),
        (ms["result"] > ms["spread_line"]).to_numpy(float), sw=_w(ms))
    tt = t[t["total"] != t["total_line"]]
    total_cal = _logistic_n(
        np.column_stack([logit(novig_first(tt["over_odds"], tt["under_odds"])),
                         tt["blend_total"] - tt["total_line"]]),
        (tt["total"] > tt["total_line"]).to_numpy(float), sw=_w(tt))
    mm = m[(m["result"] != 0) & m["home_moneyline"].notna() & m["away_moneyline"].notna()]
    ml_cal = _logistic_n(
        np.column_stack([logit(novig_first(mm["home_moneyline"], mm["away_moneyline"])),
                         mm["blend_margin"] - mm["spread_line"]]),
        (mm["result"] > 0).to_numpy(float), sw=_w(mm))

    def off(x, p_first, y, sw):
        return np.array([0.0, 1.0, _logistic_offset(np.asarray(x, float), logit(p_first), np.asarray(y, float),
                                                    sw=sw)])
    offset_cal = {
        "spread": off(ms["blend_margin"] - ms["spread_line"],
                      novig_first(ms["home_spread_odds"], ms["away_spread_odds"]),
                      (ms["result"] > ms["spread_line"]).to_numpy(float), _w(ms)),
        "total": off(tt["blend_total"] - tt["total_line"], novig_first(tt["over_odds"], tt["under_odds"]),
                     (tt["total"] > tt["total_line"]).to_numpy(float), _w(tt)),
        "ml": off(mm["blend_margin"] - mm["spread_line"], novig_first(mm["home_moneyline"], mm["away_moneyline"]),
                  (mm["result"] > 0).to_numpy(float), _w(mm)),
    }

    return FittedModel(core, k_spread, k_total, ml_a, ml_b,
                       OutcomeDist(sd, mw, MARGINS), OutcomeDist(tsd, tw, TOTALS),
                       spread_cal, total_cal, ml_cal, offset_cal, pure_sd, pure_tsd)
