"""Estimation uncertainty by a time-blocked Bayesian bootstrap of the fitting stage.

Method (pre-registered in docs/UNCERTAINTY.md):
  * Features are FIXED. The stateful ratings pipeline (ratings.py) runs once, in date
    order; no game is shuffled, duplicated or re-run through it.
  * Replicate b re-weights the training games with Exp(1) multipliers drawn per BLOCK
    of games (Bayesian bootstrap, Rubin 1981), normalized to mean 1, and refits
    everything model.fit estimates under those weights: the ridge regressions, the inner
    walk-forward calibration (each inner season still fit on earlier seasons only),
    the shrinkage k, residual scales, key-number weights and the offset calibration.
    Weights never move a game across a season or date, so every chronological
    boundary is kept, and no season is ever dropped.
  * One weight per block keeps the dependence inside a block (scheme below).
  * Held fixed: rating hyperparameters, ridge penalty, inner_start, the key-number
    prior, the feature set and the observed market inputs. Intervals are therefore
    CONDITIONAL on the market snapshot and on the ratings as computed.

Replicate b always uses the random stream (seed, b), so results do not depend on the
number of workers, and the first B replicates of a larger run equal a run with B.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import model as model_mod
from .model import FittedModel, fit, logit, sigmoid
from .ratings import FEATURES, TOTAL_FEATURES

METHOD = "time-blocked Bayesian bootstrap of the fitting stage"
METHOD_VERSION = "ubb-1"
SCHEMES = ("game", "week4", "season", "season*week4")
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "uncertainty"

# columns a fit can read: the cache key covers exactly these
_KEY_COLS = sorted(set(FEATURES + TOTAL_FEATURES + [
    "game_id", "season", "week", "gameday", "result", "total", "spread_line", "total_line",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds", "home_moneyline", "away_moneyline"]))


# ---------------------------------------------------------------- weights

def block_ids(train: pd.DataFrame, scheme: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(primary block label per row, secondary label per row or None) for a scheme."""
    season = train["season"].to_numpy(int)
    wk4 = np.array([f"{s}-{(int(w) - 1) // 4}" for s, w in zip(season, train["week"])])
    if scheme == "game":
        return train["game_id"].astype(str).to_numpy(), None
    if scheme == "week4":
        return wk4, None
    if scheme == "season":
        return season.astype(str), None
    if scheme == "season*week4":
        return season.astype(str), wk4
    raise ValueError(f"unknown bootstrap scheme {scheme!r}; choose from {SCHEMES}")


def _exp_per_label(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq = np.unique(labels)                       # sorted: deterministic given the rows
    draw = dict(zip(uniq, rng.exponential(1.0, len(uniq))))
    return np.array([draw[x] for x in labels])


def replicate_weights(train: pd.DataFrame, scheme: str, seed: int, b: int) -> np.ndarray:
    """Exp(1) weight per block (product over levels for two-level schemes), mean 1."""
    rng = np.random.default_rng([int(seed), int(b)])
    a, s = block_ids(train, scheme)
    w = _exp_per_label(a, rng)
    if s is not None:
        w = w * _exp_per_label(s, rng)
    return w / w.mean()


# ---------------------------------------------------------------- replicate fits

def code_fingerprint() -> str:
    """Hash of the code that determines a replicate fit."""
    src = inspect.getsource(model_mod) + inspect.getsource(replicate_weights) + inspect.getsource(block_ids) \
        + inspect.getsource(_exp_per_label) + METHOD_VERSION
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def train_hash(train: pd.DataFrame) -> str:
    """Identifier of the exact training rows and every column a fit reads."""
    cols = [c for c in _KEY_COLS if c in train.columns]
    h = pd.util.hash_pandas_object(train[cols].reset_index(drop=True), index=True).to_numpy()
    return hashlib.sha256(h.tobytes() + ",".join(cols).encode()).hexdigest()[:16]


def cache_key(train: pd.DataFrame, n: int, seed: int, scheme: str, fit_kwargs: dict | None = None) -> str:
    parts = [METHOD_VERSION, code_fingerprint(), train_hash(train), str(n), str(seed), scheme,
             repr(sorted((fit_kwargs or {}).items()))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def is_cached(train: pd.DataFrame, n: int, seed: int, scheme: str, fit_kwargs: dict | None = None,
              start: int = 0) -> bool:
    key = cache_key(train, n, seed, scheme, {**dict(fit_kwargs or {}), "start": start})
    return (CACHE_DIR / f"{key}.pkl").exists()


_W: dict = {}


def _init_worker(train: pd.DataFrame, scheme: str, seed: int, fit_kwargs: dict) -> None:
    _W.update(train=train, scheme=scheme, seed=seed, fit_kwargs=fit_kwargs)


def _fit_one(b: int) -> FittedModel:
    t = _W["train"]
    return fit(t, weights=replicate_weights(t, _W["scheme"], _W["seed"], b), **_W["fit_kwargs"])


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def fit_replicates(train: pd.DataFrame, n: int, seed: int, scheme: str, workers: int = 1,
                   fit_kwargs: dict | None = None, use_cache: bool = True,
                   start: int = 0) -> list[FittedModel]:
    """Replicates start..start+n-1 of the bootstrap, each a full model.fit under weights.

    Cached under data/cache/uncertainty only when the method version, model code, training
    rows (which fix the information cutoff), n, seed, scheme and fit options all match."""
    fit_kwargs = dict(fit_kwargs or {})
    if n <= 0:
        return []
    key = cache_key(train, n, seed, scheme, {**fit_kwargs, "start": start})
    path = CACHE_DIR / f"{key}.pkl"
    if use_cache and path.exists():
        try:
            with path.open("rb") as f:
                meta, models = pickle.load(f)
            if meta.get("key") == key and len(models) == n:
                return models
        except Exception:
            pass   # unreadable cache: recompute
    idx = list(range(start, start + n))
    if workers > 1 and n > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(train, scheme, seed, fit_kwargs)) as ex:
            models = list(ex.map(_fit_one, idx, chunksize=max(1, n // (4 * workers))))
    else:
        _init_worker(train, scheme, seed, fit_kwargs)
        models = [_fit_one(b) for b in idx]
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(({"key": key, "n": n, "seed": seed, "scheme": scheme, "start": start}, models), f)
        tmp.replace(path)
    return models


# ---------------------------------------------------------------- intervals

def z_for(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("interval level must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + level / 2)


def prob_interval(p_hat: float, p_reps, level: float) -> tuple[float, float, float]:
    """(lo, hi, SD of logit p) : logit(p_hat) +/- z * SD(logit of the replicates).

    Centered on the point model's estimate in logit space; NaN bounds if fewer than two
    finite replicates. This is uncertainty about the ESTIMATED probability, not a range
    of outcomes."""
    r = np.asarray(p_reps, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or not np.isfinite(p_hat):
        return np.nan, np.nan, np.nan
    sd = float(np.std(logit(r), ddof=1))
    c = float(logit(p_hat))
    z = z_for(level)
    return float(sigmoid(c - z * sd)), float(sigmoid(c + z * sd)), sd


def linear_interval(x_hat: float, x_reps, level: float) -> tuple[float, float, float]:
    """(lo, hi, SD): x_hat +/- z * SD(replicates). Used for EV at a fixed price."""
    r = np.asarray(x_reps, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or not np.isfinite(x_hat):
        return np.nan, np.nan, np.nan
    sd = float(np.std(r, ddof=1))
    z = z_for(level)
    return float(x_hat - z * sd), float(x_hat + z * sd), sd


def pmf_interval(pmf: np.ndarray, support: np.ndarray, level: float) -> tuple[float, float]:
    """Equal-tailed interval of an integer outcome distribution: the smallest support
    values whose CDF reaches (1-level)/2 and 1-(1-level)/2. Coverage is at least about
    `level` (discrete outcomes)."""
    cdf = np.cumsum(pmf) / np.sum(pmf)
    a = (1 - level) / 2
    lo = support[np.searchsorted(cdf, a - 1e-12)]
    hi = support[min(np.searchsorted(cdf, 1 - a - 1e-12), len(support) - 1)]
    return float(lo), float(hi)


__all__ = ["METHOD", "METHOD_VERSION", "SCHEMES", "replicate_weights", "fit_replicates", "train_hash",
           "cache_key", "is_cached", "prob_interval", "linear_interval", "pmf_interval", "z_for", "default_workers",
           "code_fingerprint"]
