"""Feature-group ablation under the protocol in docs/EXPERIMENTS.md.

Baseline arm: FEATURES / TOTAL_FEATURES. Experiment arm: baseline + one group.
Both arms run the same walk-forward on the same rows; metrics are paired by game.
Every run is appended to logs/experiments.jsonl (hash-chained). The holdout may be
run once per group, and only after that group passed the development criteria.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, store, validate
from .ratings import EXPERIMENT_GROUPS, FEATURES, TOTAL_FEATURES

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "experiments.jsonl"
PROTOCOL = ROOT / "docs" / "EXPERIMENTS.md"
DEV = list(range(2015, 2022))
HOLDOUT_FIRST = 2022


class ProtocolError(RuntimeError):
    pass


def _rows(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = validate._targets(p)
    return {k: v[["game_id", "season", "y", "pm"]] for k, v in t.items()}


def compare(base: pd.DataFrame, exp: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Paired metrics on identical rows (games present in both arms for each target)."""
    out = {}
    rb, re_ = _rows(base), _rows(exp)
    for target in rb:
        j = rb[target].merge(re_[target][["game_id", "pm"]], on="game_id", suffixes=("_b", "_e"))
        pb = np.clip(j["pm_b"].to_numpy(float), 1e-6, 1 - 1e-6)
        pe = np.clip(j["pm_e"].to_numpy(float), 1e-6, 1 - 1e-6)
        y = j["y"].to_numpy(float)
        ll_b = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
        ll_e = -(y * np.log(pe) + (1 - y) * np.log(1 - pe))
        j = j.assign(d_ll=ll_e - ll_b, d_br=(pe - y) ** 2 - (pb - y) ** 2)
        est, lo, hi = metrics.cluster_bootstrap(j, "game_id", "d_ll", n=n_boot)
        seasons = j.groupby("season")["d_ll"].mean()
        out[target] = {"n": int(len(j)), "logloss_base": float(ll_b.mean()), "logloss_exp": float(ll_e.mean()),
                       "d_logloss": est, "ci_lo": lo, "ci_hi": hi, "d_brier": float(j["d_br"].mean()),
                       "seasons_improved": int((seasons < 0).sum()), "seasons": int(len(seasons))}
    mae = lambda p: float(np.nanmean(np.abs(p["result"] - p["model_margin"])))  # noqa: E731
    tmae = lambda p: float(np.nanmean(np.abs(p["total"] - p["model_total"])))  # noqa: E731
    out["own_line_mae"] = {"margin_base": mae(base), "margin_exp": mae(exp),
                           "total_base": tmae(base), "total_exp": tmae(exp)}
    return out


def passes_dev(res: dict) -> tuple[bool, str]:
    targets = ["winner", "home covers", "over hits"]
    worse = [t for t in targets if res[t]["ci_lo"] > 0]
    if worse:
        return False, f"significantly worse on {worse}"
    good = [t for t in targets if res[t]["ci_hi"] < 0 and res[t]["seasons_improved"] >= 5]
    if not good:
        return False, "no target with a log-loss CI below 0 and improvement in >= 5 of 7 dev seasons"
    return True, f"passes on {good}"


def history(group: str | None = None) -> list[dict]:
    return [r["data"] for r in store.read(LOG) if group is None or r["data"]["group"] == group]


def run(feat: pd.DataFrame, group: str, holdout: bool, last_season: int, model_version: str) -> dict:
    if group not in EXPERIMENT_GROUPS:
        raise ProtocolError(f"unknown group {group}; choose from {sorted(EXPERIMENT_GROUPS)}")
    past = history(group)
    if holdout:
        if any(r["period"] == "holdout" for r in past):
            raise ProtocolError(f"{group} was already evaluated on the holdout; the protocol allows one run")
        if not any(r["period"] == "development" and r["passed"] for r in past):
            raise ProtocolError(f"{group} has not passed the development criteria; holdout not allowed")
        seasons = list(range(HOLDOUT_FIRST, last_season + 1))
    else:
        seasons = DEV
    gm, gt = EXPERIMENT_GROUPS[group]
    base, _ = validate.run(feat, seasons, with_bets=False, verbose=False)
    exp, _ = validate.run(feat, seasons, with_bets=False, verbose=False,
                          features=FEATURES + gm, total_features=TOTAL_FEATURES + gt)
    res = compare(base, exp)
    if holdout:
        dev = [r for r in past if r["period"] == "development" and r["passed"]][-1]
        targets = dev["passed_targets"]
        passed = all(res[t]["d_logloss"] <= 0 for t in targets)
        verdict = f"holdout {'confirms' if passed else 'does not confirm'} {targets}"
        passed_targets = targets if passed else []
    else:
        passed, verdict = passes_dev(res)
        passed_targets = [t for t in ("winner", "home covers", "over hits")
                          if res[t]["ci_hi"] < 0 and res[t]["seasons_improved"] >= 5] if passed else []
    rec = {"group": group, "period": "holdout" if holdout else "development", "seasons": seasons,
           "features_margin": gm, "features_total": gt, "results": res, "passed": bool(passed),
           "passed_targets": passed_targets, "verdict": verdict, "model_version": model_version,
           "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    store.append(LOG, "experiment", [rec])
    return rec
