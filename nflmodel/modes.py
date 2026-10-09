"""Compare probability models on identical rows: legacy (baseline), model, market.

Protocol: docs/EXPERIMENTS.md, section "CAL: probability model comparison". Development
seasons only (2015-2021); the holdout is not used here. Every run is appended to
logs/experiments.jsonl.

All arms use the same walk-forward fits, the same games, the same closing consensus
prices and the same Settings except probability_model. Bets are the historical replay
(closing consensus price, one price per side), so the market-only arm can never find a
bet: a single book's price is always worse than its own no-vig probability. Price
shopping needs several books' timestamped quotes, which history here does not have.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd

from . import metrics, store, validate
from .config import Settings
from .experiment import DEV, LOG, PROTOCOL

MODES = ("legacy", "model", "market")
TARGETS = ("winner", "home covers", "over hits")


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prob_metrics(preds: dict[str, pd.DataFrame], n_boot: int = 2000) -> dict:
    """Per target: metrics of each arm and of the raw market on the SAME rows, and paired
    log-loss differences (arm - legacy, arm - raw market) with game-clustered CIs."""
    out = {}
    rows = {m: validate._targets(p) for m, p in preds.items()}
    for t in TARGETS:
        base = rows["legacy"][t][["game_id", "season", "y", "pk"]].rename(columns={"pk": "p_raw"})
        j = base
        for m in MODES:
            j = j.merge(rows[m][t][["game_id", "pm"]].rename(columns={"pm": f"p_{m}"}), on="game_id")
        y = j["y"].to_numpy(float)
        res = {"n": int(len(j)), "mean_outcome": float(y.mean())}
        for arm in ("raw",) + MODES:
            p = j[f"p_{arm}"].to_numpy(float)
            res[arm] = {"log_loss": float(_ll(p, y).mean()), "brier": float(((p - y) ** 2).mean()),
                        "ece": float(metrics.ece(pd.Series(p), pd.Series(y))),
                        "mean_abs_shift_from_raw": float(np.abs(p - j["p_raw"].to_numpy(float)).mean())}
        for a, b in (("model", "legacy"), ("market", "legacy"), ("legacy", "raw"), ("model", "raw")):
            d = j.assign(d=_ll(j[f"p_{a}"], y) - _ll(j[f"p_{b}"], y))
            est, lo, hi = metrics.cluster_bootstrap(d, "game_id", "d", n=n_boot)
            res[f"{a} - {b}"] = {"d_logloss": est, "ci_lo": lo, "ci_hi": hi,
                                 "seasons_lower": int((d.groupby("season")["d"].mean() < 0).sum()),
                                 "seasons": int(d["season"].nunique())}
        out[t] = res
    return out


def bet_metrics(bets: pd.DataFrame, games: int, n_boot: int = 2000) -> dict:
    b = bets[bets["decision"] != "NO BET"]
    out = {"bets": int(len(b)), "games_scored": int(games), "bets_per_game": float(len(b) / games) if games else 0.0}
    for market, g in list(b.groupby("market")) + [("all", b)]:
        if g.empty:
            continue
        roi, lo, hi = metrics.cluster_bootstrap(g, "game_id", "flat", n=n_boot)
        out[market] = {"bets": int(len(g)), "flat_roi": roi, "ci_lo": lo, "ci_hi": hi,
                       "units": float(g["flat"].sum()), "max_drawdown": float(metrics.max_drawdown(
                           g.sort_values(["gameday", "game_id"])["flat"]))}
    return out


def coverage(preds: pd.DataFrame) -> dict:
    """Share of scored games with a closing price in each market (identical for all arms)."""
    n = len(preds)
    return {"games": int(n),
            "spread": float(preds["home_spread_odds"].notna().mean()) if n else 0.0,
            "total": float(preds["over_odds"].notna().mean()) if n else 0.0,
            "ml": float(preds["home_moneyline"].notna().mean()) if n else 0.0}


def run(feat: pd.DataFrame, model_version: str, settings: Settings | None = None,
        seasons: list[int] | None = None, record: bool = True) -> dict:
    settings = settings or Settings()
    seasons = seasons or DEV
    if max(seasons) >= 2022:
        raise ValueError("this comparison is restricted to development seasons (<= 2021)")
    preds, bets, bets_std = {}, {}, {}
    for m in MODES:
        s = replace(settings, probability_model=m)
        preds[m], bets[m] = validate.run(feat, seasons, settings=s, verbose=False)
        _, bets_std[m] = validate.run(feat, seasons, settings=s, verbose=False, bet_overround=0.0476)
    games = len(preds["legacy"])
    rec = {"group": "CAL", "period": "development", "seasons": seasons, "settings": vars(settings),
           "probabilities": prob_metrics(preds), "coverage": coverage(preds["legacy"]),
           "bets_consensus_prices": {m: bet_metrics(bets[m], games) for m in MODES},
           "bets_standard_juice": {m: bet_metrics(bets_std[m], games) for m in MODES},
           "model_version": model_version,
           "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    worse = [t for t in TARGETS if rec["probabilities"][t]["model - legacy"]["ci_lo"] > 0]
    rec["passed"] = not worse
    rec["verdict"] = ("keep probability_model=model (not significantly worse than legacy on any target)"
                      if not worse else f"model is significantly worse than legacy on {worse}: revert default")
    if record:
        store.append(LOG, "experiment", [rec])
    return rec


def to_markdown(rec: dict) -> str:
    out = [f"Seasons {rec['seasons'][0]}-{rec['seasons'][-1]} (development), commit `{rec['model_version']}`.", "",
           "Probabilities on identical rows (lower log loss / Brier is better):", "",
           "| target | n | raw market | legacy | model | market-only | model - legacy (95% CI) | "
           "legacy - raw (95% CI) | mean shift from raw: legacy / model |", "|---|---|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        r = rec["probabilities"][t]
        d, e = r["model - legacy"], r["legacy - raw"]
        out.append(f"| {t} | {r['n']} | {r['raw']['log_loss']:.5f} | {r['legacy']['log_loss']:.5f} | "
                   f"{r['model']['log_loss']:.5f} | {r['market']['log_loss']:.5f} | {d['d_logloss']:+.5f} "
                   f"({d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}) | {e['d_logloss']:+.5f} ({e['ci_lo']:+.5f}, "
                   f"{e['ci_hi']:+.5f}) | {r['legacy']['mean_abs_shift_from_raw']:.2%} / "
                   f"{r['model']['mean_abs_shift_from_raw']:.2%} |")
    out += ["", "Brier and calibration error (ECE):", "", "| target | arm | Brier | ECE |", "|---|---|---|---|"]
    for t in TARGETS:
        for arm in ("raw",) + MODES:
            x = rec["probabilities"][t][arm]
            out.append(f"| {t} | {arm} | {x['brier']:.5f} | {x['ece']:.4f} |")
    for key, label in (("bets_consensus_prices", "closing consensus prices"),
                       ("bets_standard_juice", "every pair re-priced at -110/-110 juice (4.76%)")):
        out += ["", f"Historical replay at {label} (one price per side; no price shopping possible):", "",
                "| arm | bets | bets/game | flat ROI (95% CI, game-clustered) | units | max drawdown |",
                "|---|---|---|---|---|---|"]
        for m in MODES:
            b = rec[key][m]
            a = b.get("all")
            if a is None:
                out.append(f"| {m} | 0 | 0 | n/a | 0 | 0 |")
            else:
                out.append(f"| {m} | {b['bets']} | {b['bets_per_game']:.3f} | {a['flat_roi']:+.1%} "
                           f"({a['ci_lo']:+.1%}, {a['ci_hi']:+.1%}) | {a['units']:+.1f} | {a['max_drawdown']:.1f} |")
    c = rec["coverage"]
    out += ["", f"Market coverage of the {c['games']} scored games: spread {c['spread']:.0%}, total "
            f"{c['total']:.0%}, moneyline {c['ml']:.0%} (same rows for every arm).", "",
            f"Verdict: {rec['verdict']}."]
    return "\n".join(out)
