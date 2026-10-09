"""Grading of the forecast history and the placed-bet ledger, reported separately.

1. Forecast quality: every side the model priced, scored at the prediction
   horizon (the latest forecast recorded at or before kickoff - horizon), and
   compared with the market reference on exactly the same rows.
2. Recommendations (hypothetical, NOT wagers): the best side per market,
   split into executable ("BET") and conditional ("BET IF ...") tiers, flat 1u.
3. Actual wagers: only bets recorded in the ledger with `run.py place`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, store
from .backtest import grade, profit
from .odds import implied_probability, no_vig
from .timeutil import parse_utc

ROOT = Path(__file__).resolve().parent.parent
LEGACY_LOG = ROOT / "logs" / "predictions.csv"


def model_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "nflmodel", "run.py"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (h or "unknown") + ("-modified" if dirty else "")
    except Exception:
        return "unknown"


def _results(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "home_score", "away_score", "spread_line", "total_line", "home_moneyline", "away_moneyline"]
    df = df.drop(columns=[c for c in cols[1:] if c in df.columns]).merge(games[cols], on="game_id", how="left")
    done = df["home_score"].notna()
    df["result"] = ""
    df.loc[done, "result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                              for r in df[done].itertuples(index=False)]
    return df


def _clv(r) -> float:
    """Closing-line value vs the nflverse closing consensus (points; probability for ML)."""
    if r.market == "spread" and pd.notna(r.spread_line):
        return r.point - (-r.spread_line if r.side == "home" else r.spread_line)
    if r.market == "total" and pd.notna(r.total_line):
        return (r.total_line - r.point) if r.side == "over" else (r.point - r.total_line)
    if r.market == "ml" and pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
        h, a = no_vig(r.home_moneyline, r.away_moneyline)
        return (h if r.side == "home" else a) - implied_probability(r.price)
    return np.nan


def at_horizon(fc: pd.DataFrame, horizon_minutes: float) -> pd.DataFrame:
    """Latest forecast per game/market/side recorded at or before kickoff - horizon."""
    if fc.empty:
        return fc
    rec = fc["recorded_utc"].map(parse_utc)
    cut = fc["kickoff_utc"].map(parse_utc) - pd.Timedelta(minutes=horizon_minutes)
    fc = fc[rec <= cut].assign(_rec=rec[rec <= cut])
    return fc.sort_values("_rec").groupby(["game_id", "market", "side"]).tail(1).drop(columns="_rec")


def grade_all(games: pd.DataFrame, horizon_minutes: float, forecasts_path: Path = store.FORECASTS,
              ledger_path: Path = store.LEDGER) -> dict:
    out = {}
    fc = store.forecasts_frame(forecasts_path)
    if not fc.empty:
        h = _results(at_horizon(fc, horizon_minutes), games)
        h["clv"] = h.apply(_clv, axis=1)
        h["flat_units"] = [profit(r, p) if r else np.nan for r, p in zip(h["result"], h["price"])]
        out["forecasts"] = h
    led = store.ledger_frame(ledger_path)
    if not led.empty:
        led = _results(led[led["voided"].isna()].copy(), games)
        led["clv"] = led.apply(_clv, axis=1)
        led["units"] = [profit(r, p) * s if r else np.nan
                        for r, p, s in zip(led["result"], led["price"], led["stake_units"])]
        out["ledger"] = led
    return out


def report(g: dict, horizon_minutes: float) -> None:
    fc = g.get("forecasts")
    print(f"\n1) FORECAST QUALITY at the {horizon_minutes:g}-minute horizon")
    if fc is None or fc.empty:
        print("   no forecasts recorded at or before the horizon yet")
    else:
        done = fc[fc["result"].isin(["W", "L"])].dropna(subset=["model_prob", "market_prob"])
        n_games = done["game_id"].nunique()
        print(f"   {len(fc)} sides recorded, {len(done)} graded from {n_games} game(s) (pushes and rows without "
              "a market reference excluded; model and market scored on the same rows)")
        if n_games < 50:
            print(f"   ! only {n_games} game(s): far too few to distinguish model from market")
        if len(done):
            y = (done["result"] == "W").to_numpy(float)
            for name, col in (("model", "model_prob"), ("market reference", "market_prob")):
                b, l = metrics.brier_logloss(done[col], y)
                print(f"   {name:18s} Brier {b:.4f}  log loss {l:.4f}")
    print("\n2) RECOMMENDATIONS (hypothetical, flat 1u; these are NOT wagers)")
    if fc is not None and not fc.empty:
        flag = fc["is_best_side"] if "is_best_side" in fc else pd.Series(np.nan, index=fc.index)
        derived = fc.index.isin(fc.sort_values("ev", ascending=False)
                                  .groupby(["game_id", "market"]).head(1).index)
        best = fc[flag.where(flag.notna(), derived).astype(bool)]
        for tier in ("executable", "conditional"):
            t = best[(best["tier"] == tier) & (best["result"] != "")]
            pending = ((best["tier"] == tier) & (best["result"] == "")).sum()
            if t.empty:
                print(f"   {tier:11s}: none graded ({pending} pending)")
                continue
            w, l_, p = (t["result"] == "W").sum(), (t["result"] == "L").sum(), (t["result"] == "P").sum()
            print(f"   {tier:11s}: {w}-{l_}-{p}, {t['flat_units'].sum():+.2f}u flat, "
                  f"avg CLV {t['clv'].mean():+.2f} ({pending} pending)")
    led = g.get("ledger")
    print("\n3) ACTUAL WAGERS (ledger)")
    if led is None or led.empty:
        print("   none recorded (use `python run.py place`)")
        return
    s = led[led["result"] != ""].sort_values("placed_utc")
    print(f"   {len(led)} placed, {len(s)} settled, {len(led) - len(s)} open")
    if len(s):
        w, l_, p = (s["result"] == "W").sum(), (s["result"] == "L").sum(), (s["result"] == "P").sum()
        print(f"   {w}-{l_}-{p}, staked {s['stake_units'].sum():.2f}u, profit {s['units'].sum():+.2f}u "
              f"(ROI {s['units'].sum() / s['stake_units'].sum():+.1%}), max drawdown "
              f"{metrics.max_drawdown(s['units']):.2f}u, avg CLV {s['clv'].mean():+.2f}")


def import_legacy(path: Path = LEGACY_LOG) -> int:
    """One-time import of the pre-store CSV log into the hash-chained history (marked legacy)."""
    if not path.exists() or any(r["data"].get("legacy") for r in store.read(store.FORECASTS)):
        return 0
    df = pd.read_csv(path)
    rows = []
    for r in df.to_dict("records"):
        r = {k: r.get(k) for k in store.FORECAST_FIELDS if k in r} | {
            "kickoff_utc": r["kickoff_utc"], "model_version": r["model_version"], "run_utc": r["logged_utc"],
            "legacy": True, "reference": "nflverse consensus (untimed)", "reference_live": False}
        rows.append(r)
    for run_utc, chunk in pd.DataFrame(rows).groupby("run_utc", sort=True):
        recs = chunk.to_dict("records")
        for rec in recs:
            rec["tier"] = store.TIER.get(rec.get("decision"), "pass")
        store.append(store.FORECASTS, "forecast", recs, recorded_utc=run_utc)
    return len(rows)
