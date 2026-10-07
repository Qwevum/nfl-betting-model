"""Prediction log and grading.

Every `predict` run appends one row per game, market and side to
logs/predictions.csv: bets AND passes, with the model version, odds, probabilities
and decision. The file is append-only; nothing is ever removed, so results
can't be selectively reported.

`grade` uses, for each game/market/side, the last prediction logged before
kickoff, and reports:
  * probability quality for every logged prediction (Brier score, log loss)
    next to the market's no-vig probability
  * betting results for BET rows at the logged stake, and flat 1 unit
  * closing-line value (CLV) of the bets
  * what the passes would have done, for transparency
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import grade, profit
from .data import utcnow
from .odds import implied_probability, no_vig

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "predictions.csv"
ET = ZoneInfo("America/New_York")

LOG_COLS = ["logged_utc", "model_version", "season", "week", "game_id", "kickoff_utc", "away_team",
            "home_team", "market", "side", "team", "book", "point", "price", "price_source", "odds_time",
            "p_win", "p_push", "model_prob", "implied", "market_prob", "ev", "ev[fair 0.5 worse]",
            "ev[market only]", "decision", "stake_units", "reasons"]


def model_version() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "nflmodel", "run.py"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (h or "unknown") + ("-modified" if dirty else "")
    except Exception:
        return "unknown"


def kickoff_utc(gameday, gametime) -> str:
    t = str(gametime) if isinstance(gametime, str) else "13:00"
    local = pd.Timestamp(f"{pd.Timestamp(gameday).date()} {t}").tz_localize(ET)
    return local.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def log_predictions(rows: pd.DataFrame, preds: pd.DataFrame) -> Path:
    LOG.parent.mkdir(exist_ok=True)
    g = preds.set_index("game_id")
    df = rows.copy()
    df["logged_utc"] = utcnow()
    df["model_version"] = model_version()
    for col in ("season", "week", "away_team", "home_team"):
        df[col] = df["game_id"].map(g[col])
    df["kickoff_utc"] = [kickoff_utc(g.loc[i, "gameday"], g.loc[i, "gametime"]) for i in df["game_id"]]
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[LOG_COLS].to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return LOG


def _brier_logloss(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grade_log(games: pd.DataFrame) -> pd.DataFrame | None:
    if not LOG.exists():
        print("No predictions logged yet. Run `python run.py predict` first.")
        return None
    log = pd.read_csv(LOG)
    log = log[log["logged_utc"] < log["kickoff_utc"]]           # only pre-kickoff predictions
    log = (log.sort_values("logged_utc")
              .groupby(["game_id", "market", "side"]).tail(1))  # latest pre-kickoff view
    cols = ["game_id", "home_score", "away_score", "spread_line", "total_line",
            "home_moneyline", "away_moneyline"]
    df = log.merge(games[cols], on="game_id", how="left")
    done = df["home_score"].notna()
    df["result"] = ""
    df.loc[done, "result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                              for r in df[done].itertuples(index=False)]
    df["units"] = [profit(r, p) * s if r else np.nan
                   for r, p, s in zip(df["result"], df["price"], df["stake_units"])]
    df["flat_units"] = [profit(r, p) if r else np.nan for r, p in zip(df["result"], df["price"])]

    def clv(r):
        if r.market == "spread" and pd.notna(r.spread_line):
            return r.point - (-r.spread_line if r.side == "home" else r.spread_line)
        if r.market == "total" and pd.notna(r.total_line):
            return (r.total_line - r.point) if r.side == "over" else (r.point - r.total_line)
        if r.market == "ml" and pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
            h, a = no_vig(r.home_moneyline, r.away_moneyline)
            return (h if r.side == "home" else a) - implied_probability(r.price)
        return np.nan

    df["clv"] = df.apply(clv, axis=1)
    return df


def report(df: pd.DataFrame) -> None:
    done = df[(df["result"] != "") & (df["result"] != "P")]
    print(f"\nLogged predictions (latest before kickoff): {len(df)} sides, {len(done)} graded (pushes excluded)")
    if done.empty:
        print("Nothing graded yet.")
        return
    y = (done["result"] == "W").to_numpy(float)
    print("\nProbability quality on every graded side (bets and passes):")
    print(f"  {'':22s}{'n':>6s}{'Brier':>9s}{'LogLoss':>9s}")
    b, l = _brier_logloss(done["model_prob"].to_numpy(float), y)
    print(f"  {'model':22s}{len(done):6d}{b:9.4f}{l:9.4f}")
    m = done.dropna(subset=["market_prob"])
    if len(m):
        mb, ml_ = _brier_logloss(m["market_prob"].to_numpy(float), (m["result"] == "W").to_numpy(float))
        print(f"  {'market (no-vig)':22s}{len(m):6d}{mb:9.4f}{ml_:9.4f}")

    graded = df[df["result"] != ""]
    best = graded.sort_values("ev", ascending=False).groupby(["game_id", "market"]).head(1)
    for label, sub in (("BETS", graded[graded["decision"] != "NO BET"]),
                       ("PASSES (higher-EV side of each market, not bet)", best[best["decision"] == "NO BET"])):
        if sub.empty:
            continue
        print(f"\n{label}:")
        for market, g in sub.groupby("market"):
            w, l_, p = (g["result"] == "W").sum(), (g["result"] == "L").sum(), (g["result"] == "P").sum()
            unit = "pts" if market != "ml" else "prob"
            extra = (f"  staked {g['stake_units'].sum():.1f}u -> {g['units'].sum():+.2f}u"
                     if label == "BETS" else "")
            print(f"  {market:6s} {w}-{l_}-{p}  flat 1u: {g['flat_units'].sum():+.2f}u{extra}  "
                  f"avg CLV {g['clv'].mean():+.2f} {unit}")
