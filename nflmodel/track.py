"""Grade logged picks after the games: record, units, and closing-line value (CLV).

CLV is the best early sign of a real edge: if the line consistently moves toward
your side after you bet (you got +3.5 and it closed +2.5), the market agrees with
you, regardless of how a small number of games happened to finish.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import grade, profit
from .odds import no_vig, implied

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "picks"


def grade_all(games: pd.DataFrame) -> pd.DataFrame | None:
    files = sorted(PICKS.glob("*.csv"))
    if not files:
        print("No logged picks yet. Run `predict` first; it writes picks/<season>_week<N>.csv.")
        return None
    picks = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    picks = picks[picks["status"] == "BET"]
    cols = ["game_id", "home_score", "away_score", "spread_line", "total_line",
            "home_moneyline", "away_moneyline"]
    picks = picks.drop(columns=[c for c in cols[1:] if c in picks.columns])
    df = picks.merge(games[cols], on="game_id", how="left")
    done = df["home_score"].notna()
    df["result"] = ""
    df.loc[done, "result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                              for r in df[done].itertuples(index=False)]
    df["units"] = 0.0
    df.loc[done, "units"] = [profit(r, p) * s for r, p, s in
                             zip(df.loc[done, "result"], df.loc[done, "price"], df.loc[done, "stake_units"])]

    def clv(r):
        if r.market == "spread" and pd.notna(r.spread_line):
            close = -r.spread_line if r.side == "home" else r.spread_line
            return r.point - close  # points better than the close
        if r.market == "total" and pd.notna(r.total_line):
            return (r.total_line - r.point) if r.side == "over" else (r.point - r.total_line)
        if r.market == "ml" and pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
            h, a = no_vig(r.home_moneyline, r.away_moneyline)
            close_p = h if r.side == "home" else a
            return close_p - implied(r.price)  # win-prob points better than the fair close
        return np.nan

    df["clv"] = df.apply(clv, axis=1)
    return df


def report(df: pd.DataFrame) -> None:
    done = df[df["result"] != ""]
    print(f"\nGraded {len(done)} of {len(df)} logged bets\n")
    if done.empty:
        return
    for market, g in done.groupby("market"):
        w, l, p = (g["result"] == "W").sum(), (g["result"] == "L").sum(), (g["result"] == "P").sum()
        unit = "pts" if market != "ml" else "prob"
        print(f"  {market:6s} {w}-{l}-{p}  units {g['units'].sum():+.2f}  "
              f"avg CLV {g['clv'].mean():+.2f} {unit}  beat close {(g['clv'] > 0).mean():.0%}")
    print(f"\n  TOTAL units {done['units'].sum():+.2f} on {done['stake_units'].sum():.1f} staked "
          f"(ROI {done['units'].sum() / done['stake_units'].sum():+.1%})")
