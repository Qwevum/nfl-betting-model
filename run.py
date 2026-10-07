#!/usr/bin/env python3
"""NFL spread / moneyline / total model.

  python run.py predict [--week N] [--season YYYY]   picks for the upcoming week
  python run.py backtest [--from 2015]               walk-forward test vs closing lines
  python run.py ratings                              current team power ratings
  python run.py grade                                grade logged picks + CLV
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import backtest, evaluate, track
from nflmodel.data import FIRST_PBP_SEASON, load_games, load_team_games
from nflmodel.model import fit
from nflmodel.odds import gather_offers
from nflmodel.ratings import build_features

ROOT = Path(__file__).resolve().parent
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build(refresh: bool = True):
    print("Loading schedules, scores and lines ...")
    games = load_games(refresh=refresh)
    current = int(games.loc[games["result"].isna() & (games["gameday"] >= pd.Timestamp.today().normalize()
                                                         - pd.Timedelta(days=1)), "season"].min()
                  if games["result"].isna().any() else games["season"].max())
    print("Loading play-by-play team stats ...")
    tg = load_team_games(list(range(FIRST_PBP_SEASON, current + 1)), refresh_current=refresh)
    print("Building ratings ...")
    feat, table = build_features(games, tg)
    return games, feat, table, current


def apply_adjustments(preds: pd.DataFrame, market_weight: float, k_spread: float) -> pd.DataFrame:
    """adjustments.csv: team,points,note  (points = how much better (+) or worse (-)
    the team is this week than its ratings say, e.g. KC,-6,backup QB starting)."""
    path = ROOT / "adjustments.csv"
    if not path.exists():
        return preds
    adj = pd.read_csv(path, comment="#")
    if adj.empty:
        return preds
    pts = adj.groupby("team")["points"].sum()
    h = preds["home_team"].map(pts).fillna(0)
    a = preds["away_team"].map(pts).fillna(0)
    preds = preds.copy()
    preds["model_margin"] += h - a
    # The market has usually priced the news already; only the ratings part of the
    # blend is missing it, and only the calibrated share of that counts.
    preds["fair_margin"] += (h - a) * (1 - market_weight) * k_spread
    for t, p in pts.items():
        print(f"  adjustment: {t} {p:+.1f} pts")
    return preds


def fmt_line(team_home: str, team_away: str, home_margin: float) -> str:
    if pd.isna(home_margin):
        return "-"
    if abs(home_margin) < 0.25:
        return "PK"
    fav = team_home if home_margin > 0 else team_away
    return f"{fav} -{abs(home_margin):.1f}"


def describe_pick(r, preds_idx) -> str:
    g = preds_idx.loc[r.game_id]
    team = {"home": g.home_team, "away": g.away_team}.get(r.side, r.side.upper())
    if r.market == "spread":
        what = f"{team} {r.point:+g}"
    elif r.market == "ml":
        what = f"{team} ML"
    else:
        what = f"{team} {r.point:g}"
    return f"{what} ({int(r.price):+d} @ {r.book})"


def cmd_predict(args):
    games, feat, table, current = build(refresh=not args.no_refresh)
    season = args.season or current
    played = feat[feat["result"].notna()]
    model = fit(played[played["season"] > FIRST_PBP_SEASON])
    coefs = model.blend.raw_coefs()

    upcoming = feat[(feat["season"] == season) & feat["result"].isna()]
    week = args.week or int(upcoming["week"].min())
    wk = feat[(feat["season"] == season) & (feat["week"] == week)]
    if wk.empty:
        raise SystemExit(f"No games found for {season} week {week}")
    preds = model.predict(wk)
    preds = apply_adjustments(preds, coefs["spread_line"], model.k_spread)

    print("Gathering odds ...")
    offers = gather_offers(wk)
    best = evaluate.evaluate_offers(model, preds, offers)
    picks = evaluate.pick_bets(best, preds)
    preds["mkt_home_win"] = evaluate.market_no_vig(preds)
    preds["model_home_win"] = [model.win_prob(m) for m in preds["fair_margin"]]

    idx = preds.set_index("game_id")
    picks["pick"] = [describe_pick(r, idx) for r in picks.itertuples(index=False)]

    # ---- sheet-style summary ----
    rows = []
    for g in preds.itertuples(index=False):
        row = {
            "game": f"{g.away_team} @ {g.home_team}",
            "market": fmt_line(g.home_team, g.away_team, g.spread_line),
            "model": fmt_line(g.home_team, g.away_team, g.model_margin),
            "fair": fmt_line(g.home_team, g.away_team, g.fair_margin),
            "home_win%": f"{g.model_home_win:.0%}",
            "mkt_home%": "" if pd.isna(g.mkt_home_win) else f"{g.mkt_home_win:.0%}",
            "tot mkt/fair": "" if pd.isna(g.total_line) else f"{g.total_line:g} / {g.fair_total:.1f}",
        }
        for market, label in (("spread", "SPREAD"), ("ml", "ML"), ("total", "TOTAL")):
            p = picks[(picks["game_id"] == g.game_id) & (picks["market"] == market)]
            if p.empty:
                row[label] = ""
                continue
            p = p.iloc[0]
            row[label] = ("" if p.status == "pass" else
                          f"{'⚠ ' if p.status == 'check news' else ''}{p.pick} EV {p.ev:+.1%} {p.stake_units:g}u")
        rows.append(row)
    sheet = pd.DataFrame(rows)

    print(f"\n=== {season} Week {week} ===   (edge kept after calibration: spreads "
          f"{model.k_spread:.0%}, totals {model.k_total:.0%}; margin sd {model.margin_dist.sd:.1f})\n")
    print(sheet.to_string(index=False))
    bets = picks[picks["status"] == "BET"].sort_values("ev", ascending=False)
    print(f"\nBETS ({len(bets)}):")
    for r in bets.itertuples(index=False):
        print(f"  {r.pick:40s} win {r.p_win:.1%}  EV {r.ev:+.1%}  stake {r.stake_units:g}u  "
              f"(best of {r.n_books} book{'s' if r.n_books > 1 else ''})")
    chk = picks[picks["status"] == "check news"]
    if len(chk):
        print("\nCHECK NEWS (model disagrees with market by 4+ pts; usually injuries/QB):")
        for r in chk.itertuples(index=False):
            print(f"  {r.pick}  EV {r.ev:+.1%}")

    out_dir = ROOT / "picks"
    out_dir.mkdir(exist_ok=True)
    out = picks.merge(preds[["game_id", "season", "week", "away_team", "home_team", "gameday",
                             "model_margin", "fair_margin", "fair_total"]], on="game_id")
    out["logged_at"] = dt.datetime.now().isoformat(timespec="minutes")
    path = out_dir / f"{season}_week{week:02d}.csv"
    out.to_csv(path, index=False)
    sheet.to_csv(out_dir / f"{season}_week{week:02d}_sheet.csv", index=False)
    print(f"\nSaved {path.relative_to(ROOT)} (used by `grade`) and the sheet view next to it.")


def cmd_backtest(args):
    games, feat, table, current = build(refresh=not args.no_refresh)
    last = int(feat.loc[feat["result"].notna(), "season"].max())
    seasons = list(range(args.start, last + 1))
    print(f"Walk-forward backtest {seasons[0]}-{seasons[-1]} ...")
    fits, bets = backtest.run(feat, seasons)
    print("\nAccuracy (mean absolute error of predicted home margin, lower is better):")
    print(fits.round(3).to_string(index=False))
    print("\nBetting against CLOSING lines (consensus prices):")
    summ = backtest.summarize(bets)
    print(summ.round(3).to_string(index=False))
    out = ROOT / "data" / "backtest_bets.csv"
    bets.to_csv(out, index=False)
    print(f"\nAll backtest bets: {out.relative_to(ROOT)}")
    print("Break-even at -110 is 52.4%. Treat any ROI inside the confidence range of 0 as noise.")


def cmd_ratings(args):
    _, _, table, _ = build(refresh=not args.no_refresh)
    print("points_rating = points better than an average team on a neutral field\n")
    print(table.round(3).to_string(index=False))


def cmd_grade(args):
    games = load_games(refresh=not args.no_refresh)
    df = track.grade_all(games)
    if df is not None:
        track.report(df)
        df.to_csv(ROOT / "picks" / "graded.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--week", type=int); p.add_argument("--season", type=int)
    b = sub.add_parser("backtest"); b.add_argument("--from", dest="start", type=int, default=2015)
    sub.add_parser("ratings"); sub.add_parser("grade")
    for s in sub.choices.values():
        s.add_argument("--no-refresh", action="store_true", help="use cached data, no downloads")
    args = ap.parse_args()
    {"predict": cmd_predict, "backtest": cmd_backtest, "ratings": cmd_ratings, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
