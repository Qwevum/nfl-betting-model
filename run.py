#!/usr/bin/env python3
"""NFL spread / moneyline / total model.

  python run.py predict [--week N] [--season YYYY]   full report for the upcoming week + log
  python run.py recommend [--date YYYY-MM-DD]        decision table for a game day (live or historical)
  python run.py validate [--from 2015]               out-of-sample validation vs baselines and market
  python run.py ratings                              current team power ratings
  python run.py grade                                grade the prediction log (bets and passes)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import decide, report, track, validate
from nflmodel.backtest import grade as grade_bet, profit
from nflmodel.data import (FIRST_PBP_SEASON, SOURCES, load_games, load_injuries, load_team_games,
                           save_sources)
from nflmodel.model import fit
from nflmodel.odds import consensus_offers, gather_offers
from nflmodel.report import table
from nflmodel.ratings import build_features

ROOT = Path(__file__).resolve().parent
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build(refresh: bool = True):
    print("Loading schedules, scores and lines ...")
    games = load_games(refresh=refresh)
    upcoming = games["result"].isna() & (games["gameday"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    current = int(games.loc[upcoming, "season"].min() if upcoming.any() else games["season"].max())
    print("Loading play-by-play ...")
    tg, qg = load_team_games(list(range(FIRST_PBP_SEASON, current + 1)), refresh_current=refresh)
    print("Building ratings ...")
    feat, table_, qbr = build_features(games, tg, qg)
    return games, feat, table_, current, qbr


# ---------------------------------------------------------------- user inputs

def _read(name: str) -> pd.DataFrame:
    path = ROOT / name
    return pd.read_csv(path, comment="#") if path.exists() else pd.DataFrame()


def apply_qb_overrides(rows: pd.DataFrame, games: pd.DataFrame, qbr) -> tuple[pd.DataFrame, dict]:
    """qb_overrides.csv: team,qb_name,source - a starter you have confirmed."""
    ov = _read("qb_overrides.csv")
    if ov.empty:
        return rows, {}
    ids = {}
    for side in ("home", "away"):
        for n, i in zip(games[f"{side}_qb_name"], games[f"{side}_qb_id"]):
            if isinstance(n, str) and isinstance(i, str):
                ids[n] = i
    rows = rows.copy()
    done = {}
    for r in ov.itertuples(index=False):
        for side in ("home", "away"):
            m = rows[f"{side}_team"] == r.team
            if not m.any():
                continue
            qb_id = ids.get(r.qb_name)
            rating = qbr.rating(qb_id)  # unknown QB -> prior for inexperienced QBs
            base = rows.loc[m, f"{side}_qb_base"].fillna(rating)
            rows.loc[m, f"{side}_qb_id"] = qb_id if qb_id else f"unknown:{r.qb_name}"
            rows.loc[m, f"{side}_qb_name"] = r.qb_name
            rows.loc[m, f"{side}_qb_rating"] = rating
            rows.loc[m, f"{side}_qb_delta"] = rating - base
            done[r.team] = r.qb_name + ("" if qb_id else " (no NFL dropbacks in data; rated as an inexperienced QB)")
    rows["f_qb"] = rows["home_qb_delta"].fillna(0) - rows["away_qb_delta"].fillna(0)
    return rows, done


def apply_weather(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    w = _read("weather_manual.csv")
    if w.empty:
        return rows, {}
    rows = rows.copy()
    out = {}
    for r in w.itertuples(index=False):
        m = (rows["away_team"] == r.away) & (rows["home_team"] == r.home)
        if m.any():
            rows.loc[m, "t_wind"] = float(r.wind_mph)
            out[(r.away, r.home)] = r._asdict()
    return rows, out


# ---------------------------------------------------------------- core analysis

def analyze(games, feat, qbr, day: pd.DataFrame, cutoff: pd.Timestamp, live: bool, min_edge: float,
            refresh: bool):
    """Fit on games before `cutoff`, price `day`, and build decisions with context."""
    train = feat[feat["result"].notna() & (feat["gameday"] < cutoff) & (feat["season"] > FIRST_PBP_SEASON)]
    model = fit(train)
    coefs = model.pure.raw_coefs()

    overrides, weather = {}, {}
    if live:
        day, overrides = apply_qb_overrides(day, games, qbr)
        day, weather = apply_weather(day)
    preds = model.predict(day)

    if live:
        inj = load_injuries(int(day["season"].iloc[0]), refresh=refresh)
        print("Gathering odds ...")
        offers = gather_offers(day)
    else:
        inj = pd.DataFrame()
        offers = consensus_offers(day)

    rows, contexts = [], {}
    for g in preds.itertuples(index=False):
        if live:
            ctx = decide.build_context(g, inj, SOURCES, coefs, model.k_spread, overrides, weather)
        else:  # historical: only the rules that can be applied from data available then
            ctx = decide.GameContext()
            if pd.notna(g.spread_line) and abs(g.model_margin - g.spread_line) >= decide.GAP_POINTS:
                ctx.block["spread"].append("model vs market gap"); ctx.block["ml"].append("model vs market gap")
        contexts[g.game_id] = ctx
        rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, min_edge)
    rows = pd.DataFrame(rows)
    if len(rows):
        best = decide.best_per_market(rows)
        rows["is_best_side"] = rows.set_index(["game_id", "market", "side"]).index.isin(
            best.set_index(["game_id", "market", "side"]).index)
    return model, coefs, preds, rows, contexts


# ---------------------------------------------------------------- commands

def cmd_predict(args):
    games, feat, _, current, qbr = build(refresh=not args.no_refresh)
    season = args.season or current
    upcoming = feat[(feat["season"] == season) & feat["result"].isna()]
    week = args.week or int(upcoming["week"].min())
    wk = feat[(feat["season"] == season) & (feat["week"] == week)]
    if wk.empty:
        raise SystemExit(f"No games found for {season} week {week}")
    live = wk["result"].isna().any()
    cutoff = wk["gameday"].min()
    model, coefs, preds, rows, contexts = analyze(games, feat, qbr, wk, cutoff, live, args.min_edge,
                                                  refresh=not args.no_refresh)
    if rows.empty:
        raise SystemExit("No odds available for these games yet.")
    save_sources()

    best = rows[rows["is_best_side"]]
    print(f"\n=== {season} Week {week} ===  model {track.model_version()}  |  "
          f"flag: EV > {args.min_edge:.0%} and robust to a 0.5-pt error\n")
    print(report.terminal_summary(best, preds))
    n_bet = (best["decision"] != "NO BET").sum()
    print(f"\n{n_bet} bet(s) of {len(best)} markets. Full reasoning per game in the report.")

    val = ROOT / "reports" / "validation_summary.md"
    md = report.header(season, week, SOURCES, track.model_version(), model, args.min_edge,
                       val.read_text() if val.exists() else None)
    for g in preds.itertuples(index=False):
        md += "\n" + report.game_section(g, contexts[g.game_id], rows[rows["game_id"] == g.game_id], coefs, model)
    out = ROOT / "reports" / f"{season}_week{week:02d}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"Report: {out.relative_to(ROOT)}")
    if live:
        upcoming_ids = set(wk.loc[wk["result"].isna(), "game_id"])
        log = track.log_predictions(rows[rows["game_id"].isin(upcoming_ids)], preds)
        print(f"Logged {rows['game_id'].isin(upcoming_ids).sum()} predictions (bets and passes) to "
              f"{log.relative_to(ROOT)}")


def cmd_recommend(args):
    games, feat, _, _, qbr = build(refresh=not args.no_refresh)
    today = pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.date) if args.date else None
    if start is None:
        ahead = feat.loc[(feat["gameday"] >= today) & feat["result"].isna(), "gameday"]
        if ahead.empty:
            raise SystemExit("No upcoming games in the schedule.")
        start = ahead.min()
    end = start + pd.Timedelta(days=args.days)
    day = feat[(feat["gameday"] >= start) & (feat["gameday"] < end)]
    if day.empty:
        raise SystemExit(f"No games between {start.date()} and {(end - pd.Timedelta(days=1)).date()}.")
    live = start >= today
    model, coefs, preds, rows, contexts = analyze(games, feat, qbr, day, start, live, args.min_edge,
                                                  refresh=not args.no_refresh)
    if rows.empty:
        raise SystemExit("No odds available for these games yet.")
    markets = set(args.markets.split(","))
    rows = rows[rows["market"].isin(markets)]
    show = rows if args.all else rows[rows["is_best_side"]]

    g = preds.set_index("game_id")
    head = ["Date", "Game", "Market", "Bet", "Book", "Odds", "Implied", "Mkt no-vig", "Model", "EV",
            "EV -0.5pt", "Kelly", "Stake", "Decision"]
    align = "llllllrrrrrrrl"
    graded = not live
    if graded:
        head += ["Result", "Units"]; align += "lr"
    out, total, rec = [], 0.0, []
    for _, r in show.sort_values(["decision", "ev"], ascending=[True, False]).iterrows():
        x = g.loc[r["game_id"]]
        line = [pd.Timestamp(x.gameday).strftime("%a %m/%d"), f"{x.away_team} @ {x.home_team}",
                report.MARKET[r["market"]], report._bet(r), str(r["book"]), f"{int(r['price']):+d}",
                report._pct(r["implied"]), report._pct(r["market_prob"]), report._pct(r["model_prob"]),
                f"{r['ev']:+.1%}", f"{r['ev[fair 0.5 worse]']:+.1%}", f"{r['kelly']:.1%}",
                f"{r['stake_units']:.2f}u" if r["decision"] != "NO BET" else "-", r["decision"]]
        if graded:
            res = grade_bet(r["market"], r["side"], r["point"], x.home_score, x.away_score) if pd.notna(x.home_score) else ""
            u = profit(res, r["price"]) * r["stake_units"] if res and r["decision"] != "NO BET" else np.nan
            line += [res, "" if np.isnan(u) else f"{u:+.2f}"]
            if not np.isnan(u):
                total += u; rec.append(res)
        out.append(line)
    span = start.strftime("%a %b %d, %Y") + ("" if args.days == 1 else
                                             f" - {(end - pd.Timedelta(days=1)).strftime('%a %b %d')}")
    print(f"\n{span}  |  {len(day)} games  |  "
          f"{'live odds' if live else 'historical closing lines, model fit on earlier games only'}  |  "
          f"flag: EV > {args.min_edge:.0%}\n")
    print(table(out, head, align))
    b = show[show["decision"] != "NO BET"]
    print(f"\n{len(b)} bet(s), {b['stake_units'].sum():.2f}u total stake (quarter Kelly, max "
          f"{decide.MAX_STAKE_UNITS:g}u). Decision reasons: `predict` report or the saved CSV.")
    if graded and rec:
        print(f"Result: {rec.count('W')}-{rec.count('L')}-{rec.count('P')}, {total:+.2f}u "
              "(one day is noise; see `validate` for the full record)")
    path = ROOT / "picks" / f"recommend_{start.date()}.csv"
    path.parent.mkdir(exist_ok=True)
    show.to_csv(path, index=False)
    print(f"Saved {path.relative_to(ROOT)}")
    if live:
        log = track.log_predictions(rows, preds)
        print(f"Logged {len(rows)} predictions to {log.relative_to(ROOT)}")


def cmd_validate(args):
    _, feat, _, _, _ = build(refresh=not args.no_refresh)
    last = int(feat.loc[feat["result"].notna(), "season"].max())
    seasons = list(range(args.start, last + 1))
    print(f"Validating {seasons[0]}-{seasons[-1]} (each season predicted by a model fit on earlier ones) ...")
    preds, bets = validate.run(feat, seasons, args.min_edge)
    print("Ablation: same model without the QB-change feature ...")
    no_qb = [f for f in validate.FEATURES if f != "f_qb"]
    preds_nq, _ = validate.run(feat, seasons, args.min_edge, features=no_qb, with_bets=False)

    probs = validate.prob_table(preds)
    probs_nq = validate.prob_table(preds_nq).query("predictor == 'model'").assign(predictor="model without QB feature")
    probs = pd.concat([probs, probs_nq]).sort_values(["target", "brier"]).reset_index(drop=True)
    cal = validate.calibration(preds)
    mae = validate.mae_table(preds)
    mae_nq = validate.mae_table(preds_nq)[["season", "MAE model own line"]].rename(
        columns={"MAE model own line": "MAE own line w/o QB"})
    mae = mae.merge(mae_nq, on="season")
    bt = validate.bet_table(bets)

    print("\nProbability quality (lower Brier / log loss is better):")
    print(probs.round(4).to_string(index=False))
    print("\nCalibration of model win probabilities:")
    print(cal.round(3).to_string(index=False))
    print("\nMargin error by season (points):")
    print(mae.round(2).to_string(index=False))
    print(f"\nBetting vs closing lines, decision rules as live (EV > {args.min_edge:.0%}, robust to 0.5 pt, "
          "no 4+ pt gaps), one side per market:")
    print(bt.to_string(index=False))

    md = [f"# Validation {seasons[0]}-{seasons[-1]}", "", validate.__doc__.strip(), "",
          "## Probability quality", "", validate.to_markdown(probs), "",
          "## Calibration (model win probability)", "", validate.to_markdown(cal.astype({"bin": str})), "",
          "## Margin error by season", "", validate.to_markdown(mae), "",
          f"## Betting results (staking: flat 1u, and quarter Kelly capped at {decide.MAX_STAKE_UNITS:g}u)", "",
          validate.to_markdown(bt, {"win%": "{:.1%}", "flat ROI": "{:+.1%}", "qtr-Kelly units": "{:+.1f}",
                                    "qtr-Kelly ROI": "{:+.1%}"})]
    out = ROOT / "reports" / "validation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    w = probs[(probs["target"] == "winner")].set_index("predictor")
    allb = bt[bt["market"] == "all"].iloc[0]
    summary = (f"Validation {seasons[0]}-{seasons[-1]} ({int(w.loc['model', 'n'])} games, out of sample): "
               f"winner Brier model {w.loc['model', 'brier']:.4f} vs closing market "
               f"{w.loc['market no-vig (closing)', 'brier']:.4f} (lower is better). Bets under the live rules: "
               f"{allb['bets']} ({allb['W-L-P']}), flat ROI {allb['flat ROI']:+.1%} "
               f"(95% range {allb['flat ROI 95%']}). Full tables: reports/validation.md")
    (ROOT / "reports" / "validation_summary.md").write_text(summary + "\n")
    print(f"\n{summary}")


def cmd_ratings(args):
    _, _, table_, _, _ = build(refresh=not args.no_refresh)
    print("points_rating = points better than an average team on a neutral field\n")
    print(table_.round(3).to_string(index=False))


def cmd_grade(args):
    games = load_games(refresh=not args.no_refresh)
    df = track.grade_log(games)
    if df is not None:
        track.report(df)
        out = ROOT / "logs" / "graded.csv"
        df.to_csv(out, index=False)
        print(f"\nDetail: {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--week", type=int); p.add_argument("--season", type=int)
    r = sub.add_parser("recommend", help="decision table for a game day, live or historical")
    r.add_argument("--date", help="YYYY-MM-DD (default: next game day)")
    r.add_argument("--days", type=int, default=1, help="number of days from --date (default 1)")
    r.add_argument("--markets", default="spread,ml,total", help="comma list of spread,ml,total")
    r.add_argument("--all", action="store_true", help="show both sides of every market")
    v = sub.add_parser("validate"); v.add_argument("--from", dest="start", type=int, default=2015)
    sub.add_parser("ratings"); sub.add_parser("grade")
    for s in sub.choices.values():
        s.add_argument("--no-refresh", action="store_true", help="use cached data, no downloads")
        s.add_argument("--min-edge", type=float, default=decide.MIN_EDGE,
                       help="minimum EV to bet (0.02 = 2%%)")
    args = ap.parse_args()
    {"predict": cmd_predict, "recommend": cmd_recommend, "validate": cmd_validate,
     "ratings": cmd_ratings, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
