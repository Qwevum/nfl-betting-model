#!/usr/bin/env python3
"""NFL spread / moneyline / total model.

  python run.py predict [--week N] [--season YYYY]   full report for the upcoming week + log
  python run.py recommend [--date YYYY-MM-DD]        decision table for a game day (live or historical)
  python run.py validate [--from 2015]               out-of-sample validation vs baselines and market
  python run.py ratings                              current team power ratings
  python run.py grade                                forecasts, recommendations and wagers, graded separately
  python run.py place --forecast ID --stake U        record a wager you actually placed
  python run.py verify-log                           check the hash chains of history and ledger
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import decide, inputs, market, report, store, track, validate
from nflmodel.config import load_settings
from nflmodel.timeutil import fmt, kickoff_utc, now_utc, parse_utc
from nflmodel.backtest import grade as grade_bet, profit
from nflmodel.data import (FIRST_PBP_SEASON, SOURCES, load_games, load_injuries, load_team_games,
                           save_sources)
from nflmodel.model import fit
from nflmodel.odds import consensus_offers, gather_offers, validate_offers
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
    feat["kickoff_utc"] = [_kick(d, t) for d, t in zip(feat["gameday"], feat["gametime"])]
    return games, feat, table_, current, qbr


def _kick(gameday, gametime):
    try:
        return kickoff_utc(gameday, gametime)
    except ValueError:
        return pd.NaT


# ---------------------------------------------------------------- core analysis

def analyze(games, feat, qbr, day: pd.DataFrame, cutoff: pd.Timestamp, live: bool, settings,
            refresh: bool, now: pd.Timestamp):
    """Fit on games before `cutoff`, price `day`, and build decisions with context.

    Live: bookmaker offers are validated (timestamp, freshness, pre-kickoff) before
    any best-price selection; games already started are dropped.
    """
    train = feat[feat["result"].notna() & (feat["gameday"] < cutoff) & (feat["season"] > FIRST_PBP_SEASON)]
    model = fit(train)
    coefs = model.pure.raw_coefs()
    notes = []
    art = {"offers_raw": pd.DataFrame(), "offers_rejected": pd.DataFrame(), "offers_valid": pd.DataFrame(),
           "consensus": pd.DataFrame(), "references": pd.DataFrame(), "injury_season": None}

    overrides, weather = {}, {}
    if live:
        started = day[day["kickoff_utc"].isna() | (day["kickoff_utc"] <= now)]
        for g in started.itertuples(index=False):
            notes.append(f"{g.game_id} skipped: kickoff {'unknown' if pd.isna(g.kickoff_utc) else fmt(g.kickoff_utc)} "
                         f"is not after {fmt(now)}")
        day = day.drop(started.index)
        if day.empty:
            return model, coefs, model.predict(day), pd.DataFrame(), {}, notes, art
        overrides, n1 = inputs.load_qb_overrides(day, settings.kickoff_match_minutes)
        weather, n2 = inputs.load_weather(day, settings.kickoff_match_minutes)
        notes += n1 + n2
        day = inputs.apply_qb_overrides(day, overrides, games, qbr)
        day = inputs.apply_weather(day, weather)

    refs_by_game: dict = {}
    if live:
        inj = load_injuries(int(day["season"].iloc[0]), refresh=refresh)
        print("Gathering odds ...")
        books, consensus, n3 = gather_offers(day, inputs.kickoff_map(day), settings)
        notes += n3
        # 1) validate quotes, 2) build the market reference from valid quotes, 3) predict
        valid, rejected = validate_offers(books, inputs.kickoff_map(day), now, settings)
        art.update(offers_raw=books, offers_rejected=rejected, offers_valid=valid, consensus=consensus,
                   injury_season=int(day["season"].iloc[0]))
        if len(rejected):
            notes.append(f"{len(rejected)} bookmaker quote(s) rejected before price selection "
                         f"({rejected['reject_reason'].str.split(':').str[0].value_counts().to_dict()})")
        day = day.copy()
        day["reference_kind"] = "nflverse consensus (untimed)"
        for gid, o in valid.groupby("game_id"):
            refs = market.references_for_game(o, model, settings.min_reference_books)
            refs_by_game[gid] = refs
            m = day["game_id"] == gid
            if refs[("spread", None)] is not None:
                day.loc[m, "spread_line"] = refs[("spread", None)].line
            if refs[("total", None)] is not None:
                day.loc[m, "total_line"] = refs[("total", None)].line
            if any(refs[(mk, None)] is not None for mk in ("spread", "ml", "total")):
                day.loc[m, "reference_kind"] = "live bookmaker quotes"
        art["references"] = pd.DataFrame([
            {"game_id": gid, "market": mk, "excluded_book": ex, **{k: v for k, v in vars(r).items() if k != "per_book"},
             "per_book": str(r.per_book)}
            for gid, refs in refs_by_game.items() for (mk, ex), r in refs.items() if r is not None])
        offers = pd.concat([valid, consensus], ignore_index=True)
    else:
        inj = pd.DataFrame()
        offers = consensus_offers(day)
    preds = model.predict(day)

    rows, contexts = [], {}
    for g in preds.itertuples(index=False):
        if live:
            ctx = decide.build_context(g, inj, SOURCES, coefs, model.k_spread, overrides, weather)
        else:  # historical: only the rules that can be applied from data available then
            ctx = decide.GameContext()
            if pd.notna(g.spread_line) and abs(g.model_margin - g.spread_line) >= settings.gap_points:
                ctx.block["spread"].append("model vs market gap"); ctx.block["ml"].append("model vs market gap")
        contexts[g.game_id] = ctx
        refs = refs_by_game.get(g.game_id, {}) if live else None
        rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, settings.min_edge,
                                   refs=refs, min_ref_books=settings.min_reference_books)
    rows = pd.DataFrame(rows)
    if len(rows):
        best = decide.best_per_market(rows)
        rows["is_best_side"] = rows.set_index(["game_id", "market", "side"]).index.isin(
            best.set_index(["game_id", "market", "side"]).index)
    return model, coefs, preds, rows, contexts, notes, art


# ---------------------------------------------------------------- recording

def record(rows, preds, art, settings, now, args, games) -> None:
    """Snapshot the exact inputs and append every priced side to the forecast history.

    Simulated-time runs (--now) and runs from uncommitted code are never recorded."""
    version = track.model_version()
    if args.now or version.endswith("-modified") or version == "unknown":
        print("Not recorded: simulated time (--now) or uncommitted model code.")
        return
    if rows.empty:
        return
    run_utc = fmt(now)
    run_id = run_utc.replace(":", "").replace("-", "") + "_" + version
    slate = games[games["game_id"].isin(rows["game_id"])]
    inj = art.get("injury_season")
    sid, d = store.save_snapshot(
        run_id,
        files={"sources.json": ROOT / "data" / "sources.json",
               "injuries.parquet": ROOT / "data" / f"injuries_{inj}.parquet" if inj else None,
               "odds_manual.csv": ROOT / "odds_manual.csv", "qb_overrides.csv": ROOT / "qb_overrides.csv",
               "weather_manual.csv": ROOT / "weather_manual.csv"},
        frames={"schedule_rows.csv": slate, "offers_raw.csv": art["offers_raw"],
                "offers_rejected.csv": art["offers_rejected"], "offers_valid.csv": art["offers_valid"],
                "consensus.csv": art["consensus"], "references.csv": art["references"],
                "predictions.csv": preds},
        meta={"run_utc": run_utc, "model_version": version, "settings": vars(settings)})
    kick = dict(zip(preds["game_id"], preds["kickoff_utc"].map(fmt)))
    recs = store.record_forecasts(
        rows.assign(kickoff_utc=rows["game_id"].map(kick),
                    season=rows["game_id"].map(dict(zip(preds["game_id"], preds["season"]))),
                    week=rows["game_id"].map(dict(zip(preds["game_id"], preds["week"]))),
                    away_team=rows["game_id"].map(dict(zip(preds["game_id"], preds["away_team"]))),
                    home_team=rows["game_id"].map(dict(zip(preds["game_id"], preds["home_team"])))),
        {"run_utc": run_utc, "run_id": run_id, "model_version": version, "snapshot_id": sid,
         "horizon_minutes": settings.horizon_minutes})
    ex = sum(r["data"]["tier"] == "executable" for r in recs)
    cond = sum(r["data"]["tier"] == "conditional" for r in recs)
    print(f"Recorded {len(recs)} forecasts ({ex} executable, {cond} conditional) to "
          f"{store.FORECASTS.relative_to(ROOT)}; inputs in {d.relative_to(ROOT)}")
    print("Forecast ids for `python run.py place`: first 10 characters of the hash below")
    for r in recs:
        if r["data"]["tier"] != "pass":
            x = r["data"]
            print(f"  {r['hash'][:10]}  {x['decision']:22s} {x['game_id']} {x['market']} {x['team']} "
                  f"{'' if x['point'] is None else x['point']} {x['price']:+.0f} @ {x['book']}")


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
    settings = load_settings(min_edge=args.min_edge, max_odds_age_minutes=args.max_odds_age)
    now = parse_utc(args.now) if args.now else now_utc()
    model, coefs, preds, rows, contexts, notes, art = analyze(
        games, feat, qbr, wk, cutoff, live, settings, refresh=not args.no_refresh, now=now)
    for n in notes:
        print(f"  ! {n}")
    if rows.empty:
        raise SystemExit("No odds available for these games yet.")
    save_sources()

    best = rows[rows["is_best_side"]]
    print(f"\n=== {season} Week {week} ===  model {track.model_version()}  |  "
          f"flag: EV > {settings.min_edge:.0%} and robust to a 0.5-pt error\n")
    print(report.terminal_summary(best, preds))
    n_bet = (best["decision"] != "NO BET").sum()
    print(f"\n{n_bet} bet(s) of {len(best)} markets. Full reasoning per game in the report.")

    val = ROOT / "reports" / "validation_summary.md"
    md = report.header(season, week, SOURCES, track.model_version(), model, settings.min_edge,
                       val.read_text() if val.exists() else None)
    for g in preds.itertuples(index=False):
        md += "\n" + report.game_section(g, contexts[g.game_id], rows[rows["game_id"] == g.game_id], coefs, model)
    out = ROOT / "reports" / ("dev" if args.now or track.model_version().endswith("-modified") else "") \
        / f"{season}_week{week:02d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"Report: {out.relative_to(ROOT)}")
    if live:
        record(rows, preds, art, settings, now, args, games)


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
    settings = load_settings(min_edge=args.min_edge, max_odds_age_minutes=args.max_odds_age)
    now = parse_utc(args.now) if args.now else now_utc()
    model, coefs, preds, rows, contexts, notes, art = analyze(
        games, feat, qbr, day, start, live, settings, refresh=not args.no_refresh, now=now)
    for n in notes:
        print(f"  ! {n}")
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
          f"flag: EV > {settings.min_edge:.0%}\n")
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
        record(rows, preds, art, settings, now, args, games)


def cmd_validate(args):
    _, feat, _, _, _ = build(refresh=not args.no_refresh)
    last = int(feat.loc[feat["result"].notna(), "season"].max())
    seasons = list(range(args.start, last + 1))
    print(f"Validating {seasons[0]}-{seasons[-1]} (each season predicted by a model fit on earlier ones) ...")
    min_edge = load_settings(min_edge=args.min_edge).min_edge
    preds, bets = validate.run(feat, seasons, min_edge)
    print("Ablation: same model without the QB-change feature ...")
    no_qb = [f for f in validate.FEATURES if f != "f_qb"]
    preds_nq, _ = validate.run(feat, seasons, min_edge, features=no_qb, with_bets=False)

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
    print(f"\nBetting vs closing lines, decision rules as live (EV > {min_edge:.0%}, robust to 0.5 pt, "
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


def cmd_templates(args):
    _, feat, _, current, _ = build(refresh=not args.no_refresh)
    season = args.season or current
    upcoming = feat[(feat["season"] == season) & feat["result"].isna()]
    week = args.week or int(upcoming["week"].min())
    day = feat[(feat["season"] == season) & (feat["week"] == week)]
    for p in inputs.write_templates(day, ROOT / "templates"):
        print(f"wrote {p.relative_to(ROOT)}")
    print("Copy the rows you need into odds_manual.csv / qb_overrides.csv / weather_manual.csv.")


def cmd_grade(args):
    games = load_games(refresh=not args.no_refresh)
    settings = load_settings()
    for path in (store.FORECASTS, store.LEDGER):
        ok, msg = store.verify(path)
        print(f"{path.relative_to(ROOT)}: {msg}")
        if not ok:
            raise SystemExit("integrity check failed; not grading")
    g = track.grade_all(games, settings.horizon_minutes)
    track.report(g, settings.horizon_minutes)
    for name, df in g.items():
        df.to_csv(ROOT / "logs" / f"graded_{name}.csv", index=False)


def cmd_place(args):
    rec = store.place_bet(args.forecast, args.stake, placed_utc=args.placed_utc, price=args.price,
                          point=args.point, book=args.book, note=args.note or "")
    d = rec["data"]
    print(f"Recorded bet {d['bet_id']}: {d['game_id']} {d['market']} {d['team']} "
          f"{'' if d['point'] is None else d['point']} {d['price']:+.0f} @ {d['book']}, {d['stake_units']}u "
          f"(forecast said {d['forecast_decision']} at {d['forecast_price']:+.0f})")


def cmd_void(args):
    store.void_bet(args.bet_id, args.reason)
    print(f"Voided {args.bet_id}")


def cmd_verify_log(args):
    bad = False
    for path in (store.FORECASTS, store.LEDGER):
        ok, msg = store.verify(path)
        bad |= not ok
        print(f"{path.relative_to(ROOT)}: {'OK' if ok else 'FAILED'} - {msg}")
    if bad:
        raise SystemExit(1)


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
    pl = sub.add_parser("place", help="record a wager you actually placed")
    pl.add_argument("--forecast", required=True, help="forecast id (hash prefix) printed by predict")
    pl.add_argument("--stake", type=float, required=True, help="units staked")
    pl.add_argument("--price", type=float, help="price you got, if different from the forecast")
    pl.add_argument("--point", type=float, help="line you got, if different")
    pl.add_argument("--book"); pl.add_argument("--note")
    pl.add_argument("--placed-utc", help="when you placed it (ISO-8601 UTC); default now")
    vo = sub.add_parser("void", help="void a recorded wager (e.g. cancelled by the book)")
    vo.add_argument("--bet-id", required=True); vo.add_argument("--reason", required=True)
    sub.add_parser("verify-log", help="check the forecast history and ledger hash chains")
    t = sub.add_parser("templates", help="write input templates (game_id, kickoff_utc) for a week")
    t.add_argument("--week", type=int); t.add_argument("--season", type=int)
    for s in sub.choices.values():
        s.add_argument("--no-refresh", action="store_true", help="use cached data, no downloads")
        s.add_argument("--min-edge", type=float, default=None,
                       help="minimum EV to bet (0.02 = 2%%); default from settings")
        s.add_argument("--max-odds-age", type=float, default=None,
                       help="reject bookmaker quotes older than this many minutes; default from settings")
        s.add_argument("--now", help="evaluate as of this UTC time (ISO-8601 with Z), for reproducible runs")
    args = ap.parse_args()
    {"predict": cmd_predict, "recommend": cmd_recommend, "validate": cmd_validate,
     "ratings": cmd_ratings, "grade": cmd_grade, "templates": cmd_templates,
     "place": cmd_place, "void": cmd_void, "verify-log": cmd_verify_log}[args.cmd](args)


if __name__ == "__main__":
    main()
