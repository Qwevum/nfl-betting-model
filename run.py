#!/usr/bin/env python3
"""NFL spread / moneyline / total model.

  python run.py predict [--week N] [--season YYYY]   full report for the upcoming week + log
  python run.py recommend [--date YYYY-MM-DD]        decision table for a game day (live or historical)
  python run.py validate [--from 2015]               out-of-sample validation vs baselines and market
  python run.py ratings                              current team power ratings
  python run.py grade [--season S --weeks A-B]       forecasts, recommendations and wagers, graded separately
  python run.py coverage --season S --weeks A-B      which slate games have horizon-eligible forecasts
  python run.py place --forecast ID --stake U        record a wager you actually placed
  python run.py verify-log                           check the hash chains of history and ledger
  python run.py collect [--days 7]                   snapshot odds + injuries for upcoming games (no model)
  python run.py check-live                           is the live bookmaker feed configured and usable?
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import decide, inputs, pricing, report, runtime, store, track, validate
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
            refresh: bool, clock):
    """Fit on games before `cutoff`, price `day`, and build decisions with context.

    Live: bookmaker offers are validated (timestamp, freshness, pre-kickoff) before
    any best-price selection; games already started are dropped.
    """
    train = feat[feat["result"].notna() & (feat["gameday"] < cutoff) & (feat["season"] > FIRST_PBP_SEASON)]
    model = fit(train)
    coefs = model.pure.raw_coefs()
    notes = []
    art = {"offers_raw": pd.DataFrame(), "offers_rejected": pd.DataFrame(), "offers_valid": pd.DataFrame(),
           "consensus": pd.DataFrame(), "references": pd.DataFrame(), "injury_season": None,
           "inputs_rejected": pd.DataFrame(columns=inputs.REJECT_COLS), "weather_accepted": pd.DataFrame(),
           "qb_accepted": pd.DataFrame()}

    overrides, weather = {}, {}
    if live:
        now = clock.stamp("inputs_read")   # decision time for user inputs
        started = day[day["kickoff_utc"].isna() | (day["kickoff_utc"] <= now)]
        for g in started.itertuples(index=False):
            notes.append(f"{g.game_id} skipped: kickoff {'unknown' if pd.isna(g.kickoff_utc) else fmt(g.kickoff_utc)} "
                         f"is not after {fmt(now)}")
        day = day.drop(started.index)
        if day.empty:
            return model, coefs, model.predict(day), pd.DataFrame(), {}, notes, art
        # validate user inputs against the decision time and kickoff; only accepted rows are applied
        overrides, rej_qb = inputs.load_qb_overrides(day, settings, now)
        weather, rej_wx = inputs.load_weather(day, settings, now)
        rejected_inputs = pd.concat([rej_qb, rej_wx], ignore_index=True)
        art.update(inputs_rejected=rejected_inputs, qb_accepted=pd.DataFrame(list(overrides.values())),
                   weather_accepted=pd.DataFrame(list(weather.values())))
        for x in rejected_inputs.itertuples(index=False):
            notes.append(f"{x.file} row rejected ({x.game_id}{', ' + x.team if isinstance(x.team, str) else ''}): "
                         f"{x.reason}")
        day = inputs.apply_qb_overrides(day, overrides, games, qbr)
        # Weather is NOT applied to the model: wind/temperature are not features. A valid
        # forecast only lifts the "no outdoor total without a forecast" rule (decide.py).

    if not live:
        # historical replay: untimed consensus only; only rules applicable from data available then
        preds = model.predict(day)
        offers = consensus_offers(day)
        rows, contexts = [], {}
        for g in preds.itertuples(index=False):
            ctx = decide.GameContext()
            if pd.notna(g.spread_line) and abs(g.model_margin - g.spread_line) >= settings.gap_points:
                ctx.block["spread"].append("model vs market gap"); ctx.block["ml"].append("model vs market gap")
            contexts[g.game_id] = ctx
            rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, settings, refs=None)
        rows = pd.DataFrame(rows)
        if len(rows):
            best = decide.best_per_market(rows)
            rows["is_best_side"] = rows.set_index(["game_id", "market", "side"]).index.isin(
                best.set_index(["game_id", "market", "side"]).index)
        clock.stamp("prediction_completed")
        art["timeline"] = clock.as_dict()
        return model, coefs, preds, rows, contexts, notes, art

    inj = load_injuries(int(day["season"].iloc[0]), refresh=refresh)
    clock.stamp("injuries_retrieved")
    print("Gathering odds ...")
    kick = inputs.kickoff_map(day)
    books, consensus, n3 = gather_offers(day, kick, settings)
    collected = clock.stamp("odds_collected")   # quotes are validated against the time they were fetched
    notes += n3
    # 1) validate quotes, 2) build references from valid quotes, 3) predict and decide
    valid, rejected = validate_offers(books, kick, collected, settings)
    art.update(offers_raw=books, offers_rejected=rejected, offers_valid=valid, consensus=consensus,
               injury_season=int(day["season"].iloc[0]))
    if len(rejected):
        notes.append(f"{len(rejected)} bookmaker quote(s) rejected before price selection "
                     f"({rejected['reject_reason'].str.split(':').str[0].value_counts().to_dict()})")

    def context_fn(g):
        ctx = decide.build_context(g, inj, SOURCES, coefs, model.k_spread, overrides, weather, settings)
        rej = art["inputs_rejected"]
        for x in rej[rej["game_id"] == g.game_id].itertuples(index=False):
            ctx.missing.append(f"{x.file} row rejected and NOT applied"
                               f"{' (' + x.team + ')' if isinstance(x.team, str) else ''}: {x.reason}")
        return ctx

    first = {}

    def price(v):
        p = pricing.price_slate(model, day, v, consensus, settings, context_fn)
        first.setdefault("references", p.references)
        return p

    # price, then revalidate EVERY quote (selected offers and reference quotes) at completion
    # and re-price from the quotes still valid; finally downgrade anything stale or post-kickoff
    priced, info = runtime.finalize(price, books, kick, clock, settings, valid)
    if info["repriced"]:
        notes.append(f"{info['expired_quotes']} quote(s) expired while the run was processing; the slate was "
                     f"re-priced {info['repriced']} time(s) from the quotes still valid at completion")
    if not info["stable"]:
        notes.append("quotes kept expiring during re-pricing; executable bets relying on any stale quote were "
                     "downgraded to conditional")
    art.update(references=priced.references, references_at_collection=first.get("references"),
               offers_valid_final=priced.valid, reprice=info)
    rows = priced.rows
    if len(rows):
        n_late = int(rows["post_kickoff"].sum())
        if n_late:
            notes.append(f"{n_late} side(s) dropped: kickoff passed before the run completed")
    art["timeline"] = clock.as_dict()
    return model, coefs, priced.preds, rows, priced.contexts, notes, art


# ---------------------------------------------------------------- recording

def record(rows, preds, art, settings, clock, args, games) -> None:
    """Snapshot the exact inputs and append every priced side to the forecast history.

    Simulated-time runs (--now) and runs from uncommitted code are never recorded.
    Forecasts carry every stage timestamp; their recorded time is the actual write
    time, never earlier than prediction completion. Sides whose game kicked off
    before the run completed are not recorded."""
    version = track.model_version()
    if clock.simulated or args.now or version.endswith("-modified") or version == "unknown":
        print("Not recorded: simulated time (--now) or uncommitted model code.")
        return
    if "post_kickoff" in rows:
        rows = rows[~rows["post_kickoff"].astype(bool)]
    if rows.empty:
        return
    timeline = art["timeline"]
    run_utc = timeline["prediction_completed_utc"]
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
                "references_at_collection.csv": art.get("references_at_collection"),
                "offers_valid_final.csv": art.get("offers_valid_final"),
                "inputs_rejected.csv": art["inputs_rejected"], "qb_accepted.csv": art["qb_accepted"],
                "weather_accepted.csv": art["weather_accepted"], "predictions.csv": preds},
        meta={"model_version": version, "settings": vars(settings), "reprice": art.get("reprice"), **timeline})
    kick = dict(zip(preds["game_id"], preds["kickoff_utc"].map(fmt)))
    recs = store.record_forecasts(
        rows.assign(kickoff_utc=rows["game_id"].map(kick),
                    season=rows["game_id"].map(dict(zip(preds["game_id"], preds["season"]))),
                    week=rows["game_id"].map(dict(zip(preds["game_id"], preds["week"]))),
                    away_team=rows["game_id"].map(dict(zip(preds["game_id"], preds["away_team"]))),
                    home_team=rows["game_id"].map(dict(zip(preds["game_id"], preds["home_team"])))),
        {"run_id": run_id, "model_version": version, "snapshot_id": sid,
         "horizon_minutes": settings.horizon_minutes, **timeline})
    if len(art.get("weather_accepted", [])):
        n = store.archive_weather({w["game_id"]: w for w in art["weather_accepted"].to_dict("records")}, run_id)
        print(f"Archived {n} new weather forecast(s) to {store.WEATHER_ARCHIVE.relative_to(ROOT)}")
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
    clock = runtime.Clock(parse_utc(args.now) if args.now else None)
    clock.stamp("run_started")
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
    model, coefs, preds, rows, contexts, notes, art = analyze(
        games, feat, qbr, wk, cutoff, live, settings, refresh=not args.no_refresh, clock=clock)
    for n in notes:
        print(f"  ! {n}")
    if rows.empty:
        raise SystemExit("No odds available for these games yet.")
    save_sources()

    best = rows[rows["is_best_side"]]
    print(f"\n=== {season} Week {week} ===  model {track.model_version()}  |  "
          f"flag: EV > {settings.min_edge:.1%} and robust to a 0.5-pt error; stakes "
          f"{settings.kelly_fraction:g}x Kelly capped at {settings.max_stake_units:g}u; gap rule {settings.gap_points:g} pts\n")
    print(report.terminal_summary(best, preds))
    n_bet = (best["decision"] != "NO BET").sum()
    print(f"\n{n_bet} bet(s) of {len(best)} markets. Full reasoning per game in the report.")

    val = ROOT / "reports" / "validation_summary.md"
    md = report.header(season, week, SOURCES, track.model_version(), model, settings,
                       val.read_text() if val.exists() else None, art.get("timeline", {}), clock.simulated)
    for g in preds.itertuples(index=False):
        md += "\n" + report.game_section(g, contexts[g.game_id], rows[rows["game_id"] == g.game_id], coefs, model,
                                         settings)
    out = ROOT / "reports" / ("dev" if args.now or track.model_version().endswith("-modified") else "") \
        / f"{season}_week{week:02d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(f"Report: {out.relative_to(ROOT)}")
    if live:
        record(rows, preds, art, settings, clock, args, games)


def cmd_recommend(args):
    clock = runtime.Clock(parse_utc(args.now) if args.now else None)
    clock.stamp("run_started")
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
    model, coefs, preds, rows, contexts, notes, art = analyze(
        games, feat, qbr, day, start, live, settings, refresh=not args.no_refresh, clock=clock)
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
          f"flag: EV > {settings.min_edge:.1%}; stakes {settings.kelly_fraction:g}x Kelly capped at "
          f"{settings.max_stake_units:g}u; gap rule {settings.gap_points:g} pts\n")
    print(table(out, head, align))
    b = show[show["decision"] != "NO BET"]
    print(f"\n{len(b)} bet(s), {b['stake_units'].sum():.2f}u total stake ({settings.kelly_fraction:g}x Kelly, "
          f"max {settings.max_stake_units:g}u per bet). Decision reasons: `predict` report or the saved CSV.")
    if graded and rec:
        print(f"Result: {rec.count('W')}-{rec.count('L')}-{rec.count('P')}, {total:+.2f}u "
              "(one day is noise; see `validate` for the full record)")
    path = ROOT / "picks" / f"recommend_{start.date()}.csv"
    path.parent.mkdir(exist_ok=True)
    show.to_csv(path, index=False)
    print(f"Saved {path.relative_to(ROOT)}")
    if live:
        record(rows, preds, art, settings, clock, args, games)


def cmd_validate(args):
    _, feat, _, _, _ = build(refresh=not args.no_refresh)
    last = int(feat.loc[feat["result"].notna(), "season"].max())
    seasons = list(range(args.start, last + 1))
    settings = load_settings(min_edge=args.min_edge)
    min_edge = settings.min_edge
    print(f"Settings: {settings.describe()}")
    print(f"Validating {seasons[0]}-{seasons[-1]} (each season predicted by a model fit on earlier ones) ...")
    preds, bets = validate.run(feat, seasons, settings)
    print("Replay at standard retail juice (4.76% overround, as -110/-110) ...")
    _, bets_retail = validate.run(feat, seasons, settings, bet_overround=0.0476)
    print("Ablation: same model without the QB-change feature ...")
    no_qb = [f for f in validate.FEATURES if f != "f_qb"]
    preds_nq, _ = validate.run(feat, seasons, settings, features=no_qb, with_bets=False)

    probs = validate.prob_table(preds)
    # ablation scored on exactly the same rows as the full model
    abl = preds.drop(columns=["p_model", "p_cover", "p_over"]).merge(
        preds_nq[["game_id", "p_model", "p_cover", "p_over"]], on="game_id")
    probs_nq = validate.prob_table(abl, n_boot=200)
    probs_nq = probs_nq[probs_nq["predictor"] == "model"].assign(predictor="model without QB feature")
    probs = pd.concat([probs, probs_nq]).sort_values(["target", "predictor"]).reset_index(drop=True)
    seas = validate.season_diffs(preds)
    cal = validate.calibration(preds)
    mae = validate.mae_table(preds)
    mae_nq = validate.mae_table(preds_nq)[["season", "MAE model own line"]].rename(
        columns={"MAE model own line": "MAE own line w/o QB"})
    mae = mae.merge(mae_nq, on="season")
    bt = validate.bet_table(bets)
    bs = validate.bet_seasons(bets).merge(validate.overround_by_season(feat), on="season", how="right").fillna(
        {"bets": 0, "flat_units": 0.0})
    bt_r = validate.bet_table(bets_retail)
    bs_r = validate.bet_seasons(bets_retail).rename(columns={"bets": "bets @4.76%", "flat_units": "units @4.76%",
                                                             "flat_roi": "roi @4.76%"})
    bs = bs.merge(bs_r, on="season", how="left")
    bs = bs[bs["season"] >= seasons[0]]

    pd.set_option("display.max_colwidth", 40)
    print("\nProbability quality on identical rows (lower is better; ECE = calibration error):")
    print(probs.round(5).to_string(index=False))
    print("\nModel minus market Brier by season (negative = model better):")
    print(seas.round(5).to_string(index=False))
    print("\nCalibration of win probabilities (same games):")
    print(cal.round(3).to_string(index=False))
    print("\nMargin error by season (points):")
    print(mae.round(2).to_string(index=False))
    print(f"\nHistorical replay at CLOSING consensus prices (EV > {min_edge:.1%}, robust to 0.5 pt, "
          f"{settings.gap_points:g}-pt gap rule;"
          " live-only rules NOT applied), one side per market:")
    print(bt.round(4).to_string(index=False))
    print("\nSame rules, every bet priced at standard retail juice (4.76% overround):")
    print(bt_r.round(4).to_string(index=False))
    print("\nBy season, with the median overround of the recorded consensus prices:")
    print(bs.round(4).to_string(index=False))

    fmt_b = {"flat ROI": "{:+.1%}", "flat units": "{:+.1f}", "flat max drawdown": "{:.1f}",
             "qtr-Kelly ROI": "{:+.1%}", "qtr-Kelly max drawdown": "{:.1f}"}
    md = [f"# Validation {seasons[0]}-{seasons[-1]}", "", validate.__doc__.strip(), "",
          f"Settings used: `{settings.describe()}`", "",
          "## Probability quality (identical rows for every predictor)", "",
          "`model minus market (Brier)`: negative means the model beat the closing market; the CI resamples "
          "whole games.", "", validate.to_markdown(probs), "",
          "## Model minus market Brier, by season", "", validate.to_markdown(seas), "",
          "## Calibration of win probabilities", "", validate.to_markdown(cal), "",
          "## Margin error by season", "", validate.to_markdown(mae), "",
          f"## Historical replay at closing consensus prices (staking: flat 1u, and {settings.kelly_fraction:g}x "
          f"Kelly capped at {settings.max_stake_units:g}u)", "",
          f"Rules applied: EV above {settings.min_edge:.1%}, robustness to a 0.5-pt error, "
          f"{settings.gap_points:g}-pt gap. Not applied: quote "
          "freshness, multi-book reference, starter availability, weather. CIs resample whole games, so a "
          "spread, moneyline and total on the same game are not treated as independent.", "",
          validate.to_markdown(bt, fmt_b), "",
          "### Same rules at standard retail juice (4.76% overround)", "",
          "The recorded consensus prices carry about 2.4% overround through 2022 and about 4.7% from 2023 "
          "(see the season table). Most bettors pay the latter. Here every consensus pair is re-priced "
          "proportionally at 4.76% (the -110/-110 margin), keeping its no-vig probabilities, and the decision "
          "rules are re-applied at those prices.", "",
          validate.to_markdown(bt_r, fmt_b), "", "### By season", "",
          validate.to_markdown(bs, {"flat_units": "{:+.1f}", "flat_roi": "{:+.1%}", "units @4.76%": "{:+.1f}",
                                    "roi @4.76%": "{:+.1%}", "spread_or": "{:.2%}", "ml_or": "{:.2%}",
                                    "total_or": "{:.2%}"})]
    out = ROOT / "reports" / "validation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    w = probs[probs["target"] == "winner"].set_index("predictor")
    allb = bt[bt["market"] == "all"].iloc[0]
    retail = bt_r[bt_r["market"] == "all"].iloc[0] if len(bt_r) and (bt_r["market"] == "all").any() else \
        {"bets": 0, "flat ROI": float("nan"), "flat ROI 95% (game-clustered)": "n/a"}
    summary = (f"Validation {seasons[0]}-{seasons[-1]} (out of sample, closing information set, "
               f"{int(w.loc['model', 'n'])} games scored identically): winner Brier model "
               f"{w.loc['model', 'brier']:.4f} vs closing market {w.loc['market no-vig (closing)', 'brier']:.4f}, "
               f"difference {w.loc['model minus market (Brier)', 'brier']:+.5f} "
               f"(95% CI {w.loc['model minus market (Brier)', 'ci95']}). Historical replay at closing consensus "
               f"prices (live-only rules not applied): {allb['bets']} bets on {allb['games']} games, flat ROI "
               f"{allb['flat ROI']:+.1%} ({allb['flat ROI 95% (game-clustered)']}, game-clustered), max drawdown "
               f"{allb['flat max drawdown']:.1f}u; at standard 4.76% juice: {retail['bets']} bets, flat ROI "
               f"{retail['flat ROI']:+.1%} ({retail['flat ROI 95% (game-clustered)']}). Full tables: reports/validation.md")
    (ROOT / "reports" / "validation_summary.md").write_text(summary + "\n")
    print(f"\n{summary}")


def cmd_ratings(args):
    _, _, table_, _, _ = build(refresh=not args.no_refresh)
    print("points_rating = points better than an average team on a neutral field\n")
    print(table_.round(3).to_string(index=False))


def cmd_collect(args):
    """Snapshot market and injury data for upcoming games, without the model.

    Run it on a schedule (e.g. hourly, and around kickoff - 60 minutes) to build the
    timestamped history that horizon-matched evaluation needs. Nothing is backfilled."""
    settings = load_settings(max_odds_age_minutes=args.max_odds_age)
    clock = runtime.Clock()
    now = clock.stamp("run_started")
    games = load_games(refresh=not args.no_refresh)
    games = games.assign(kickoff_utc=[_kick(d, t) for d, t in zip(games["gameday"], games["gametime"])])
    window = games[(games["kickoff_utc"] > now) & (games["kickoff_utc"] <= now + pd.Timedelta(days=args.days))]
    if window.empty:
        raise SystemExit(f"No games kicking off in the next {args.days} days.")
    kick = inputs.kickoff_map(window)
    books, consensus, notes = gather_offers(window, kick, settings)
    collected = clock.stamp("odds_collected")
    valid, rejected = validate_offers(books, kick, collected, settings)
    season = int(window["season"].iloc[0])
    inj = load_injuries(season, refresh=not args.no_refresh)
    clock.stamp("injuries_retrieved")
    weather, rej_wx = inputs.load_weather(window, settings, clock.stamp("inputs_read"))
    save_sources()
    run_utc = fmt(collected)
    run_id = "collect_" + run_utc.replace(":", "").replace("-", "")
    sid, d = store.save_snapshot(
        run_id,
        files={"sources.json": ROOT / "data" / "sources.json", "injuries.parquet": ROOT / "data" / f"injuries_{season}.parquet",
               "odds_manual.csv": ROOT / "odds_manual.csv", "weather_manual.csv": ROOT / "weather_manual.csv"},
        frames={"schedule_rows.csv": window, "offers_raw.csv": books, "offers_valid.csv": valid,
                "offers_rejected.csv": rejected, "consensus.csv": consensus, "inputs_rejected.csv": rej_wx,
                "weather_accepted.csv": pd.DataFrame(list(weather.values()))},
        meta={"settings": vars(settings), "notes": notes, **clock.as_dict()})
    store.append(store.COLLECTIONS, "collection", [{
        "run_id": run_id, "snapshot_id": sid, **clock.as_dict(), "games": int(len(window)),
        "valid_quotes": int(len(valid)), "rejected_quotes": int(len(rejected)),
        "books": int(valid["book"].nunique()) if len(valid) else 0, "injury_rows": int(len(inj))}],
        )
    n_wx = store.archive_weather(weather, run_id) if weather else 0
    for x in rej_wx.itertuples(index=False):
        notes.append(f"{x.file} row rejected ({x.game_id}): {x.reason}")
    for n in notes:
        print(f"  ! {n}")
    print(f"Weather: {len(weather)} valid forecast(s), {n_wx} newly archived, {len(rej_wx)} rejected")
    print(f"Collected {len(window)} games, {len(valid)} valid / {len(rejected)} rejected quotes, "
          f"{len(inj)} injury rows -> {d.relative_to(ROOT)}")


def cmd_experiment(args):
    from nflmodel import availability, experiment
    version = track.model_version()
    if version.endswith("-modified") or version == "unknown":
        raise SystemExit("Experiments are recorded against a commit; commit your changes first.")
    _, feat, _, _, _ = build(refresh=not args.no_refresh)
    last = int(feat.loc[feat["result"].notna(), "season"].max())
    if args.group in ("G3", "G4"):
        print("Building snap-count availability / OL continuity table ...")
        tbl = availability.team_game_table(list(range(availability.FIRST_SEASON, last + 1)))
        feat = availability.add_features(feat, tbl)
    print(f"Running {args.group} on {'HOLDOUT' if args.holdout else 'development'} seasons ...")
    rec = experiment.run(feat, args.group, args.holdout, last, version)
    print(f"\n{rec['group']} ({rec['period']}, seasons {rec['seasons'][0]}-{rec['seasons'][-1]}): {rec['verdict']}")
    for t in ("winner", "home covers", "over hits"):
        x = rec["results"][t]
        print(f"  {t:12s} n={x['n']}  log loss {x['logloss_base']:.5f} -> {x['logloss_exp']:.5f}  "
              f"diff {x['d_logloss']:+.5f} (95% CI {x['ci_lo']:+.5f} to {x['ci_hi']:+.5f})  "
              f"improved {x['seasons_improved']}/{x['seasons']} seasons")
    m = rec["results"]["own_line_mae"]
    print(f"  own-line MAE margin {m['margin_base']:.3f} -> {m['margin_exp']:.3f}, "
          f"total {m['total_base']:.3f} -> {m['total_exp']:.3f}")


def cmd_check_live(args):
    """Diagnose the live bookmaker feed. Prints statistics only; writes nothing."""
    from nflmodel import livecheck
    from nflmodel.odds import fetch_odds_api
    settings = load_settings(max_odds_age_minutes=args.max_odds_age)
    ok, desc = livecheck.key_status()
    print(desc)
    if not ok:
        print("\nLive feed: NOT AVAILABLE. No quotes were fetched or invented; nflverse consensus lines are not a "
              "live feed.\n")
        print(livecheck.SETUP)
        return
    clock = runtime.Clock()
    clock.stamp("run_started")
    games = load_games(refresh=not args.no_refresh)
    games = games.assign(kickoff_utc=[_kick(d, t) for d, t in zip(games["gameday"], games["gametime"])])
    now = clock.now()
    window = games[(games["kickoff_utc"] > now) & (games["kickoff_utc"] <= now + pd.Timedelta(days=args.days))]
    if window.empty:
        raise SystemExit(f"No games kicking off in the next {args.days} days to check against.")
    kick = inputs.kickoff_map(window)
    print(f"Requesting odds once for {len(window)} games in the next {args.days:g} days ...")
    try:
        books = fetch_odds_api(window, kick, os.environ["ODDS_API_KEY"])
    except Exception as exc:  # noqa: BLE001 - report any transport/API failure without the key
        print(f"\nLive feed: FAILED. {livecheck.describe_failure(exc)}")
        return
    collected = clock.stamp("odds_collected")
    valid, rejected = validate_offers(books, kick, collected, settings)
    summary = livecheck.summarize(valid, rejected, kick, collected, settings.min_reference_books)
    print("\nLive feed: " + ("WORKING" if summary["valid_quotes"] else "REACHABLE BUT NO VALID QUOTES"))
    print(livecheck.format_summary(summary, settings.min_reference_books))


def cmd_templates(args):
    _, feat, _, current, _ = build(refresh=not args.no_refresh)
    season = args.season or current
    upcoming = feat[(feat["season"] == season) & feat["result"].isna()]
    week = args.week or int(upcoming["week"].min())
    day = feat[(feat["season"] == season) & (feat["week"] == week)]
    for p in inputs.write_templates(day, ROOT / "templates"):
        print(f"wrote {p.relative_to(ROOT)}")
    print("Copy the rows you need into odds_manual.csv / qb_overrides.csv / weather_manual.csv.")


def _slate_from_args(args, games):
    """Explicit evaluation slate from --season/--week(s) or --from/--to."""
    if args.date_from or args.date_to:
        season = args.season
        desc = f"{args.date_from or '...'} to {args.date_to or '...'}" + (f", season {season}" if season else "")
        return track.define_slate(games, season=season, date_from=args.date_from, date_to=args.date_to), desc
    season = args.season or int(games.loc[games["result"].notna(), "season"].max())
    weeks = None
    if args.weeks:
        a, _, b = args.weeks.partition("-")
        weeks = (int(a), int(b or a))
    desc = f"season {season}" + (f", weeks {weeks[0]}-{weeks[1]}" if weeks else ", all weeks")
    return track.define_slate(games, season=season, weeks=weeks), desc


def cmd_grade(args):
    games = load_games(refresh=not args.no_refresh)
    settings = load_settings()
    for path in (store.FORECASTS, store.LEDGER):
        ok, msg = store.verify(path)
        print(f"{path.relative_to(ROOT)}: {msg}")
        if not ok:
            raise SystemExit("integrity check failed; not grading")
    slate, desc = _slate_from_args(args, games)
    markets = tuple(args.markets.split(","))
    now = parse_utc(args.now) if args.now else now_utc()
    g = track.grade_all(games, settings, slate=slate, now=now, markets=markets)
    track.report(g, settings, f"{desc}; markets {','.join(markets)}")
    for name, df in g.items():
        df.to_csv(ROOT / "logs" / f"graded_{name}.csv", index=False)


def cmd_coverage(args):
    """Coverage of the forecast history against an explicit slate (writes nothing)."""
    games = load_games(refresh=not args.no_refresh)
    settings = load_settings()
    ok, msg = store.verify(store.FORECASTS)
    print(f"{store.FORECASTS.relative_to(ROOT)}: {msg}")
    if not ok:
        raise SystemExit("integrity check failed")
    slate, desc = _slate_from_args(args, games)
    markets = tuple(args.markets.split(","))
    now = parse_utc(args.now) if args.now else now_utc()
    games_cov, mk, _ = track.slate_coverage(slate, store.forecasts_frame(), settings, now, markets)
    track.print_coverage(f"{desc}; markets {','.join(markets)}", games_cov, mk, settings)
    if args.list:
        print(games_cov.to_string(index=False))


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
    from nflmodel import experiment
    for path in (store.FORECASTS, store.LEDGER, store.COLLECTIONS, experiment.LOG, store.WEATHER_ARCHIVE):
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
    sub.add_parser("ratings")
    gr = sub.add_parser("grade", help="forecasts on an explicit slate, recommendations, wagers")
    cv = sub.add_parser("coverage", help="forecast coverage of an explicit slate (writes nothing)")
    cv.add_argument("--list", action="store_true", help="print every game's status")
    for x in (gr, cv):
        x.add_argument("--season", type=int, help="default: latest season with results")
        x.add_argument("--weeks", help="week or range, e.g. 5 or 5-8 (default: all weeks)")
        x.add_argument("--from", dest="date_from", help="first game date YYYY-MM-DD (instead of weeks)")
        x.add_argument("--to", dest="date_to", help="last game date YYYY-MM-DD")
        x.add_argument("--markets", default="spread,ml,total", help="comma list of spread,ml,total")
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
    ex = sub.add_parser("experiment", help="ablate one feature group (docs/EXPERIMENTS.md)")
    ex.add_argument("--group", required=True, choices=["G1", "G2", "G3", "G4"])
    ex.add_argument("--holdout", action="store_true", help="the one-time holdout evaluation (2022+)")
    ck = sub.add_parser("check-live", help="check the live bookmaker feed (never shows the key; writes nothing)")
    ck.add_argument("--days", type=float, default=7, help="games kicking off within this many days")
    co = sub.add_parser("collect", help="snapshot odds + injury data for upcoming games (no model)")
    co.add_argument("--days", type=float, default=7, help="games kicking off within this many days")
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
     "place": cmd_place, "void": cmd_void, "verify-log": cmd_verify_log,
     "collect": cmd_collect, "experiment": cmd_experiment, "check-live": cmd_check_live,
     "coverage": cmd_coverage}[args.cmd](args)


if __name__ == "__main__":
    main()
