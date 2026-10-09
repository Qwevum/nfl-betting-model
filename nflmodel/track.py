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


def issue_time(fc: pd.DataFrame) -> pd.Series:
    """When a forecast was actually available: the later of its recorded time and its
    prediction-completion time. Legacy rows (before the timestamp fix) only carry the
    run START time, so their issue time is unknown (NaT)."""
    rec = pd.to_datetime(fc["recorded_utc"].map(parse_utc), utc=True)
    if "prediction_completed_utc" not in fc:
        return pd.Series(pd.NaT, index=fc.index, dtype="datetime64[ns, UTC]")
    done = pd.to_datetime(fc["prediction_completed_utc"].map(lambda v: parse_utc(v) if isinstance(v, str) else None),
                          utc=True)
    issued = rec.where(rec >= done, done)
    return issued.where(done.notna(), pd.NaT)


def at_horizon(fc: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select, per game, ONE run whose forecasts were issued inside the horizon window.

    Window: [kickoff - horizon - tolerance, kickoff - horizon]. Forecasts issued after
    the cutoff (including runs that completed late) or before the window start (e.g.
    days earlier) are excluded. Among eligible runs the latest is used for every side
    of that game, so sides from different runs are never mixed.

    Returns (selected rows with lead_minutes, per-game coverage table with reasons)."""
    cols = ["game_id", "kickoff_utc", "status", "runs_considered", "selected_run", "lead_minutes", "reason"]
    if fc.empty:
        return fc, pd.DataFrame(columns=cols)
    fc = fc.copy()
    fc["_issued"] = issue_time(fc)
    fc["_kick"] = pd.to_datetime(fc["kickoff_utc"].map(parse_utc), utc=True)
    fc["_run"] = fc["run_id"] if "run_id" in fc else None
    fc["_run"] = fc["_run"].where(fc["_run"].notna(), "legacy@" + fc["recorded_utc"])
    cutoff = fc["_kick"] - pd.Timedelta(minutes=settings.horizon_minutes)
    start = cutoff - pd.Timedelta(minutes=settings.horizon_tolerance_minutes)
    fc["_lead"] = (fc["_kick"] - fc["_issued"]).dt.total_seconds() / 60
    fc["_ok"] = fc["_issued"].notna() & (fc["_issued"] >= start) & (fc["_issued"] <= cutoff)

    picked, cov = [], []
    for gid, g in fc.groupby("game_id"):
        runs = g.groupby("_run").agg(issued=("_issued", "max"), ok=("_ok", "all"), lead=("_lead", "min"))
        ok = runs[runs["ok"]]
        row = {"game_id": gid, "kickoff_utc": g["kickoff_utc"].iloc[0], "runs_considered": len(runs)}
        if len(ok):
            run = ok["issued"].idxmax()
            sel = g[g["_run"] == run]
            picked.append(sel.assign(lead_minutes=sel["_lead"]))
            cov.append({**row, "status": "eligible", "selected_run": run, "lead_minutes": float(ok.loc[run, "lead"]),
                        "reason": ""})
            continue
        if runs["issued"].isna().all():
            why = "only legacy forecasts (pre-fix run-start timestamps; completion time unknown)"
        elif (runs["lead"] < settings.horizon_minutes).all():
            why = f"all forecasts issued after the cutoff (kickoff - {settings.horizon_minutes:g} min)"
        else:
            why = (f"no forecast issued inside the {settings.horizon_tolerance_minutes:g}-min window "
                   f"(closest lead {runs['lead'].dropna().min():.0f} min)" if runs["lead"].notna().any()
                   else "no usable issue time")
        cov.append({**row, "status": "excluded", "selected_run": None, "lead_minutes": np.nan, "reason": why})
    sel = pd.concat(picked, ignore_index=True) if picked else fc.iloc[0:0].assign(lead_minutes=np.nan)
    sel = sel.drop(columns=[c for c in sel.columns if c.startswith("_")])
    return sel, pd.DataFrame(cov, columns=cols)


# ---------------------------------------------------------------- explicit evaluation slate

MARKETS = ("spread", "ml", "total")
STATUS_ORDER = ["eligible", "forecasts exist, none qualify", "no forecast recorded", "window not closed",
                "unknown kickoff"]


def define_slate(games: pd.DataFrame, season: int | None = None, weeks: tuple[int, int] | None = None,
                 date_from=None, date_to=None) -> pd.DataFrame:
    """Every scheduled game in the requested slate (season + week range, or a date range).

    Returns game_id, season, week, kickoff_utc (NaT if unknown), away/home teams, played."""
    from .timeutil import kickoff_utc
    g = games.copy()
    if season is not None:
        g = g[g["season"] == season]
    if weeks is not None:
        g = g[(g["week"] >= weeks[0]) & (g["week"] <= weeks[1])]
    if date_from is not None:
        g = g[pd.to_datetime(g["gameday"]) >= pd.Timestamp(date_from)]
    if date_to is not None:
        g = g[pd.to_datetime(g["gameday"]) <= pd.Timestamp(date_to)]

    def kick(d, t):
        try:
            return kickoff_utc(d, t)
        except ValueError:
            return pd.NaT
    out = pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                        "away_team": g["away_team"], "home_team": g["home_team"],
                        "kickoff_utc": [kick(d, t) for d, t in zip(g["gameday"], g["gametime"])],
                        "played": g["result"].notna() if "result" in g else False})
    return out.reset_index(drop=True)


def slate_coverage(slate: pd.DataFrame, fc: pd.DataFrame, settings, now: pd.Timestamp,
                   markets=MARKETS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-game status, per-market coverage among eligible games, selected forecast rows).

    Every slate game gets exactly one status. A game whose evaluation window
    [kickoff - horizon - tolerance, kickoff - horizon] has not closed yet is never counted
    as missed. Forecasts are never created or backfilled here."""
    fc = fc if fc is not None else pd.DataFrame()
    if len(fc):
        fc = fc[fc["game_id"].isin(slate["game_id"]) & fc["market"].isin(markets)]
    sel, cov = at_horizon(fc, settings) if len(fc) else (pd.DataFrame(), pd.DataFrame())
    elig = set(cov.loc[cov["status"] == "eligible", "game_id"]) if len(cov) else set()
    why = dict(zip(cov["game_id"], cov["reason"])) if len(cov) else {}
    lead = dict(zip(cov["game_id"], cov["lead_minutes"])) if len(cov) else {}
    has_fc = set(fc["game_id"]) if len(fc) else set()
    rows = []
    for g in slate.itertuples(index=False):
        k = g.kickoff_utc
        closes = None if pd.isna(k) else k - pd.Timedelta(minutes=settings.horizon_minutes)
        if pd.isna(k):
            status, reason = "unknown kickoff", "kickoff time not listed in the schedule"
        elif g.game_id in elig:
            status, reason = "eligible", ""
        elif now < closes:
            status, reason = "window not closed", f"evaluation window closes {fmt_ts(closes)}"
        elif g.game_id in has_fc:
            status, reason = "forecasts exist, none qualify", why.get(g.game_id, "")
        else:
            status, reason = "no forecast recorded", ""
        rows.append({"game_id": g.game_id, "week": g.week, "kickoff_utc": None if pd.isna(k) else fmt_ts(k),
                     "status": status, "reason": reason, "played": bool(g.played),
                     "lead_minutes": lead.get(g.game_id) if status == "eligible" else None})
    games_cov = pd.DataFrame(rows, columns=["game_id", "week", "kickoff_utc", "status", "reason", "played",
                                            "lead_minutes"])
    mk = []
    if len(sel):
        sel = sel[sel["game_id"].isin(elig)]
    for m in markets:
        s_m = sel[sel["market"] == m] if len(sel) else pd.DataFrame()
        with_fc = set(s_m["game_id"]) if len(s_m) else set()
        with_ref = set(s_m.dropna(subset=["market_prob"])["game_id"]) if len(s_m) else set()
        mk.append({"market": m, "eligible_games": len(elig), "with_forecast": len(with_fc),
                   "with_market_reference": len(with_ref), "missing": len(elig - with_fc),
                   "missing_games": ",".join(sorted(elig - with_fc))})
    return games_cov, pd.DataFrame(mk), sel


def fmt_ts(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def coverage_summary(games_cov: pd.DataFrame) -> pd.DataFrame:
    """Counts and percentages. Denominator: games whose window has closed (unknown kickoffs
    and still-open windows are listed separately and excluded from the denominator)."""
    closed = games_cov[games_cov["status"].isin(STATUS_ORDER[:3])]
    denom = len(closed)
    rows = []
    for st in STATUS_ORDER:
        n = int((games_cov["status"] == st).sum())
        pct = (n / denom) if st in STATUS_ORDER[:3] and denom else None
        rows.append({"status": st, "games": n, "pct_of_closed_windows": pct})
    return pd.DataFrame(rows)


def print_coverage(slate_desc: str, games_cov: pd.DataFrame, mk: pd.DataFrame, settings) -> None:
    lo = settings.horizon_minutes + settings.horizon_tolerance_minutes
    summ = coverage_summary(games_cov)
    closed = int(summ.loc[summ["status"].isin(STATUS_ORDER[:3]), "games"].sum())
    print(f"\nCOVERAGE for slate: {slate_desc} ({len(games_cov)} scheduled games)")
    print(f"  horizon window: {lo:g}-{settings.horizon_minutes:g} min before kickoff; "
          f"denominator = {closed} game(s) whose window has closed")
    for r in summ.itertuples(index=False):
        pct = "" if r.pct_of_closed_windows is None or pd.isna(r.pct_of_closed_windows) \
            else f" ({r.pct_of_closed_windows:.1%})"
        note = "" if r.status in STATUS_ORDER[:3] else "  [not in denominator]"
        print(f"  {r.status:30s} {r.games:4d}{pct}{note}")
    ex = games_cov[games_cov["status"] == "forecasts exist, none qualify"]
    for why, n in ex["reason"].value_counts().items():
        print(f"    - {n} game(s): {why}")
    el = games_cov[games_cov["status"] == "eligible"]
    if len(el):
        lead = el["lead_minutes"].astype(float)
        print(f"  lead time of selected runs: min {lead.min():.0f}, median {lead.median():.0f}, "
              f"max {lead.max():.0f} min")
    if len(mk) and mk["eligible_games"].iloc[0]:
        print("  market coverage within eligible games:")
        for r in mk.itertuples(index=False):
            print(f"    {r.market:6s} forecast {r.with_forecast}/{r.eligible_games}, with market reference "
                  f"{r.with_market_reference}/{r.eligible_games}"
                  + (f", missing: {r.missing_games}" if r.missing else ""))


def grade_all(games: pd.DataFrame, settings, slate: pd.DataFrame | None = None, now: pd.Timestamp | None = None,
              markets=MARKETS, forecasts_path: Path = store.FORECASTS, ledger_path: Path = store.LEDGER) -> dict:
    """Forecast quality on an explicit slate, recommendations, and actual wagers (separately).

    slate: from define_slate(); when None, the slate is every game that appears in the
    forecast history (games with no forecast are then invisible, so pass a slate)."""
    from .timeutil import now_utc
    out = {}
    fc = store.forecasts_frame(forecasts_path)
    if slate is None:   # fallback: games that appear in the history, kickoffs as recorded there
        k = fc[["game_id", "kickoff_utc"]].drop_duplicates("game_id") if len(fc) else \
            pd.DataFrame(columns=["game_id", "kickoff_utc"])
        played = set(games.loc[games["result"].notna(), "game_id"]) if "result" in games else set()
        slate = pd.DataFrame({"game_id": k["game_id"], "week": None,
                              "kickoff_utc": [parse_utc(x) if isinstance(x, str) else pd.NaT for x in k["kickoff_utc"]],
                              "played": k["game_id"].isin(played)})
    games_cov, mk, sel = slate_coverage(slate, fc, settings, now or now_utc(), markets)
    out["coverage"], out["market_coverage"] = games_cov, mk
    if len(sel):
        h = _results(sel, games)
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


def score_table(fc: pd.DataFrame) -> pd.DataFrame:
    """Model vs market on identical graded rows, by market and model version."""
    done = fc[fc["result"].isin(["W", "L"])].dropna(subset=["model_prob", "market_prob"])
    done = done.assign(model_version=done["model_version"].fillna("unknown") if "model_version" in done
                       else "unknown")
    rows = []
    if done.empty:
        return pd.DataFrame(rows)
    for (market, version), g in done.groupby(["market", "model_version"]):
        y = (g["result"] == "W").to_numpy(float)
        bm, lm = metrics.brier_logloss(g["model_prob"], y)
        bk, lk = metrics.brier_logloss(g["market_prob"], y)
        rows.append({"market": market, "model_version": version, "games": g["game_id"].nunique(), "sides": len(g),
                     "model_brier": bm, "market_brier": bk, "model_logloss": lm, "market_logloss": lk})
    return pd.DataFrame(rows)


def report(g: dict, settings, slate_desc: str = "games in forecast history") -> None:
    fc = g.get("forecasts")
    print("\n1) FORECAST QUALITY (one run per game inside the horizon window)")
    if "coverage" in g:
        print_coverage(slate_desc, g["coverage"], g.get("market_coverage", pd.DataFrame()), settings)
    if fc is not None and len(fc):
        t = score_table(fc)
        if t.empty:
            print("  no graded eligible rows with both model and market probabilities yet")
        else:
            print("  model vs market on identical graded rows:")
            print("  " + t.round(4).to_string(index=False).replace("\n", "\n  "))
            if t["games"].sum() < 50:
                print("  ! fewer than 50 games: far too few to distinguish model from market")
    print("\n2) RECOMMENDATIONS (hypothetical, flat 1u; these are NOT wagers), from the selected runs")
    if fc is None or fc.empty:
        print("   none")
    else:
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
        print_arms(fc, settings)
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


def arm_table(fc: pd.DataFrame, settings) -> pd.DataFrame:
    """Prospective arms on the same eligible runs and rows (hypothetical flat 1u, NOT wagers):
      price shopping (market only): raw ex-book market EV above the threshold, at a timestamped
                                    quote with a live reference (the same execution requirements)
      model-assisted:               decision BET
    Records written before ev_raw_market existed are not counted."""
    if fc is None or fc.empty or "ev_raw_market" not in fc:
        return pd.DataFrame()
    f = fc.dropna(subset=["ev_raw_market"])
    live = f["reference_live"].fillna(False).astype(bool) & f["price_source"].isin(["odds_api", "manual"])
    arms = {"price shopping (market only)": f[live & (f["ev_raw_market"] > settings.min_edge)],
            "model-assisted (BET)": f[f["decision"] == "BET"]}
    rows = []
    for name, a in arms.items():
        done = a[a["result"].isin(["W", "L", "P"])]
        rows.append({"arm": name, "sides flagged": len(a), "graded": len(done),
                     "W-L-P": f"{(done['result'] == 'W').sum()}-{(done['result'] == 'L').sum()}-"
                              f"{(done['result'] == 'P').sum()}",
                     "flat units": float(done["flat_units"].sum()) if len(done) else 0.0,
                     "avg CLV": float(done["clv"].mean()) if len(done) else np.nan})
    return pd.DataFrame(rows)


def print_arms(fc: pd.DataFrame, settings) -> None:
    t = arm_table(fc, settings)
    print("\n   Arms on identical eligible rows (hypothetical flat 1u, NOT wagers):")
    if t.empty:
        print("   no forecasts with the raw-market EV field yet (recorded from model version 2026-10 on)")
        return
    print("   " + t.round(3).to_string(index=False).replace("\n", "\n   "))
    print("   ! small samples are noise; neither more flags nor a higher ROI alone shows an improvement")


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
