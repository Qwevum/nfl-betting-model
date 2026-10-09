"""User-supplied inputs, all keyed by game_id + kickoff_utc (never by team names).

  qb_overrides.csv    game_id,kickoff_utc,team,qb_name,source,confirmed_utc
  weather_manual.csv  game_id,kickoff_utc,wind_mph,temp_f,source,forecast_utc,retrieved_utc,valid_for_utc
  odds_manual.csv     see odds.MANUAL_COLS

`python run.py templates` writes rows with the right game_id/kickoff_utc for a slate.

Every row is validated against an explicit DECISION TIME (the moment the run reads
its inputs) and the game's kickoff before anything is applied. A row is rejected,
with the reason kept for the report and the snapshot, when:
  * its game_id is not in the slate or its kickoff_utc doesn't match the schedule
  * a required field (source, qb_name, team) is blank
  * a timestamp is missing, naive or unparseable
  * it was not yet available at the decision time (timestamp after decision + skew)
  * it is from or after kickoff
  * it is older than its freshness limit (qb_confirm_max_age_minutes,
    weather_max_age_minutes)
  * weather values are not finite numbers in range (wind >= 0, plausible temperature)
  * a forecast's valid-for time is not within weather_valid_window_minutes of kickoff
Rejected rows are never applied, so they cannot clear a starter-availability
condition or unlock an outdoor total.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .odds import check_game_keys
from .timeutil import fmt, parse_utc

ROOT = Path(__file__).resolve().parent.parent

QB_COLS = ["game_id", "kickoff_utc", "team", "qb_name", "source", "confirmed_utc"]
WEATHER_COLS = ["game_id", "kickoff_utc", "wind_mph", "temp_f", "source", "forecast_utc", "retrieved_utc",
                "valid_for_utc"]
WIND_MAX_MPH = 100.0
TEMP_RANGE_F = (-60.0, 130.0)
REJECT_COLS = ["file", "game_id", "team", "reason"]


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; run `python run.py templates`")
    return df.replace({"": None})


def kickoff_map(day: pd.DataFrame) -> dict:
    return dict(zip(day["game_id"], day["kickoff_utc"]))


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or not str(v).strip()


def _check_time(value, label: str, decision: pd.Timestamp, kickoff: pd.Timestamp, skew: pd.Timedelta,
                max_age: pd.Timedelta | None) -> tuple[pd.Timestamp | None, str]:
    """(timestamp, '') if usable at the decision time, else (None, reason)."""
    try:
        t = parse_utc(value)
    except ValueError as exc:
        return None, f"{label}: {exc}"
    if t > decision + skew:
        return None, f"{label} {fmt(t)} is after the decision time {fmt(decision)} (not yet available)"
    if t >= kickoff:
        return None, f"{label} {fmt(t)} is at/after kickoff {fmt(kickoff)}"
    if max_age is not None and decision - t > max_age:
        return None, f"{label} {fmt(t)} is stale: older than {max_age.total_seconds() / 60:g} min at {fmt(decision)}"
    return t, ""


def _reject(rows: list, file: str, r, reason: str, team=None):
    rows.append({"file": file, "game_id": getattr(r, "game_id", None), "team": team, "reason": reason})


def load_qb_overrides(day: pd.DataFrame, settings, decision_utc: pd.Timestamp, path: Path | None = None):
    """Validated starter confirmations: ({(game_id, team): row}, rejected DataFrame)."""
    path = path or ROOT / "qb_overrides.csv"
    df = _read(path, QB_COLS)
    kick = kickoff_map(day)
    ok, bad = check_game_keys(df, kick, settings.kickoff_match_minutes)
    rejected = [{"file": path.name, "game_id": r.game_id, "team": r.team, "reason": r.reject_reason}
                for r in bad.itertuples(index=False)]
    teams = {g.game_id: {g.home_team, g.away_team} for g in day.itertuples(index=False)}
    skew = pd.Timedelta(minutes=settings.clock_skew_minutes)
    max_age = pd.Timedelta(minutes=settings.qb_confirm_max_age_minutes)
    out = {}
    for r in ok.itertuples(index=False):
        if _blank(r.team) or r.team not in teams[r.game_id]:
            _reject(rejected, path.name, r, f"team {r.team!r} is not in this game", r.team)
            continue
        if _blank(r.qb_name):
            _reject(rejected, path.name, r, "qb_name is blank", r.team)
            continue
        if _blank(r.source):
            _reject(rejected, path.name, r, "source is blank (say where the confirmation came from)", r.team)
            continue
        t, why = _check_time(r.confirmed_utc, "confirmed_utc", decision_utc, kick[r.game_id], skew, max_age)
        if why:
            _reject(rejected, path.name, r, why, r.team)
            continue
        if (r.game_id, r.team) in out:
            _reject(rejected, path.name, r, "duplicate override for this game and team; the first row is used", r.team)
            continue
        out[(r.game_id, r.team)] = {**r._asdict(), "kickoff_utc": fmt(kick[r.game_id]), "confirmed_utc": fmt(t)}
    return out, pd.DataFrame(rejected, columns=REJECT_COLS)


def load_weather(day: pd.DataFrame, settings, decision_utc: pd.Timestamp, path: Path | None = None):
    """Validated forecasts: ({game_id: row}, rejected DataFrame).

    A forecast only unlocks outdoor totals; wind is NOT a model feature."""
    path = path or ROOT / "weather_manual.csv"
    df = _read(path, WEATHER_COLS)
    kick = kickoff_map(day)
    ok, bad = check_game_keys(df, kick, settings.kickoff_match_minutes)
    rejected = [{"file": path.name, "game_id": r.game_id, "team": None, "reason": r.reject_reason}
                for r in bad.itertuples(index=False)]
    skew = pd.Timedelta(minutes=settings.clock_skew_minutes)
    max_age = pd.Timedelta(minutes=settings.weather_max_age_minutes)
    window = pd.Timedelta(minutes=settings.weather_valid_window_minutes)
    out = {}
    for r in ok.itertuples(index=False):
        k = kick[r.game_id]
        if _blank(r.source):
            _reject(rejected, path.name, r, "source is blank (name the forecast provider)")
            continue
        try:
            wind, temp = float(r.wind_mph), float(r.temp_f)
        except (TypeError, ValueError):
            _reject(rejected, path.name, r, f"wind_mph/temp_f not numeric ({r.wind_mph!r}, {r.temp_f!r})")
            continue
        if not (math.isfinite(wind) and 0 <= wind <= WIND_MAX_MPH):
            _reject(rejected, path.name, r, f"wind_mph {r.wind_mph!r} must be finite and between 0 and {WIND_MAX_MPH:g}")
            continue
        if not (math.isfinite(temp) and TEMP_RANGE_F[0] <= temp <= TEMP_RANGE_F[1]):
            _reject(rejected, path.name, r, f"temp_f {r.temp_f!r} must be finite and between "
                                            f"{TEMP_RANGE_F[0]:g} and {TEMP_RANGE_F[1]:g}")
            continue
        issued, why = _check_time(r.forecast_utc, "forecast_utc", decision_utc, k, skew, max_age)
        if why:
            _reject(rejected, path.name, r, why)
            continue
        retrieved, why = _check_time(r.retrieved_utc, "retrieved_utc", decision_utc, k, skew, None)
        if why:
            _reject(rejected, path.name, r, why)
            continue
        if retrieved < issued:
            _reject(rejected, path.name, r, f"retrieved_utc {fmt(retrieved)} is before forecast_utc {fmt(issued)}")
            continue
        try:
            valid_for = parse_utc(r.valid_for_utc)
        except ValueError as exc:
            _reject(rejected, path.name, r, f"valid_for_utc: {exc}")
            continue
        if abs(valid_for - k) > window:
            _reject(rejected, path.name, r, f"valid_for_utc {fmt(valid_for)} is more than "
                                            f"{settings.weather_valid_window_minutes:g} min from kickoff {fmt(k)}")
            continue
        if r.game_id in out:
            _reject(rejected, path.name, r, "duplicate forecast for this game; the first row is used")
            continue
        out[r.game_id] = {**r._asdict(), "kickoff_utc": fmt(k), "wind_mph": wind, "temp_f": temp,
                          "forecast_utc": fmt(issued),
                          "retrieved_utc": fmt(retrieved), "valid_for_utc": fmt(valid_for)}
    return out, pd.DataFrame(rejected, columns=REJECT_COLS)


def apply_qb_overrides(rows: pd.DataFrame, overrides: dict, games: pd.DataFrame, qbr) -> pd.DataFrame:
    """Replace the projected starter for the exact game the override names."""
    if not overrides:
        return rows
    ids = {}
    for side in ("home", "away"):
        for n, i in zip(games[f"{side}_qb_name"], games[f"{side}_qb_id"]):
            if isinstance(n, str) and isinstance(i, str):
                ids[n] = i
    rows = rows.copy()
    for (gid, team), r in overrides.items():
        for side in ("home", "away"):
            m = (rows["game_id"] == gid) & (rows[f"{side}_team"] == team)
            if not m.any():
                continue
            qb_id = ids.get(r["qb_name"])
            rating = qbr.rating(qb_id)  # unknown QB -> prior for inexperienced QBs
            base = rows.loc[m, f"{side}_qb_base"].fillna(rating)
            rows.loc[m, f"{side}_qb_id"] = qb_id if qb_id else f"unknown:{r['qb_name']}"
            rows.loc[m, f"{side}_qb_name"] = r["qb_name"]
            rows.loc[m, f"{side}_qb_rating"] = rating
            rows.loc[m, f"{side}_qb_delta"] = rating - base
    rows["f_qb"] = rows["home_qb_delta"].fillna(0) - rows["away_qb_delta"].fillna(0)
    return rows


def write_templates(day: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Template rows (game_id, kickoff_utc, teams) for each input file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame({"game_id": day["game_id"], "kickoff_utc": [fmt(k) for k in day["kickoff_utc"]],
                         "away": day["away_team"], "home": day["home_team"]})
    paths = []
    odds = base.assign(book="", market="", side="", point="", price="", odds_time_utc="")
    qb = base.assign(team="", qb_name="", source="", confirmed_utc="")
    wx = base.assign(wind_mph="", temp_f="", source="", forecast_utc="", retrieved_utc="",
                     valid_for_utc=base["kickoff_utc"])
    for name, df in (("odds_manual", odds), ("qb_overrides", qb), ("weather_manual", wx)):
        p = out_dir / f"{name}_template.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
