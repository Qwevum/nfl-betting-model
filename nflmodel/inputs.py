"""User-supplied inputs, all keyed by game_id + kickoff_utc (never by team names).

  qb_overrides.csv    game_id,kickoff_utc,team,qb_name,source,confirmed_utc
  weather_manual.csv  game_id,kickoff_utc,wind_mph,temp_f,source,forecast_utc
  odds_manual.csv     see odds.MANUAL_COLS

`python run.py templates` writes rows with the right game_id/kickoff_utc for a slate.
Rows that don't match a game in the slate, or whose kickoff doesn't match the
schedule (e.g. a rescheduled game), are rejected and reported, not applied.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .odds import check_game_keys
from .timeutil import fmt, parse_utc

ROOT = Path(__file__).resolve().parent.parent

QB_COLS = ["game_id", "kickoff_utc", "team", "qb_name", "source", "confirmed_utc"]
WEATHER_COLS = ["game_id", "kickoff_utc", "wind_mph", "temp_f", "source", "forecast_utc"]


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, comment="#", dtype=str)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; run `python run.py templates`")
    return df


def kickoff_map(day: pd.DataFrame) -> dict:
    return dict(zip(day["game_id"], day["kickoff_utc"]))


def load_qb_overrides(day: pd.DataFrame, tol_minutes: float, path: Path | None = None):
    """{(game_id, team): row} and a list of rejection notes."""
    df = _read(path or ROOT / "qb_overrides.csv", QB_COLS)
    ok, bad = check_game_keys(df, kickoff_map(day), tol_minutes)
    notes = [f"qb_overrides.csv row rejected ({r.game_id}): {r.reject_reason}" for r in bad.itertuples(index=False)]
    teams = {g.game_id: {g.home_team, g.away_team} for g in day.itertuples(index=False)}
    out = {}
    for r in ok.itertuples(index=False):
        if r.team not in teams[r.game_id]:
            notes.append(f"qb_overrides.csv row rejected ({r.game_id}): team {r.team} is not in this game")
            continue
        try:
            parse_utc(r.confirmed_utc)
        except ValueError as exc:
            notes.append(f"qb_overrides.csv row rejected ({r.game_id}, {r.team}): confirmed_utc {exc}")
            continue
        out[(r.game_id, r.team)] = r._asdict()
    return out, notes


def load_weather(day: pd.DataFrame, tol_minutes: float, path: Path | None = None):
    """{game_id: row} and rejection notes."""
    df = _read(path or ROOT / "weather_manual.csv", WEATHER_COLS)
    ok, bad = check_game_keys(df, kickoff_map(day), tol_minutes)
    notes = [f"weather_manual.csv row rejected ({r.game_id}): {r.reject_reason}" for r in bad.itertuples(index=False)]
    out = {}
    for r in ok.itertuples(index=False):
        try:
            parse_utc(r.forecast_utc)
            float(r.wind_mph)
        except (ValueError, TypeError) as exc:
            notes.append(f"weather_manual.csv row rejected ({r.game_id}): {exc}")
            continue
        out[r.game_id] = r._asdict()
    return out, notes


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


def apply_weather(rows: pd.DataFrame, weather: dict) -> pd.DataFrame:
    if not weather:
        return rows
    rows = rows.copy()
    for gid, r in weather.items():
        rows.loc[rows["game_id"] == gid, "t_wind"] = float(r["wind_mph"])
    return rows


def write_templates(day: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Template rows (game_id, kickoff_utc, teams) for each input file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame({"game_id": day["game_id"], "kickoff_utc": [fmt(k) for k in day["kickoff_utc"]],
                         "away": day["away_team"], "home": day["home_team"]})
    paths = []
    odds = base.assign(book="", market="", side="", point="", price="", odds_time_utc="")
    qb = base.assign(team="", qb_name="", source="", confirmed_utc="")
    wx = base.assign(wind_mph="", temp_f="", source="", forecast_utc="")
    for name, df in (("odds_manual", odds), ("qb_overrides", qb), ("weather_manual", wx)):
        p = out_dir / f"{name}_template.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
