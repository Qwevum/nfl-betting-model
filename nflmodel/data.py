"""Download and cache nflverse data, recording where and when each piece came from.

Sources (all free, maintained by the nflverse project, github.com/nflverse):
  * games.csv   - every game since 1999: scores, closing spread/total/moneyline
                  (consensus, no per-book timestamp), rest days, roof, game-time
                  weather (filled in after the game), starting QBs (projected for
                  upcoming games).
  * play-by-play parquet per season: EPA, success, win probability, passer.
  * injuries parquet per season: official NFL injury reports (game status and
                  practice participation) by week.

Play-by-play is reduced to one row per team per game and one row per QB per game,
cached as CSV, so the large raw files are downloaded once per past season.

Every download is recorded in SOURCES (url, retrieved time, server Last-Modified
when the server provides one) and written to data/sources.json.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)

INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.parquet"
)

FIRST_PBP_SEASON = 2010
CACHE_VERSION = "v2"   # bump when the cached per-game tables change shape

SOURCES: dict[str, dict] = {}

# Relocated franchises are mapped to their current abbreviation so ratings carry over.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type",
    "epa", "success", "wp", "qb_dropback", "passer_player_id", "passer_player_name",
]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(url: str, label: str | None = None) -> bytes:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        body = resp.read()
        if label:
            SOURCES[label] = {"url": url.split("apiKey=")[0] + ("apiKey=***" if "apiKey=" in url else ""),
                              "retrieved_utc": utcnow(),
                              "last_modified": resp.headers.get("Last-Modified")}
        return body


def note_cached(label: str, path: Path, url: str) -> None:
    """Record a source read from the local cache: keep the details saved when it was downloaded."""
    if label in SOURCES or not path.exists():
        return
    prev = {}
    saved = CACHE / "sources.json"
    if saved.exists():
        try:
            prev = json.loads(saved.read_text()).get(label, {})
        except ValueError:
            prev = {}
    rec = {**prev, "url": url}
    rec["retrieved_utc"] = (prev.get("retrieved_utc") or dt.datetime.fromtimestamp(
        path.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")).replace(" (cached)", "") + " (cached)"
    rec.setdefault("last_modified", None)
    SOURCES[label] = rec


def save_sources() -> Path:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "sources.json"
    path.write_text(json.dumps(SOURCES, indent=2))
    return path


def fix_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)


def load_games(refresh: bool = True) -> pd.DataFrame:
    """All games (past and scheduled) with closing lines, from nflverse."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "games.csv"
    if refresh or not path.exists():
        path.write_bytes(_fetch(GAMES_URL, "schedule_scores_lines"))
    else:
        note_cached("schedule_scores_lines", path, GAMES_URL)
    g = pd.read_csv(path, low_memory=False)
    g["away_team"] = fix_team(g["away_team"])
    g["home_team"] = fix_team(g["home_team"])
    g["gameday"] = pd.to_datetime(g["gameday"])
    played = g.loc[g["result"].notna(), "gameday"]
    if len(played) and "schedule_scores_lines" in SOURCES:
        SOURCES["schedule_scores_lines"]["latest_result_in_data"] = str(played.max().date())
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_injuries(season: int, refresh: bool = True) -> pd.DataFrame:
    """Official injury reports for a season (empty frame if not published)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"injuries_{season}.parquet"
    url = INJURY_URL.format(season=season)
    if refresh or not path.exists():
        try:
            path.write_bytes(_fetch(url, "injury_reports"))
        except Exception as exc:
            print(f"  ! injury reports unavailable: {exc}")
            SOURCES["injury_reports"] = {"url": url, "retrieved_utc": utcnow(),
                                         "error": str(exc)}
            return pd.DataFrame()
    else:
        note_cached("injury_reports", path, url)
    inj = pd.read_parquet(path)
    inj["team"] = fix_team(inj["team"])
    if "injury_reports" in SOURCES and len(inj):
        SOURCES["injury_reports"]["latest_week_in_data"] = int(inj["week"].max())
    return inj


def _team_games_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    pbp["posteam"] = fix_team(pbp["posteam"])
    pbp["defteam"] = fix_team(pbp["defteam"])
    # Garbage time distorts efficiency; keep plays where the game is still competitive.
    comp = pbp[(pbp["wp"] > 0.05) & (pbp["wp"] < 0.95)]
    keys = ["game_id", "season", "week", "posteam", "defteam"]
    eff = comp.groupby(keys).agg(epa=("epa", "mean"), sr=("success", "mean")).reset_index()
    pass_eff = (
        comp[comp["qb_dropback"] == 1].groupby(keys)["epa"].mean().rename("pass_epa").reset_index()
    )
    plays = pbp.groupby(keys).size().rename("plays").reset_index()
    out = eff.merge(pass_eff, on=keys, how="left").merge(plays, on=keys, how="left")
    return out.rename(columns={"posteam": "team", "defteam": "opp"})


def _qb_games_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per passer per game: dropbacks and EPA/dropback (competitive game states)."""
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["epa"].notna() & pbp["passer_player_id"].notna()
             & (pbp["wp"] > 0.05) & (pbp["wp"] < 0.95)].copy()
    db["posteam"] = fix_team(db["posteam"])
    out = (db.groupby(["game_id", "season", "week", "posteam", "passer_player_id"])
             .agg(dropbacks=("epa", "size"), epa=("epa", "mean"),
                  name=("passer_player_name", "first"))
             .reset_index())
    return out.rename(columns={"posteam": "team", "passer_player_id": "qb_id"})


def load_team_games(seasons: list[int], refresh_current: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team_games, qb_games).

    team_games: one row per (game, offense) with EPA/play, success rate, pass EPA, plays.
    qb_games:   one row per (game, passer) with dropbacks and EPA/dropback.
    """
    CACHE.mkdir(exist_ok=True)
    current = max(seasons)
    teams, qbs = [], []
    for season in seasons:
        tpath = CACHE / f"team_games_{CACHE_VERSION}_{season}.csv"
        qpath = CACHE / f"qb_games_{CACHE_VERSION}_{season}.csv"
        url = PBP_URL.format(season=season)
        if tpath.exists() and qpath.exists() and not (refresh_current and season == current):
            teams.append(pd.read_csv(tpath)); qbs.append(pd.read_csv(qpath))
            if season == current:
                note_cached("play_by_play_current_season", tpath, url)
            continue
        print(f"  downloading play-by-play {season} ...", flush=True)
        try:
            raw = _fetch(url, "play_by_play_current_season" if season == current else None)
        except Exception as exc:  # season not published yet
            print(f"  ! could not fetch {season}: {exc}")
            continue
        pbp = pd.read_parquet(io.BytesIO(raw), columns=PBP_COLS)
        tg, qg = _team_games_from_pbp(pbp), _qb_games_from_pbp(pbp)
        tg.to_csv(tpath, index=False); qg.to_csv(qpath, index=False)
        teams.append(tg); qbs.append(qg)
        if season == current and len(pbp):
            SOURCES["play_by_play_current_season"]["latest_week_in_data"] = int(pbp["week"].max())
    if not teams:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(teams, ignore_index=True), pd.concat(qbs, ignore_index=True)
