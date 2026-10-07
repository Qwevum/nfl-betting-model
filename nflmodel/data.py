"""Download and cache nflverse data: schedules/results/lines and play-by-play.

Sources (all free, updated nightly by the nflverse project):
  * games.csv  - every game since 1999 with scores, closing spread/total/moneyline,
                 rest days, roof, weather, QBs.   (github.com/nflverse/nfldata)
  * play-by-play parquet per season with EPA / success / win probability.
                 (github.com/nflverse/nflverse-data releases)

Play-by-play is reduced to one row per team per game and cached as CSV, so the
large raw files are only downloaded once per past season.
"""
from __future__ import annotations

import io
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

FIRST_PBP_SEASON = 2010

# Relocated franchises are mapped to their current abbreviation so ratings carry over.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type",
    "epa", "success", "wp", "qb_dropback",
]


def _fetch(url: str) -> bytes:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        return resp.read()


def fix_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)


def load_games(refresh: bool = True) -> pd.DataFrame:
    """All games (past and scheduled) with closing lines, from nflverse."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "games.csv"
    if refresh or not path.exists():
        path.write_bytes(_fetch(GAMES_URL))
    g = pd.read_csv(path, low_memory=False)
    g["away_team"] = fix_team(g["away_team"])
    g["home_team"] = fix_team(g["home_team"])
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


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


def load_team_games(seasons: list[int], refresh_current: bool = True) -> pd.DataFrame:
    """One row per (game, offense) with EPA/play, success rate, pass EPA, plays."""
    CACHE.mkdir(exist_ok=True)
    current = max(seasons)
    frames = []
    for season in seasons:
        path = CACHE / f"team_games_{season}.csv"
        if path.exists() and not (refresh_current and season == current):
            frames.append(pd.read_csv(path))
            continue
        print(f"  downloading play-by-play {season} ...", flush=True)
        try:
            raw = _fetch(PBP_URL.format(season=season))
        except Exception as exc:  # season not published yet
            print(f"  ! could not fetch {season}: {exc}")
            continue
        pbp = pd.read_parquet(io.BytesIO(raw), columns=PBP_COLS)
        tg = _team_games_from_pbp(pbp)
        tg.to_csv(path, index=False)
        frames.append(tg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
