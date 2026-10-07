"""Odds: price math, sources of lines, and picking the best line per side.

Lines are kept in one long format, one row per offer:
    game_id, book, market ('spread'|'ml'|'total'), side ('home'|'away'|'over'|'under'),
    point (spread from that side's view, e.g. -3.5; total number; NaN for ml), price (American),
    source ('consensus'|'odds_api'|'manual'|'manual_untimed'), odds_time (book's update time, if known)

Sources, in the order they're merged:
  1. nflverse consensus lines (always available, book = 'consensus')
  2. The Odds API, every US book, if ODDS_API_KEY is set  (free tier: the-odds-api.com)
  3. odds_manual.csv, for lines you type in from your own sportsbook apps
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SOURCES, _fetch, utcnow

ROOT = Path(__file__).resolve().parent.parent

ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    "?apiKey={key}&regions=us,us2&markets=h2h,spreads,totals&oddsFormat=american"
)

FULL_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def decimal(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def implied_probability(american: float) -> float:
    """Break-even win probability of an American price, vig included.

    -110 -> 110/210 = 52.4%      +150 -> 100/250 = 40.0%
    """
    if american == 0 or abs(american) < 100:
        raise ValueError(f"not a valid American price: {american}")
    if american < 0:
        return -american / (-american + 100)
    return 100 / (american + 100)


def no_vig(p1_american: float, p2_american: float) -> tuple[float, float]:
    """Both sides' implied probabilities with the bookmaker's margin removed."""
    a, b = implied_probability(p1_american), implied_probability(p2_american)
    return a / (a + b), b / (a + b)


def ev(p_win: float, p_push: float, american: float) -> float:
    """Expected profit per 1 unit staked: the bet's edge. A push returns the stake."""
    p_lose = 1 - p_win - p_push
    return p_win * (decimal(american) - 1) - p_lose


def kelly(p_win: float, p_push: float, american: float) -> float:
    """Full-Kelly bankroll fraction, f* = (b*p - q) / b, never below 0."""
    b = decimal(american) - 1
    p_lose = 1 - p_win - p_push
    return max(0.0, (b * p_win - p_lose) / b)


# ---------------------------------------------------------------- sources

def consensus_offers(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        sp = lambda v: v if pd.notna(v) else -110.0  # noqa: E731
        if pd.notna(g.spread_line):  # nflverse spread_line: + means home favored
            rows.append((g.game_id, "consensus", "spread", "home", -g.spread_line, sp(g.home_spread_odds)))
            rows.append((g.game_id, "consensus", "spread", "away", g.spread_line, sp(g.away_spread_odds)))
        if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            rows.append((g.game_id, "consensus", "ml", "home", np.nan, g.home_moneyline))
            rows.append((g.game_id, "consensus", "ml", "away", np.nan, g.away_moneyline))
        if pd.notna(g.total_line):
            rows.append((g.game_id, "consensus", "total", "over", g.total_line, sp(g.over_odds)))
            rows.append((g.game_id, "consensus", "total", "under", g.total_line, sp(g.under_odds)))
    out = pd.DataFrame(rows, columns=["game_id", "book", "market", "side", "point", "price"])
    out["source"] = "consensus"
    out["odds_time"] = None   # nflverse does not timestamp its lines
    return out


def odds_api_offers(games: pd.DataFrame, key: str) -> pd.DataFrame:
    events = json.loads(_fetch(ODDS_API_URL.format(key=key), "odds_api"))
    idx = {(g.home_team, g.away_team): g.game_id for g in games.itertuples(index=False)}
    rows = []
    for ev_ in events:
        home, away = FULL_NAMES.get(ev_["home_team"]), FULL_NAMES.get(ev_["away_team"])
        gid = idx.get((home, away))
        if gid is None:
            continue
        for bk in ev_.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(mk["key"])
                for oc in mk.get("outcomes", []):
                    if market == "total":
                        side = oc["name"].lower()
                    else:
                        side = "home" if FULL_NAMES.get(oc["name"]) == home else "away"
                    rows.append((gid, bk["key"], market, side, oc.get("point", np.nan), oc["price"],
                                 "odds_api", mk.get("last_update") or bk.get("last_update")))
    return pd.DataFrame(rows, columns=["game_id", "book", "market", "side", "point", "price",
                                       "source", "odds_time"])


def manual_offers(games: pd.DataFrame, path: Path) -> pd.DataFrame:
    """odds_manual.csv columns: book,away,home,market,side,point,price[,retrieved_at]."""
    m = pd.read_csv(path, comment="#")
    cols = ["game_id", "book", "market", "side", "point", "price", "source", "odds_time"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    if "retrieved_at" not in m.columns:
        m["retrieved_at"] = None
    m["odds_time"] = m["retrieved_at"]
    m["source"] = np.where(m["retrieved_at"].notna(), "manual", "manual_untimed")
    SOURCES["manual_odds"] = {"url": str(path), "retrieved_utc": utcnow(), "rows": int(len(m))}
    idx = {(g.home_team, g.away_team): g.game_id for g in games.itertuples(index=False)}
    m["game_id"] = [idx.get((h, a)) for h, a in zip(m["home"], m["away"])]
    missing = m[m["game_id"].isna()]
    if len(missing):
        print(f"  ! {len(missing)} manual odds rows don't match a game this week (check team codes)")
    return m.dropna(subset=["game_id"])[cols]


def gather_offers(week_games: pd.DataFrame) -> pd.DataFrame:
    frames = [consensus_offers(week_games)]
    key = os.environ.get("ODDS_API_KEY")
    if key:
        try:
            api = odds_api_offers(week_games, key)
            print(f"  The Odds API: {len(api)} offers from {api['book'].nunique()} books")
            frames.append(api)
        except Exception as exc:
            print(f"  ! The Odds API failed: {exc}")
            SOURCES["odds_api"] = {"url": "api.the-odds-api.com", "retrieved_utc": utcnow(), "error": str(exc)}
    manual = ROOT / "odds_manual.csv"
    if manual.exists():
        frames.append(manual_offers(week_games, manual))
    return pd.concat(frames, ignore_index=True)
