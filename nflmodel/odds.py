"""Odds: price math, sources of lines, and picking the best line per side.

Lines are kept in one long format, one row per offer:
    game_id, book, market ('spread'|'ml'|'total'), side ('home'|'away'|'over'|'under'),
    point (spread from that side's view, e.g. -3.5; total number; NaN for ml), price (American),
    source ('consensus'|'odds_api'|'manual'|'manual_untimed'), odds_time (book's update time, if known)

Sources:
  * The Odds API, every US book, if ODDS_API_KEY is set (free tier: the-odds-api.com);
    quotes carry the book's last_update time.
  * odds_manual.csv: quotes you copy from your own sportsbook apps, keyed by
    game_id + kickoff_utc, with the UTC time you saw them.
  * nflverse consensus lines: untimed, so never treated as an executable price.

Bookmaker offers pass validate_offers() before anything else looks at them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import SOURCES, _fetch, utcnow
from .timeutil import fmt, parse_utc

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

OFFER_COLS = ["game_id", "book", "market", "side", "point", "price", "source", "odds_time"]
VALID_SIDES = {"spread": {"home", "away"}, "ml": {"home", "away"}, "total": {"over", "under"}}


def consensus_offers(games: pd.DataFrame) -> pd.DataFrame:
    """nflverse consensus lines. Untimed: usable as a conditional reference, never executable."""
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
    out = pd.DataFrame(rows, columns=OFFER_COLS[:6])
    out["source"] = "consensus"
    out["odds_time"] = None   # nflverse does not timestamp its lines
    return out


def odds_api_offers(events: list, games: pd.DataFrame, kickoffs: dict,
                    match_minutes: float = 360.0) -> pd.DataFrame:
    """Parse The Odds API events. Games are matched on both teams AND kickoff time."""
    by_teams: dict[tuple[str, str], list[str]] = {}
    for g in games.itertuples(index=False):
        by_teams.setdefault((g.home_team, g.away_team), []).append(g.game_id)
    rows = []
    for ev_ in events:
        home, away = FULL_NAMES.get(ev_.get("home_team")), FULL_NAMES.get(ev_.get("away_team"))
        try:
            commence = parse_utc(ev_.get("commence_time"))
        except ValueError:
            continue
        gid = None
        for cand in by_teams.get((home, away), []):
            k = kickoffs.get(cand)
            if k is not None and abs((commence - k).total_seconds()) <= match_minutes * 60:
                gid = cand
        if gid is None:
            continue
        for bk in ev_.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(mk.get("key"))
                if market is None:
                    continue
                for oc in mk.get("outcomes", []):
                    if market == "total":
                        side = str(oc.get("name", "")).lower()
                    else:
                        side = "home" if FULL_NAMES.get(oc.get("name")) == home else "away"
                    rows.append((gid, bk.get("key"), market, side, oc.get("point", np.nan), oc.get("price"),
                                 "odds_api", mk.get("last_update") or bk.get("last_update")))
    return pd.DataFrame(rows, columns=OFFER_COLS)


def fetch_odds_api(games: pd.DataFrame, kickoffs: dict, key: str) -> pd.DataFrame:
    events = json.loads(_fetch(ODDS_API_URL.format(key=key), "odds_api"))
    return odds_api_offers(events, games, kickoffs)


MANUAL_COLS = ["game_id", "kickoff_utc", "book", "market", "side", "point", "price", "odds_time_utc"]


def manual_offers(path: Path, kickoffs: dict, kickoff_match_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """odds_manual.csv rows keyed by game_id + kickoff_utc. Returns (offers, rejected rows)."""
    m = pd.read_csv(path, comment="#", dtype=str)
    if m.empty:
        return pd.DataFrame(columns=OFFER_COLS), pd.DataFrame()
    missing = [c for c in MANUAL_COLS if c not in m.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; run `python run.py templates` "
                         "for the current format (games are identified by game_id and kickoff_utc)")
    SOURCES["manual_odds"] = {"url": str(path), "retrieved_utc": utcnow(), "rows": int(len(m))}
    ok, bad = check_game_keys(m, kickoffs, kickoff_match_minutes)
    out = ok.rename(columns={"odds_time_utc": "odds_time"})
    out["point"] = pd.to_numeric(out["point"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["source"] = "manual"
    return out[OFFER_COLS], bad


def check_game_keys(rows: pd.DataFrame, kickoffs: dict, tol_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows whose game_id exists and whose kickoff_utc matches the schedule."""
    reasons = []
    for r in rows.itertuples(index=False):
        k = kickoffs.get(r.game_id)
        if k is None:
            reasons.append("unknown game_id (not in this slate)")
            continue
        try:
            given = parse_utc(r.kickoff_utc)
        except ValueError as exc:
            reasons.append(f"kickoff_utc: {exc}")
            continue
        if abs((given - k).total_seconds()) > tol_minutes * 60:
            reasons.append(f"kickoff_utc {fmt(given)} does not match schedule {fmt(k)}")
        else:
            reasons.append("")
    rows = rows.assign(reject_reason=reasons)
    return rows[rows["reject_reason"] == ""].drop(columns="reject_reason"), rows[rows["reject_reason"] != ""]


def validate_offers(offers: pd.DataFrame, kickoffs: dict, now: pd.Timestamp, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split offers into (valid, rejected). Runs BEFORE any best-price selection.

    A valid offer has: a game in the slate with a known kickoff, a known market/side,
    a valid American price, a numeric point where needed, an odds timestamp in UTC
    that is not in the future, no older than settings.max_odds_age_minutes, and
    both the quote and `now` strictly before kickoff.
    """
    max_age = pd.Timedelta(minutes=settings.max_odds_age_minutes)
    skew = pd.Timedelta(minutes=settings.clock_skew_minutes)
    reasons, times = [], []
    for r in offers.itertuples(index=False):
        reason, t = "", pd.NaT
        k = kickoffs.get(r.game_id)
        if k is None:
            reason = "game not in slate or kickoff unknown"
        elif r.side not in VALID_SIDES.get(r.market, set()):
            reason = f"invalid market/side {r.market}/{r.side}"
        elif not _valid_price(r.price):
            reason = f"invalid price {r.price!r}"
        elif r.market != "ml" and not _valid_point(r.market, r.point):
            reason = f"invalid point {r.point!r}"
        else:
            try:
                t = parse_utc(r.odds_time)
            except ValueError as exc:
                reason = f"odds timestamp: {exc}"
            else:
                if t > now + skew:
                    reason = f"odds timestamp {fmt(t)} is in the future"
                elif now - t > max_age:
                    reason = f"stale: quoted {fmt(t)}, older than {settings.max_odds_age_minutes:g} min"
                elif t >= k or now >= k:
                    reason = f"at/after kickoff {fmt(k)}"
        reasons.append(reason)
        times.append(t)
    out = offers.assign(odds_time_utc=times, reject_reason=reasons)
    valid = out[out["reject_reason"] == ""].drop(columns="reject_reason")
    return valid, out[out["reject_reason"] != ""]


def _valid_price(p) -> bool:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return np.isfinite(p) and abs(p) >= 100


def _valid_point(market: str, point) -> bool:
    try:
        x = float(point)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(x) or abs(x * 2 - round(x * 2)) > 1e-9:
        return False
    return x > 0 if market == "total" else abs(x) < 60


def gather_offers(week_games: pd.DataFrame, kickoffs: dict, settings) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(timestamped bookmaker offers, consensus offers, notes). Not yet validated."""
    notes, frames = [], []
    key = os.environ.get("ODDS_API_KEY")
    if key:
        try:
            api = fetch_odds_api(week_games, kickoffs, key)
            notes.append(f"The Odds API: {len(api)} offers from {api['book'].nunique()} books")
            frames.append(api)
        except Exception as exc:
            notes.append(f"The Odds API failed: {exc}")
            SOURCES["odds_api"] = {"url": "api.the-odds-api.com", "retrieved_utc": utcnow(), "error": str(exc)}
    else:
        notes.append("ODDS_API_KEY not set: no live bookmaker feed")
    manual = ROOT / "odds_manual.csv"
    if manual.exists():
        mo, bad = manual_offers(manual, kickoffs, settings.kickoff_match_minutes)
        frames.append(mo)
        for r in bad.itertuples(index=False):
            notes.append(f"odds_manual.csv row rejected ({r.game_id}): {r.reject_reason}")
    books = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OFFER_COLS)
    return books, consensus_offers(week_games), notes
