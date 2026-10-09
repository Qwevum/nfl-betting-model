"""Experimental feature groups G3 (snap-weighted availability) and G4 (OL continuity).

Sources (nflverse-data releases, CC-BY 4.0, free): snap counts (from Pro Football
Reference, 2013+), injury reports (2013+ verified), players table (gsis <-> pfr id).

G3  For each team and game: the expected snap share of players listed Out or
    Doubtful on that week's injury report, excluding QBs (handled by the QB
    feature). Expected share = mean offense_pct / defense_pct over the team's
    previous (up to) 4 games this season. Units: "players' worth of snaps"
    (11 = a whole unit). Timing: the final injury report precedes kickoff by NFL
    rule, but the feed carries no publish timestamp, so this timing is inferred.
G4  Share of the team's offensive-line snaps in its most recent game taken by its
    five highest-snap linemen over the previous 4 games. Uses completed games only.

Games without coverage (seasons before 2013, week 1) get neutral values.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .data import CACHE, _fetch, fix_team, load_injuries

SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
OL = {"T", "G", "C", "OL", "OT", "OG"}
OUT = {"Out", "Doubtful"}
FIRST_SEASON = 2013
OLC_NEUTRAL = 0.96   # fill for games with no prior OL data: 2013-2014 median (before the dev period)


def _load(url: str, path, refresh: bool) -> pd.DataFrame:
    if refresh or not path.exists():
        path.write_bytes(_fetch(url))
    return pd.read_parquet(path)


def load_snaps(season: int, refresh: bool = False) -> pd.DataFrame:
    d = _load(SNAP_URL.format(season=season), CACHE / f"snap_counts_{season}.parquet", refresh)
    if d.empty:
        return d
    d["team"] = fix_team(d["team"])
    return d


def players_crosswalk(refresh: bool = False) -> dict:
    p = _load(PLAYERS_URL, CACHE / "players.parquet", refresh)
    p = p.dropna(subset=["gsis_id", "pfr_id"])
    return dict(zip(p["gsis_id"], p["pfr_id"]))


def team_game_table(seasons: list[int], refresh_current: bool = False) -> pd.DataFrame:
    """One row per (game_id, team): avail_off, avail_def, olc."""
    current = max(seasons)
    xwalk = players_crosswalk(refresh=refresh_current)
    out = []
    for season in seasons:
        if season < FIRST_SEASON:
            continue
        snaps = load_snaps(season, refresh=refresh_current and season == current)
        if snaps.empty:
            continue
        inj = load_injuries(season, refresh=refresh_current and season == current)
        if len(inj):
            inj = inj[inj["report_status"].isin(OUT) & (inj["position"] != "QB")].copy()
            inj["pfr_id"] = inj["gsis_id"].map(xwalk)
        out.append(_season_table(snaps, inj))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["game_id", "team", "avail_off",
                                                                              "avail_def", "olc"])


def _season_table(snaps: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, t in snaps.groupby("team"):
        games = t[["game_id", "week"]].drop_duplicates().sort_values("week")
        gl = games.to_dict("records")
        for i, g in enumerate(gl):
            prev = [x["game_id"] for x in gl[max(0, i - 4):i]]
            row = {"game_id": g["game_id"], "team": team, "avail_off": 0.0, "avail_def": 0.0, "olc": np.nan}
            if prev:
                p = t[t["game_id"].isin(prev)]
                n = len(prev)
                exp_off = p.groupby("pfr_player_id")["offense_pct"].sum() / n
                exp_def = p.groupby("pfr_player_id")["defense_pct"].sum() / n
                if len(inj):
                    listed = inj[(inj["team"] == team) & (inj["week"] == g["week"])]["pfr_id"].dropna()
                    row["avail_off"] = float(exp_off.reindex(listed).fillna(0).sum())
                    row["avail_def"] = float(exp_def.reindex(listed).fillna(0).sum())
                ol = p[p["position"].isin(OL)]
                top5 = ol.groupby("pfr_player_id")["offense_snaps"].sum().nlargest(5).index
                last = t[(t["game_id"] == prev[-1]) & t["position"].isin(OL)]
                tot = last["offense_snaps"].sum()
                if tot > 0 and len(top5):
                    row["olc"] = float(last.loc[last["pfr_player_id"].isin(top5), "offense_snaps"].sum() / tot)
            rows.append(row)
    return pd.DataFrame(rows)


def add_features(feat: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Merge G3/G4 columns onto game rows (home - away for margin, sums for totals)."""
    t = table.set_index(["game_id", "team"])
    def get(col, side, fill):
        idx = list(zip(feat["game_id"], feat[f"{side}_team"]))
        return t[col].reindex(idx).fillna(fill).to_numpy(float) if len(t) else np.full(len(feat), fill)
    h_off, a_off = get("avail_off", "home", 0.0), get("avail_off", "away", 0.0)
    h_def, a_def = get("avail_def", "home", 0.0), get("avail_def", "away", 0.0)
    h_olc, a_olc = get("olc", "home", OLC_NEUTRAL), get("olc", "away", OLC_NEUTRAL)
    return feat.assign(f_avail=(a_off + a_def) - (h_off + h_def), t_avail=h_off + a_off,
                       f_olc=h_olc - a_olc, t_olc=h_olc + a_olc - 2 * OLC_NEUTRAL,
                       home_avail_off=h_off, away_avail_off=a_off, home_olc=h_olc, away_olc=a_olc)
