"""Optional historical OPEN and CLOSE lines (e.g. the AusSportsBetting NFL file), mapped to game_id.

nflverse games.csv only has closing consensus lines. This loader reads a user-supplied file
`data/historical_lines.csv` (or .xlsx with openpyxl installed) WITHOUT assuming its layout:

1. Columns are matched to roles by name patterns (case/punctuation-insensitive). A role that
   matches several columns, or a required role that matches none, is an error that lists the
   columns found. An explicit mapping {role: column} (data/historical_lines.mapping.json or
   the `mapping` argument) overrides inference.
2. Team names (full names, nicknames or nflverse codes) are mapped to the nflverse code of
   that season (OAK/LV, SD/LAC, STL/LA, Washington's names).
3. Rows are matched to schedule games on (home, away, date within +/- 1 day); rows matching
   only with home/away swapped (neutral sites) are matched and flagged. Unmatched rows are
   returned for the report.
4. The sign convention is not assumed: on matched games the file's closing line is compared
   with the nflverse spread_line (home margin, positive = home favored). If they correlate
   negatively the file uses handicap form (home -3.5 = home favored) and is negated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "data" / f"historical_lines.{ext}" for ext in ("csv", "xlsx", "xls")]
MAPPING_FILE = ROOT / "data" / "historical_lines.mapping.json"


class HistLinesError(RuntimeError):
    pass


# role -> regexes over normalized column names (lowercase, alphanumerics only). Order = priority.
ROLES = {
    "date": [r"^date$", r"^gamedate$", r"^gameday$"],
    "home_team": [r"^hometeam$", r"^home$"],
    "away_team": [r"^awayteam$", r"^away$", r"^visitor(team)?$"],
    "home_line_open": [r"^homelineopen(ing)?$", r"^homespreadopen(ing)?$", r"^openinghomeline$"],
    "home_line_close": [r"^homelineclos(e|ing)$", r"^homespreadclos(e|ing)$", r"^closinghomeline$"],
    "home_line_odds_open": [r"^homelineoddsopen(ing)?$"],
    "away_line_odds_open": [r"^awaylineoddsopen(ing)?$"],
    "home_line_odds_close": [r"^homelineoddsclos(e|ing)$"],
    "away_line_odds_close": [r"^awaylineoddsclos(e|ing)$"],
    "total_open": [r"^totalscoreopen(ing)?$", r"^totalopen(ing)?$"],
    "total_close": [r"^totalscoreclos(e|ing)$", r"^totalclos(e|ing)$"],
}
REQUIRED = ("date", "home_team", "away_team", "home_line_open", "home_line_close")

NAMES = {
    "arizona cardinals": "ARI", "cardinals": "ARI", "atlanta falcons": "ATL", "falcons": "ATL",
    "baltimore ravens": "BAL", "ravens": "BAL", "buffalo bills": "BUF", "bills": "BUF",
    "carolina panthers": "CAR", "panthers": "CAR", "chicago bears": "CHI", "bears": "CHI",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cleveland browns": "CLE", "browns": "CLE",
    "dallas cowboys": "DAL", "cowboys": "DAL", "denver broncos": "DEN", "broncos": "DEN",
    "detroit lions": "DET", "lions": "DET", "green bay packers": "GB", "packers": "GB",
    "houston texans": "HOU", "texans": "HOU", "indianapolis colts": "IND", "colts": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "kansas city chiefs": "KC", "chiefs": "KC",
    "miami dolphins": "MIA", "dolphins": "MIA", "minnesota vikings": "MIN", "vikings": "MIN",
    "new england patriots": "NE", "patriots": "NE", "new orleans saints": "NO", "saints": "NO",
    "new york giants": "NYG", "giants": "NYG", "new york jets": "NYJ", "jets": "NYJ",
    "philadelphia eagles": "PHI", "eagles": "PHI", "pittsburgh steelers": "PIT", "steelers": "PIT",
    "san francisco 49ers": "SF", "49ers": "SF", "seattle seahawks": "SEA", "seahawks": "SEA",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "tennessee titans": "TEN", "titans": "TEN",
    "washington redskins": "WAS", "redskins": "WAS", "washington football team": "WAS",
    "washington commanders": "WAS", "commanders": "WAS", "washington": "WAS",
}
# relocated franchises: name -> [(last season with old code, old code), new code]
MOVED = {"raiders": ("OAK", 2019, "LV"), "chargers": ("SD", 2016, "LAC"), "rams": ("STL", 2015, "LA")}
CODES = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX",
         "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "OAK", "PHI", "PIT", "SD", "SEA", "SF",
         "STL", "TB", "TEN", "WAS"}


def _norm(c) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def team_code(name, season: int) -> str | None:
    """nflverse code for a team name in a given season (None if unknown)."""
    if not isinstance(name, str):
        return None
    s = name.strip()
    if s.upper() in CODES:
        return s.upper()
    low = re.sub(r"\s+", " ", s.lower())
    for key, (old, last, new) in MOVED.items():
        if re.search(r"(^|\s)" + key + r"($|\s)", low):
            return old if season <= last else new
    return NAMES.get(low) or NAMES.get(low.split(" ")[-1])


def season_of(date: pd.Timestamp) -> int:
    """NFL season of a date (January/February games belong to the previous season)."""
    return date.year if date.month >= 3 else date.year - 1


def odds_format(values: np.ndarray) -> str:
    """'decimal' (all finite values in (1, 20)), 'american' (all |v| >= 100), 'none' or 'mixed'."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "none"
    if np.all((v > 1) & (v < 20)):
        return "decimal"
    if np.all(np.abs(v) >= 100):
        return "american"
    raise HistLinesError(f"line odds are neither all decimal nor all American (e.g. {sorted(set(v.round(2)))[:8]})")


def decimal_to_american(d: float) -> float:
    if not np.isfinite(d):
        return np.nan
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def find_file() -> Path | None:
    return next((p for p in CANDIDATES if p.exists()), None)


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise HistLinesError(f"{path.name} is an Excel file; run `pip install openpyxl` or save it as "
                                 "data/historical_lines.csv") from e
        df = pd.read_excel(path)
    # some exports put a note above the header: use the first row that has a date and team columns
    if not _has_core(df.columns):
        for i in range(min(5, len(df))):
            hdr = df.iloc[i].tolist()
            if _has_core(hdr):
                df = df.iloc[i + 1:].reset_index(drop=True)
                df.columns = hdr
                break
    return df


def _has_core(cols) -> bool:
    n = [_norm(c) for c in cols]
    return any(re.match(p, c) for c in n for p in ROLES["date"]) and \
        any(re.match(p, c) for c in n for p in ROLES["home_team"])


def infer_columns(columns, mapping: dict | None = None) -> dict:
    """{role: column}. Raises HistLinesError for ambiguous or missing required roles."""
    mapping = dict(mapping or {})
    missing_cols = [c for c in mapping.values() if c not in list(columns)]
    if missing_cols:
        raise HistLinesError(f"mapping names columns not in the file: {missing_cols}")
    out, problems = dict(mapping), []
    for role, pats in ROLES.items():
        if role in out:
            continue
        for pat in pats:
            hits = [c for c in columns if re.match(pat, _norm(c))]
            if len(hits) == 1:
                out[role] = hits[0]
                break
            if len(hits) > 1:
                problems.append(f"{role}: ambiguous {hits}")
                break
    miss = [r for r in REQUIRED if r not in out]
    if miss or problems:
        raise HistLinesError("cannot map columns: " + "; ".join(problems + [f"{r}: not found" for r in miss])
                             + f". Columns in the file: {list(columns)}. Provide {MAPPING_FILE.name} as "
                             "{role: column}.")
    return out


def load(path: Path | None = None, games: pd.DataFrame | None = None, mapping: dict | None = None):
    """(lines, report). lines: game_id, season, open_line, close_line (nflverse convention: expected
    home margin), open/close home and away line odds (NaN if absent), total_open, total_close, swapped.
    report: mapping used, sign convention, matched/unmatched counts and the unmatched rows."""
    path = path or find_file()
    if path is None or not Path(path).exists():
        raise HistLinesError(f"no historical lines file found ({path or 'data/historical_lines.csv|.xlsx'})")
    if mapping is None and MAPPING_FILE.exists():
        mapping = json.loads(MAPPING_FILE.read_text())
    raw = read_raw(path)
    cols = infer_columns(raw.columns, mapping)
    df = pd.DataFrame({role: raw[col] for role, col in cols.items()})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    for c in df.columns:
        if c not in ("date", "home_team", "away_team"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])
    df["season"] = df["date"].map(season_of)
    df["home"] = [team_code(n, s) for n, s in zip(df["home_team"], df["season"])]
    df["away"] = [team_code(n, s) for n, s in zip(df["away_team"], df["season"])]
    if games is None:
        games = pd.read_csv(ROOT / "data" / "games.csv", low_memory=False)
    g = games[["game_id", "season", "gameday", "home_team", "away_team", "spread_line"]].copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    matched, unmatched = [], []
    by_teams: dict = {}
    for r in g.itertuples(index=False):
        by_teams.setdefault((r.home_team, r.away_team), []).append(r)
    for i, r in df.iterrows():
        hit, swapped = None, False
        for key, sw in (((r["home"], r["away"]), False), ((r["away"], r["home"]), True)):
            cands = [x for x in by_teams.get(key, []) if abs((x.gameday - r["date"]).days) <= 1]
            if len(cands) == 1:
                hit, swapped = cands[0], sw
                break
        if hit is None:
            why = ("unknown team name" if pd.isna(r["home"]) or pd.isna(r["away"]) else "no schedule game within 1 day")
            unmatched.append({"row": int(i), "date": r["date"].date(), "home_team": r["home_team"],
                              "away_team": r["away_team"], "reason": why})
            continue
        matched.append({"idx": i, "game_id": hit.game_id, "swapped": swapped, "nfl_close": hit.spread_line})
    m = pd.DataFrame(matched, columns=["idx", "game_id", "swapped", "nfl_close"])
    lines = df.loc[m["idx"]].reset_index(drop=True).assign(game_id=m["game_id"].to_numpy(),
                                                           swapped=m["swapped"].to_numpy(),
                                                           nfl_close=m["nfl_close"].to_numpy())
    # a row matched with home/away swapped has its lines from the other team's view
    flip = np.where(lines["swapped"], -1.0, 1.0)
    # sign convention, checked against nflverse closing lines on matched games
    ok = pd.DataFrame({"x": flip * lines["home_line_close"].to_numpy(float),
                       "y": lines["nfl_close"].to_numpy(float)}).dropna()
    sign, conv = -1.0, "unknown (fewer than 20 comparable games; handicap form assumed - CHECK)"
    if len(ok) >= 20:
        corr = float(np.corrcoef(ok["x"], ok["y"])[0, 1])
        sign = -1.0 if corr < 0 else 1.0
        conv = (f"{'handicap form (negated)' if sign < 0 else 'home-margin form'}; correlation with nflverse "
                f"closing line {corr:+.3f} on {len(ok)} games")
    lines["open_line"] = sign * flip * lines["home_line_open"]
    lines["close_line"] = sign * flip * lines["home_line_close"]
    for a, b in (("home_line_odds_open", "away_line_odds_open"), ("home_line_odds_close", "away_line_odds_close")):
        if a not in lines:
            lines[a] = np.nan
        if b not in lines:
            lines[b] = np.nan
        sw = lines["swapped"].to_numpy()
        lines.loc[sw, [a, b]] = lines.loc[sw, [b, a]].to_numpy()
    for c in ("total_open", "total_close"):
        if c not in lines:
            lines[c] = np.nan
    odds_cols = ["home_line_odds_open", "away_line_odds_open", "home_line_odds_close", "away_line_odds_close"]
    odds_fmt = odds_format(lines[odds_cols].to_numpy(float).ravel())
    if odds_fmt == "decimal":
        for c in odds_cols:
            lines[c] = lines[c].map(decimal_to_american)
    gap = (lines["close_line"] - lines["nfl_close"]).abs()
    out = lines[["game_id", "season", "open_line", "close_line", "home_line_odds_open", "away_line_odds_open",
                 "home_line_odds_close", "away_line_odds_close", "total_open", "total_close", "swapped",
                 "nfl_close"]].drop_duplicates("game_id")
    report = {"file": str(path.name), "rows": int(len(df)), "matched": int(len(out)), "unmatched": unmatched,
              "mapping": cols, "sign_convention": conv,
              "median |close - nflverse close|": float(gap.median()) if gap.notna().any() else None,
              "swapped": int(out["swapped"].sum()), "odds_format": odds_fmt}
    return out, report


__all__ = ["odds_format", "decimal_to_american", "load", "infer_columns", "team_code", "season_of", "find_file", "HistLinesError", "ROLES", "REQUIRED"]
