"""Sequential, leak-free team ratings.

Games are processed in date order. Before each game the current ratings are
recorded as that game's features; only afterwards are ratings updated with the
game's result. So every feature only uses information available before kickoff.

Ratings kept per team (all opponent-adjusted except pace/points):
  mrtg     points rating (margin Elo): expected margin = mrtg_home - mrtg_away + HFA
  off/def  EPA/play produced / allowed relative to league average
  osr/dsr  success rate produced / allowed relative to league average
  opass    pass EPA/dropback relative to league average (QB-sensitive)
  pace     offensive plays per game
  pf/pa    points for / against per game

QBs are rated separately (opponent-adjusted EPA per dropback, shrunk toward the
average of inexperienced QBs). Because a team's pass ratings already contain its
usual QB, the QB feature is only the *change*: the listed starter's rating minus
the rating of the QBs the team has actually been playing (qb_base). A team with
its regular starter gets ~0, so the QB is not counted twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = {
    "k_margin": 0.07,      # margin-Elo step size
    "margin_cap": 24.0,    # cap blowouts when updating
    "hfa_update": 1.7,     # home edge assumed inside the Elo update
    "a_eff": 0.10,         # EPA / success rate step size
    "a_pace": 0.10,        # pace / points EWMA weight
    "carry": 0.60,         # share of a rating kept from one season to the next
    "qb_decay": 0.97,      # per-game weight decay of a QB's past dropbacks
    "qb_shrink": 150.0,    # dropbacks of prior weight in a QB rating
    "qb_new": 150.0,       # career dropbacks below which a QB counts as inexperienced
}

FEATURES = ["f_mrtg", "f_epa", "f_sr", "f_pass", "f_qb", "f_hfa", "f_rest", "f_div"]
TOTAL_FEATURES = ["t_off", "t_def", "t_pace", "t_pts", "t_dome", "t_wind"]


class _Team:
    __slots__ = ("mrtg", "off", "def_", "osr", "dsr", "opass", "dpass", "pace", "pf", "pa", "season",
                 "qb_base", "last_qb")

    def __init__(self, season: int, pace: float = 62.0, pts: float = 22.0):
        self.mrtg = self.off = self.def_ = self.osr = self.dsr = self.opass = self.dpass = 0.0
        self.pace, self.pf, self.pa, self.season = pace, pts, pts, season
        self.qb_base = None
        self.last_qb = None

    def new_season(self, season: int, carry: float, lg_pace: float, lg_pts: float) -> None:
        for name in ("mrtg", "off", "def_", "osr", "dsr", "opass", "dpass"):
            setattr(self, name, getattr(self, name) * carry)
        self.pace = lg_pace + carry * (self.pace - lg_pace)
        self.pf = lg_pts + carry * (self.pf - lg_pts)
        self.pa = lg_pts + carry * (self.pa - lg_pts)
        self.season = season


class QBRatings:
    """Opponent-adjusted EPA/dropback per QB, decayed and shrunk to a data-derived prior."""

    def __init__(self, p: dict):
        self.p = p
        self.S: dict[str, float] = {}   # decayed sum of dropbacks * adjusted EPA
        self.N: dict[str, float] = {}   # decayed dropbacks
        self.career: dict[str, float] = {}
        self.new_S, self.new_N = 0.0, 0.0   # pooled inexperienced-QB performance

    @property
    def prior(self) -> float:
        return self.new_S / self.new_N if self.new_N > 300 else -0.05

    def rating(self, qb_id) -> float:
        if not isinstance(qb_id, str):
            return self.prior
        k = self.p["qb_shrink"]
        return (self.S.get(qb_id, 0.0) + k * self.prior) / (self.N.get(qb_id, 0.0) + k)

    def update(self, qb_id: str, dropbacks: float, adj_epa: float) -> None:
        if self.career.get(qb_id, 0.0) < self.p["qb_new"]:
            self.new_S += dropbacks * adj_epa
            self.new_N += dropbacks
        d = self.p["qb_decay"]
        self.S[qb_id] = d * self.S.get(qb_id, 0.0) + dropbacks * adj_epa
        self.N[qb_id] = d * self.N.get(qb_id, 0.0) + dropbacks
        self.career[qb_id] = self.career.get(qb_id, 0.0) + dropbacks

def build_features(games: pd.DataFrame, team_games: pd.DataFrame, qb_games: pd.DataFrame | None = None,
                   params: dict | None = None):
    """Return (games-with-features, final team ratings table, QBRatings)."""
    p = {**PARAMS, **(params or {})}
    tg = team_games.set_index(["game_id", "team"]) if len(team_games) else None
    qbg: dict[tuple[str, str], list] = {}
    if qb_games is not None and len(qb_games):
        for r in qb_games.itertuples(index=False):
            qbg.setdefault((r.game_id, r.team), []).append((r.qb_id, r.dropbacks, r.epa))
    qbr = QBRatings(p)
    first_season = int(team_games["season"].min()) if len(team_games) else int(games["season"].min())
    games = games[games["season"] >= first_season].copy()

    teams: dict[str, _Team] = {}
    lg = {"epa": 0.0, "sr": 0.44, "pass": 0.05, "pace": 62.0, "pts": 22.0}
    rows = []

    for g in games.itertuples(index=False):
        h, a, season = g.home_team, g.away_team, int(g.season)
        for t in (h, a):
            if t not in teams:
                teams[t] = _Team(season, lg["pace"], lg["pts"])
            elif teams[t].season != season:
                teams[t].new_season(season, p["carry"], lg["pace"], lg["pts"])
        H, A = teams[h], teams[a]
        neutral = str(g.location) == "Neutral"
        rest = (g.home_rest if pd.notna(g.home_rest) else 7) - (g.away_rest if pd.notna(g.away_rest) else 7)
        dome = str(g.roof) in ("dome", "closed")
        wind = 0.0 if dome or pd.isna(g.wind) else float(g.wind)

        def qb_delta(T: _Team, qb_id):
            if not isinstance(qb_id, str):
                return 0.0, np.nan
            r = qbr.rating(qb_id)
            base = T.qb_base if T.qb_base is not None else r
            return r - base, r
        h_qbd, h_qbr = qb_delta(H, g.home_qb_id)
        a_qbd, a_qbr = qb_delta(A, g.away_qb_id)

        rows.append({
            "game_id": g.game_id,
            "f_mrtg": H.mrtg - A.mrtg,
            "f_epa": (H.off - H.def_) - (A.off - A.def_),
            "f_sr": (H.osr - H.dsr) - (A.osr - A.dsr),
            "f_pass": (H.opass - H.dpass) - (A.opass - A.dpass),
            "f_qb": h_qbd - a_qbd,
            "f_hfa": 0.0 if neutral else 1.0,
            "f_rest": float(np.clip(rest, -7, 7)),
            "f_div": float(g.div_game) if pd.notna(g.div_game) else 0.0,
            "t_off": H.off + A.off,
            "t_def": H.def_ + A.def_,
            "t_pace": H.pace + A.pace,
            "t_pts": (H.pf + H.pa + A.pf + A.pa) / 2,
            "t_dome": 1.0 if dome else 0.0,
            "t_wind": wind,
            # pre-game ratings, for the ratings table / report
            "home_mrtg": H.mrtg, "away_mrtg": A.mrtg,
            "home_qb_rating": h_qbr, "away_qb_rating": a_qbr,
            "home_qb_base": H.qb_base, "away_qb_base": A.qb_base,
            "home_qb_delta": h_qbd, "away_qb_delta": a_qbd,
            "home_last_qb": H.last_qb, "away_last_qb": A.last_qb,
        })

        # ---- update with the result (only for completed games) ----
        if pd.isna(g.result):
            continue
        margin = float(g.result)
        exp = H.mrtg - A.mrtg + (0.0 if neutral else p["hfa_update"])
        err = float(np.clip(margin, -p["margin_cap"], p["margin_cap"])) - exp
        H.mrtg += p["k_margin"] * err
        A.mrtg -= p["k_margin"] * err

        a_p = p["a_pace"]
        H.pf += a_p * (g.home_score - H.pf); H.pa += a_p * (g.away_score - H.pa)
        A.pf += a_p * (g.away_score - A.pf); A.pa += a_p * (g.home_score - A.pa)
        lg["pts"] += 0.01 * ((g.home_score + g.away_score) / 2 - lg["pts"])

        if tg is None:
            continue
        try:
            hs, as_ = tg.loc[(g.game_id, h)], tg.loc[(g.game_id, a)]
        except KeyError:
            continue
        a_e = p["a_eff"]
        for O, D, s in ((H, A, hs), (A, H, as_)):
            r = s["epa"] - (lg["epa"] + O.off + D.def_)
            O.off += a_e * r; D.def_ += a_e * r
            r = s["sr"] - (lg["sr"] + O.osr + D.dsr)
            O.osr += a_e * r; D.dsr += a_e * r
            if pd.notna(s["pass_epa"]):
                r = s["pass_epa"] - (lg["pass"] + O.opass + D.dpass)
                O.opass += a_e * r; D.dpass += a_e * r
            O.pace += a_p * (s["plays"] - O.pace)
        lg["epa"] += 0.01 * ((hs["epa"] + as_["epa"]) / 2 - lg["epa"])
        lg["sr"] += 0.01 * ((hs["sr"] + as_["sr"]) / 2 - lg["sr"])
        if pd.notna(hs["pass_epa"]) and pd.notna(as_["pass_epa"]):
            lg["pass"] += 0.01 * ((hs["pass_epa"] + as_["pass_epa"]) / 2 - lg["pass"])
        lg["pace"] += 0.01 * ((hs["plays"] + as_["plays"]) / 2 - lg["pace"])

        # QBs: rate on this game, then move each team's QB baseline toward the
        # dropback-weighted rating of whoever actually played.
        for T, O, D, team in ((H, H, A, h), (A, A, H, a)):
            plays = qbg.get((g.game_id, team), [])
            tot = sum(db for _, db, _ in plays)
            if not tot:
                continue
            played_rating = sum(db * qbr.rating(q) for q, db, _ in plays) / tot
            T.qb_base = played_rating if T.qb_base is None else T.qb_base + a_e * 2 * (played_rating - T.qb_base)
            T.last_qb = max(plays, key=lambda x: x[1])[0]
            for q, db, e in plays:
                if pd.notna(e):
                    qbr.update(q, db, e - lg["pass"] - D.dpass)

    feats = pd.DataFrame(rows)
    out = games.merge(feats, on="game_id", how="left")

    table = pd.DataFrame([
        {"team": t, "points_rating": s.mrtg, "off_epa": s.off, "def_epa_allowed": s.def_,
         "net_epa": s.off - s.def_, "pass_off": s.opass, "pass_def": s.dpass,
         "pace": s.pace, "pts_for": s.pf, "pts_against": s.pa}
        for t, s in teams.items()
    ]).sort_values("points_rating", ascending=False).reset_index(drop=True)
    return out, table, qbr
