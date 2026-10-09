"""Game forecasts: one predictive distribution per game, separate from betting decisions.

For each game the forecast is ONE integer distribution of the home-minus-away margin
(and one of the total). Win / tie / loss probabilities, the projected margin and its
prediction interval all come from that same distribution, so they cannot disagree.

Center of the margin distribution, in order of preference:
  1. market-informed (spread): the BOOK-INDEPENDENT spread reference (live median across
     all books, else the untimed nflverse consensus) at its line, moved by the model's
     calibrated edge exactly as the betting probabilities are (decide.AnchoredPricer:
     sigmoid(a + b logit p_market + c edge), edge = blend model at that line minus the
     line), then converted back to a mean margin through the key-number distribution.
  2. market-informed (moneyline only): the same, from the moneyline reference.
  3. team ratings alone: the ratings-only model margin, with the out-of-sample residual
     scale of the ratings-only model (wider than the market-informed scale).
  With a market line but incomplete ratings the model edge is 0 and the basis says so.
Totals follow the same order (total reference, else ratings alone).

Ties: regular-season games can end tied, so the margin distribution keeps P(margin = 0).
Postseason games cannot; that mass is removed and the rest renormalized.

Betting rows (decide.py) price each offer against a reference built WITHOUT that
offer's book, so a moneyline row's probability can differ slightly from the forecast's
win probability. The forecast never uses a bookmaker-specific reference. Moneyline rows
also condition on the game not ending tied (a tie refunds a two-way moneyline at US
books), so they are compared with P(win) / (1 - P(tie)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import uncertainty as unc
from .model import MARGINS, TOTALS, FittedModel, _norm_cdf, logit, novig_first, sigmoid

LIMITS = {"spread": (-60.0, 60.0), "total": (5.0, 125.0)}
LIVE = "live median of all books (no book excluded)"
CONSENSUS = "nflverse consensus line (untimed)"


# ---------------------------------------------------------------- vectorized distribution math

def pmf_matrix(dist, mu: np.ndarray, sd: np.ndarray | float | None = None) -> np.ndarray:
    """Row i = dist.pmf(mu[i]) (optionally with scale sd[i]); same formula as OutcomeDist.pmf."""
    mu = np.asarray(mu, float)
    sd = np.full(len(mu), dist.sd) if sd is None else np.broadcast_to(np.asarray(sd, float), mu.shape)
    edges = np.append(dist.support - 0.5, dist.support[-1] + 0.5)
    p = np.diff(_norm_cdf((edges[None, :] - mu[:, None]) / sd[:, None]), axis=1) * dist.w[None, :]
    return p / p.sum(axis=1, keepdims=True)


def _cond_over(dist, mu: np.ndarray, line: np.ndarray) -> np.ndarray:
    p = pmf_matrix(dist, mu)
    s = dist.support[None, :]
    over = (p * (s > line[:, None])).sum(axis=1)
    under = (p * (s < line[:, None])).sum(axis=1)
    return over / (over + under)


def invert(dist, line: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorized market._invert: mu with P(X > line | no push; mu) = p (same bisection)."""
    line = np.asarray(line, float)
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    a, b = np.full(len(p), lo), np.full(len(p), hi)
    for _ in range(50):
        mid = (a + b) / 2
        below = _cond_over(dist, mid, line) < p
        a = np.where(below, mid, a)
        b = np.where(below, b, mid)
    return (a + b) / 2


# ---------------------------------------------------------------- anchors (same rules as AnchoredPricer)

def _anchors(model: FittedModel, pred: pd.DataFrame, refs_by_game: dict | None) -> dict:
    """Per game: market anchor (line, p, live?) for spread/total, moneyline p, and the model edge
    at the anchor line (blend recomputed at that line when it differs from the slate line)."""
    n = len(pred)
    out = {k: np.full(n, np.nan) for k in ("L_s", "p_s", "L_t", "p_t", "p_ml")}
    for k in ("live_s", "live_t", "live_ml"):
        out[k] = np.zeros(n, bool)
    refs_by_game = refs_by_game or {}
    for i, g in enumerate(pred.itertuples(index=False)):
        refs = refs_by_game.get(g.game_id) or {}
        for mk, L, a, b, kL, kp, kl in (("spread", g.spread_line, g.home_spread_odds, g.away_spread_odds,
                                         "L_s", "p_s", "live_s"),
                                        ("total", g.total_line, g.over_odds, g.under_odds, "L_t", "p_t", "live_t")):
            r = refs.get(mk)
            if r is not None:
                out[kL][i], out[kp][i], out[kl][i] = r.line, r.p, True
            elif pd.notna(L):
                out[kL][i], out[kp][i] = L, float(novig_first(a, b))
        r = refs.get("ml")
        if r is not None:
            out["p_ml"][i], out["live_ml"][i] = r.p, True
        elif pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            out["p_ml"][i] = float(novig_first(g.home_moneyline, g.away_moneyline))
    for mk, kL, blend_col, line_col, ridge in (("spread", "L_s", "blend_margin", "spread_line", model.core.blend),
                                               ("total", "L_t", "blend_total", "total_line", model.core.total_blend)):
        L = out[kL]
        blend = pred[blend_col].to_numpy(float).copy()
        redo = np.isfinite(L) & ~(pred[line_col].to_numpy(float) == L)
        if redo.any():
            blend[redo] = ridge.predict(pred[redo].assign(**{line_col: L[redo]}))
        e = blend - L
        out[f"edge_{mk}"] = np.where(np.isfinite(e), e, 0.0)
    return out


# ---------------------------------------------------------------- forecasts under one model

@dataclass
class Batch:
    """Predictive distributions for every game of `pred` under one fitted model."""
    margin_basis: np.ndarray
    total_basis: np.ndarray
    margin_source: np.ndarray
    total_source: np.ndarray
    mu_margin: np.ndarray
    mu_total: np.ndarray
    pmf_margin: np.ndarray      # games x len(MARGINS); NaN rows where unavailable
    pmf_total: np.ndarray

    @property
    def has_margin(self) -> np.ndarray:
        return np.isfinite(self.mu_margin)

    @property
    def p_home(self) -> np.ndarray:
        return np.where(self.has_margin, np.nansum(self.pmf_margin[:, MARGINS > 0], axis=1), np.nan)

    @property
    def p_away(self) -> np.ndarray:
        return np.where(self.has_margin, np.nansum(self.pmf_margin[:, MARGINS < 0], axis=1), np.nan)

    @property
    def p_tie(self) -> np.ndarray:
        return np.where(self.has_margin, self.pmf_margin[:, MARGINS == 0][:, 0], np.nan)


def postseason_mask(pred: pd.DataFrame) -> np.ndarray:
    if "game_type" not in pred:
        return np.zeros(len(pred), bool)
    gt = pred["game_type"]
    return (gt.notna() & (gt != "REG")).to_numpy()


def batch(model: FittedModel, pred: pd.DataFrame, refs_by_game: dict | None = None, mode: str = "model",
          force_ratings_only: bool = False) -> Batch:
    """pred: model.predict(day) for THIS model. refs_by_game: {game_id: {market: Reference}} built
    from all books (no exclusion), or None (untimed consensus from the slate rows)."""
    n = len(pred)
    A = _anchors(model, pred, refs_by_game)
    ratings_m = pred["model_margin"].to_numpy(float)
    ratings_t = pred["model_total"].to_numpy(float)
    mb, ms = np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    tb, ts = np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    mu_m, mu_t = np.full(n, np.nan), np.full(n, np.nan)
    sd_m, sd_t = np.full(n, model.margin_dist.sd), np.full(n, model.total_dist.sd)

    # margin
    use_s = np.isfinite(A["L_s"]) & np.isfinite(A["p_s"]) & (not force_ratings_only)
    use_ml = ~use_s & np.isfinite(A["p_ml"]) & (not force_ratings_only)
    use_r = ~use_s & ~use_ml & np.isfinite(ratings_m)
    cs, cm = model.cal_for("spread", mode), model.cal_for("ml", mode)
    if use_s.any():
        q = sigmoid(cs[0] + cs[1] * logit(A["p_s"][use_s]) + cs[2] * A["edge_spread"][use_s])
        mu_m[use_s] = invert(model.margin_dist, A["L_s"][use_s], q, *LIMITS["spread"])
        mb[use_s] = np.where(np.isfinite(ratings_m[use_s]), "market-informed (spread)",
                             "market reference only (model inputs incomplete)")
        ms[use_s] = np.where(A["live_s"][use_s], LIVE, CONSENSUS)
    if use_ml.any():
        q = sigmoid(cm[0] + cm[1] * logit(A["p_ml"][use_ml]) + cm[2] * 0.0)
        mu_m[use_ml] = invert(model.margin_dist, np.zeros(use_ml.sum()), q, *LIMITS["spread"])
        mb[use_ml] = np.where(np.isfinite(ratings_m[use_ml]), "market-informed (moneyline only)",
                              "market reference only (model inputs incomplete)")
        ms[use_ml] = np.where(A["live_ml"][use_ml], LIVE, CONSENSUS)
    if use_r.any():
        mu_m[use_r] = ratings_m[use_r]
        sd_m[use_r] = model.pure_margin_sd or model.margin_dist.sd
        mb[use_r] = "team ratings only"
        ms[use_r] = "market ignored (baseline)" if force_ratings_only else "no market line"
    none_m = ~(use_s | use_ml | use_r)
    ms[none_m] = "insufficient data: no market line and incomplete team ratings"

    # total
    use_t = np.isfinite(A["L_t"]) & np.isfinite(A["p_t"]) & (not force_ratings_only)
    use_rt = ~use_t & np.isfinite(ratings_t)
    ct = model.cal_for("total", mode)
    if use_t.any():
        q = sigmoid(ct[0] + ct[1] * logit(A["p_t"][use_t]) + ct[2] * A["edge_total"][use_t])
        mu_t[use_t] = invert(model.total_dist, A["L_t"][use_t], q, *LIMITS["total"])
        tb[use_t] = np.where(np.isfinite(ratings_t[use_t]), "market-informed (total)",
                             "market reference only (model inputs incomplete)")
        ts[use_t] = np.where(A["live_t"][use_t], LIVE, CONSENSUS)
    if use_rt.any():
        mu_t[use_rt] = ratings_t[use_rt]
        sd_t[use_rt] = model.pure_total_sd or model.total_dist.sd
        tb[use_rt] = "team ratings only"
        ts[use_rt] = "market ignored (baseline)" if force_ratings_only else "no market total"
    ts[~(use_t | use_rt)] = "insufficient data: no market total and incomplete team ratings"

    pm = np.full((n, len(MARGINS)), np.nan)
    ok = np.isfinite(mu_m)
    if ok.any():
        pm[ok] = pmf_matrix(model.margin_dist, mu_m[ok], sd_m[ok])
        post = postseason_mask(pred) & ok
        if post.any():
            pm[np.ix_(post, MARGINS == 0)] = 0.0
            pm[post] /= pm[post].sum(axis=1, keepdims=True)
    pt = np.full((n, len(TOTALS)), np.nan)
    ok = np.isfinite(mu_t)
    if ok.any():
        pt[ok] = pmf_matrix(model.total_dist, mu_t[ok], sd_t[ok])
    return Batch(mb, tb, ms, ts, mu_m, mu_t, pm, pt)


# ---------------------------------------------------------------- slate forecasts with uncertainty

def _median(pmf: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.full(len(pmf), np.nan)
    for i, p in enumerate(pmf):
        if np.all(np.isfinite(p)):
            out[i] = support[np.searchsorted(np.cumsum(p), 0.5 - 1e-12)]
    return out


def forecast_slate(model: FittedModel, day: pd.DataFrame, refs_all: dict | None, settings,
                   replicates: list[FittedModel] | None = None, level: float = 0.9,
                   pi_levels: tuple = ()) -> pd.DataFrame:
    """One row per game: the point forecast from `model`; probability intervals and full
    predictive outcome intervals from the bootstrap replicates (if any).

    day: slate rows (features + market columns). refs_all: {game_id: {market: Reference}}
    built from ALL books (no exclusion), or None for consensus-only runs."""
    mode = settings.probability_model
    replicates = replicates or []
    pred = model.predict(day)
    pt = batch(model, pred, refs_all, mode)
    reps = [batch(m, m.predict(day), refs_all, mode) for m in replicates]
    levels = sorted(set([level, *pi_levels]))
    df = pd.DataFrame({
        "game_id": pred["game_id"].to_numpy(), "season": pred["season"].to_numpy(),
        "week": pred["week"].to_numpy(), "gameday": pred["gameday"].to_numpy(),
        "kickoff_utc": pred["kickoff_utc"].to_numpy() if "kickoff_utc" in pred else pd.NaT,
        "away_team": pred["away_team"].to_numpy(), "home_team": pred["home_team"].to_numpy(),
        "postseason": postseason_mask(pred),
        "margin_basis": np.where(pt.margin_basis == "", "unavailable", pt.margin_basis),
        "total_basis": np.where(pt.total_basis == "", "unavailable", pt.total_basis),
        "margin_source": pt.margin_source, "total_source": pt.total_source,
        "p_home": pt.p_home, "p_away": pt.p_away, "p_tie": pt.p_tie,
        "margin_mean": np.where(pt.has_margin, np.nansum(pt.pmf_margin * MARGINS, axis=1), np.nan),
        "margin_median": _median(pt.pmf_margin, MARGINS),
        "total_mean": np.where(np.isfinite(pt.mu_total), np.nansum(pt.pmf_total * TOTALS, axis=1), np.nan),
        "total_median": _median(pt.pmf_total, TOTALS),
        "replicates": len(reps), "level": level})
    home_fav = df["p_home"] >= df["p_away"]
    df["winner"] = np.where(pt.has_margin, np.where(home_fav, df["home_team"], df["away_team"]), None)
    df["p_winner"] = np.where(pt.has_margin, np.maximum(df["p_home"], df["p_away"]), np.nan)

    # 1) uncertainty in the ESTIMATED probabilities (not outcome variability)
    for k in ("p_home_lo", "p_home_hi", "p_home_sd_logit", "p_away_lo", "p_away_hi", "p_away_sd_logit"):
        df[k] = np.nan
    if len(reps) >= 2:
        ph = np.array([r.p_home for r in reps])
        pa = np.array([r.p_away for r in reps])
        for i in np.flatnonzero(pt.has_margin):
            df.loc[i, ["p_home_lo", "p_home_hi", "p_home_sd_logit"]] = unc.prob_interval(pt.p_home[i], ph[:, i], level)
            df.loc[i, ["p_away_lo", "p_away_hi", "p_away_sd_logit"]] = unc.prob_interval(pt.p_away[i], pa[:, i], level)
        df["unc_status"] = np.where(pt.has_margin, "ok", "unavailable: no forecast distribution")
    else:
        df["unc_status"] = np.where(pt.has_margin, "not computed (uncertainty off)",
                                    "unavailable: no forecast distribution")

    # 2) prediction intervals for the ACTUAL outcome: full predictive distribution
    #    (mixture over replicates when available, else the point model's distribution)
    for name, support, P, R in (("margin", MARGINS, pt.pmf_margin, [r.pmf_margin for r in reps]),
                                ("total", TOTALS, pt.pmf_total, [r.pmf_total for r in reps])):
        mix = np.nanmean(np.array(R), axis=0) if len(R) >= 2 else P
        kind = ("predictive mixture over replicates" if len(R) >= 2
                else "plug-in (point model; parameter uncertainty not included)")
        valid = np.all(np.isfinite(mix), axis=1) & np.all(np.isfinite(P), axis=1)
        df[f"{name}_pi_kind"] = np.where(valid, kind, "unavailable")
        for lv in levels:
            tag = f"{round(lv * 100):d}"
            lo, hi = np.full(len(df), np.nan), np.full(len(df), np.nan)
            for i in np.flatnonzero(valid):
                lo[i], hi[i] = unc.pmf_interval(mix[i], support, lv)
            df[f"{name}_lo_{tag}"], df[f"{name}_hi_{tag}"] = lo, hi
    return df


__all__ = ["Batch", "batch", "forecast_slate", "pmf_matrix", "invert", "postseason_mask"]
