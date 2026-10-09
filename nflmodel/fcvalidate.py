"""Evaluation of game forecasts and their uncertainty (protocol: docs/UNCERTAINTY.md).

Two parts, never mixed:

1. simulate(): probability-interval behaviour where the truth is KNOWN. The point model
   fit on real training seasons is taken as the truth; training outcomes are simulated
   from its predictive distribution with team-season and season shocks whose variances
   are estimated from real residuals (method of moments), so the simulated history has
   realistic dependence. Each simulation refits the point model and the bootstrap
   replicates under every candidate block scheme and forecasts real test games. Reported
   per scheme: mean bootstrap SD / true sampling SD of the estimator (logit scale), and
   coverage of (a) the estimator's expectation and (b) the truth model's probability.
   Limits: the truth is the model itself (well specified by construction except for ridge
   and calibration bias), market inputs are held fixed and do not react to the simulated
   shocks, and the shock structure is a simple random-effects approximation.

2. evaluate(): real games, chronological out of sample. Every test season is forecast by a
   point model and replicates fit only on earlier seasons, with closing consensus markets
   (the only historical market data), and compared with baselines on identical games.
   A single win or loss cannot show whether an interval contains the latent probability,
   so for probabilities only calibration and scoring are checked here.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast, metrics, store, uncertainty as unc
from .model import MARGINS, TOTALS, OutcomeDist, fit, logit, novig_first

LEVELS = (0.5, 0.8, 0.9, 0.95)


def _decided_p(fc_row_home, fc_row_away):
    """P(home wins | not tied)."""
    s = fc_row_home + fc_row_away
    return np.where(s > 0, fc_row_home / s, np.nan)


def _game_mu(model, pred: pd.DataFrame, mode: str = "model") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per game: margin center (mu), total center, P(home | decided) under `model`."""
    b = forecast.batch(model, pred, None, mode)
    return b.mu_margin, b.mu_total, _decided_p(b.p_home, b.p_away)


# ---------------------------------------------------------------- dependence (method of moments)

def estimate_shocks(train: pd.DataFrame, mu: np.ndarray, mt: np.ndarray) -> dict:
    """Variances of team-season and season shocks in real residuals.

    margin residual r = result - mu; signed per team (home +r, away -r). Team-season mean over
    n games ~ u_T - mean(u_opp) + noise, so var(mean) ~ tau^2 (1 + 1/n) + sigma^2 / n.
    Season mean of r ~ h_s + noise (home-field shift). Totals likewise (team effects add,
    season effect removed first). Negative estimates are set to 0."""
    d = train.assign(r=train["result"].to_numpy(float) - mu, rt=train["total"].to_numpy(float) - mt)
    d = d.dropna(subset=["r", "rt"])
    s2, st2 = float(d["r"].var()), float(d["rt"].var())
    long = pd.concat([d.assign(team=d["home_team"], sr=d["r"]), d.assign(team=d["away_team"], sr=-d["r"])])
    ts = long.groupby(["season", "team"])["sr"].agg(["mean", "size"])
    n = float(ts["size"].mean())
    tau_u2 = max(0.0, (ts["mean"].var() - s2 / n) / (1 + 1 / n))
    ss = d.groupby("season")["r"].agg(["mean", "size"])
    tau_h2 = max(0.0, ss["mean"].var() - s2 / ss["size"].mean())
    lt = pd.concat([d.assign(team=d["home_team"]), d.assign(team=d["away_team"])])
    lt = lt.assign(rc=lt["rt"] - lt.groupby("season")["rt"].transform("mean"))
    tt = lt.groupby(["season", "team"])["rc"].agg(["mean", "size"])
    tau_w2 = max(0.0, (tt["mean"].var() - st2 / float(tt["size"].mean())) / (1 + 1 / float(tt["size"].mean())))
    st = d.groupby("season")["rt"].agg(["mean", "size"])
    tau_v2 = max(0.0, st["mean"].var() - st2 / st["size"].mean())
    return {"sd_margin_resid": s2 ** 0.5, "sd_total_resid": st2 ** 0.5, "tau_team_margin": tau_u2 ** 0.5,
            "tau_season_hfa": tau_h2 ** 0.5, "tau_team_total": tau_w2 ** 0.5, "tau_season_total": tau_v2 ** 0.5}


def _sample(pmfs: np.ndarray, support: np.ndarray, rng) -> np.ndarray:
    cdf = np.cumsum(pmfs, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(len(pmfs))[:, None]
    return support[(cdf < u).sum(axis=1).clip(0, len(support) - 1)]


def simulate_outcomes(train: pd.DataFrame, truth, mu: np.ndarray, mt: np.ndarray, shocks: dict,
                      rng: np.random.Generator) -> pd.DataFrame:
    seasons = train["season"].to_numpy(int)
    keys = sorted(set(zip(seasons, train["home_team"])) | set(zip(seasons, train["away_team"])))
    u = dict(zip(keys, rng.normal(0, shocks["tau_team_margin"], len(keys))))
    w = dict(zip(keys, rng.normal(0, shocks["tau_team_total"], len(keys))))
    us = sorted(set(seasons))
    h = dict(zip(us, rng.normal(0, shocks["tau_season_hfa"], len(us))))
    v = dict(zip(us, rng.normal(0, shocks["tau_season_total"], len(us))))
    m_shift = np.array([u[(s, a)] - u[(s, b)] + h[s] for s, a, b in zip(seasons, train["home_team"], train["away_team"])])
    t_shift = np.array([w[(s, a)] + w[(s, b)] + v[s] for s, a, b in zip(seasons, train["home_team"], train["away_team"])])
    post = (train["game_type"] != "REG").to_numpy() if "game_type" in train else np.zeros(len(train), bool)
    pm = np.array([truth.margin_dist.pmf(x) for x in mu + m_shift])
    pm[np.ix_(post, MARGINS == 0)] = 0.0
    pm /= pm.sum(axis=1, keepdims=True)
    pt = np.array([truth.total_dist.pmf(x) for x in mt + t_shift])
    return train.assign(result=_sample(pm, MARGINS, rng).astype(float), total=_sample(pt, TOTALS, rng).astype(float))


_S: dict = {}


def _sim_init(payload: dict) -> None:
    _S.update(payload)


def _sim_one(r: int) -> dict:
    P = _S
    rng = np.random.default_rng([P["seed"], 7, r])
    sim = simulate_outcomes(P["train"], P["truth"], P["mu"], P["mt"], P["shocks"], rng)
    point = fit(sim)
    _, _, p_hat = _game_mu(point, point.predict(P["test"]))
    out = {"p_hat": p_hat}
    for scheme in P["schemes"]:
        reps = unc.fit_replicates(sim, P["B"], P["seed"] + 1000 * (r + 1), scheme, workers=1, use_cache=False)
        lp = np.array([_game_mu(m, m.predict(P["test"]))[2] for m in reps])
        out[scheme] = np.std(logit(lp), axis=0, ddof=1)
    return out


def simulate(feat: pd.DataFrame, train_seasons: tuple[int, int], test_season: int, R: int, B: int,
             schemes=unc.SCHEMES, seed: int = 20261009, workers: int = 1, levels=LEVELS,
             shocks_override: dict | None = None) -> dict:
    """shocks_override: fixed shock SDs (sensitivity scenario) instead of the estimated ones."""
    played = feat[feat["result"].notna()]
    train = played[played["season"].between(*train_seasons)].reset_index(drop=True)
    test = played[played["season"] == test_season].reset_index(drop=True)
    t0 = time.time()
    truth = fit(train)
    mu, mt, _ = _game_mu(truth, truth.predict(train))
    ok = np.isfinite(mu) & np.isfinite(mt)
    train, mu, mt = train[ok].reset_index(drop=True), mu[ok], mt[ok]
    shocks = estimate_shocks(train, mu, mt)
    if shocks_override:
        shocks = {**shocks, **shocks_override, "override": True}
    _, _, p_true = _game_mu(truth, truth.predict(test))
    payload = {"train": train, "test": test, "truth": truth, "mu": mu, "mt": mt, "shocks": shocks,
               "schemes": list(schemes), "B": B, "seed": seed}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_sim_init, initargs=(payload,)) as ex:
            res = list(ex.map(_sim_one, range(R)))
    else:
        _sim_init(payload)
        res = [_sim_one(r) for r in range(R)]
    ph = np.array([x["p_hat"] for x in res])                 # R x games
    lph = logit(ph)
    true_sd = np.std(lph, axis=0, ddof=1)                     # sampling SD of the estimator, per game
    centre = lph.mean(axis=0)                                 # its expectation (logit)
    lt = logit(p_true)
    rows = []
    for scheme in schemes:
        sd = np.array([x[scheme] for x in res])               # R x games bootstrap SDs
        row = {"scheme": scheme, "sd_ratio (pooled)": float(np.sqrt(np.mean(sd ** 2) / np.mean(true_sd ** 2))),
               "sd_ratio (median game)": float(np.median(sd.mean(axis=0) / true_sd))}
        for lv in levels:
            z = unc.z_for(lv)
            lo, hi = lph - z * sd, lph + z * sd
            row[f"cover E[est] {lv:.0%}"] = float(np.mean((lo <= centre) & (centre <= hi)))
            row[f"cover truth {lv:.0%}"] = float(np.mean((lo <= lt) & (lt <= hi)))
        rows.append(row)
    return {"table": pd.DataFrame(rows), "shocks": shocks, "R": R, "B": B, "train_seasons": train_seasons,
            "test_season": test_season, "games": len(test), "seconds": round(time.time() - t0, 1),
            "true_sd_median_logit": float(np.median(true_sd)),
            "bias_median_logit": float(np.median(centre - lt)),
            "seed": seed}


# ---------------------------------------------------------------- real games, out of sample

def _intervals(pmf: np.ndarray, support: np.ndarray, level: float) -> np.ndarray:
    out = np.full((len(pmf), 2), np.nan)
    for i, p in enumerate(pmf):
        if np.all(np.isfinite(p)):
            out[i] = unc.pmf_interval(p, support, level)
    return out


def _plain(dist, mu: np.ndarray, post: np.ndarray | None = None) -> np.ndarray:
    """Normal pmf on the integers with the same center and scale, no key-number weights."""
    d = OutcomeDist(dist.sd, np.ones(len(dist.support)), dist.support)
    out = np.full((len(mu), len(dist.support)), np.nan)
    ok = np.isfinite(mu)
    if ok.any():
        out[ok] = forecast.pmf_matrix(d, mu[ok])
        if post is not None and (post & ok).any():
            m = post & ok
            out[np.ix_(m, dist.support == 0)] = 0.0
            out[m] /= out[m].sum(axis=1, keepdims=True)
    return out


def evaluate(feat: pd.DataFrame, seasons: list[int], settings, B: int, workers: int = 1,
             levels=LEVELS, use_cache: bool = True, verbose: bool = True) -> dict:
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    frames = []
    t0 = time.time()
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        test = played[played["season"] == s].reset_index(drop=True)
        point = fit(train)
        reps = unc.fit_replicates(train, B, settings.uncertainty_seed, settings.uncertainty_scheme,
                                  workers=workers, use_cache=use_cache)
        fc = forecast.forecast_slate(point, test, None, settings, reps, settings.uncertainty_level,
                                     pi_levels=levels)
        mk = forecast.forecast_slate(point, test, None, replace(settings, probability_model="market"), None,
                                     settings.uncertainty_level, pi_levels=levels)
        pred = point.predict(test)
        ro = forecast.batch(point, pred, None, "model", force_ratings_only=True)
        pb = forecast.batch(point, pred, None, settings.probability_model)
        post = forecast.postseason_mask(pred)
        fc["p_market_only"] = _decided_p(mk["p_home"], mk["p_away"])
        fc["p_ratings_only"] = _decided_p(ro.p_home, ro.p_away)
        fc["p_ml_novig"] = np.where(test["home_moneyline"].notna() & test["away_moneyline"].notna(),
                                    novig_first(test["home_moneyline"], test["away_moneyline"]), np.nan)
        tr = train[train["result"] != 0]
        fc["p_home_rate"] = float((tr["result"] > 0).mean())
        fc["result"], fc["total"] = test["result"].to_numpy(float), test["total"].to_numpy(float)
        # outcome-interval baselines: plug-in key-number pmf (no parameter uncertainty), and a plain
        # normal with the same mean and scale (no key numbers); ratings-only predictive intervals
        nm_pmf, nt_pmf = _plain(point.margin_dist, pb.mu_margin, post), _plain(point.total_dist, pb.mu_total)
        for lv in levels:
            tag = f"{round(lv * 100):d}"
            # plain normal, same center and scale (no key numbers)
            fc[[f"nmargin_lo_{tag}", f"nmargin_hi_{tag}"]] = _intervals(nm_pmf, MARGINS, lv)
            fc[[f"ntotal_lo_{tag}", f"ntotal_hi_{tag}"]] = _intervals(nt_pmf, TOTALS, lv)
            # ratings-only predictive distribution
            fc[[f"rmargin_lo_{tag}", f"rmargin_hi_{tag}"]] = _intervals(ro.pmf_margin, MARGINS, lv)
            fc[[f"rtotal_lo_{tag}", f"rtotal_hi_{tag}"]] = _intervals(ro.pmf_total, TOTALS, lv)
            # plug-in: point model's key-number distribution (no parameter uncertainty)
            fc[[f"pmargin_lo_{tag}", f"pmargin_hi_{tag}"]] = _intervals(pb.pmf_margin, MARGINS, lv)
            fc[[f"ptotal_lo_{tag}", f"ptotal_hi_{tag}"]] = _intervals(pb.pmf_total, TOTALS, lv)
        frames.append(fc)
        if verbose:
            print(f"  {s}: {len(test)} games, {len(reps)} replicates, {time.time() - t0:.0f}s", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["p_forecast"] = _decided_p(d["p_home"], d["p_away"])
    return {"games": d, "seasons": seasons, "B": B, "seconds": round(time.time() - t0, 1)}


PREDICTORS = {"forecast (market-informed + model)": "p_forecast", "market-only forecast (model edge 0)": "p_market_only",
              "closing moneyline no-vig": "p_ml_novig", "team ratings only": "p_ratings_only",
              "home-team base rate": "p_home_rate"}


def _scored(d: pd.DataFrame) -> pd.DataFrame:
    """Decided games where EVERY predictor has a probability (identical rows)."""
    x = d[d["result"] != 0].dropna(subset=list(PREDICTORS.values()))
    return x.assign(y=(x["result"] > 0).astype(float))


def score_table(d: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    x = _scored(d)
    rows = []
    for name, col in PREDICTORS.items():
        b, l = metrics.brier_logloss(x[col], x["y"])
        rows.append({"predictor": name, "games": len(x), "brier": b, "log_loss": l, "ece": metrics.ece(x[col], x["y"])})
    rows.append({"predictor": "coin flip", "games": len(x), "brier": 0.25, "log_loss": float(np.log(2)), "ece": np.nan})
    diff = x.assign(sq=(x["p_forecast"] - x["y"]) ** 2 - (x["p_ml_novig"] - x["y"]) ** 2)
    est, lo, hi = metrics.cluster_bootstrap(diff, "game_id", "sq", n=n_boot)
    out = pd.DataFrame(rows)
    out.attrs["forecast_minus_ml"] = (est, lo, hi)
    return out


def by_season(d: pd.DataFrame) -> pd.DataFrame:
    x = _scored(d)
    out = []
    for s, g in x.groupby("season"):
        r = {"season": s, "games": len(g)}
        for name, col in (("forecast", "p_forecast"), ("market-only", "p_market_only"), ("ML no-vig", "p_ml_novig"),
                          ("ratings only", "p_ratings_only")):
            r[f"Brier {name}"] = float(((g[col] - g["y"]) ** 2).mean())
        out.append(r)
    return pd.DataFrame(out)


def calibration(d: pd.DataFrame) -> pd.DataFrame:
    x = _scored(d)
    return metrics.calibration_table(x["p_forecast"], x["y"])


def by_range(d: pd.DataFrame) -> pd.DataFrame:
    """Brier by the forecast favorite's probability."""
    x = _scored(d)
    fav = np.maximum(x["p_forecast"], 1 - x["p_forecast"])
    bands = pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    out = []
    for band, g in x.groupby(bands, observed=True):
        f = np.maximum(g["p_forecast"], 1 - g["p_forecast"])
        won = np.where(g["p_forecast"] >= 0.5, g["y"], 1 - g["y"])
        out.append({"favorite prob": str(band), "games": len(g), "mean forecast": float(f.mean()),
                    "favorite won": float(won.mean()),
                    "Brier forecast": float(((g["p_forecast"] - g["y"]) ** 2).mean()),
                    "Brier ML no-vig": float(((g["p_ml_novig"] - g["y"]) ** 2).mean())})
    return pd.DataFrame(out)


def interval_table(d: pd.DataFrame, levels=LEVELS) -> pd.DataFrame:
    """Coverage and mean width of outcome prediction intervals, identical games per row group."""
    rows = []
    for name, actual, kinds in (("margin", "result", (("predictive mixture (method)", "margin"),
                                                      ("plug-in key-number", "pmargin"),
                                                      ("normal, no key numbers", "nmargin"),
                                                      ("team ratings only", "rmargin"))),
                                ("total", "total", (("predictive mixture (method)", "total"),
                                                    ("plug-in key-number", "ptotal"),
                                                    ("normal, no key numbers", "ntotal"),
                                                    ("team ratings only", "rtotal")))):
        for lv in levels:
            tag = f"{round(lv * 100):d}"
            cols = [f"{k}_{s}_{tag}" for _, k in kinds for s in ("lo", "hi")]
            x = d.dropna(subset=cols + [actual])
            for label, k in kinds:
                lo, hi = x[f"{k}_lo_{tag}"], x[f"{k}_hi_{tag}"]
                rows.append({"outcome": name, "level": f"{lv:.0%}", "interval": label, "games": len(x),
                             "coverage": float(((x[actual] >= lo) & (x[actual] <= hi)).mean()),
                             "mean width": float((hi - lo).mean())})
    return pd.DataFrame(rows)


def prob_interval_summary(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["p_home_lo", "p_home_hi"])
    w = x["p_home_hi"] - x["p_home_lo"]
    return pd.DataFrame([{"games": len(x), "level": f"{x['level'].iloc[0]:.0%}" if len(x) else "",
                          "median width (pct points)": float(np.median(w) * 100) if len(x) else np.nan,
                          "90th pct width": float(np.quantile(w, 0.9) * 100) if len(x) else np.nan,
                          "median SD (logit)": float(x["p_home_sd_logit"].median()) if len(x) else np.nan}])


# ---------------------------------------------------------------- selection, records, reports

PROTOCOL = Path(__file__).resolve().parent.parent / "docs" / "UNCERTAINTY.md"
SIMPLICITY = {"game": 0, "week4": 1, "season": 2, "season*week4": 3}


def select_scheme(table: pd.DataFrame, floor: float = 0.9) -> tuple[str, str]:
    """Pre-registered rule: among schemes with pooled SD ratio >= floor, the ratio closest to 1;
    if none reaches the floor, the highest ratio. Ties go to the simpler scheme."""
    t = table.assign(simple=table["scheme"].map(SIMPLICITY))
    ok = t[t["sd_ratio (pooled)"] >= floor]
    if len(ok):
        best = ok.assign(d=(ok["sd_ratio (pooled)"] - 1).abs().round(3)).sort_values(["d", "simple"]).iloc[0]
        return best["scheme"], (f"closest to 1 among ratios >= {floor} "
                                f"(ratio {best['sd_ratio (pooled)']:.2f})")
    best = t.sort_values(["sd_ratio (pooled)", "simple"], ascending=[False, True]).iloc[0]
    return best["scheme"], (f"no scheme reached {floor}; highest ratio {best['sd_ratio (pooled)']:.2f} "
                            "(intervals UNDERSTATE sampling variability)")


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() if PROTOCOL.exists() else ""


def past_records(period: str | None = None) -> list[dict]:
    from .experiment import LOG
    return [r["data"] for r in store.read(LOG)
            if r["data"].get("group") == "UNC" and (period is None or r["data"].get("period") == period)]


def record(rec: dict) -> None:
    from .experiment import LOG
    store.append(LOG, "experiment", [{"group": "UNC", "protocol_sha256": _protocol_sha(), **rec}])


def _md(df: pd.DataFrame, pct=(), f3=(), f4=()) -> str:
    from .validate import to_markdown
    fmt = {c: "{:.0f}" for c in df.columns if pd.api.types.is_integer_dtype(df[c])}
    fmt.update({c: "{:.1%}" for c in pct})
    fmt.update({c: "{:.3f}" for c in f3})
    fmt.update({c: "{:.4f}" for c in f4})
    return to_markdown(df, fmt)


def simulation_md(runs: list[tuple[str, dict]], chosen: tuple[str, str]) -> str:
    out = ["# Probability-interval simulation", "",
           "Protocol: `docs/UNCERTAINTY.md`. Truth = the point model fit on real training seasons; training "
           "outcomes simulated from its predictive distribution with team-season and season shocks; each "
           "simulation refits the point model and the bootstrap replicates under every block scheme and "
           "forecasts the real test-season games (closing consensus markets held fixed).", "",
           "* **SD ratio**: mean bootstrap SD / true sampling SD of the estimator (logit scale). 1 = right size, "
           "< 1 = intervals too narrow.",
           "* **cover E[est]**: coverage of the estimator's own expected value, i.e. of sampling variability only.",
           "* **cover truth**: coverage of the truth model's probability. Lower than cover E[est] when the "
           "estimator is biased (ridge shrinkage, calibration); the bootstrap does not correct bias.", "",
           "Limits: the truth is the model itself (well specified apart from estimation bias); market inputs are "
           "fixed and do not react to the shocks; the dependence is a simple random-effects approximation; real "
           "games also carry information neither the model nor this simulation contains.", ""]
    for name, r in runs:
        sh = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["shocks"].items()}
        out += [f"## {name}", "",
                f"Train {r['train_seasons'][0]}-{r['train_seasons'][1]}, test {r['test_season']} "
                f"({r['games']} games), {r['R']} simulations x {r['B']} replicates per scheme, seed {r['seed']}, "
                f"{r['seconds']:.0f}s.", "",
                f"Shock SDs (points): `{sh}`", "",
                f"True sampling SD of the estimator, median game: {r['true_sd_median_logit']:.4f} (logit); "
                f"median bias of its expectation vs truth: {r['bias_median_logit']:+.4f} (logit).", "",
                _md(r["table"], f3=[c for c in r["table"].columns if c != "scheme"]), ""]
    out += [f"**Selected scheme (pre-registered rule, estimated-dependence scenario): `{chosen[0]}`**, "
            f"{chosen[1]}.", ""]
    return "\n".join(out)


def validation_md(res: dict, settings, period: str) -> str:
    d = res["games"]
    st = score_table(d)
    est, lo, hi = st.attrs["forecast_minus_ml"]
    pis = prob_interval_summary(d)
    out = [f"# Game-forecast validation ({period}, seasons {res['seasons'][0]}-{res['seasons'][-1]})", "",
           "Chronological out of sample: each season is forecast by a point model and bootstrap replicates fit "
           "only on earlier seasons. Market inputs are the nflverse CLOSING consensus lines (the only historical "
           "market data), so the market-informed forecasts are evaluated at the market's best information set.", "",
           f"Uncertainty: `{unc.METHOD}` ({unc.METHOD_VERSION}), scheme `{settings.uncertainty_scheme}`, "
           f"{res['B']} replicates, seed {settings.uncertainty_seed}, level {settings.uncertainty_level:.0%}. "
           f"Runtime {res['seconds']:.0f}s.", "",
           "## Win probability: scoring on identical games", "",
           "P(home wins | not tied), decided games where every predictor has a value.", "",
           _md(st, f4=["brier", "log_loss", "ece"]), "",
           f"Forecast minus closing moneyline Brier: {est:+.5f} (95% CI {lo:+.5f} to {hi:+.5f}, game bootstrap); "
           "negative = forecast better.", "",
           "## By season (Brier)", "", _md(by_season(d), f4=[c for c in by_season(d).columns if c.startswith("Brier")]), "",
           "## By forecast favorite probability", "",
           _md(by_range(d), pct=["mean forecast", "favorite won"], f4=["Brier forecast", "Brier ML no-vig"]), "",
           "## Calibration (forecast, with counts)", "", _md(calibration(d), f3=["predicted", "actual"]), "",
           "## Actual margin and total: prediction-interval coverage and width", "",
           "Each interval is evaluated on the same games within an outcome/level group. Width in points.", "",
           _md(interval_table(d), pct=["coverage"], f3=["mean width"]), "",
           "## Width of the win-probability intervals", "",
           "These describe uncertainty in the ESTIMATED probability under this method (conditional on the closing "
           "market). They are narrow because the forecast stays close to the market; that does not make the game "
           "predictable and does not cover market or model misspecification. A single win or loss cannot test "
           "them; see the simulation report.", "",
           _md(pis, f3=["median width (pct points)", "90th pct width"], f4=["median SD (logit)"]), ""]
    return "\n".join(out)


def summary_record(res: dict, settings, period: str, model_version: str) -> dict:
    d = res["games"]
    st = score_table(d, n_boot=500)
    it = interval_table(d)
    return {"period": period, "seasons": res["seasons"], "model_version": model_version, "passed": None,
            "verdict": "reported (no acceptance test: uncertainty method evaluation)",
            "method": unc.METHOD_VERSION, "scheme": settings.uncertainty_scheme, "replicates": res["B"],
            "seed": settings.uncertainty_seed, "level": settings.uncertainty_level,
            "scores": st.round(6).to_dict("records"), "forecast_minus_ml_brier": list(st.attrs["forecast_minus_ml"]),
            "intervals": it.round(4).to_dict("records"),
            "prob_interval_widths": prob_interval_summary(d).round(5).to_dict("records")}


__all__ = ["simulate", "evaluate", "score_table", "by_season", "calibration", "by_range", "interval_table",
           "prob_interval_summary", "estimate_shocks", "select_scheme", "simulation_md", "validation_md",
           "summary_record", "record", "past_records", "LEVELS"]
