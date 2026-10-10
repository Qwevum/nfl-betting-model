"""OL1: bet the opening spread toward the model (protocol: docs/EXPERIMENTS.md, section OL1).

Historical only; nothing here changes live predictions or decisions.

g = own line - open (ratings-only margin of the season's walk-forward model, nflverse
convention: positive = home better). If |g| >= T, bet the model's side at the OPEN line.
The expected move a + b*g uses (a, b) fit on EARLIER seasons only; the expected close is
treated as fair to turn it into a predicted cover probability with the season's
key-number margin distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import _invert
from .model import fit
from .odds import decimal

DEV = list(range(2015, 2022))
HOLDOUT_FIRST = 2022
THRESHOLDS = (2.0, 3.0, 4.0)
KEYS = (-7.0, -3.0, 3.0, 7.0)
MIN_BETS = 100
EXCLUDE_NOTE = "data supply issue"
OPEN_TIMING_FLAG = ("The file does not say when its 'open' line was recorded (no timestamps; one bookmaker per era: "
                    "Pinnacle to Feb 2018, bet365 Sep 2018 - Feb 2025, betr from Sep 2025). It cannot be confirmed to "
                    "fall in the live early window (168-84 h before kickoff). If it was posted earlier (look-ahead "
                    "lines), the model knew more than the opener and these results are flattered.")


def source_era(season: int) -> str:
    return "Pinnacle" if season <= 2017 else ("bet365" if season <= 2024 else "betr")


def season_rows(feat: pd.DataFrame, lines: pd.DataFrame, seasons: list[int], min_train: int = 4):
    """(rows, models): one row per game with an opening and closing line, from the walk-forward
    model of its season (fit on earlier seasons only, >= min_train training seasons)."""
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    ln = lines.drop(columns=[c for c in ("season",) if c in lines])
    if "notes" in ln:
        ln = ln[~ln["notes"].astype(str).str.lower().str.contains(EXCLUDE_NOTE, na=False)]
    out, models = [], {}
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        if train["season"].nunique() < min_train:
            continue
        test = played[played["season"] == s].merge(ln, on="game_id").dropna(subset=["open_line", "close_line"])
        if test.empty:
            continue
        model = fit(train)
        models[s] = model
        pred = model.predict(test)
        out.append(pd.DataFrame({
            "game_id": pred["game_id"].to_numpy(), "season": s, "result": pred["result"].to_numpy(float),
            "open": test["open_line"].to_numpy(float), "close": test["close_line"].to_numpy(float),
            "own_line": pred["model_margin"].to_numpy(float),
            "home_qb_delta": pred["home_qb_delta"].to_numpy(float) if "home_qb_delta" in pred else np.nan,
            "away_qb_delta": pred["away_qb_delta"].to_numpy(float) if "away_qb_delta" in pred else np.nan,
            "open_price_home": test["home_line_odds_open"].to_numpy(float),
            "open_price_away": test["away_line_odds_open"].to_numpy(float)}))
    rows = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(rows):
        rows["g"] = rows["own_line"] - rows["open"]
        rows = rows.dropna(subset=["g"])
    return rows, models


def fit_move(rows: pd.DataFrame) -> tuple[float, float]:
    """OLS (a, b) of close - open on g."""
    x = rows["g"].to_numpy(float)
    y = (rows["close"] - rows["open"]).to_numpy(float)
    xc = x - x.mean()
    b = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc)) if np.sum(xc * xc) > 0 else 0.0
    return float(y.mean() - b * x.mean()), b


def move_params(rows: pd.DataFrame, seasons: list[int]) -> dict:
    """{season: (a, b)} fit on all rows from EARLIER seasons (expanding window)."""
    out = {}
    for s in seasons:
        prior = rows[rows["season"] < s]
        if len(prior) >= 50:
            out[s] = fit_move(prior)
    return out


def crosses_key(open_line: float, exp_close: float) -> bool:
    return any((open_line < k <= exp_close) or (exp_close <= k < open_line) for k in KEYS)


def _profit(res: str, american: float) -> float:
    if res == "P":
        return 0.0
    if res == "L":
        return -1.0
    return decimal(american) - 1 if np.isfinite(american) else np.nan


def bets(rows: pd.DataFrame, T: float, moves: dict, models: dict) -> pd.DataFrame:
    """One bet per game with |g| >= T, at the opening line, graded and with predictions."""
    out = []
    for r in rows.itertuples(index=False):
        if abs(r.g) < T or r.season not in models:
            continue
        home = r.g > 0
        v = (r.result - r.open) if home else (r.open - r.result)
        res = "W" if v > 0 else ("P" if v == 0 else "L")
        if r.season in moves:   # expected-move metrics need earlier seasons (none for the first season)
            a, b = moves[r.season]
            exp_close = r.open + a + b * r.g
            dist = models[r.season].margin_dist
            mu = _invert(dist, exp_close, 0.5, -60.0, 60.0)
            over, push, under = dist.prob_over(mu, r.open)
            p_home = over / (over + under)
        else:
            a = b = exp_close = push = p_home = np.nan
        price_open = r.open_price_home if home else r.open_price_away
        qd = (abs(r.home_qb_delta) < 0.02) and (abs(r.away_qb_delta) < 0.02)
        out.append({"game_id": r.game_id, "season": r.season, "era": source_era(r.season),
                    "side": "home" if home else "away", "g": r.g, "open": r.open, "close": r.close,
                    "result": res, "profit_110": _profit(res, -110.0), "profit_105": _profit(res, -105.0),
                    "profit_open": _profit(res, price_open) if np.isfinite(price_open) else np.nan,
                    "clv_points": (r.close - r.open) if home else (r.open - r.close),
                    "exp_move": a + b * r.g, "exp_close": exp_close, "crosses_key": crosses_key(r.open, exp_close) if np.isfinite(exp_close) else None,
                    "p_cover_pred": p_home if home else 1 - p_home, "p_push_pred": push, "usual_qbs": bool(qd)})
    return pd.DataFrame(out)


def _boot(x: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize(b: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Bets, record, cover rate, ROI at -110 / -105 / the file's open price, CLV and predicted cover,
    each with a 95% bootstrap CI (one bet per game)."""
    if b.empty:
        return {"bets": 0}
    dec = b[b["result"] != "P"]
    cover = (dec["result"] == "W").to_numpy(float)
    out = {"bets": int(len(b)), "W": int((b.result == "W").sum()), "L": int((b.result == "L").sum()),
           "P": int((b.result == "P").sum())}
    for key, x in (("cover", cover), ("roi_110", b["profit_110"]), ("roi_105", b["profit_105"]),
                   ("roi_open", b["profit_open"]), ("clv_points", b["clv_points"]),
                   ("pred_cover", dec["p_cover_pred"])):
        est, lo, hi = _boot(np.asarray(x, float), n, seed)
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = est, lo, hi
    out["open_price_bets"] = int(b["profit_open"].notna().sum())
    out["mean_exp_move"] = float(b["exp_move"].abs().mean())
    return out


def choose_threshold(by_T: dict) -> float | None:
    """Highest ROI at -110 among thresholds with >= MIN_BETS bets; ties -> larger T."""
    ok = [(s["roi_110"], T) for T, s in by_T.items() if s.get("bets", 0) >= MIN_BETS]
    if not ok:
        return None
    best = max(r for r, _ in ok)
    return max(T for r, T in ok if r == best)


def breakdown(b: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Descriptive splits: key-number crossing, source era, season, usual QBs."""
    rows = []
    for name, col in (("crosses 3 or 7", "crosses_key"), ("era", "era"), ("season", "season"),
                      ("both usual QBs", "usual_qbs")):
        for val, g in b.groupby(col):
            s = summarize(g, n)
            rows.append({"split": name, "group": str(val), **{k: s.get(k) for k in (
                "bets", "W", "L", "P", "cover", "cover_lo", "cover_hi", "pred_cover", "roi_110", "roi_110_lo",
                "roi_110_hi", "roi_105", "clv_points")}})
    return pd.DataFrame(rows)


__all__ = ["season_rows", "fit_move", "move_params", "crosses_key", "bets", "summarize", "choose_threshold",
           "breakdown", "source_era", "DEV", "HOLDOUT_FIRST", "THRESHOLDS", "OPEN_TIMING_FLAG"]
