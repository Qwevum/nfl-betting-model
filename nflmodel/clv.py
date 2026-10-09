"""Opening lines and closing-line value (backtest). Needs histlines.load() output.

Walk-forward: each test season uses a model fit on earlier seasons only, and the game's
pre-game ratings (built from games already played; the opener is posted after the previous
week's games). Caveats, stated in the report: the starting QB used is the actual starter
(possibly unknown when the line opened), and the closing line used for CLV is the file's.

1. Does the line move toward the model? OLS of (close - open) on (model - open), where
   "model" is (a) the ratings-only own line and (b) the blend recomputed AT THE OPENING LINE.
   Slope 0: the market ignores what the model knows; slope 1: the close moves all the way to
   the model. Game-bootstrap CI on the slope.
2. Bet at the open when EV > min_edge: probabilities from the live offset calibration anchored
   at the opening line and opening no-vig pair (-110/-110 when the file has no line odds),
   edge = blend at the open line minus the open line. Reported: count, CLV in points and in
   no-vig probability (closing distribution evaluated at the line bet), and flat ROI at the
   opening price and at -110, each with a game-bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .decide import AnchoredPricer
from .market import _invert
from .model import fit, novig_first
from .odds import decimal, ev

LIM = (-60.0, 60.0)


def _boot_slope(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    def slope(a, b):
        a = a - a.mean()
        return float(np.sum(a * (b - b.mean())) / np.sum(a * a)) if np.sum(a * a) > 0 else np.nan
    rng = np.random.default_rng(seed)
    est = slope(x, y)
    reps = [slope(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return est, float(lo), float(hi)


def game_rows(feat: pd.DataFrame, lines: pd.DataFrame, seasons: list[int], settings) -> pd.DataFrame:
    """One row per matched game: open/close, own line, blend at open, and both sides' EV at the open."""
    played = feat[feat["result"].notna()]
    first = int(played["season"].min())
    out = []
    for s in seasons:
        train = played[(played["season"] < s) & (played["season"] > first)]
        test = played[played["season"] == s].merge(lines.drop(columns=["season"]), on="game_id")
        test = test.dropna(subset=["open_line", "close_line"])
        if test.empty or train["season"].nunique() < 4:
            continue
        model = fit(train)
        at_open = test.assign(spread_line=test["open_line"],
                              home_spread_odds=test["home_line_odds_open"].fillna(-110.0),
                              away_spread_odds=test["away_line_odds_open"].fillna(-110.0))
        pred = model.predict(at_open)
        for g, raw in zip(pred.itertuples(index=False), test.itertuples(index=False)):
            pr = AnchoredPricer(model, g, mode=settings.probability_model)
            row = {"game_id": g.game_id, "season": s, "result": g.result, "open": raw.open_line,
                   "close": raw.close_line, "own_line": g.model_margin, "blend_open": g.blend_margin,
                   "odds_from_file": pd.notna(raw.home_line_odds_open) and pd.notna(raw.away_line_odds_open)}
            for side, point, price in (("home", -raw.open_line, at_open_price(raw, "home")),
                                       ("away", raw.open_line, at_open_price(raw, "away"))):
                pw, pp = pr.probs("spread", side, point)
                row[f"ev_{side}"], row[f"pw_{side}"], row[f"pp_{side}"], row[f"price_{side}"] = ev(pw, pp, price), pw, pp, price
            row["p_open_home"] = float(novig_first(at_open_price(raw, "home"), at_open_price(raw, "away")))
            close_h = raw.home_line_odds_close if pd.notna(raw.home_line_odds_close) else -110.0
            close_a = raw.away_line_odds_close if pd.notna(raw.away_line_odds_close) else -110.0
            mu_c = _invert(model.margin_dist, raw.close_line, float(novig_first(close_h, close_a)), *LIM)
            over, push, under = model.margin_dist.prob_over(mu_c, raw.open_line)
            row["p_close_home_at_open"] = over / (over + under)    # closing view of the OPEN home bet (no push)
            out.append(row)
    return pd.DataFrame(out)


def at_open_price(raw, side: str) -> float:
    v = raw.home_line_odds_open if side == "home" else raw.away_line_odds_open
    return float(v) if pd.notna(v) else -110.0


def line_move(rows: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    out = []
    y = (rows["close"] - rows["open"]).to_numpy(float)
    for name, col in (("ratings-only own line", "own_line"), ("blend at the opening line", "blend_open")):
        ok = rows[col].notna().to_numpy()
        x = (rows[col] - rows["open"]).to_numpy(float)[ok]
        est, lo, hi = _boot_slope(x, y[ok], n)
        out.append({"model": name, "games": int(ok.sum()), "slope": est, "lo": lo, "hi": hi,
                    "CI includes 0": lo <= 0 <= hi, "mean |model - open|": float(np.mean(np.abs(x)))})
    return pd.DataFrame(out)


def bets(rows: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    """The better side of each game if its EV at the open exceeds min_edge, graded and with CLV."""
    out = []
    for r in rows.itertuples(index=False):
        side = "home" if r.ev_home >= r.ev_away else "away"
        e = getattr(r, f"ev_{side}")
        if e <= min_edge:
            continue
        home = side == "home"
        margin = r.result
        v = (margin - r.open) if home else (r.open - margin)
        res = "W" if v > 0 else ("P" if v == 0 else "L")
        price = getattr(r, f"price_{side}")
        p_open = r.p_open_home if home else 1 - r.p_open_home
        p_close = r.p_close_home_at_open if home else 1 - r.p_close_home_at_open
        out.append({"game_id": r.game_id, "season": r.season, "side": side, "line": -r.open if home else r.open,
                    "price": price, "ev": e, "result": res,
                    "clv_points": (r.close - r.open) if home else (r.open - r.close),
                    "clv_prob": p_close - p_open,
                    "profit_open": {"W": decimal(price) - 1, "P": 0.0, "L": -1.0}[res],
                    "profit_110": {"W": 100 / 110, "P": 0.0, "L": -1.0}[res]})
    return pd.DataFrame(out, columns=["game_id", "season", "side", "line", "price", "ev", "result", "clv_points",
                                      "clv_prob", "profit_open", "profit_110"])


def bet_summary(b: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    if b.empty:
        return pd.DataFrame([{"metric": "bets", "value": 0, "95% CI": "n/a (no side cleared the threshold)"}])
    rows = [{"metric": "bets", "value": len(b), "95% CI": ""},
            {"metric": "W-L-P", "value": f"{(b.result == 'W').sum()}-{(b.result == 'L').sum()}-{(b.result == 'P').sum()}",
             "95% CI": ""}]
    for col, name in (("clv_points", "mean CLV (points)"), ("clv_prob", "mean CLV (no-vig probability)"),
                      ("profit_open", "ROI at the opening price"), ("profit_110", "ROI at -110")):
        est, lo, hi = metrics.cluster_bootstrap(b, "game_id", col, n=n)
        rows.append({"metric": name, "value": est, "95% CI": f"{lo:+.4f} to {hi:+.4f}"})
    return pd.DataFrame(rows)


__all__ = ["game_rows", "line_move", "bets", "bet_summary"]
