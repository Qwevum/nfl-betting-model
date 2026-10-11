"""OL1-paper: prospective paper tracking of the OL1 rule (protocol: docs/EXPERIMENTS.md, OL1-paper).

Separate from live decisions in both directions: nothing here changes a BET / NO BET, and the
live rules (including gap_points) do not suppress these records. No stakes.

At each `collect` run, for games 84-168 h before kickoff with a multi-book spread reference
from that run's validated quotes: g = ratings-only own line - reference line; if |g| >= 4,
record the side the model favors with the best price available across books. The earliest
record per game is graded on CLV against the last valid pre-kickoff reference (closing.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import closing, store
from .odds import decimal, implied_probability

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "logs" / "paper_forecasts.jsonl"
KIND = "OL1-paper"
GAP = 4.0
LOOKS = (50, 100)


def _best_quote(q: pd.DataFrame, side: str) -> dict | None:
    """Best line for the side (most points), then the best price at that line."""
    s = q[(q["market"] == "spread") & (q["side"] == side)].dropna(subset=["point", "price"])
    if s.empty:
        return None
    s = s[s["point"] == s["point"].max()]
    r = s.loc[s["price"].idxmax()]
    return {"book": r["book"], "point": float(r["point"]), "price": float(r["price"]),
            "quote_utc": pd.Timestamp(r["odds_time_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ")}


def candidates(preds: pd.DataFrame, valid: pd.DataFrame, now: pd.Timestamp, settings) -> tuple[pd.DataFrame, list]:
    """(OL1-paper records, notes). preds: model.predict rows (model_margin = own line, kickoff_utc).
    valid: this run's validated quotes (odds_time_utc)."""
    lo, hi = settings.early_horizon_min_hours, settings.early_horizon_max_hours
    out, notes = [], []
    for g in preds.itertuples(index=False):
        if pd.isna(g.kickoff_utc):
            continue
        lead = (g.kickoff_utc - now).total_seconds() / 3600
        if not (lo <= lead <= hi):
            continue
        q = valid[valid["game_id"] == g.game_id] if len(valid) else valid
        ref = closing._reference(q, "spread", settings.min_reference_books) if len(q) else None
        if ref is None:
            notes.append(f"{g.game_id}: no multi-book spread reference in this run")
            continue
        if pd.isna(g.model_margin):
            notes.append(f"{g.game_id}: ratings incomplete")
            continue
        gap = float(g.model_margin - ref["line"])
        if abs(gap) < GAP:
            continue
        side = "home" if gap > 0 else "away"
        best = _best_quote(q, side)
        if best is None:
            notes.append(f"{g.game_id}: no {side} spread quote")
            continue
        bet_line = -best["point"] if side == "home" else best["point"]          # nflverse convention
        p_over = ref["p"] if bet_line == ref["line"] else closing._shift_prob(ref["p"], ref["line"], bet_line)
        out.append({"game_id": g.game_id, "season": int(g.season), "week": int(g.week),
                    "away_team": g.away_team, "home_team": g.home_team,
                    "kickoff_utc": g.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "lead_hours": round(lead, 2),
                    "market": "spread", "side": side, "team": g.home_team if side == "home" else g.away_team,
                    "own_line": float(g.model_margin), "reference_line": ref["line"], "reference_p_home": ref["p"],
                    "reference_books": int(ref["n_books"]),
                    "reference_newest_utc": pd.Timestamp(ref["as_of_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "gap": gap, **best, "market_prob": p_over if side == "home" else 1 - p_over,
                    "market_prob_method": "exact" if bet_line == ref["line"] else "normal approximation (line differs)",
                    "implied_at_price": implied_probability(best["price"])})
    return pd.DataFrame(out), notes


def record(cands: pd.DataFrame, run: dict, path: Path = STORE) -> list[dict]:
    if cands.empty:
        return []
    return store.append(path, KIND, [{**r, **run} for r in cands.to_dict("records")])


def frame(path: Path = STORE) -> pd.DataFrame:
    recs = [r for r in store.read(path) if r.get("kind") == KIND]
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame([{**r["data"], "recorded_utc": r["recorded_utc"], "hash": r["hash"]} for r in recs])


def graded(fc: pd.DataFrame, refs: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Earliest record per game, with CLV against the last pre-kickoff reference and the result."""
    if fc.empty:
        return fc
    first = fc.sort_values("recorded_utc").groupby("game_id").head(1).reset_index(drop=True)
    c = closing.clv(first, refs)
    res = games.set_index("game_id")[["home_score", "away_score"]]
    out = []
    for r in c.itertuples(index=False):
        hs, as_ = res.loc[r.game_id] if r.game_id in res.index else (np.nan, np.nan)
        if pd.isna(hs):
            out.append(("", np.nan))
            continue
        margin = hs - as_
        v = (margin + r.point) if r.side == "home" else (-margin + r.point)
        w = "W" if v > 0 else ("P" if v == 0 else "L")
        out.append((w, {"W": decimal(r.price) - 1, "P": 0.0, "L": -1.0}[w]))
    return c.assign(result=[x for x, _ in out], profit=[y for _, y in out])


def _ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def decision(n: int, lo: float, hi: float) -> str:
    """Pre-registered rule: formal looks at n = 50 and n = 100 only."""
    if n < LOOKS[0]:
        return f"collecting ({n}/{LOOKS[0]} graded); no conclusion before the first look"
    look = LOOKS[1] if n >= LOOKS[1] else LOOKS[0]
    if np.isfinite(hi) and hi < 0:
        return f"look at n={look}: CI entirely below 0 -> retire OL1"
    if np.isfinite(lo) and lo > 0:
        return (f"look at n={look}: CI above 0 -> small real stakes may be considered, ONLY at -105 or better "
                "(log them with `place`); paper tracking continues")
    if look == LOOKS[1]:
        return f"look at n={look}: CI includes 0 -> retire OL1"
    return f"look at n={look}: CI includes 0 -> continue to n={LOOKS[1]}"


def summary(g: pd.DataFrame) -> dict:
    """Running CLV summary of graded forecasts (CLV available), in kickoff order."""
    x = g.dropna(subset=["clv_prob"]).sort_values("kickoff_utc") if len(g) else g
    n = len(x)
    # the decision uses the sample as of the most recent formal look, not later forecasts
    look_n = LOOKS[1] if n >= LOOKS[1] else (LOOKS[0] if n >= LOOKS[0] else n)
    at_look = x.iloc[:look_n] if n else x
    p, plo, phi = _ci(x["clv_prob"].to_numpy()) if n else (np.nan,) * 3
    pts, tlo, thi = _ci(x["clv_points"].to_numpy()) if n else (np.nan,) * 3
    _, llo, lhi = _ci(at_look["clv_prob"].to_numpy()) if n else (np.nan,) * 3
    settled = x[x["result"].isin(["W", "L", "P"])] if n else x
    return {"recorded_games": int(len(g)), "graded_with_clv": n, "mean_clv_prob": p, "clv_prob_lo": plo,
            "clv_prob_hi": phi, "mean_clv_points": pts, "clv_points_lo": tlo, "clv_points_hi": thi,
            "settled": int(len(settled)),
            "record": (f"{(settled.result == 'W').sum()}-{(settled.result == 'L').sum()}-{(settled.result == 'P').sum()}"
                       if len(settled) else "0-0-0"),
            "roi_at_recorded_price": float(settled["profit"].mean()) if len(settled) else np.nan,
            "decision": decision(n if n < LOOKS[0] else look_n, llo, lhi)}


__all__ = ["candidates", "record", "frame", "graded", "summary", "decision", "STORE", "KIND", "GAP", "LOOKS"]
