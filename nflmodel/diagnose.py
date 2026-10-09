"""Why each candidate is (not) a bet: categories, overlapping reasons, a sequential funnel,
the EV distribution, and the conditional watchlist.

Every flag is derived from the decision row itself (decide.decide_game + the completion
recheck), and the funnel's last stage equals the number of BET rows by construction.

Categories (one per side, first match wins):
  actionable                           decision == BET: every requirement met
  negative estimated value             EV <= 0 at the best price
  small positive value below threshold 0 < EV <= min_edge, or EV gone if the fair line is 0.5 pt worse
  blocked by rule                      EV clears, but a decision rule blocks it (gap rule, starter
                                       Out/Questionable, price only at a book you don't use)
  insufficient information             EV clears, but something needed is missing: a timestamped
                                       quote, a live reference without the book, a confirmed
                                       starter, a weather forecast, or quotes went stale/kickoff
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decide import BOOK_SOURCES_WITH_TIME, _fmt_price

CATEGORIES = ["actionable", "negative estimated value", "small positive value below threshold",
              "blocked by rule", "insufficient information"]

FLAGS = {
    "ev_not_positive": "EV <= 0",
    "ev_below_threshold": "0 < EV <= threshold",
    "too_sensitive": "EV gone if fair line 0.5 pt worse",
    "gap_rule": "model vs market gap rule",
    "starter_out": "projected starter Out/Questionable/unlisted",
    "missing_weather": "no weather forecast (outdoor total)",
    "other_block": "other rule",
    "not_actionable_book": "best price at a book you don't use",
    "no_live_quote": "no timestamped bookmaker quote",
    "no_live_reference": "no live reference without the book",
    "uncertain_starter": "starter not confirmed",
    "stale_or_kickoff": "stale at completion / kickoff passed",
}

# Sequential funnel: (stage label, flags that must be clear to survive it)
FUNNEL = [
    ("EV > 0", ["ev_not_positive"]),
    ("EV above threshold", ["ev_below_threshold"]),
    ("robust to a 0.5-pt error", ["too_sensitive"]),
    ("no rule blocks (gap, starter out, weather, other)", ["gap_rule", "starter_out", "missing_weather",
                                                          "other_block"]),
    ("timestamped bookmaker quote", ["no_live_quote"]),
    ("live reference without that book", ["no_live_reference"]),
    ("starters confirmed", ["uncertain_starter"]),
    ("at a book you can use", ["not_actionable_book"]),
    ("still valid at completion", ["stale_or_kickoff"]),
]


def _s(x) -> str:
    return x if isinstance(x, str) else ""


def flags(rows: pd.DataFrame, settings) -> pd.DataFrame:
    """One boolean column per FLAGS key, aligned with rows."""
    r = rows
    ev = r["ev"].astype(float)
    sens = r["ev[fair 0.5 worse]"].astype(float)
    blocks = r.get("blocks", pd.Series("", index=r.index)).map(_s)
    reasons = r["reasons"].map(_s)
    f = pd.DataFrame(index=r.index)
    f["ev_not_positive"] = ev <= 0
    f["ev_below_threshold"] = (ev > 0) & (ev <= settings.min_edge)
    f["too_sensitive"] = (ev > settings.min_edge) & (sens <= 0)
    f["gap_rule"] = blocks.str.contains("differs from market")
    f["starter_out"] = blocks.str.contains("starter|starting QB unknown", regex=True)
    f["missing_weather"] = blocks.str.contains("weather")
    f["other_block"] = (blocks != "") & ~(f["gap_rule"] | f["starter_out"] | f["missing_weather"])
    f["not_actionable_book"] = ~r.get("venue_ok", pd.Series(True, index=r.index)).fillna(True).astype(bool)
    f["no_live_quote"] = ~r["price_source"].isin(BOOK_SOURCES_WITH_TIME)
    f["no_live_reference"] = ~r["reference_live"].fillna(False).astype(bool)
    f["uncertain_starter"] = r.get("conditions", pd.Series("", index=r.index)).map(_s) != ""
    others = f.drop(columns=[]).any(axis=1)
    f["stale_or_kickoff"] = (r["decision"] != "BET") & ~others | reasons.str.contains(
        "stale by completion|expired by completion|kickoff passed", regex=True)
    return f


def categorize(rows: pd.DataFrame, settings) -> pd.Series:
    f = flags(rows, settings)
    cat = pd.Series("insufficient information", index=rows.index)
    blocked = f[["gap_rule", "starter_out", "other_block", "not_actionable_book"]].any(axis=1)
    cat[blocked] = "blocked by rule"
    cat[f["ev_below_threshold"] | f["too_sensitive"]] = "small positive value below threshold"
    cat[f["ev_not_positive"]] = "negative estimated value"
    cat[rows["decision"] == "BET"] = "actionable"
    return cat


def overlap(rows: pd.DataFrame, settings) -> pd.DataFrame:
    """How many sides carry each reason (a side can carry several)."""
    f = flags(rows, settings)
    return pd.DataFrame({"reason": [FLAGS[k] for k in f.columns], "sides": [int(f[k].sum()) for k in f.columns]})


def funnel(rows: pd.DataFrame, settings) -> pd.DataFrame:
    f = flags(rows, settings)
    alive = pd.Series(True, index=rows.index)
    out = [{"stage": "candidate sides", "remaining": int(alive.sum()), "removed": 0}]
    for label, keys in FUNNEL:
        before = int(alive.sum())
        alive &= ~f[keys].any(axis=1)
        out.append({"stage": label, "remaining": int(alive.sum()), "removed": before - int(alive.sum())})
    return pd.DataFrame(out)


def ev_distribution(rows: pd.DataFrame, settings) -> pd.DataFrame:
    out = []
    for market, g in list(rows.groupby("market")) + [("all", rows)]:
        e = g["ev"].astype(float)
        out.append({"market": market, "sides": len(g), "min": e.min(), "p25": e.quantile(.25),
                    "median": e.median(), "p75": e.quantile(.75), "max": e.max(),
                    "> 0": int((e > 0).sum()), f"> {settings.min_edge:.0%}": int((e > settings.min_edge).sum())})
    return pd.DataFrame(out)


def watchlist(rows: pd.DataFrame, settings) -> pd.DataFrame:
    """The closest non-actionable candidates (best side per market), highest EV first.

    Prices here are CONDITIONAL on the current probability estimate. They are not
    recommendations; a later quote needs a fresh run before it can be acted on."""
    if rows.empty or settings.watchlist_size == 0:
        return rows.iloc[0:0]
    best = rows[rows["is_best_side"].astype(bool) & (rows["decision"] != "BET")].copy()
    if best.empty:
        return best
    best["category"] = categorize(best, settings)
    f = flags(best, settings)
    best["why"] = [", ".join(FLAGS[k] for k in f.columns if f.loc[i, k]) for i in best.index]
    best["needs_price"] = best["min_price"].map(_fmt_price)
    return best.sort_values("ev", ascending=False).head(settings.watchlist_size)


def primary_cause(rows: pd.DataFrame, settings) -> str:
    """Plain-language attribution of why there are no actionable bets."""
    if rows.empty:
        return "no priced candidates"
    f = flags(rows, settings)
    n = len(rows)
    ev_fail = int((f["ev_not_positive"] | f["ev_below_threshold"] | f["too_sensitive"]).sum())
    info_fail = int((f["no_live_quote"] | f["no_live_reference"]).sum())
    if ev_fail == n and info_fail == n:
        return ("prices and data coverage together: no side clears the EV threshold, and none has a timestamped "
                "multi-book quote, so even a large edge could not be executable")
    if ev_fail == n:
        return "prices: no side clears the EV threshold at the available prices"
    if info_fail == n:
        return "data coverage: sides with enough EV lack a timestamped quote or a live reference"
    return "a mix of prices and decision rules; see the funnel"


__all__ = ["CATEGORIES", "FLAGS", "flags", "categorize", "overlap", "funnel", "ev_distribution", "watchlist",
           "primary_cause", "np"]
