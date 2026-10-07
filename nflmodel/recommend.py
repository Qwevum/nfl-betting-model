"""Daily +EV recommendations: model probability vs the book's implied probability.

For every offer (book, market, side, line, price):
  implied   = break-even probability of the price          (odds.implied_probability)
  model     = model's probability the bet wins, among outcomes that don't push
  edge      = expected profit per unit staked = p_win * (decimal - 1) - p_lose
  kelly     = full-Kelly bankroll fraction (b*p - q) / b

A bet is flagged when edge > min_edge (default 2%). For a bet with no push,
edge > 0 exactly when model > implied, and edge = model * decimal - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate
from .backtest import grade, profit
from .model import FittedModel
from .odds import implied_probability

KELLY_FRACTION = evaluate.KELLY_FRACTION
MAX_STAKE_UNITS = evaluate.MAX_STAKE_UNITS


def compare_offers(model: FittedModel, preds: pd.DataFrame, offers: pd.DataFrame,
                   min_edge: float = 0.02) -> pd.DataFrame:
    """Best book per game/market/side, with implied vs model probability and the flag."""
    best = evaluate.evaluate_offers(model, preds, offers)
    if best.empty:
        return best
    best["implied"] = best["price"].map(implied_probability)
    decided = 1 - best["p_push"]
    best["model"] = np.where(decided > 0, best["p_win"] / decided, np.nan)
    best["edge"] = best["ev"]
    best["stake_units"] = np.minimum(best["kelly"] * KELLY_FRACTION * 100, MAX_STAKE_UNITS).round(2)

    p = preds.set_index("game_id")
    gap = []
    for r in best.itertuples(index=False):
        g = p.loc[r.game_id]
        if r.market == "total":
            gap.append(abs(g.model_total - g.total_line) if pd.notna(g.total_line) else 0.0)
        else:
            gap.append(abs(g.model_margin - g.spread_line) if pd.notna(g.spread_line) else 0.0)
    best["model_vs_market"] = gap
    best["flag"] = np.where(best["edge"] <= min_edge, "",
                   np.where(best["model_vs_market"] >= evaluate.CHECK_NEWS_POINTS, "CHECK NEWS", "+EV"))

    cols = ["game_id", "gameday", "gametime", "away_team", "home_team", "home_score", "away_score"]
    return best.merge(preds[[c for c in cols if c in preds.columns]], on="game_id")


def add_results(rec: pd.DataFrame) -> pd.DataFrame:
    """For past dates: grade each bet and its units at the recommended stake."""
    rec = rec.copy()
    done = rec["home_score"].notna()
    rec["result"] = ""
    rec["units"] = np.nan
    if done.any():
        rec.loc[done, "result"] = [grade(r.market, r.side, r.point, r.home_score, r.away_score)
                                   for r in rec[done].itertuples(index=False)]
        rec.loc[done, "units"] = [profit(res, price) * st for res, price, st in
                                  zip(rec.loc[done, "result"], rec.loc[done, "price"],
                                      rec.loc[done, "stake_units"])]
    return rec


# ---------------------------------------------------------------- table output

def _bet_label(r) -> str:
    team = {"home": r.home_team, "away": r.away_team}.get(r.side, r.side.capitalize())
    if r.market == "spread":
        return f"{team} {r.point:+g}"
    if r.market == "ml":
        return f"{team} ML"
    return f"{team} {r.point:g}"


def table(rows: list[list[str]], header: list[str], align: str) -> str:
    """Box-drawn table. align: one char per column, 'l' or 'r'."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]

    def line(cells):
        out = []
        for c, w, a in zip(cells, widths, align):
            out.append(c.rjust(w) if a == "r" else c.ljust(w))
        return "│ " + " │ ".join(out) + " │"

    def rule(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r

    body = [rule("┌", "┬", "┐"), line(header), rule("├", "┼", "┤")]
    body += [line(r) for r in rows]
    body.append(rule("└", "┴", "┘"))
    return "\n".join(body)


def render(rec: pd.DataFrame, show_all: bool = False) -> str:
    show = rec if show_all else rec[rec["flag"] != ""]
    show = show.sort_values(["gameday", "edge"], ascending=[True, False])
    graded = "result" in show.columns and (show["result"] != "").any()

    header = ["Date", "Game", "Market", "Bet", "Book", "Odds", "Implied", "Model", "Edge",
              "Kelly", "Stake", "Flag"]
    align = "llllllrrrrrl"
    if graded:
        header += ["Result", "Units"]
        align += "lr"
    rows = []
    for r in show.itertuples(index=False):
        row = [
            pd.Timestamp(r.gameday).strftime("%a %m/%d"),
            f"{r.away_team} @ {r.home_team}",
            {"spread": "Spread", "ml": "Moneyline", "total": "Total"}[r.market],
            _bet_label(r),
            str(r.book),
            f"{int(r.price):+d}",
            f"{r.implied:.1%}",
            f"{r.model:.1%}",
            f"{r.edge:+.1%}",
            f"{r.kelly:.1%}",
            f"{r.stake_units:.2f}u" if r.flag == "+EV" else "-",
            r.flag,
        ]
        if graded:
            row += [r.result, "" if pd.isna(r.units) or r.flag != "+EV" else f"{r.units:+.2f}"]
        rows.append(row)
    if not rows:
        return "(no bets clear the edge threshold)"
    return table(rows, header, align)
