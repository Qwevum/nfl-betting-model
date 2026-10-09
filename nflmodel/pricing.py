"""Price a slate from a given set of validated quotes.

Everything that depends on the quotes is computed here: leave-one-book-out market
references, the slate's market lines, model predictions, decision contexts, and
per-side decisions (probabilities, EV, sensitivity, stakes). Re-running it with a
smaller quote set (e.g. after quotes expired) recomputes all of these from
scratch, so nothing from an expired reference is carried over.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import decide, market


@dataclass
class PricedSlate:
    preds: pd.DataFrame
    rows: pd.DataFrame
    contexts: dict = field(default_factory=dict)
    refs_by_game: dict = field(default_factory=dict)
    references: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid: pd.DataFrame = field(default_factory=pd.DataFrame)


def price_slate(model, day: pd.DataFrame, valid: pd.DataFrame, consensus: pd.DataFrame, settings,
                context_fn) -> PricedSlate:
    """context_fn(g) -> decide.GameContext for one prediction row."""
    day = day.copy()
    day["reference_kind"] = "nflverse consensus (untimed)"
    refs_by_game: dict = {}
    for gid, o in valid.groupby("game_id"):
        refs = market.references_for_game(o, model, settings.min_reference_books)
        refs_by_game[gid] = refs
        m = day["game_id"] == gid
        if refs[("spread", None)] is not None:
            day.loc[m, "spread_line"] = refs[("spread", None)].line
        if refs[("total", None)] is not None:
            day.loc[m, "total_line"] = refs[("total", None)].line
        if any(refs[(mk, None)] is not None for mk in ("spread", "ml", "total")):
            day.loc[m, "reference_kind"] = "live bookmaker quotes"
    offers = pd.concat([valid, consensus], ignore_index=True)
    preds = model.predict(day)
    rows, contexts = [], {}
    for g in preds.itertuples(index=False):
        ctx = context_fn(g)
        contexts[g.game_id] = ctx
        rows += decide.decide_game(model, g, offers[offers["game_id"] == g.game_id], ctx, settings,
                                   refs=refs_by_game.get(g.game_id, {}))
    rows = pd.DataFrame(rows)
    if len(rows):
        best = decide.best_per_market(rows)
        rows["is_best_side"] = rows.set_index(["game_id", "market", "side"]).index.isin(
            best.set_index(["game_id", "market", "side"]).index)
    return PricedSlate(preds, rows, contexts, refs_by_game, market.references_frame(refs_by_game), valid)
