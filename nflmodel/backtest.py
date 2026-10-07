"""Grading helpers shared by validation, `recommend` and the prediction log."""
from __future__ import annotations

from .odds import decimal


def grade(market: str, side: str, point: float, home: float, away: float) -> str:
    """W / L / P for one bet given the final score."""
    margin, total = home - away, home + away
    if market == "spread":
        v = (margin if side == "home" else -margin) + point
    elif market == "ml":
        v = margin if side == "home" else -margin
    else:
        v = (total - point) if side == "over" else (point - total)
    return "W" if v > 0 else ("P" if v == 0 else "L")


def profit(result: str, price: float) -> float:
    """Profit per 1 unit staked at an American price."""
    return {"W": decimal(price) - 1, "P": 0.0, "L": -1.0}[result]
