"""Settings for live decisions and evaluation.

Defaults live here; `settings.toml` in the repo root overrides any of them, e.g.

    max_odds_age_minutes = 20
    min_reference_books = 3

Every Settings object is validated when created: wrong types, non-finite numbers
and values outside their sensible range raise SettingsError, so a typo cannot
silently change how bets are sized or filtered.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    # Odds validity
    max_odds_age_minutes: float = 30.0     # older quotes are stale and rejected
    clock_skew_minutes: float = 2.0        # tolerated future timestamps (clock drift)
    kickoff_match_minutes: float = 10.0    # input rows must match the schedule kickoff this closely
    # Market reference
    min_reference_books: int = 2           # other books needed for a reference (leave-one-book-out)
    # Prediction horizon: forecasts are meant to be made this long before kickoff
    horizon_minutes: float = 60.0
    # Decisions
    min_edge: float = 0.02                 # EV per unit staked required to bet
    gap_points: float = 4.0                # model-vs-market gap (pts) that blocks a bet as unexplained
    kelly_fraction: float = 0.25           # share of full Kelly staked
    max_stake_units: float = 2.0           # stake cap per bet (1 unit = 1% of bankroll)

    def __post_init__(self):
        validate_settings(self)

    def describe(self) -> str:
        return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in asdict(self).items())


# (name, kind, lower, lower_inclusive, upper, upper_inclusive)
_RULES = [
    ("max_odds_age_minutes", float, 0, False, 24 * 60, True),
    ("clock_skew_minutes", float, 0, True, 60, True),
    ("kickoff_match_minutes", float, 0, True, 24 * 60, True),
    ("min_reference_books", int, 1, True, 50, True),
    ("horizon_minutes", float, 0, False, 7 * 24 * 60, True),
    ("min_edge", float, 0, True, 1, False),
    ("gap_points", float, 0, False, 60, True),
    ("kelly_fraction", float, 0, False, 1, True),
    ("max_stake_units", float, 0, False, 100, True),
]


def validate_settings(s: Settings) -> None:
    errors = []
    for name, kind, lo, lo_inc, hi, hi_inc in _RULES:
        v = getattr(s, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{name} must be a number, got {v!r}")
            continue
        if kind is int and (not float(v).is_integer()):
            errors.append(f"{name} must be a whole number, got {v!r}")
            continue
        if not math.isfinite(v):
            errors.append(f"{name} must be finite, got {v!r}")
            continue
        if (v < lo) or (v == lo and not lo_inc) or (v > hi) or (v == hi and not hi_inc):
            rng = f"{'[' if lo_inc else '('}{lo}, {hi}{']' if hi_inc else ')'}"
            errors.append(f"{name}={v!r} outside {rng}")
    if not errors and s.clock_skew_minutes >= s.max_odds_age_minutes:
        errors.append("clock_skew_minutes must be smaller than max_odds_age_minutes")
    if errors:
        raise SettingsError("invalid settings: " + "; ".join(errors))


def load_settings(path: Path | None = None, **overrides) -> Settings:
    path = path or ROOT / "settings.toml"
    data = {}
    if path.exists():
        data = tomllib.loads(path.read_text())
        unknown = set(data) - {f.name for f in fields(Settings)}
        if unknown:
            raise SettingsError(f"unknown settings in {path.name}: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return replace(Settings(), **data)
