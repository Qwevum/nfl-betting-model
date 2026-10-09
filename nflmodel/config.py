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
import re
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
    # User inputs (qb_overrides.csv, weather_manual.csv): how old they may be at the decision time
    qb_confirm_max_age_minutes: float = 48 * 60    # a starter confirmation older than this is stale
    weather_max_age_minutes: float = 12 * 60       # forecasts update often; older issues are stale
    weather_valid_window_minutes: float = 180      # forecast's valid-for time must be this close to kickoff
    # Market reference
    min_reference_books: int = 2           # other books needed for a reference (leave-one-book-out)
    # Prediction horizon: forecasts are meant to be made this long before kickoff
    horizon_minutes: float = 60.0
    horizon_tolerance_minutes: float = 15.0   # eligible forecasts: [kickoff-horizon-tol, kickoff-horizon]
    # Decisions
    min_edge: float = 0.02                 # EV per unit staked required to bet
    gap_points: float = 4.0                # model-vs-market gap (pts) that blocks a bet as unexplained
    kelly_fraction: float = 0.25           # share of full Kelly staked
    max_stake_units: float = 2.0           # stake cap per bet (1 unit = 1% of bankroll)
    # The Odds API: bookmaker regions per request. Cost = 3 markets x number of regions
    # credits per request, so "us" (3 credits) suits the free plan; "us,us2" adds books.
    odds_api_regions: str = "us"
    # Probability used for decisions:
    #   "model"  - the ex-book market reference at the exact line, moved only by the model's
    #              calibrated edge (offset calibration: market log-odds kept with weight 1)
    #   "market" - the ex-book market reference alone (price shopping baseline)
    #   "legacy" - the pre-2026-10 calibration (free intercept and slope on the market's
    #              log-odds); kept only as the comparison baseline
    probability_model: str = "model"
    # Sportsbooks you can actually bet at (Odds API bookmaker keys, comma-separated, e.g.
    # "draftkings,fanduel"). Every book still contributes to market references; only these
    # are actionable. Empty = every book counts as actionable (the report says so).
    actionable_books: str = ""
    watchlist_size: int = 10               # closest non-actionable candidates shown in the watchlist

    def __post_init__(self):
        validate_settings(self)

    def describe(self) -> str:
        return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in asdict(self).items())


ODDS_API_REGIONS = {"us", "us2", "us_dfs", "us_ex", "uk", "eu", "au"}

# (name, kind, lower, lower_inclusive, upper, upper_inclusive)
_RULES = [
    ("max_odds_age_minutes", float, 0, False, 24 * 60, True),
    ("clock_skew_minutes", float, 0, True, 60, True),
    ("kickoff_match_minutes", float, 0, True, 24 * 60, True),
    ("min_reference_books", int, 1, True, 50, True),
    ("qb_confirm_max_age_minutes", float, 0, False, 14 * 24 * 60, True),
    ("weather_max_age_minutes", float, 0, False, 7 * 24 * 60, True),
    ("weather_valid_window_minutes", float, 0, False, 24 * 60, True),
    ("horizon_minutes", float, 0, False, 7 * 24 * 60, True),
    ("horizon_tolerance_minutes", float, 0, False, 7 * 24 * 60, True),
    ("min_edge", float, 0, True, 1, False),
    ("gap_points", float, 0, False, 60, True),
    ("kelly_fraction", float, 0, False, 1, True),
    ("max_stake_units", float, 0, False, 100, True),
    ("watchlist_size", int, 0, True, 200, True),
]

PROBABILITY_MODELS = {"model", "market", "legacy"}
_BOOK_KEY = re.compile(r"^[a-z0-9_]+$")


def actionable_set(settings) -> set[str]:
    """Books the user can bet at; empty set = all books."""
    return {b.strip().lower() for b in settings.actionable_books.split(",") if b.strip()}


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
    regions = s.odds_api_regions.split(",") if isinstance(s.odds_api_regions, str) else None
    if not regions or any(r.strip() not in ODDS_API_REGIONS for r in regions):
        errors.append(f"odds_api_regions must be a comma-separated subset of {sorted(ODDS_API_REGIONS)}, "
                      f"got {s.odds_api_regions!r}")
    if s.probability_model not in PROBABILITY_MODELS:
        errors.append(f"probability_model must be one of {sorted(PROBABILITY_MODELS)}, got {s.probability_model!r}")
    if not isinstance(s.actionable_books, str):
        errors.append(f"actionable_books must be a comma-separated string, got {s.actionable_books!r}")
    else:
        bad = [b for b in (x.strip().lower() for x in s.actionable_books.split(",")) if b and not _BOOK_KEY.match(b)]
        if bad:
            errors.append(f"actionable_books has invalid bookmaker keys {bad} (use Odds API keys like draftkings)")
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
