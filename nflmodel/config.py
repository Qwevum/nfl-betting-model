"""Settings for live decisions and evaluation.

Defaults live here; `settings.toml` in the repo root overrides any of them, e.g.

    max_odds_age_minutes = 20
    min_reference_books = 3
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    min_edge: float = 0.02
    gap_points: float = 4.0
    kelly_fraction: float = 0.25
    max_stake_units: float = 2.0


def load_settings(path: Path | None = None, **overrides) -> Settings:
    s = Settings()
    path = path or ROOT / "settings.toml"
    if path.exists():
        data = tomllib.loads(path.read_text())
        known = {f.name for f in fields(Settings)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown settings in {path.name}: {sorted(unknown)}")
        s = replace(s, **data)
    return replace(s, **{k: v for k, v in overrides.items() if v is not None})
