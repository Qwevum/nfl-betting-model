"""NFL spread, moneyline and total model built on nflverse data.

Requires Python 3.11+ (the standard-library `tomllib` reads settings.toml)."""
import sys

MIN_PYTHON = (3, 11)
if sys.version_info < MIN_PYTHON:  # pragma: no cover - exercised only on old interpreters
    raise ImportError(f"nflmodel requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ (found "
                      f"{sys.version_info.major}.{sys.version_info.minor}); tomllib is not available earlier")
