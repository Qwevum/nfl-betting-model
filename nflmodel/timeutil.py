"""Strict UTC time handling. Naive or unparseable timestamps are errors, never guessed."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now(dt.timezone.utc))


def parse_utc(value) -> pd.Timestamp:
    """ISO-8601 timestamp with an explicit zone ('Z' or +hh:mm), converted to UTC.

    Raises ValueError for missing, unparseable or timezone-naive values.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        raise ValueError("missing timestamp")
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        s = str(value).strip()
        if not s:
            raise ValueError("missing timestamp")
        try:
            ts = pd.Timestamp(s)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unparseable timestamp {s!r}") from exc
    if ts is pd.NaT:
        raise ValueError("missing timestamp")
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no timezone (use Z or +hh:mm)")
    return ts.tz_convert("UTC")


def kickoff_utc(gameday, gametime) -> pd.Timestamp:
    """Kickoff from the nflverse schedule (date + Eastern clock time) as UTC.

    Raises ValueError if the kickoff time is not listed: without it nothing about
    the game can be checked against the pre-kickoff cutoff.
    """
    if not isinstance(gametime, str) or ":" not in gametime:
        raise ValueError(f"kickoff time not listed for {gameday}")
    local = pd.Timestamp(f"{pd.Timestamp(gameday).date()} {gametime}").tz_localize(ET)
    return local.tz_convert("UTC")


def fmt(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
