"""Live bookmaker-feed check: is the feed configured, reachable and good enough?

Never prints or logs the API key. Never treats nflverse consensus lines as a live
feed. Writes nothing: it is a diagnostic, not a collection run.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .market import book_pairs
from .timeutil import fmt

SETUP = """How to enable the live bookmaker feed:
  1. Create an account at https://the-odds-api.com and check its current plans and
     credit costs (each request is charged per market and region; this tool requests
     h2h, spreads and totals for the us and us2 regions in one call).
  2. Export the key in the shell that runs the model, without committing it anywhere:
         export ODDS_API_KEY=...        # macOS/Linux
         setx ODDS_API_KEY ...          # Windows (new terminal afterwards)
  3. Run:  python run.py check-live
  Without a key, odds come only from odds_manual.csv (your own app quotes) and the
  untimed nflverse consensus, which is never treated as executable."""


def key_status() -> tuple[bool, str]:
    """(configured, description) without revealing the value."""
    v = os.environ.get("ODDS_API_KEY")
    if v is None:
        return False, "ODDS_API_KEY is not set"
    if not v.strip():
        return False, "ODDS_API_KEY is set but empty"
    return True, "ODDS_API_KEY is set (value not shown)"


def redact(text: str) -> str:
    v = os.environ.get("ODDS_API_KEY")
    text = str(text)
    if v and v.strip():
        text = text.replace(v, "***")
    return text


def summarize(valid: pd.DataFrame, rejected: pd.DataFrame, kickoffs: dict, collected: pd.Timestamp,
              min_reference_books: int) -> dict:
    """Statistics for a batch of quotes that already went through validate_offers()."""
    out = {"collected_utc": fmt(collected), "valid_quotes": int(len(valid)), "rejected_quotes": int(len(rejected)),
           "reject_reasons": {}, "books": 0, "games_with_quotes": 0, "games_in_slate": len(kickoffs),
           "age_minutes": None, "reference_ready": {}}
    if len(rejected):
        out["reject_reasons"] = rejected["reject_reason"].str.split(":").str[0].value_counts().to_dict()
    if valid.empty:
        return out
    out["books"] = int(valid["book"].nunique())
    out["games_with_quotes"] = int(valid["game_id"].nunique())
    age = (collected - pd.to_datetime(valid["odds_time_utc"], utc=True)).dt.total_seconds() / 60
    out["age_minutes"] = {"min": float(age.min()), "median": float(age.median()), "max": float(age.max())}
    # A game/market is reference-ready when at least min_reference_books OTHER books have
    # two-sided quotes for every book's offer, i.e. min_reference_books + 1 books in total.
    need = min_reference_books + 1
    for market in ("spread", "ml", "total"):
        ready = 0
        for gid, g in valid.groupby("game_id"):
            pairs = book_pairs(g, market)
            if len({p["book"] for p in pairs}) >= need:
                ready += 1
        out["reference_ready"][market] = ready
    return out


def format_summary(s: dict, min_reference_books: int) -> str:
    lines = [f"Collected at {s['collected_utc']}: {s['valid_quotes']} valid / {s['rejected_quotes']} rejected quotes, "
             f"{s['books']} distinct books, {s['games_with_quotes']} of {s['games_in_slate']} games quoted"]
    if s["reject_reasons"]:
        lines.append("  rejected: " + ", ".join(f"{k} ({v})" for k, v in s["reject_reasons"].items()))
    if s["age_minutes"]:
        a = s["age_minutes"]
        lines.append(f"  quote age at collection: min {a['min']:.1f}, median {a['median']:.1f}, max {a['max']:.1f} min")
    if s["reference_ready"]:
        lines.append(f"  games with >= {min_reference_books + 1} books quoting both sides (a live reference "
                     "excluding any one book): " + ", ".join(f"{k} {v}" for k, v in s["reference_ready"].items()))
    return "\n".join(lines)


def describe_failure(exc: Exception) -> str:
    msg = redact(exc)
    if "403" in msg or "Tunnel" in msg or "CONNECT" in msg:
        return f"api.the-odds-api.com is not reachable from this network ({msg}); check firewall/proxy rules"
    if "401" in msg:
        return "the API rejected the key (HTTP 401): check that ODDS_API_KEY is correct and active"
    if "429" in msg:
        return "rate limit or monthly quota exhausted (HTTP 429)"
    return f"request failed: {msg}"


__all__ = ["SETUP", "key_status", "redact", "summarize", "format_summary", "describe_failure", "np"]
