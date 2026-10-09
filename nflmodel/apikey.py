"""The Odds API key: where it comes from, without ever printing or storing it.

Lookup order:
  1. the ODDS_API_KEY environment variable
  2. ODDS_API_KEY=... in `.env` at the repo root (gitignored; never committed)

The value is only ever passed to the request URL. Everything that could reach the
terminal, a report, a snapshot or a log goes through redact() first.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
NAME = "ODDS_API_KEY"


def _from_env_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().removeprefix("export ").strip() == NAME:
            return v.strip().strip("'\"")
    return None


def lookup(env_file: Path | None = None) -> tuple[str | None, str]:
    """(key or None, source description). Source never contains the value."""
    v = os.environ.get(NAME)
    if v is not None:
        return (v.strip() or None), "environment variable"
    path = env_file or ENV_FILE
    v = _from_env_file(path)
    if v is not None:
        return (v or None), f"{path.name} file"
    return None, "not set"


def get(env_file: Path | None = None) -> str | None:
    return lookup(env_file)[0]


def redact(text, env_file: Path | None = None) -> str:
    text = str(text)
    key = get(env_file)
    if key:
        text = text.replace(key, "***")
    return text
