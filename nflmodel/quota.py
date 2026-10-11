"""The Odds API credit check and collect-schedule usage estimate.

remaining() calls GET /v4/sports, which The Odds API documents as not counting against the
usage quota; its response headers report credits used and remaining this month. The key is
only placed in the request URL and never printed (errors are redacted).
"""
from __future__ import annotations

import ssl
import urllib.request

SPORTS_URL = "https://api.the-odds-api.com/v4/sports/?apiKey={key}"
WEEKS_PER_MONTH = 52 / 12


def remaining(key: str, timeout: float = 30) -> dict:
    from .apikey import redact
    try:
        req = urllib.request.Request(SPORTS_URL.format(key=key), headers={"User-Agent": "nfl-model"})
        with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=timeout) as r:
            h = r.headers
            used, left = h.get("x-requests-used"), h.get("x-requests-remaining")
            return {"used": int(float(used)) if used is not None else None,
                    "remaining": int(float(left)) if left is not None else None}
    except Exception as e:   # never echo the URL with the key
        return {"error": redact(f"{type(e).__name__}: {e}")}


def weekly_runs(every_hours: float, gameday_hourly: bool) -> float:
    """Runs per NFL week: every `every_hours` from Sunday 20:00 ET to Monday-night kickoff (about
    7 days = 168 h), plus, optionally, hourly 08:00-20:00 ET on the three game days (Thu, Sun, Mon)
    not already covered by the base schedule."""
    base = 168 / every_hours
    extra = 0.0
    if gameday_hourly:
        extra = 3 * max(0.0, 12 - 12 / every_hours)
    return base + extra


def monthly_credits(per_run: int, every_hours: float, gameday_hourly: bool) -> float:
    return weekly_runs(every_hours, gameday_hourly) * per_run * WEEKS_PER_MONTH


def report(status: dict | None, per_run: int, every_hours: float, gameday_hourly: bool) -> str:
    lines = []
    if status is None:
        lines.append("No ODDS_API_KEY: credits unknown (set it in .env).")
        plan = None
    elif "error" in status:
        lines.append(f"Credit check failed: {status['error']}")
        plan = None
    else:
        plan = (status["used"] or 0) + (status["remaining"] or 0)
        lines.append(f"The Odds API this month: {status['used']} credits used, {status['remaining']} remaining "
                     f"(monthly allowance about {plan}). This check spent 0 credits.")
    lines.append(f"Each collect run = 1 request = {per_run} credits (3 markets x regions).")
    lines.append("")
    lines.append("schedule (in season)                         runs/week  credits/week  credits/month")
    for h, gd in sorted({(every_hours, gameday_hourly), (2.0, False), (3.0, False), (4.0, False), (6.0, False),
                         (8.0, False), (4.0, True), (6.0, True)}):
        w = weekly_runs(h, gd)
        m = monthly_credits(per_run, h, gd)
        mark = "  <- requested" if (h, gd) == (every_hours, gameday_hourly) else ""
        fits = "" if plan is None else ("  fits" if m <= plan else "  EXCEEDS the allowance")
        lines.append(f"every {h:g} h{' + hourly on game days' if gd else '':<26s} {w:9.0f}  {w * per_run:12.0f}  "
                     f"{m:13.0f}{fits}{mark}")
    lines.append("")
    lines.append("Not included: predict / recommend / check-live runs (each one request), and any reruns.")
    return "\n".join(lines)


__all__ = ["remaining", "weekly_runs", "monthly_credits", "report"]
