"""Run clock and the final pre-issue recheck.

A run passes through stages that can take minutes (model fit, downloads, odds
fetch). Each stage is stamped with the time it actually happened, and checks use
the stage's own time:

  run_started_utc           command start
  inputs_read_utc           QB confirmations / weather validated against this
  odds_collected_utc        quotes validated (freshness, kickoff) against this
  injuries_retrieved_utc    injury feed fetched
  prediction_completed_utc  decisions finished; executable recommendations are
                            rechecked against THIS time before being issued
  recorded_utc              (store) when the forecast was written: never earlier
                            than prediction_completed_utc

Simulated runs (--now) use a fixed clock and are never recorded prospectively.
"""
from __future__ import annotations

import pandas as pd

from .timeutil import fmt, now_utc, parse_utc


class Clock:
    def __init__(self, simulated: pd.Timestamp | None = None):
        self._fixed = simulated
        self.stamps: dict[str, pd.Timestamp] = {}

    @property
    def simulated(self) -> bool:
        return self._fixed is not None

    def now(self) -> pd.Timestamp:
        return self._fixed if self._fixed is not None else now_utc()

    def stamp(self, stage: str) -> pd.Timestamp:
        t = self.now()
        self.stamps[stage] = t
        return t

    def as_dict(self) -> dict[str, str]:
        return {f"{k}_utc": fmt(v) for k, v in self.stamps.items()}


EXECUTABLE = "BET"


def recheck_before_issue(rows: pd.DataFrame, kickoffs: dict, completed: pd.Timestamp, settings) -> pd.DataFrame:
    """Re-validate decisions at prediction-completion time.

    * Any row whose game has kicked off by `completed` becomes NO BET and is marked
      post_kickoff (never recorded as a forecast).
    * An executable BET whose quote is older than max_odds_age_minutes at `completed`
      (it was fresh when collected, but the run took too long) is downgraded to the
      conditional "BET IF PRICE AVAILABLE".
    """
    if rows.empty:
        return rows.assign(post_kickoff=pd.Series(dtype=bool))
    rows = rows.copy()
    max_age = pd.Timedelta(minutes=settings.max_odds_age_minutes)
    post, decisions, reasons = [], [], []
    for r in rows.itertuples(index=False):
        k = kickoffs.get(r.game_id)
        dec, why = r.decision, r.reasons or ""
        late = k is None or pd.isna(k) or completed >= k
        if late:
            dec = "NO BET"
            why = (why + " | " if why else "") + f"kickoff passed before the run completed at {fmt(completed)}"
        elif dec in (EXECUTABLE, "BET IF CONFIRMED") and isinstance(r.odds_time, str):
            try:
                age = completed - parse_utc(r.odds_time)
            except ValueError:
                age = None
            if age is None or age > max_age:
                dec = "BET IF PRICE AVAILABLE"
                why = (why + " | " if why else "") + (
                    f"quote from {r.odds_time} was stale by completion ({fmt(completed)}); reconfirm the price")
        post.append(late)
        decisions.append(dec)
        reasons.append(why)
    rows["decision"], rows["reasons"], rows["post_kickoff"] = decisions, reasons, post
    rows.loc[rows["decision"] == "NO BET", "stake_units"] = 0.0
    return rows
