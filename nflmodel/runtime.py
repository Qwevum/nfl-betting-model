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

Completion check (finalize): at prediction completion ALL raw quotes are
revalidated, not only the selected offers. If the set of valid quotes changed
(a comparison quote or one side of a reference pair expired, or kickoff passed),
the whole slate is re-priced from the remaining quotes: references are rebuilt
(two-sided pairing and leave-one-book-out preserved) and probabilities, EV,
sensitivity, decisions and stakes are recomputed. This repeats up to
`max_reprice` times. A final safeguard (recheck_before_issue) then downgrades any
executable bet whose selected quote OR any reference quote is stale at the final
time, and drops games that have kicked off; so if re-pricing cannot settle, the
result is conservative, never a stale-reference BET.
"""
from __future__ import annotations

import pandas as pd

from .odds import validate_offers
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
        elif dec in (EXECUTABLE, "BET IF CONFIRMED"):
            if isinstance(r.odds_time, str):
                try:
                    age = completed - parse_utc(r.odds_time)
                except ValueError:
                    age = None
                if age is None or age > max_age:
                    dec = "BET IF PRICE AVAILABLE"
                    why = (why + " | " if why else "") + (
                        f"quote from {r.odds_time} was stale by completion ({fmt(completed)}); reconfirm the price")
            live_ref = bool(getattr(r, "reference_live", False))
            ref_oldest = getattr(r, "reference_oldest_utc", None)
            if dec in (EXECUTABLE, "BET IF CONFIRMED") and live_ref:
                try:
                    ref_age = completed - parse_utc(ref_oldest)
                except (ValueError, TypeError):
                    ref_age = None
                if ref_age is None or ref_age > max_age:
                    dec = "BET IF PRICE AVAILABLE"
                    why = (why + " | " if why else "") + (
                        f"market reference quotes expired by completion ({fmt(completed)}; oldest "
                        f"{ref_oldest}); EV is not current, reconfirm the market")
        post.append(late)
        decisions.append(dec)
        reasons.append(why)
    rows["decision"], rows["reasons"], rows["post_kickoff"] = decisions, reasons, post
    rows.loc[rows["decision"] == "NO BET", "stake_units"] = 0.0
    return rows


def _quote_keys(valid: pd.DataFrame) -> frozenset:
    if valid is None or valid.empty:
        return frozenset()
    cols = ["game_id", "book", "market", "side", "point", "price", "odds_time"]
    return frozenset(tuple(None if pd.isna(v) else v for v in row) for row in valid[cols].itertuples(index=False))


def finalize(price, books: pd.DataFrame, kickoffs: dict, clock: Clock, settings, initial_valid: pd.DataFrame,
             max_reprice: int = 2):
    """Price, then revalidate every quote at completion and re-price until stable.

    price(valid) -> object with .rows (DataFrame) and .valid; it must compute everything
    from `valid` alone. Returns (priced, info) where priced.rows have been rechecked."""
    priced = price(initial_valid)
    valid, info = initial_valid, {"repriced": 0, "stable": False, "expired_quotes": 0}
    while True:
        t = clock.stamp("prediction_completed")
        now_valid, _ = validate_offers(books, kickoffs, t, settings) if len(books) else (books, None)
        if _quote_keys(now_valid) == _quote_keys(valid):
            info["stable"] = True
            break
        if info["repriced"] >= max_reprice:
            break
        info["expired_quotes"] += len(_quote_keys(valid) - _quote_keys(now_valid))
        valid = now_valid
        priced = price(valid)
        info["repriced"] += 1
    priced.rows = recheck_before_issue(priced.rows, kickoffs, t, settings)
    info["completed_utc"] = fmt(t)
    return priced, info
