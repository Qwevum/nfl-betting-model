# Change log: measured, reproducible changes

Each entry gives the problem, the files changed, how it was verified, and the
measured effect. Numbers come from commands in this repo on cached data unless
stated otherwise.

## 0. Baseline (commit `34c198f`)

**Environment:** Python 3.11.15, pandas 3.0.6, numpy 2.4.6, pyarrow 25.0.1.

**Tests:** `python -m unittest discover tests` gives 10 tests, all passing. They
cover odds math, grading and the outcome distribution only.

**Data coverage** (nflverse `games.csv`, seasons used by the model):

| seasons | games | closing spread/total/ML | spread prices | starting-QB ids | game-time wind |
|---|---|---|---|---|---|
| 2010–2025 | 267–285 per season | 100% | 100% (except 1 game in 2017) | 100% | 40–75% (domes have none) |
| 2026 (to wk 4) | 64 played | 93 games with lines | 93 | 79 | 45 |

* Play-by-play is cached per season for 2010–2026, in the `team_games_v2_*` and
  `qb_games_v2_*` files.
* The injury reports for 2026 are cached. The feed has a `week` field but **no
  publication timestamp**. On 2026-10-07 it held 19 week-5 rows, all for the
  Thursday game.
* nflverse lines have no per-book or time-of-snapshot field. The documented
  meaning is closing or consensus.

**Reproduction:** `python run.py validate --no-refresh` regenerated
`reports/validation.md` byte-identically.
* Winner Brier: model 0.2126, closing market 0.2126.
* Bets under the backtest rules: 1,333 bets (646-663-24), flat ROI +2.3%
  (95% CI −3.5% to +8.0%).

**Not reproducible, or not checked here:**
* **The Odds API and the Open-Meteo weather API.** Both are blocked by this
  sandbox's network policy, so the live multi-book path has never run end to end.
* **Weekly reports.** They depend on the moment the data was refreshed, and
  only the latest cache is kept, not per-run snapshots. Re-running `predict`
  today does not reproduce `reports/2026_week05.md`.
* **README rounding.** The README lists the closing-market Brier for "home
  covers" as 0.2500; the output is 0.2499.

**Issues found in the audit** (addressed below unless noted):

1. **Untimed market reference.** The reference is the nflverse consensus line,
   which has no timestamp. Live bookmaker quotes are only ever treated as
   offers, never as the reference. A moved market (for example DAL −9.5 at
   books vs −8.5 consensus) therefore leaves a stale reference.
2. **No offer validation.** Offers are not checked for timestamp format,
   timezone, freshness, or kickoff before the best price is chosen.
3. **Games matched by team names.** Manual odds, QB overrides and weather are
   matched by team names, not game id and kickoff. QB overrides apply to any
   game of that team in the window.
4. **Missing injury data reads as healthy.** A missing or failed injury feed,
   or a starter absent from a partial report, is treated as "no designation",
   which in practice means healthy.
5. **One log for everything.** Forecasts and "bets" share one append-only CSV
   with no integrity check. Nothing records what was actually wagered, so
   graded "BETS" are hypothetical.
6. **Validation does not match live use.**
   * It uses closing consensus prices, actual starters and **observed game-time
     wind** (postgame knowledge) as a feature.
   * It applies only the EV, sensitivity and gap rules, yet the README calls
     these "the live rules".
   * The model and the market are scored on slightly different game sets
     (3,082 vs 3,081).
   * Confidence intervals treat correlated bets on one game as independent.
   * Drawdown is not reported.

## 1. Validate quotes before price selection; identify games by id and kickoff

**Problem:** audit issues 2 and 3. Any quote, stale or untimed, could win the
best-price selection. Inputs matched games by team names, so a QB override
applied to every game of that team.

**Files:**
* new: `nflmodel/config.py`, `nflmodel/timeutil.py`, `nflmodel/inputs.py`,
  `tests/test_offers.py`
* changed: `nflmodel/odds.py`, `nflmodel/decide.py`, `run.py`, the three input
  CSVs, `README.md`

**Change:**
* `validate_offers()` runs before any selection and rejects quotes that have:
  * a missing, naive or future timestamp
  * an age above `max_odds_age_minutes`
  * a quote time at or after kickoff, or a run at or after kickoff
  * an invalid price, point or side
* Kickoffs are computed in UTC from the schedule (Eastern time converted).
* Odds API events are matched on teams **and** commence time.
* Manual odds, QB overrides and weather require `game_id` + `kickoff_utc`,
  checked against the schedule (10-minute tolerance).
* Games already started are skipped.
* `--now` gives reproducible replays, and those runs are never logged.

**Verification:**
* 14 new tests cover stale, naive, future and post-kickoff quotes, the
  configurable age limit, a stale better price losing to a fresh one, kickoff
  mismatch, unknown games, overrides scoped to one game and team, and
  kickoff-based API matching. The full suite passes: 24 tests.
* End to end, a replay at 2026-10-11T12:30Z with 5 manual quotes rejected:
  * one stale quote
  * one quote with a naive timestamp
  * one row with a wrong kickoff

  It also skipped the Thursday game.

**Measured effect:** none on model accuracy, and none expected; this changes
which quotes may be used. The run also exposed the next problem: one fresh quote
at MIA +8 became a "BET" against the untimed Oct 7 consensus reference.

## 2. Market reference from contemporaneous two-sided quotes, leave-one-book-out

**Problem:** audit issue 1. The market probability that anchors every estimate
came from the untimed nflverse consensus, even when fresh bookmaker quotes were
available. One quote could also define its own "market".

**Files:**
* new: `nflmodel/market.py`, `tests/test_market.py`
* changed: `nflmodel/decide.py`, `nflmodel/report.py`, `run.py`, `README.md`

**Change:**
* `analyze()` now runs in order:
  1. fetch quotes
  2. validate them (change 1)
  3. build references from valid quotes only
  4. set the slate's market lines from the references
  5. predict and price
* **Pairs:** a book's latest quote for each side, at the same line, with both
  sides quoted within 5 minutes of each other.
* **Vig removal:** proportional normalization.
* **Different lines:** each book's no-vig probability at its line is inverted
  through the key-number distribution into an implied mean (`mu`). The
  reference is the median `mu` across books, and the reference line is the
  median quoted line. Moneylines use the median of the logit probability.
* **Self-reference:** each offer is priced against a reference that excludes
  its own book and needs `min_reference_books` other books. Without one, the
  decision is at most "BET IF PRICE AVAILABLE".
* When live quotes exist, untimed consensus lines are not offers.
* Every decision shows the reference it used: number of books, quote times and
  dispersion.

**Verification:**
* 9 new tests cover:
  * vig removal at −110/−110
  * three books at different lines recovering the same mean within 0.15 pts
  * the evaluated book excluded from its reference
  * the median resisting an outlier book
  * no reference when too few other books remain
  * one-sided quotes 10 minutes apart not paired
  * the latest quote superseding an older one
  * stale quotes never reaching the reference
  * moneyline median-logit with an outlier

  The suite passes: 33 tests.
* Historical path unchanged: `validate --no-refresh` regenerates
  `reports/validation.md` byte-identically.
* Replay at 2026-10-11T12:30Z with 3 books two-sided on PHI @ JAX and one
  one-sided quote on CIN @ MIA:
  * the CIN @ MIA quote went from **BET** (change 1) to **BET IF PRICE
    AVAILABLE**, because there is no live reference without its own book
  * PHI +8 at book C is priced against books A and B only (EV +0.9%, no bet)

**Measured effect on accuracy:** none measurable. There are no historical
multi-book snapshots to evaluate the reference against (see change 5). Untested
in this sandbox: the reference built from real Odds API data, because the API
is blocked here.
