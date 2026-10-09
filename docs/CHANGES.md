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

## 3. Missing injury information means unknown availability

**Problem:** audit issue 4. A projected starter with no injury designation was
treated as healthy even when the feed had failed, or had no report for that
team and week. On 2026-10-07 the feed held week-5 rows only for TB and DAL,
so every Sunday starter counted as "available" by default.

**Files:**
* changed: `nflmodel/decide.py`, `README.md`
* new: `tests/test_availability.py`

**Change:** `starter_status()` returns one of four states:
* `available`: final game statuses are issued for that team and week, and the
  QB has no designation
* `ruled_out`: listed Out or Doubtful
* `questionable`: listed Questionable
* `unknown`: the feed is unavailable, there is no report for the team and week,
  or game statuses aren't issued yet

Ruled-out and questionable starters block the bet. An unknown starter adds a
condition, which downgrades an otherwise executable bet to "BET IF CONFIRMED".
A confirmed starter from `qb_overrides.csv` settles the question.

**Verification:**
* 9 new tests:
  * a failed or missing feed gives `unknown`
  * no report for the team and week gives `unknown`
  * a practice-only report gives `unknown`
  * a final report without a designation gives `available`
  * Out, Doubtful and Questionable map correctly
  * a missing starter id gives `unknown`
  * at decision level with a synthetic market-only model, a +EV quote (>5%)
    becomes "BET IF CONFIRMED" when the feed is missing, "BET" once the final
    report clears the starter, and "NO BET" when the starter is Questionable

  The suite passes: 42 tests.
* Replay at 2026-10-11T12:30Z with the cached Oct 7 feed: Hurts and Lawrence
  are now listed as "availability unknown: no week 5 injury report for PHI/JAX".

**Measured effect:** none on accuracy (a decision rule). Expect fewer executable
recommendations early in the week.

## 4. Immutable forecast history, input snapshots, and a separate placed-bet ledger

**Problem:** audit issue 5. Forecasts went into a plain CSV that could be edited
without detection, and graded "BETS" were recommendations, not wagers. Nothing
recorded what was actually bet, and the exact inputs of a run were not kept.

**Files:**
* new: `nflmodel/store.py`, `nflmodel/metrics.py`, `tests/test_store.py`,
  `logs/forecasts.jsonl`
* rewritten: `nflmodel/track.py`
* changed: `run.py`, `.gitignore`

**Change:**
* **Forecast history** (`logs/forecasts.jsonl`): an append-only, hash-chained
  JSONL log. Each record holds a sequence number and the previous record's
  hash. `verify-log` and `grade` check the chain, and `append` refuses to write
  to a broken chain.
  * Every priced side is recorded, passes included, with a tier: executable
    (`BET`), conditional (`BET IF CONFIRMED` / `BET IF PRICE AVAILABLE`) or pass.
  * Each record also carries the model version, settings horizon, run id and
    snapshot id.
* **Snapshots** (`snapshots/<run_id>/`): every recorded run copies:
  * schedule rows for the slate
  * raw, rejected and valid quotes
  * references and predictions
  * the injury file and source timestamps
  * the three input CSVs

  A manifest of file hashes gives the `snapshot_id`.
* **Ledger** (`logs/ledger.jsonl`, separate and hash-chained): `place` records
  an actual wager linked to its forecast hash, with the price, line, book and
  stake actually obtained. It rejects:
  * placement at or after kickoff
  * placement before the forecast existed
  * non-positive stakes and invalid prices
  * duplicates and ambiguous forecast ids

  `void` records cancellations, which never delete anything.
* **`grade`** reports three sections separately:
  1. forecast quality at the 60-minute horizon, model vs market on identical rows
  2. hypothetical recommendations by tier (flat 1u, labelled "NOT wagers")
  3. actual wagers from the ledger: ROI, max drawdown and CLV
* Runs from uncommitted code or with `--now` are never recorded.
* The 90 legacy CSV forecasts (Oct 7, model `d3525f5`) were imported once,
  flagged `legacy`. `logs/predictions.csv` is kept unchanged as the original.

**Verification:**
* 11 new tests:
  * the chain verifies
  * an edited price, a deleted line and reordered lines are each detected, and
    append is refused after tampering
  * NaN and timestamps serialize
  * placement records the actual price and the forecast link
  * placement is rejected after kickoff, before the forecast, with bad stakes
    or prices, and for duplicates and ambiguous ids
  * settlement: a 2u win at −110 gives +1.818u, a push gives 0, and a voided
    bet is excluded
  * horizon selection ignores forecasts recorded inside 60 minutes
  * drawdown values are correct
  * a cluster bootstrap over 3 identical bets per game is as wide as one bet
    per game, and >1.5× the naive iid interval
  * legacy rows without a best-side flag are handled

  The suite passes: 53 tests.
* `verify-log` gives 90 records, chain intact. `grade` on fresh data graded the
  6 TB @ DAL sides and prints the small-sample warning.

**Measured effect:** none on accuracy. One game is far too few to compare model
and market; that comparison accumulates from here.

## 5. Remove observed game-time wind (postgame knowledge) from the totals model

**Problem:** `t_wind` was the wind measured at the game. Training and validation
used it, but it is unknown at prediction time, and there are no historical
forecasts to substitute. Live, upcoming games have no wind value, so the model
silently used 0 mph for every outdoor game.

**Files:** `nflmodel/ratings.py` (`TOTAL_FEATURES`).

**Verification and measured effect** (walk-forward 2015–2026, same cached data
for both arms, 3,065 totals):

| | over-hits Brier | log loss | own-total MAE |
|---|---|---|---|
| with observed wind | 0.25022 | 0.69359 | 10.687 |
| without wind | 0.25029 | 0.69373 | 10.697 |

* Observed wind bought almost nothing, and that small gain was leakage.
* The live bias it caused: the own-total coefficient was −0.224 pts/mph and
  outdoor games average 7.7 mph. Unknown wind (0 mph) therefore raised an
  average outdoor total by **+1.7 pts** in the model's own line, and **+0.4
  pts** in the calibrated fair total.
* Weather forecasts from `weather_manual.csv` are still required before an
  outdoor total can be bet. They are reported as facts but don't enter the
  model.

## 6. Evaluation that matches live use, and honest uncertainty

**Problem:** audit issue 6.
* Model and market were scored on different game sets.
* Confidence intervals treated bets on the same game as independent.
* Drawdown was not reported.
* The README called the historical replay "the live rules".
* There was no statement of what each historical input knew relative to the
  prediction horizon.

**Files:**
* changed: `nflmodel/validate.py`, `nflmodel/metrics.py` (vectorized cluster
  bootstrap), `nflmodel/store.py`, `run.py` (`validate`, new `collect`),
  `README.md`, `reports/validation*.md`
* new: `tests/test_validation.py`

**Change:**
* `validate.py` documents the 60-minute horizon and, per input, whether it was
  available by then. Market prices are the binding gap: only closing lines
  exist historically.
* Every predictor is scored on identical rows. Results include Brier, log loss
  and ECE.
* The model − market Brier difference gets a game-level bootstrap CI and a
  per-season table.
* Calibration is shown for model and market.
* Betting replay:
  * game-clustered ROI CIs
  * flat and quarter-Kelly max drawdown
  * per-season results with each season's median overround
  * a sensitivity replay at standard retail juice (4.76%)
* The text states which decision rules the replay can and can't apply.
* `collect` snapshots validated quotes, injury reports and source times for
  upcoming games without the model. Each snapshot is indexed in hash-chained
  `logs/collections.jsonl`. Nothing is backfilled.

**Verification:**
* 3 new tests:
  * a missing market value removes the row for every predictor
  * the difference row equals the Brier gap
  * re-pricing hits the target overround while keeping no-vig probabilities

  The cluster-bootstrap test now checks the vectorized version. The suite
  passes: 56 tests.

**Measured results** (2015 to 2026 week 5, cached data):
* **Winner:** model 0.21283 vs market 0.21271 (+0.00012, 95% CI −0.00051 to
  +0.00068).
* **Covers:** +0.00026 (−0.00064 to +0.00118).
* **Overs:** +0.00034 (−0.00047 to +0.00111).
* **Replay at recorded prices:** +3.0% (−3.1% to +8.9%, game-clustered; 1,264
  bets on 1,008 games).
* **Replay at 4.76% juice:** +1.4% (−7.6% to +10.7%; 537 bets).

**New finding:** the consensus prices' overround is about 2.4% through 2022 and
about 4.7% from 2023. Every replay bet but one falls in 2015–2022, so the earlier
headline ROI depended on reduced-juice prices. **No claim of improved accuracy
or profitability is supported.**

**Numbers changed vs the baseline** (2,126 → 2,128 winner Brier, 1,333 → 1,264
replay bets) because of:
* change 5 (no wind)
* one more played game in the refreshed schedule (TB @ DAL)
* identical-row scoring

## 7. Additional data, tested one group at a time against the unchanged baseline

**Problem:** Phase 3. Test candidate data without contaminating the holdout or
counting the same information twice.

**Source checks:** all four groups come from nflverse-data releases.
* **Access:** verified from this environment.
* **Licence and cost:** CC-BY 4.0, free. Snap counts originate at Pro Football
  Reference and are redistributed by nflverse.
* **Coverage:**
  * play-by-play: 2010–2026 used
  * snap counts: 2013–2026 (the 2012 file is empty)
  * injury reports: 2013–2026, all 32 teams, every week
  * gsis ↔ pfr id crosswalk: 99.9% of Out/Doubtful players map
* **Update frequency:** daily during the season. Server Last-Modified on
  2026-10-09 for snap counts, 2026-10-07 for injuries. The nflverse schedule
  page is blocked here, so frequency was inferred from those timestamps.
* **Timestamps:** the injury feed has **no publish times**, so G3's timing is
  inferred from NFL reporting rules.

**Files:**
* new: `nflmodel/availability.py`, `nflmodel/experiment.py`,
  `tests/test_experiment.py`, `docs/EXPERIMENTS.md`, `logs/experiments.jsonl`
* changed: `nflmodel/data.py` (cache v3 with rushing EPA and neutral pace),
  `nflmodel/ratings.py` (experimental ratings and `EXPERIMENT_GROUPS`),
  `nflmodel/model.py` / `validate.py` (`total_features`), `run.py` (`experiment`)

**Protocol:** pre-registered in `docs/EXPERIMENTS.md` before any run, and
committed separately (`d3260e0`).
* Development seasons are 2015–2021; the holdout is 2022–2026.
* Only one group is added at a time, with no tuning.
* The acceptance rule is fixed in advance.
* The holdout can run once per group and only after a development pass. Code
  enforces this (`ProtocolError`), and 2 new tests cover it.

**Verification:**
* Baseline development-season results are unchanged after adding the new
  rating fields (max difference 5e-5, the report's rounding).
* The suite passes: 58 tests.

**Measured effect:** none of G1–G4 passes (table in `docs/EXPERIMENTS.md`).
No holdout run was made, and no group was promoted. G2 and G3 slightly lower
the model's own-line error but don't improve the market-anchored probabilities.

---

# Reliability fixes (second review)

## 8. Validate settings and pass them through every decision

**Problem:**
* `decide.py` hardcoded the Kelly fraction (0.25), the stake cap (2u) and the
  4-point gap rule.
* `settings.toml` could change `min_edge`, but not how bets were sized or
  blocked.
* Reports printed fixed text ("quarter Kelly", "4-pt gap") regardless of
  configuration.
* Settings were never range-checked.

**Files:** `nflmodel/config.py`, `nflmodel/decide.py`, `nflmodel/validate.py`,
`nflmodel/report.py`, `run.py`, `README.md`, `tests/test_settings.py` (new),
`tests/test_availability.py`.

**Change:**
* `Settings` validates itself on creation. It rejects wrong types (including
  booleans and strings), NaN and infinite values, out-of-range values, unknown
  `settings.toml` keys, and a clock skew no smaller than the odds age limit.
* `build_context`, `decide_game` and `validate.run` take the `Settings` object.
* Stakes come from `stake_units(full_kelly, settings)`.
* The weekly and validation reports print every setting actually used.

**Verification:**
* 8 new tests:
  * invalid values are rejected
  * `settings.toml` overrides and unknown keys are handled
  * on a synthetic market, lowering `max_stake_units` caps the stake
  * doubling `kelly_fraction` doubles it
  * `gap_points` 10 vs 4 turns a BET into NO BET
  * raising `min_edge` removes the bet

  The suite passes: 66 tests.
* A historical replay of 2020 with `max_stake_units=0.5` gives the same 104
  bets, with the largest stake 0.5u instead of 2u.

## 9. Validate QB confirmations and weather forecasts against the decision time and kickoff

**Problem:**
* `confirmed_utc` and `forecast_utc` only had to parse. A confirmation dated
  after the run, after kickoff or a week earlier was accepted.
* A forecast had no retrieval time, valid-for time or freshness limit.
* Negative, NaN or infinite wind and non-numeric temperatures were accepted.
* Any such row could clear a starter-availability condition or unlock an
  outdoor total.

**Files:** `nflmodel/inputs.py`, `nflmodel/config.py`, `run.py`,
`qb_overrides.csv` / `weather_manual.csv` (headers), `tests/test_inputs.py`
(new), `tests/test_offers.py`.

**Change:**
* `load_qb_overrides(day, settings, decision_utc)` and `load_weather(...)`
  validate every row before anything is applied. A row is rejected for:
  * a game or kickoff that doesn't match the schedule
  * a blank `source`, `qb_name` or `team`
  * a missing or naive timestamp
  * a timestamp after the decision time (plus clock skew)
  * a timestamp at or after kickoff
  * staleness: `qb_confirm_max_age_minutes` (48 h) or
    `weather_max_age_minutes` (12 h)
  * wind that isn't finite or isn't between 0 and 100 mph
  * temperature that isn't finite or isn't between −60 and 130 °F
  * a retrieval time before the issue time
  * a `valid_for_utc` more than `weather_valid_window_minutes` (180) from
    kickoff
  * a duplicate row
* Weather rows need two new columns, `retrieved_utc` and `valid_for_utc`.
* Rejections are kept with their reasons: printed, listed per game under
  "Missing or stale information" ("row rejected and NOT applied"), and saved to
  `inputs_rejected.csv` in the snapshot. Accepted rows are saved as
  `qb_accepted.csv` / `weather_accepted.csv`.

**Verification:**
* 12 new tests:
  * future, post-kickoff and stale confirmations are rejected, and the age
    limit is configurable
  * blank source, blank name, naive and missing times are rejected
  * future, post-kickoff and stale forecasts are rejected
  * eight kinds of invalid wind/temperature values are rejected
  * missing source, retrieval-before-issue and a valid-for time far from
    kickoff are rejected
  * decision level: a rejected confirmation leaves the bet "BET IF CONFIRMED",
    while valid ones give "BET"
  * decision level: a rejected forecast leaves the outdoor total blocked, while
    a valid one unblocks it

  The suite passes: 78 tests.
* Replay at 2026-10-11T12:30Z with three QB rows and three weather rows: the
  blank-source and future-dated QB rows, the negative-wind forecast and the
  stale forecast were rejected and listed; only the valid rows were applied.

## 10. Explicit weather handling and an archive of valid forecasts

**Problem:**
* Reports didn't say that weather is not a model feature.
* `apply_weather()` still wrote forecast wind into a column the model no longer
  reads, which wrongly suggested it affected predictions.
* Forecasts were not archived, so a weather feature could never be evaluated on
  real pre-game forecasts.

**Files:** `nflmodel/inputs.py` (removed `apply_weather`), `nflmodel/decide.py`,
`nflmodel/store.py`, `run.py` (`record`, `collect`, `verify-log`), `README.md`,
`tests/test_inputs.py`.

**Change:**
* Every report with a forecast states that weather is not a model input and
  only permits betting the total. The no-forecast block says the same.
* Accepted forecasts go to `logs/weather_forecasts.jsonl` (hash-chained, no
  duplicates) with:
  * game id and kickoff
  * valid-for time
  * wind and temperature
  * source
  * issue and retrieval times
  * run id
* Both `predict` (prospective runs only) and `collect` write to it, and
  `collect` validates weather too.
* No weather adjustment was added, and observed weather was not restored to
  training.

**Verification:** 3 new tests:
* `t_wind` is in no feature list or experiment group, and `apply_weather` is gone
* the report states that the forecast only unlocks the total, and shows its
  issue and retrieval times
* the archive stores all times, doesn't duplicate an identical forecast and
  verifies

The suite passes: 81 tests.

## 11. Run timestamps: stage clock, recheck at completion, true recorded time

**Audit:**
* `predict` captured one `now` at the start. That was before the model fit
  (about a minute), the injury download and the odds fetch.
* Quote freshness, "not in the future" and kickoff checks all used that start
  time, so quotes fetched mid-run could be rejected as future-dated.
* `record()` stamped every forecast with the start time, so a run that started
  at kickoff − 61 min and finished at kickoff − 55 min looked like a
  pre-horizon forecast.
* `collect` had the same issue.

**Files:**
* new: `nflmodel/runtime.py`, `tests/test_runtime.py`
* changed: `run.py`, `nflmodel/store.py`, `nflmodel/report.py`

**Change:**
* `Clock` stamps each stage when it actually happens:
  * `run_started`
  * `inputs_read` (the decision time for QB and weather rows)
  * `injuries_retrieved`
  * `odds_collected` (quotes are validated against this)
  * `prediction_completed`

  Every forecast record, snapshot meta and collection record carries these.
* `recheck_before_issue()` runs at prediction completion:
  * a BET whose quote has aged past `max_odds_age_minutes` is downgraded to
    "BET IF PRICE AVAILABLE"
  * a game whose kickoff has passed becomes NO BET and is **not recorded**
* `store.record_forecasts()` stamps `recorded_utc` with the actual write time.
  It raises if that would precede `prediction_completed_utc`, or if the
  completion time is in the future.
* Simulated runs use a fixed clock, are labelled "SIMULATED RUN" in the report,
  go to `reports/dev/`, and are never recorded.

**Verification:**
* 9 new tests:
  * the completion recheck: fresh → stays BET, stale-by-completion →
    downgraded, kickoff passed → NO BET, flagged and zero stake
  * conditional rows are left alone
  * the simulated clock is fixed; the real clock stamps in order
  * a quote timed after run start is valid against its collection time but
    would have been wrongly rejected against run start
  * the recorded time is the write time, not the run start
  * a future completion time is refused

  The suite passes: 90 tests.
* A simulated replay shows the banner and timeline.

**Historical records:** the 90 legacy forecasts keep their original
run-start stamps (unchanged, flagged `legacy`). They predate this fix, and all
fall days before kickoff.
