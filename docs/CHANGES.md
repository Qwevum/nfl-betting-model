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
