# NFL betting model

Prices NFL spreads, moneylines and totals, compares them with sportsbook prices,
and returns **BET / NO BET with the evidence**: verified facts with their sources
and timestamps, assumptions, missing information, EV at the actual price, and
how EV changes if the model is wrong. Every prediction, including passes, is
logged before kickoff and graded afterwards.

## Setup

```bash
pip install -r requirements.txt       # requires Python 3.11+ (uses tomllib)
python -m unittest discover tests     # regression tests
```

## Weekly workflow

```bash
python run.py check-live              # is the live odds feed configured and usable? (writes nothing)
python run.py collect                 # snapshot quotes + injury reports (run on a schedule)
python run.py predict                 # GAME FORECASTS + BETTING OPPORTUNITIES for the upcoming week; records both
python run.py forecast                # game forecasts only, from nflverse consensus lines (no odds requests, free)
python run.py recommend               # compact table for the next game day
python run.py place --forecast ID --stake 1   # record a wager you actually placed
python run.py coverage --season 2026 --weeks 5-8   # which slate games have horizon-eligible forecasts
python run.py grade --season 2026 --weeks 5-8      # forecasts, recommendations and wagers, graded separately
python run.py verify-log              # integrity check of all hash-chained logs
python run.py validate                # out-of-sample validation (about 6 minutes)
python run.py compare-models          # legacy vs model vs market-only probabilities, dev seasons
python run.py validate-forecasts      # out-of-sample check of forecasts and their intervals (dev seasons)
python run.py simulate-intervals      # known-truth simulation of the probability intervals (~45 min)
python run.py clv-backtest            # opening lines vs close and CLV (needs data/historical_lines.*; see below)
python run.py experiment --group G5   # pre-registered experiments G1-G6 (docs/EXPERIMENTS.md)
# predict / forecast / recommend take --uncertainty fast (default, 100 replicates, cached) | deep (1000) | off
```

**Latest report: [reports/LATEST.md](reports/LATEST.md).** Old pick files are kept in
`picks/archive/` and are not current.

Every report starts with three sections:
* **Actionable recommendations**: only sides meeting *every* requirement. Often empty;
  that is a valid result.
* **Watchlist (conditional, NOT recommendations)**: the closest candidates, why
  each fails, and the worst price at the *same line* that would clear the
  threshold *if the probability estimate stays the same*. A later quote at that
  price still needs a fresh run.
* **Why these decisions**: each side is categorised as negative estimated value,
  small positive value below threshold, blocked by rule, or insufficient
  information. Then come the overlapping reasons, a sequential filter funnel and
  the EV distribution by market.

`predict` writes `reports/<season>_week<NN>.md`, with one section per game:

* **Verified facts**: kickoff, stadium, roof, rest, projected starting QBs, the
  official injury report (or that game statuses are not issued yet), each tagged
  with its source and retrieval time.
* **Model estimate**: market line, the model's own line, the calibrated fair
  line, the home win probability (the same final probability as the moneyline
  rows, next to the raw market), and the inputs that moved the model's line most.
* **Per market and side**: best price found, implied (break-even) probability, and
  three separately labelled probabilities: the **raw market** (no-vig, without the
  priced book), the **calibrated market-only** probability, and the **final**
  probability used for EV. Then EV, EV if the line is 0.5 pt worse, EV at the raw
  market, and the status.
* **Decisions** with reasons, minimum acceptable price, sensitivity, and why the
  bet could be wrong.
* **Assumptions** and **missing or stale information**, listed separately.

## Game forecasts and their uncertainty (details: docs/UNCERTAINTY.md)

Every report now has two separate parts:
* **GAME FORECASTS.** These cover every game with enough data, whether or not there is a bet:
  * the predicted winner, and each team's win and tie probability
  * the projected home-minus-away margin and the projected total
  * the basis (market-informed, or team ratings only), data times, model version and missing inputs
* **BETTING OPPORTUNITIES.** For each market, the best side's book, line and odds, then:
  * P(win), P(push) and P(lose)
  * EV, with an EV range
  * the existing decision and its reasons

A team can be the predicted winner while every bet on its game is NO BET.

Two kinds of interval, computed separately:
* **Win probability ± interval.** This shows how much the *estimated* probability moves
  when the model is refit on re-weighted history. The method is a season-blocked
  Bayesian bootstrap of the whole fit, including calibration. Ratings and market
  quotes are held fixed, so the interval is conditional on the market snapshot.
  * A 90% interval is **not** a 90% chance of winning.
  * It is narrow (about ±1 point) because the forecast stays close to the market.
* **Margin / total prediction interval.** The range of the actual final score,
  from the full predictive distribution (game-to-game variation and key numbers
  such as 3 and 7). Out of sample, 90% intervals contained the actual margin in
  91.2–91.4% of games and the actual total in 90.5–91.1%.

**EV range.** Each replicate re-prices the side at the same fixed price and
reference, using its own joint win/push probabilities.
* The range does not change any decision.
* An experimental rule that also requires the EV lower bound to be above 0
  (`experimental_ev_lower_bound_rule`) is **off** by default.

Validation results:
* **Win probability:** the forecast scores the same as the closing moneyline
  (Brier difference −0.0002; the CI includes 0 in development and in the holdout).

Runtime and caching:
* **Fast run:** 100 replicates, about 40 s on 7 cores the first time; then cached
  per training cutoff in `data/cache/uncertainty/`.
* **`--uncertainty deep`:** 1,000 replicates.

## Your inputs (all optional)

Every row names its game by `game_id` and `kickoff_utc`, never by team names.
`python run.py templates` writes those rows for the next week into `templates/`.
Rows for unknown games, or whose kickoff doesn't match the schedule (for
example a rescheduled game), are rejected and listed in the output.

| file | use |
|---|---|
| `odds_manual.csv` | quotes from your sportsbook apps, with `odds_time_utc` (ISO-8601 UTC) |
| `qb_overrides.csv` | a starting QB you have confirmed for one game, with `confirmed_utc` |
| `weather_manual.csv` | wind/temperature forecast with `forecast_utc`; required to bet an outdoor total |
| `.env` file or `ODDS_API_KEY` env var | your the-odds-api.com key: every US book with per-book update times. Put `ODDS_API_KEY=...` in `.env` (gitignored, never commit it); `python run.py check-live` verifies it and prints remaining credits |
| `odds_api_regions` setting | regions per request (default `us`). Each request costs 3 credits per region (h2h, spreads, totals), so one `predict` or `check-live` costs 3 credits on the default |
| `settings.toml` | overrides for `nflmodel/config.py`, e.g. `max_odds_age_minutes = 20` |

Bookmaker quotes are validated **before** the best price is chosen. A quote is
rejected if:
* its timestamp is missing, naive (no timezone) or in the future
* it is older than `max_odds_age_minutes` (default 30)
* it was quoted at or after kickoff, or the run itself is at or after kickoff
* its price, line or market/side is invalid

Games that have started are skipped.

**Market reference.** The market's probability for each bet is built from the
validated quotes in four steps:
1. **Pairing.** Each book's two sides must be at the same line and quoted within
   5 minutes of each other.
2. **Vig removal.** Proportional: the two sides are scaled to sum to 100%.
3. **Different lines.** Each book's probability at its own line is converted
   into an implied mean margin or total using the key-number outcome
   distribution, so +3 and +3.5 quotes become comparable.
4. **Median across books.** The reference is the median, and the evaluated
   book is left out of its own reference.

At least `min_reference_books` (default 2) *other* books are needed. Otherwise
the bet can only be "BET IF PRICE AVAILABLE", and the untimed nflverse
consensus is used as a fallback. `--now 2026-10-11T15:00:00Z` replays a run
at a fixed time; such runs, and runs from uncommitted code, write to `reports/dev/`
and are never logged.

**Completion check.** When predictions finish, **all** quotes are revalidated:
the selected offers and every quote inside each market reference, on both
sides of each pair.
* If any quote has expired, the slate is re-priced from the quotes still valid.
  References are rebuilt with the same pairing and leave-one-book-out rules,
  and probabilities, EV, sensitivity, decisions and stakes are recomputed, up
  to 2 times.
* If fewer than `min_reference_books` other books remain, the bet becomes
  "BET IF PRICE AVAILABLE" with the reason stated.
* A final safeguard downgrades any executable bet whose selected or reference
  quote is still stale, and drops games that have kicked off. If re-pricing
  can't settle, the result is conservative, never a stale-reference BET.
* The report, recorded forecasts and snapshot use the final pricing. The
  snapshot also keeps `references_at_collection.csv` for audit.

## Weather

* **Not a model input.** Wind and temperature are not features: observed
  game-time weather was removed because it is postgame knowledge, and no
  historical forecasts exist to train on.
* **What a forecast does.** A valid row in `weather_manual.csv` only permits
  betting the total of an outdoor or unknown-roof game. The model's total is
  identical with or without it.
* **Validation.** Forecasts are checked against the decision time and kickoff
  (issued at most `weather_max_age_minutes` = 12 h earlier; valid-for time
  within 3 h of kickoff; wind ≥ 0; plausible temperature).
* **Archive.** Accepted forecasts are archived in `logs/weather_forecasts.jsonl`
  with issue time, retrieval time, valid-for time, kickoff and source. A future
  weather feature can then be evaluated on forecasts that really existed before
  each game.

## Data sources

All from the [nflverse](https://github.com/nflverse) project. Each run records
URL, retrieval time and server Last-Modified in `data/sources.json` and in the report.

| data | used for | limits |
|---|---|---|
| schedule/results (`games.csv`) | scores, consensus closing lines, rest, roof, projected starting QBs | lines have no per-book timestamp; projected QBs are not official |
| play-by-play | EPA, success rate, QB EPA per dropback | updated nightly |
| injury reports | game status and practice participation | game statuses usually appear Friday (Wednesday for Thursday games) |

Not available here, so reported as missing instead of guessed: weather forecasts,
travel distance, confirmed inactives, and timestamped book prices unless you supply them.

## How the probability is calculated

1. **Ratings** (`nflmodel/ratings.py`). Games are replayed in date order. Each
   team has opponent-adjusted ratings for points margin, EPA/play, success rate
   and pass EPA (offense and defense), plus pace and points. Only games before
   kickoff are used, and 60% of each rating carries into the next season.
2. **QB change**. Each QB has an opponent-adjusted EPA/dropback rating, shrunk
   toward the measured level of inexperienced QBs. A team's pass ratings already
   reflect its usual QB, so the feature is only *starter rating minus the QB play
   the team ratings reflect*. A team with its usual starter gets about 0, so the
   QB is not counted twice.
3. **Model line** (`nflmodel/model.py`). Ridge regression on those features gives
   the model's own margin and total. A second regression adds the market line.
4. **Market-anchored probability** (`probability_model = "model"`, the default).
   The market reference comes first: the live multi-book reference built without
   the offer's own book, or the untimed nflverse consensus as a labelled fallback.
   It gives an implied mean margin/total. A key-number outcome distribution (3, 7,
   10, ... happen more than a bell curve says) turns that mean into the market's
   probability at the **exact line being priced**, with push chances. This is what
   makes +7.5 vs +7 worth paying for. The model then moves it by
   `sigmoid(logit(market) + c·edge)`:
   * the market's log-odds keep weight 1, with no intercept;
   * `edge` is the blend's margin/total minus the reference line, with the blend
     recomputed at the line built without the priced book;
   * `c` (the value of one point of edge) is fit only on past seasons'
     out-of-sample predictions. It is about 0 for spreads and totals, and about
     0.03 per point for moneylines.
5. **Comparison arms.** `probability_model = "market"` uses the reference alone
   (pure price shopping). `"legacy"` is the previous calibration
   `sigmoid(a + b·logit(market) + c·edge)` with a free intercept and slope; it is
   kept only as a baseline (see "Probability audit" below).
6. **EV** = `p_win × (decimal − 1) − p_lose` at the actual price; pushes return the stake.

## Bet / no-bet rules (`nflmodel/decide.py`)

**NO BET** if any of these hold:
* EV ≤ 2% (`--min-edge`)
* EV ≤ 0 if the true line is 0.5 pt worse (too sensitive)
* the projected starting QB is Out, Doubtful or Questionable, did not practice
  with no status issued yet, or is unknown
* the model's own line is `gap_points` (default 4) or more from the market (unexplained)
* the bet is a total for an outdoor or unknown-roof game with no forecast

There are three passing tiers:
* **BET** (executable): a validated, timestamped book price; a live reference
  built without that book; and nothing left to confirm.
* **BET IF CONFIRMED** (conditional): an executable price, but something must
  be confirmed first. The usual case is a starter whose availability is
  unknown: the injury feed failed, the team's report for the week isn't out,
  or game statuses aren't issued yet. Missing injury data is never read as
  "healthy". A confirmed starter in `qb_overrides.csv` settles it.
* **BET IF PRICE AVAILABLE** (conditional): the price is unverified, either an
  untimed consensus price or no live reference without that book. The report
  gives the worst price that is still +EV.

Stakes are `kelly_fraction` × full Kelly (default 0.25), capped at `max_stake_units` (default 2;
1u = 1% of bankroll). Every report prints the settings it actually used.

**Your sportsbooks.** In `settings.toml`, set `actionable_books = "draftkings,fanduel"`
(Odds API bookmaker keys). Every book still feeds the market references, but a
bet is chosen only among your books. A side whose best price is only at other
books is NO BET ("best price only at books you don't use"). If unset, every book
counts as actionable and the report says so. `check-live` shows each game's lines
and prices across books, marking reference-only books.

## Probability audit (2026-10, details in docs/CHANGES.md §20)

* The legacy calibration moved probabilities 1.5–2.4 points from the market on
  average (development seasons). This did not significantly improve log loss.
* In live use it was applied at the stale consensus line. A 1.5-point line move
  created about 6 points of spurious probability in the regression test.
* The shared model inputs (the blend's market line) included the priced book's
  own quote.
* All three are fixed. On 2015–2021, the corrected model's log loss is not
  significantly different from legacy on any target
  ([reports/probability_models.md](reports/probability_models.md)). Bet counts
  and historical ROI were reported but **not** used to decide.

## Validation (`python run.py validate`, full tables in `reports/validation.md`)

**What is validated, and what isn't:**
* Each season is predicted by a model whose ratings, regressions and
  calibration use only earlier seasons.
* Live forecasts target **kickoff − 60 minutes** (`horizon_minutes`), but the
  only historical market prices available are nflverse **closing** consensus
  lines, which come later than that.
* So model and market are compared at the closing information set, on exactly
  the same games. That is the market's best case.
* Observed weather is not used. Injury reports are not used historically,
  because the feed has no publish times.
* Starters are the actual starters. Inactives are normally public by the
  horizon.
* Nothing is backfilled: `python run.py collect` and every recorded `predict`
  snapshot the inputs available at the time, building horizon-matched history
  from now on.

Latest run, 2015 to week 5 of 2026, 3,082 games scored identically (live
`probability_model = model`):

| target | model Brier | closing market | model − market (95% CI, game bootstrap) |
|---|---|---|---|
| winner | 0.2128 | 0.2127 | +0.0001 (−0.00039 to +0.00049) |
| home covers | 0.2500 | 0.2499 | +0.0001 (−0.00013 to +0.00030) |
| over hits | 0.2501 | 0.2500 | +0.0001 (−0.00022 to +0.00045) |

The model is **not** better than the closing market on any target, and
season-by-season differences go both ways.

**Value of the model's edge.** The live probability is
`sigmoid(logit p_market + c × edge)`, where edge is the model's line minus the
market line. Its 95% CI on `c` (game bootstrap, per walk-forward season):
* **spreads:** includes 0 in **12 of 12** seasons
* **totals:** includes 0 in **12 of 12** seasons
* **moneylines:** includes 0 in **11 of 12** seasons

So the model's disagreement with the closing line is not distinguishable from
no information. (Each season's model uses all earlier seasons, so these
intervals overlap rather than being independent.)

**Historical betting replay.** The replay places one side per market at the
closing consensus price. It applies only the EV, 0.5-point robustness and gap
rules. It cannot apply the live-only rules (quote freshness, the multi-book
reference, starter availability, weather), so these are not live results.

| prices | bets (games) | flat ROI | 95% CI (game-clustered) | max drawdown |
|---|---|---|---|---|
| recorded consensus | 76 (72) | +1.1% | −26.0% to +31.3% | 13.5u |
| standard retail juice (4.76%) | 13 (13) | +23.7% | −41.7% to +93.2% | 3.0u |

With the live calibration the replay almost never bets. Every bet falls in
2015–2021, when the recorded prices carried about **2.4% overround**; from
2023 they carry about **4.7%** and the replay places none. Both ROI intervals
include 0. (The previous report, 1,264 bets, predates the live calibration.)

**Additional data** (`python run.py experiment --group G1..G4`, protocol and
results in `docs/EXPERIMENTS.md`):
* G1 rushing efficiency, G2 neutral-situation pace, G3 snap-weighted player
  availability and G4 offensive-line continuity were each tested against the
  unchanged baseline under a pre-registered protocol.
* None improved out-of-sample probabilities on the 2015–2021 development
  seasons. None was evaluated on the holdout, and none is used live.
* **G5, informed preseason prior.** At a team's first game of a new season, a
  change of starting QB moves the team's rating. It **did not pass**:
  * every log-loss CI includes 0
  * own-line MAE got slightly worse, 10.212 → 10.248
  * the preseason win-total variant (G5a) needs `data/preseason_win_totals.csv` and was not run
* **G6, ratings hyperparameter search.** 13 configurations were tried, and all are
  recorded. The best changed k_margin 0.05, a_eff 0.07, carry 0.70, qb_shrink 100
  and qb_decay 0.985. It **did not pass**:
  * own-line MAE 10.212 → 10.183
  * log loss −0.00048 (winner), −0.00022 (home covers), −0.00031 (over), with
    every CI including 0
  * picking the best of 13 configurations on these same seasons favors it
* No holdout runs for G5 or G6. Both remain available as ratings parameters,
  0 / current by default, through `experiment --group`; neither is live.

## Spreads: calibration, opening lines and closing-line value

**Calibration diagnosis** (`reports/spread_calibration_diagnosis.md`, read-only).
The legacy calibration `sigmoid(a + b·logit p + c·edge)` is kept only as a baseline:
* **Fitted values** on the inner out-of-sample rows: b = 0.36 (95% CI −0.50 to
  +1.19) and c ≈ 0 (−0.055 to +0.047).
* **What drives its "edges".** On the 2026 week 5 slate, its shifts from the
  no-vig price average 0.95 points. Of that:
  * the market remapping (a, b) contributes up to 2.35 points
  * the model (c × edge) contributes 0.02 points
* **The live calibration** keeps a = 0, b = 1 and fits only c. It moves cover
  probabilities by at most 0.12 points on that slate.
* **Head to head:** on home covers, development seasons, the two calibrations are
  indistinguishable (log loss difference +0.0003, 95% CI −0.0024 to +0.0028).

**Opening lines and CLV** (`python run.py clv-backtest`). nflverse has closing lines only.
* **What to supply:** a historical open/close file such as the AusSportsBetting
  NFL spreadsheet, saved as `data/historical_lines.csv`, or as `.xlsx` with
  `pip install openpyxl`.
* **How it's read:**
  * columns are inferred, or set by `data/historical_lines.mapping.json`
  * teams are mapped per season, and decimal odds are detected
  * the line sign is checked against nflverse
  * unmatched rows are listed
* **What it reports:**
  * whether the closing line moves toward the model (the slope of close − open
    on model − open, with a CI)
  * the result of betting at the open when EV exceeds `min_edge`: CLV in
    points and in no-vig probability, plus ROI at the opening price and at −110
* **Result, development seasons 2015–2021** (AusSportsBetting file; 1,889 games,
  all matched; `reports/clv_backtest.md`):
  * **The close moves toward the model.** Regressing close − open on own line − open
    gives a slope of **0.17 (95% CI 0.14 to 0.21)**, so about a sixth of the model's
    disagreement with the opener is gone by kickoff.
  * **It survives removing QB hindsight.** The backtest uses the actual starter, which
    may not be known at the open. On games where both teams started their usual QB
    (622), the slope is still 0.15 (0.09 to 0.21).
  * **Large disagreements move most.** When the model and the opener differed by 3+
    points (118 of those games), the line moved 0.61 points toward the model on
    average: toward it 51% of the time, away 26%.
  * **The live rule bets almost never.** "Bet at the open when EV > 2%" placed only
    6 bets (6-0; CLV +0.58 pts, CI −0.50 to +1.67). The live calibration `c` is fit
    against CLOSING lines and is about 0, so it gives the model almost no weight
    at the open. Six bets show nothing either way.
  * **Not yet a betting rule.** Turning this into one, for example by calibrating
    `c` against opening lines, is a new decision rule. It would need its own
    pre-registered experiment and a single holdout run.

**Early decision horizon** (live). Forecasts issued between `early_horizon_max_hours`
(168) and `early_horizon_min_hours` (84) before kickoff are tagged `early`.
* For a Sunday game, that window runs from roughly Sunday night to Wednesday.
* Kickoff − 60 forecasts are tagged `kickoff-60`.
* Both are recorded, and `grade` evaluates each window separately.

**CLV in `grade`.** For each selected forecast, at both horizons, and for each
wager, `grade` reports CLV against the last valid pre-kickoff market reference.
* That reference is rebuilt from the latest snapshot of validated quotes before
  kickoff, using at least `min_reference_books` books.
* It reports points and no-vig probability, with game-clustered 95% CIs.
* It needs `collect` runs with live quotes, scheduled below.

### Collecting quotes on a schedule (not installed for you)

CLV and the early horizon need snapshots from the opener until kickoff. Each
`collect` with a live key costs about 3 credits per region per run.

A practical schedule:
* every 4 hours from Sunday 20:00 ET through Saturday
* hourly on game days from 08:00 ET
* one run about 90 minutes before each kickoff window

Windows (Task Scheduler), from the repo folder:

```powershell
schtasks /Create /TN "nfl-collect" /SC HOURLY /MO 4 /ST 00:00 `
  /TR "cmd /c cd /d \"C:\path\to\nfl-betting-model\" && python run.py collect >> logs\collect_task.log 2>&1"
schtasks /Query /TN "nfl-collect"     # check;  schtasks /Delete /TN "nfl-collect" /F  to remove
```

macOS / Linux (cron, `crontab -e`):

```
0 */4 * * *  cd /path/to/nfl-betting-model && python run.py collect >> logs/collect_task.log 2>&1
0 8-23 * * 0  cd /path/to/nfl-betting-model && python run.py collect >> logs/collect_task.log 2>&1
```

## Coverage of an explicit slate

`python run.py coverage` (and the first section of `grade`) compares the forecast
history with **every scheduled game** in a slate.
* Choose the slate with `--season` plus `--weeks 5` or `--weeks 5-8`, or with
  `--from YYYY-MM-DD --to YYYY-MM-DD`; choose markets with
  `--markets spread,ml,total`.
* Each game gets exactly one status:
  * eligible: a run was issued inside the horizon window
  * forecasts exist, none qualify: with the reason
  * no forecast recorded
  * window not closed: not counted as missed
  * unknown kickoff
* Percentages use a stated denominator: the games whose window has closed.
* Within eligible games, the report lists markets with no forecast or no market
  reference.
* It works with an empty history, writes nothing, and never creates or backfills
  forecasts. `--list` prints every game's status.

## Forecast history, snapshots and the bet ledger

* **`logs/forecasts.jsonl`**: every side of every market priced by a live run
  from committed code, recorded before kickoff. Passes are included, and each
  record carries its tier: executable, conditional or pass. The file is
  append-only and hash-chained, so `python run.py verify-log` detects any edit,
  deletion or reordering.
* **`snapshots/<run_id>/`**: the exact inputs of each recorded run: schedule
  rows, raw/rejected/valid quotes, references, predictions, the injury file,
  source timestamps and your input files, plus a manifest of hashes.
* **`logs/ledger.jsonl`**: only wagers you actually placed, recorded with
  `python run.py place --forecast <id> --stake <units> [--price ...]`. The id
  is printed by `predict`. `void` records cancellations.
* **`python run.py grade`** reports three things separately:
  1. forecast quality at the horizon. Only forecasts **issued** inside
     `[kickoff − horizon − tolerance, kickoff − horizon]` count (default 75–60
     min). Issued means the later of recording and prediction completion, so a
     late-finishing run can't qualify. The latest eligible run is used for all
     of a game's sides (never mixed), and the report lists eligible and
     excluded games with reasons and the actual lead times. Model and market
     are scored on identical rows, by market and model version
  2. hypothetical recommendations by tier (not wagers)
  3. actual wagers: ROI, drawdown and CLV

`logs/predictions.csv` holds the original Oct 7 log; its rows were imported
into the history and flagged `legacy`.

Bet only what you can afford to lose.
