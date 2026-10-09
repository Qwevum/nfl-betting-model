# NFL betting model

Prices NFL spreads, moneylines and totals, compares them with sportsbook prices,
and returns **BET / NO BET with the evidence**: verified facts with their sources
and timestamps, assumptions, missing information, EV at the actual price, and
how EV changes if the model is wrong. Every prediction, including passes, is
logged before kickoff and graded afterwards.

## Setup

```bash
pip install -r requirements.txt       # requires Python 3.11+ (uses tomllib)
python -m unittest discover tests     # 58 regression tests
```

## Weekly workflow

```bash
python run.py check-live              # is the live odds feed configured and usable? (writes nothing)
python run.py collect                 # snapshot quotes + injury reports (run on a schedule)
python run.py predict                 # report for the upcoming week; records every forecast
python run.py recommend               # compact table for the next game day
python run.py place --forecast ID --stake 1   # record a wager you actually placed
python run.py grade                   # forecasts, recommendations and wagers, graded separately
python run.py verify-log              # integrity check of all hash-chained logs
python run.py validate                # out-of-sample validation (about 6 minutes)
```

`predict` writes `reports/<season>_week<NN>.md`, with one section per game:

* **Verified facts**: kickoff, stadium, roof, rest, projected starting QBs, the
  official injury report (or that game statuses are not issued yet), each tagged
  with its source and retrieval time.
* **Model estimate**: market line, the model's own line, the calibrated fair
  line, win probability, and the inputs that moved the model's line most.
* **Per market and side**: best price found, implied probability, the market's
  no-vig probability, the model's probability, EV, EV if the line is 0.5 pt worse,
  EV using the market alone, and the decision.
* **Decisions** with reasons, minimum acceptable price, sensitivity, and why the
  bet could be wrong.
* **Assumptions** and **missing or stale information**, listed separately.

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
| `ODDS_API_KEY` env var | every US book from the-odds-api.com with per-book update times; `python run.py check-live` verifies it |
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
4. **Market-anchored probability**. The estimate starts from the market's
   no-vig probability: the live multi-book reference built without the offer's
   own book, or the untimed nflverse consensus as a labelled fallback. A calibration fit only on past seasons' out-of-sample
   predictions, `sigmoid(a + b·logit(market) + c·(model − line))`, decides how far
   the model may move it. Currently `c` is about 0 for spreads, so the model adds
   nothing there; it is small and positive for totals and moneylines.
5. **Other lines and prices**. A key-number outcome distribution (3, 7, 10, ...
   happen more than a bell curve says) converts the probability to other lines,
   including push chances. This is what makes +7.5 vs +7 worth paying for.
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

Latest run, 2015 to week 5 of 2026, 3,082 games scored identically:

| target | model Brier | closing market | model − market (95% CI, game bootstrap) |
|---|---|---|---|
| winner | 0.21283 | 0.21271 | +0.00012 (−0.00051 to +0.00068) |
| home covers | 0.25016 | 0.24990 | +0.00026 (−0.00064 to +0.00118) |
| over hits | 0.25029 | 0.24996 | +0.00034 (−0.00047 to +0.00111) |

The model is **not** better than the closing market on any target, and
season-by-season differences go both ways. The QB feature lowers the model's
own-line error in 9 of 12 seasons; its effect on final probabilities is within
noise. Calibration error (ECE) is 0.017 for the model vs 0.016 for the market.

**Historical betting replay.** The replay places one side per market at the
closing consensus price. It applies only the EV, 0.5-point robustness and gap
rules. It cannot apply the live-only rules (quote freshness, the multi-book
reference, starter availability, weather), so these are not live results.

| prices | bets (games) | flat ROI | 95% CI (game-clustered) | max drawdown |
|---|---|---|---|---|
| recorded consensus | 1,264 (1,008) | +3.0% | −3.1% to +8.9% | 30.5u |
| standard retail juice (4.76%) | 537 (486) | +1.4% | −7.6% to +10.7% | 27.6u |

**Caution: the price source changed in 2023.** The recorded consensus prices
carry about **2.4% overround through 2022 and about 4.7% from 2023**. All
replay bets except one fall in 2015–2022, so the positive ROI comes from
reduced-juice prices most bettors can't get. At standard juice it is
indistinguishable from zero.

**Additional data** (`python run.py experiment --group G1..G4`, protocol and
results in `docs/EXPERIMENTS.md`):
* G1 rushing efficiency, G2 neutral-situation pace, G3 snap-weighted player
  availability and G4 offensive-line continuity were each tested against the
  unchanged baseline under a pre-registered protocol.
* None improved out-of-sample probabilities on the 2015–2021 development
  seasons. None was evaluated on the holdout, and none is used live.

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
