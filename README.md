# NFL betting model

Prices NFL spreads, moneylines and totals, compares them with sportsbook prices,
and returns **BET / NO BET with the evidence**: verified facts with their sources
and timestamps, assumptions, missing information, EV at the actual price, and
how EV changes if the model is wrong. Every prediction, including passes, is
logged before kickoff and graded afterwards.

## Setup

```bash
pip install -r requirements.txt       # Python 3.10+
python -m unittest discover tests     # betting-math checks
```

## Weekly workflow

```bash
python run.py predict                 # report for the upcoming week + log every prediction
python run.py recommend               # compact table for the next game day
python run.py grade                   # after games: probability quality, bets, passes, CLV
python run.py validate                # out-of-sample validation (about 2 minutes)
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
| `ODDS_API_KEY` env var | every US book from the-odds-api.com (free tier) with per-book update times |
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
4. **Market-anchored probability**. The estimate starts from the consensus
   no-vig probability. A calibration fit only on past seasons' out-of-sample
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
* the model's own line is 4+ pts from the market (unexplained)
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

Stakes are a quarter of Kelly, capped at 2 units (1u = 1% of bankroll).

## Validation (`python run.py validate`, full tables in `reports/validation.md`)

Walk-forward: each season is predicted by a model fit and calibrated only on
earlier seasons. Latest run, 2015 to week 4 of 2026, 3,082 games:

| target | model | closing market (no-vig) | baseline |
|---|---|---|---|
| winner (Brier) | 0.2126 | 0.2126 | ratings only 0.2189, home rate 0.2479 |
| home covers (Brier) | 0.2502 | 0.2500 | coin flip 0.2500 |
| over hits (Brier) | 0.2502 | 0.2500 | coin flip 0.2500 |

Bets under the live rules against closing consensus prices: 1,333 bets,
646-663-24, flat-stake ROI **+2.3% (95% range −3.5% to +8.0%)**: spreads +3.4%,
moneylines +10.7% (on only 219 bets, range −7.5% to +28.8%), totals −3.5%.
Quarter-Kelly staking returned +1.4% on the amount staked.

**What this means:** the model is about as accurate as the closing line, not
better. No result above is statistically significant. The realistic edge is
**line shopping**: betting a number better than the market consensus. That is
why consensus prices alone almost never produce a bet. The QB feature lowered
the model's own-line error in 9 of 12 seasons, and its contribution to final
probabilities is small.

**Information timing:** historical predictions use closing lines (available
before kickoff) and actual starting QBs (normally known about 90 minutes before
kickoff). Historical weather is the observed value, not a forecast. Model design
choices were made after seeing 2015–2025 results, so the only pristine holdout is
the prediction log from now on.

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
  1. forecast quality at the 60-minute horizon
  2. hypothetical recommendations by tier (not wagers)
  3. actual wagers: ROI, drawdown and CLV

`logs/predictions.csv` holds the original Oct 7 log; its rows were imported
into the history and flagged `legacy`.

Bet only what you can afford to lose.
