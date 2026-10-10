# Feature experiments: protocol (written before any experiment was run)

## Periods
* **Development:** walk-forward test seasons 2015–2021. All comparisons and any
  decision to continue use only these seasons.
* **Holdout:** test seasons 2022–2026 (to date). It is evaluated **once** per
  feature group, and only for groups that pass the development criteria.
  `run.py experiment --holdout` refuses a second holdout run for the same group;
  every run is recorded in the hash-chained `logs/experiments.jsonl`.
* Caveat: earlier work in this repo (before this protocol) looked at 2015–2025
  results, so the holdout is not pristine for the baseline model itself. It is
  fresh for the new feature groups.

## Method
* One feature group at a time is added to the unchanged baseline. Both arms use:
  * identical training rows
  * the same walk-forward fitting and inner calibration
  * the same ridge penalty
* No hyperparameters are tuned for the new features: they use the same
  smoothing (EWMA step, season carry-over) as the existing ratings.
* Seasons without coverage for a group (for example snap counts before 2013)
  get the neutral value 0 in **both** arms' feature tables. The baseline
  ignores the column.

## Metrics
All are computed on identical rows for both arms:
* log loss and Brier of the final (market-anchored) probability for winner,
  home covers and over hits
* the model's own-line MAE, a signal check that does not affect decisions
* a paired, game-level bootstrap 95% CI of the log-loss difference
  (experiment − baseline)
* the number of development seasons where the experiment's log loss is lower

## Acceptance criteria (development)
A group passes only if **all** of these hold:
1. for at least one target, the pooled log-loss difference is < 0 and its 95%
   CI excludes 0
2. for no target is the difference > 0 with a CI excluding 0
3. the improving target improves in at least 5 of the 7 development seasons

A group that passes is evaluated once on the holdout. It becomes a default
feature only if the holdout log-loss difference for the same target is ≤ 0.
Otherwise it stays experimental: available via `--features`, off by default.

## Feature groups

| id | group | source | timing |
|---|---|---|---|
| G1 | separate rushing efficiency (opp.-adjusted rush EPA/play, offense & defense; passing already in the model) | play-by-play | prior games only |
| G2 | neutral-situation pace (seconds per play, Q1–Q3, score within 7) | play-by-play | prior games only |
| G3 | snap-weighted availability: expected snap share (prior 4 games) of players listed Out/Doubtful, non-QB | injury reports + snap counts | the final injury report precedes kickoff by NFL rule, but the feed has **no publish timestamps**; labelled as inferred timing |
| G4 | offensive-line continuity: share of last game's OL snaps taken by the team's top-5 OL of the prior 4 games | snap counts | prior games only |

## Results: development period (2015–2021), run once each at commit `266d256`

Recorded in `logs/experiments.jsonl`. Differences are experiment − baseline
log loss, with a paired game-level bootstrap 95% CI. Negative means the
experiment is better. Own-line MAE is a signal check only.

| group | winner | home covers | over hits | own-line MAE (margin / total) | verdict |
|---|---|---|---|---|---|
| G1 rushing efficiency | −0.00008 (−0.00042, +0.00028), 3/7 | +0.00001 (−0.00021, +0.00023), 5/7 | −0.00000 (−0.00032, +0.00030), 3/7 | 10.212→10.224 / 10.831→10.836 | fail |
| G2 neutral pace | 0 (totals-only group) | 0 | +0.00034 (−0.00021, +0.00090), 4/7 | — / 10.831→10.819 | fail |
| G3 snap-weighted availability | +0.00024 (−0.00033, +0.00085), 2/7 | +0.00003 (−0.00040, +0.00042), 3/7 | +0.00002 (−0.00030, +0.00035), 2/7 | 10.212→10.206 / 10.831→10.837 | fail |
| G4 OL continuity | −0.00001 (−0.00040, +0.00037), 4/7 | +0.00002 (−0.00030, +0.00036), 2/7 | +0.00023 (−0.00055, +0.00107), 3/7 | 10.212→10.209 / 10.831→10.834 | fail |

No group meets the criteria, so **no holdout evaluation was run** and all four
stay experimental. They are available to `validate.run(features=...,
total_features=...)` and `run.py experiment`, but are off in live predictions.

How to read this:
* G2 (pace) and G3 (availability) slightly reduce the model's *own-line* error.
  That shows they carry some signal about scores.
* But once anchored to the market, they don't improve the final probability:
  the market already prices that information.
* These are null results on 1,843–1,881 games. They do not show the data is
  useless. They show it adds nothing measurable beyond closing prices.

## CAL: probability model comparison (protocol written 2026-10-09, before the recorded run)

**Why.** An audit of the 2026 week 5 run found the legacy calibration
`sigmoid(a + b*logit(p_market) + c*edge)`:
* has a free intercept and slope on the market's log-odds that move every
  probability by about 1–2.4 points (mean, development seasons) with no
  significant log-loss gain, and a spread slope whose sign changes across seasons;
* in live use, is applied at the stale consensus line to references that moved,
  an input range it was never fit on (up to 6 points of spurious probability in the
  regression test);
* measures the model's edge against a line that includes the evaluated book.

**Disclosure.** Before writing this protocol, the development-period log loss of
the raw market vs the legacy calibration was looked at during the audit (scratch
analysis, not recorded). The holdout (2022+) was not looked at and is not used.

**Arms** (identical walk-forward fits, games, prices and Settings except
`probability_model`):
* `legacy`: the baseline, unchanged
* `model`: offset calibration: the market's log-odds kept with weight 1, no
  intercept; only `c` (value of one point of model edge) is fit out of sample;
  probabilities anchored at the exact line priced
* `market`: the market reference alone (`c = 0`)

**Metrics** on identical development rows (2015–2021): log loss, Brier, ECE per
target; paired game-clustered bootstrap CIs of log-loss differences; bet
frequency, flat ROI with game-clustered CIs and max drawdown at closing consensus
prices and at a fixed −110/−110 juice; market coverage.

**Decision rule.** `model` is a correctness fix and stays the default unless it is
significantly worse than `legacy` (paired log-loss CI entirely above 0) on any
target. Bet counts and ROI are reported but are NOT a criterion: more bets or a
higher historical ROI are not evidence of improvement, and the single consensus
price per side cannot test price shopping (the `market` arm cannot bet on it by
construction). No thresholds are tuned on these results.


## G5 and G6: preseason prior and ratings hyperparameters (protocol written 2026-10-09, before any G5/G6 run)

Both groups change how the ratings are BUILT (ratings.py), not which features the
regressions use. Each arm rebuilds the full feature table in date order with its own
ratings settings; both arms then use the same features list, the same walk-forward fits
and calibration, and are scored on identical games. Development seasons 2015–2021 only;
the general method, metrics and acceptance criteria above apply unchanged (paired
game-level bootstrap of log-loss differences, criteria 1–3, holdout once and only after a
development pass).

**Disclosure.** Before writing this, the spread calibration was diagnosed on the current
model (reports/spread_calibration_diagnosis.md): the offset c is about 0 with a CI that
includes 0. No G5/G6 configuration was run or looked at.

### G5: informed preseason prior

Currently every team rating keeps `carry` = 60% at a new season and regresses to 0.
* **G5b (runnable now): prior-season rating adjusted for a change of starting QB.** At a
  team's first game of a new season, if the starter differs from the QB who played most of
  the team's last game of the previous season, the points rating moves by
  `preseason_qb_pts` x (new QB rating - previous QB rating), and the team's QB baseline is
  reset to the new QB, so `f_qb` does not count the same change again. In-season QB changes
  are unchanged. `preseason_qb_pts = 35` (about 35 dropbacks per game x EPA per dropback =
  points per game), fixed in advance and not tuned.
* **G5a (needs data): preseason win totals.** `data/preseason_win_totals.csv` with season,
  team, win_total, odds, retrieved_utc. Rows retrieved at or after the season's first
  kickoff are rejected (they could contain information from the season). A win total
  becomes a prior margin `13.5 * Phi^-1(win_total / games)`, centered on the league each
  season, and the new-season points rating is `(1 - w) * carry * rating + w * prior` with
  `w = 0.5`, fixed in advance. Not run unless the file is supplied.

### G6: ratings hyperparameter search

Search space (one parameter moved at a time from the current values; 12 configurations):

| parameter | current | tried |
|---|---|---|
| k_margin | 0.07 | 0.05, 0.09 |
| a_eff | 0.10 | 0.07, 0.13 |
| carry | 0.60 | 0.50, 0.70 |
| qb_shrink | 150 | 100, 250 |
| qb_decay | 0.97 | 0.95, 0.985 |
| a_pace | 0.10 | 0.07, 0.13 |

Then one combined configuration: every single-parameter move that lowered the score,
applied together (if two moves of the same parameter lowered it, the lower one). That
makes at most 13 configurations besides the current one. Nothing else is tried.

**Score** of a configuration, development seasons 2015–2021, walk-forward: the
ratings-only own-line MAE of the margin (primary) and the home-covers log loss of the
live probability (reported). Every configuration and its scores are recorded in
logs/experiments.jsonl (group `G6-search`).

**Selection.** The configuration with the lowest own-line MAE is compared with the current
values under the general criteria (group `G6`). Selecting the best of several
configurations on the development seasons and then testing it on the same seasons favors
it; the holdout is the guard: run once, only if the development criteria pass.


### G5 / G6 results (appended 2026-10-09; protocol above unchanged)

Recorded in `logs/experiments.jsonl`, development seasons 2015–2021, commit `6455057`.
Differences are experiment minus current log loss, with paired game-bootstrap 95% CIs.

| group | winner | home covers | over hits | own-line MAE (margin / total) | verdict |
|---|---|---|---|---|---|
| G5b (QB change at a new season, 35 pts per EPA/dropback) | +0.00015 (−0.00066, +0.00098), 3/7 | +0.00010 (−0.00052, +0.00072), 3/7 | 0.00000 (0, 0), 0/7 | 10.212 → 10.248 / 10.831 → 10.831 | fails: no CI below 0 |
| G6 best of 13 (k_margin 0.05, a_eff 0.07, carry 0.70, qb_shrink 100, qb_decay 0.985) | −0.00048 (−0.00106, +0.00007), 5/7 | −0.00022 (−0.00067, +0.00024), 6/7 | −0.00031 (−0.00076, +0.00020), 4/7 | 10.212 → 10.183 / 10.831 → 10.840 | fails: no CI below 0 |

G6 search, own-line MAE (margin) per configuration:

| configuration | MAE |
|---|---|
| current | 10.2121 |
| k_margin 0.05 / 0.09 | 10.1972 / 10.2336 |
| a_eff 0.07 / 0.13 | 10.2019 / 10.2203 |
| carry 0.50 / 0.70 | 10.2273 / 10.2067 |
| qb_shrink 100 / 250 | 10.2114 / 10.2138 |
| qb_decay 0.95 / 0.985 | 10.2165 / 10.2120 |
| a_pace 0.07 / 0.13 | 10.2121 / 10.2121 (a_pace only affects totals) |
| combined (all improving moves) | 10.1825 |

* **Neither group passed, so neither was evaluated on the holdout.** Live
  defaults are unchanged.
* **G6's best configuration is an optimistic estimate.** It was chosen on the
  same seasons it was scored on.
* **The changes are small next to the gap to the market.** The best own-line
  MAE gain is 0.03 points. In 2015–2021 the closing line's MAE is 0.15–0.36
  points lower than the model's own line, about 0.24 on average
  (reports/validation.md).
* **G5a (win totals) was not run.** It needs `data/preseason_win_totals.csv`.


## OL1: bet the opening spread toward the model (protocol written 2026-10-09, before any OL1 run)

**Why.** The opening-line backtest found that the closing line moves toward the model's
ratings-only line. OL1 tests whether betting the opener in that direction is profitable
after juice. It is a historical test only. **Nothing changes live predictions or
decisions, whatever the result.**

**Disclosure: development results seen before writing this protocol.**
* The slope of (close − open) on (own line − open): 0.17 (95% CI 0.14 to 0.21). For
  the blend at the open line: 0.18 (0.12 to 0.24).
* The same slope on games where both teams started their usual QB: 0.15 (0.09 to 0.21).
* For those usual-QB games with |own line − open| ≥ 3 (118 games): an average move of
  +0.61 points toward the model. The line moved toward the model in 51% of them and
  away in 26%.
* The live rule (bet the open when EV > 2%) placed 6 bets and went 6–0.
* **Not seen:** the cover rate, ROI or CLV of any threshold rule, and anything from
  2022 onward.

**Data.**
* The AusSportsBetting spreadsheet, loaded by `nflmodel/histlines.py`. Every row
  matches a schedule game.
* Each era's lines come from **one bookmaker**: Pinnacle from 2014 to Feb 2018,
  bet365 from Sep 2018 to Feb 2025, and betr from Sep 2025.
* **When the "open" was recorded is not documented,** and the file has no timestamps.
  So it cannot be confirmed to fall inside the live early window (168–84 h before
  kickoff). If it was posted earlier (look-ahead lines), the model would hold more
  information than the opener did, which would flatter OL1. This mismatch is flagged
  in every report.
* Excluded: games without an opening line, and games marked "Data supply issue this
  week".

**Signal.** g = own line − open (points; positive = the model rates the home team
better).
* The own line is the ratings-only margin from the walk-forward model for that
  season, fit on earlier seasons only.
* The QB in the ratings is the actual starter, which may not have been known at the
  open. Results on games where both teams started their usual QB are reported as a
  check.

**Rule.** If |g| ≥ T, bet the side the model favors at the opening line: home when
g > 0, away otherwise. At most one bet per game.
* T is one of **2, 3 or 4 points**.
* **T is chosen on the development seasons (2015–2021) only:** the highest flat ROI at
  −110 among thresholds with at least 100 bets. A tie goes to the larger T.
* Choosing the best of three on the same seasons flatters the chosen T. The holdout
  is the guard.

**Scoring** (one bet per game, so the bootstrap resamples bets/games, 2000 resamples):
* **Prices:** flat 1u at **−110** (the main price; break-even 52.4%) and at **−105**
  (break-even 51.2%), each with a 95% CI. ROI at the file's own opening price is
  reported but is not a criterion.
* **Cover rate:** wins / (wins + losses), with a 95% CI, plus the number of pushes.
* **Realized CLV:** close − open in the bet's direction (points), with a CI.

**Expected move and predicted cover rate.**
* **Expected move:** a + β·g. The slope β is fit by OLS of (close − open) on g over
  the **earlier** seasons' games only, an expanding window starting with 2014.
  2014 is predicted by a model fit on 2011–2013, the minimum three seasons that
  `validate` uses.
* **Expected close** = open + expected move.
* **Predicted cover probability:** the season's key-number margin distribution
  (`model.margin_dist`) is centered so that P(margin > expected close | no push) = 0.5.
  That is, the expected close is treated as fair. The predicted P(cover | no push)
  and P(push) for the bet at the open line follow from it.
* **Reported:** mean predicted vs actual cover rate (with a CI), overall and in each
  subgroup below.

**Key-number subgroups.** A bet's expected move crosses a key number if a value k in
{−7, −3, 3, 7} satisfies open < k ≤ expected close, or expected close ≤ k < open. All
metrics are reported separately for crossing and not-crossing bets. The split is
descriptive, not a criterion.

**Also reported (not criteria):**
* by season
* by source era: Pinnacle 2015–2017, bet365 2018–2021
* the usual-QB subset

**Development pass.** At the chosen T, the 95% CI of the flat ROI at −110 is entirely
above 0.

**Holdout** (2022 to the latest completed game, run **once** and only after a
development pass): the holdout **confirms** if the flat ROI at −110, at the same T, is
above 0. The CI is reported either way.

**Either way:** OL1 is reported in `reports/open_line_experiment_*.md` and
`logs/experiments.jsonl` (group `OL1`). No setting, default or live decision changes.
A positive result would justify collecting timestamped early-week quotes (the live
`early` horizon) to test the rule prospectively. It would not justify betting it.

**OL1 amendment (2026-10-09, before any OL1 run; no OL1 result has been seen).** The model
needs at least four training seasons (three for its inner walk-forward plus one), so 2014
cannot be predicted from 2011–2013 as written above.
* **What changes:** the first season with walk-forward rows is 2015 (fit on 2011–2014),
  and 2015 has no earlier rows for the expected move (a, b).
* **2015 bets are still placed and graded.** The rule uses only |g| ≥ T, so the
  development criterion (ROI at −110, 2015–2021) is unchanged.
* **Only the descriptive metrics lose 2015.** Expected move, predicted cover and the
  key-number split cover 2016 onward.
