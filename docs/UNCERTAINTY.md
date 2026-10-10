# Forecast uncertainty: method and evaluation protocol

Written 2026-10-09, before any uncertainty result was computed. Results are appended
at the end. The protocol section is not edited after results are seen. If it ever
needs to change, the change gets a new dated section that explains why.

## What is estimated

Two different things get two different calculations and two different labels.

1. **Uncertainty in the estimated win probability.** This is how much the model's
   probability for a game would move if the model had been fit on a different,
   equally plausible draw of history. It is *not* a chance that a team wins.
   A 90% interval of 58–63% for the home team does not mean a 90% chance that
   the home team wins.
2. **Prediction intervals for the actual margin and total.** These describe the
   range of final scores. They come from the full predictive distribution: the
   key-number outcome model around the forecast mean, mixed over the bootstrap
   replicates. They are never built from the spread of fitted means alone.

## Method: time-blocked Bayesian bootstrap of the fitting stage

* **Features are held fixed.** Team and QB ratings are computed once, in date
  order, by the existing stateful pipeline (`ratings.py`). Games are never
  shuffled, duplicated or re-run through it.
* **Replicates re-weight training games.** Each replicate draws random weights
  for the training games: an Exp(1) multiplier per block, normalized to mean 1
  (the Bayesian bootstrap, Rubin 1981). It then refits **everything**
  `model.fit` estimates under those weights:
  * the four ridge regressions
  * the inner walk-forward (each inner season predicted by a fit on earlier
    seasons only, with the same weights)
  * the shrinkage k
  * the residual scales and key-number weights
  * the offset and legacy calibrations
  Weights never move a game to a different season or date, so every
  chronological boundary in fitting and calibration is kept. Continuous weights
  never drop a season, so the inner walk-forward keeps its structure.
* **Dependence** is handled by giving one weight to a whole block of games.
  Candidate blocks:
  * `game`: each game on its own (assumes independence; the reference point)
  * `week4`: four consecutive weeks of one season
  * `season`: a whole season
  * `season*week4`: a season weight times a week-block weight (two levels)
* **Held fixed:** rating hyperparameters (`ratings.PARAMS`), the ridge penalty,
  `inner_start`, the key-number prior, the feature set, and the observed market
  inputs (lines, no-vig probabilities, live references). Intervals are therefore
  **conditional on the observed market snapshot** and on the ratings as computed.
* **Omitted sources of uncertainty:**
  * noise in the ratings themselves, beyond what the regressions absorb
  * the choice of hyperparameters and features
  * model misspecification
  * market quotes moving after the snapshot
  * information the market has and the model lacks (injuries, weather)
* **Interval for a probability.** The interval is centered on the point model's
  probability in logit space: logit(p̂) ± z · SD of the replicates' logits.
  This is more stable than percentiles with ~100 replicates and stays inside
  (0, 1). Home and away intervals are consistent by construction.
* **EV uncertainty.** EV is computed in each replicate at the same fixed offered
  price, with that replicate's joint (win, push) probabilities. The interval is
  EV̂ ± z · SD(EV replicates). No marginal bounds are combined.
* **Monte Carlo settings.** Seeds are fixed. Replicate b always uses the stream
  `(seed, b)`, so the first 100 replicates of a deep run equal the fast run.

## Selection on development data only

Development test seasons are 2015–2021. The holdout is 2022 onward and is
evaluated **once**, after the method is chosen and committed.

### Simulation (justification for the probability interval)

A single win or loss cannot tell whether an interval contains the latent true
probability, so interval behavior is checked where the truth is known:

* **Truth.** The truth is the point model fit on real 2011–2020 games. Training
  outcomes for 2011–2020 are simulated from its predictive distribution. The
  real features and closing markets are kept.
* **Dependence.** The simulated outcomes include team-season and season shocks.
  Their variances are estimated by method of moments from real out-of-sample
  residuals, so the simulated data has realistic dependence.
* **Refit.** Each simulation refits the point model and B replicates under each
  candidate block scheme, then forecasts the real 2021 games.
* **Metrics per scheme:**
  * (a) ratio of the mean bootstrap SD to the true sampling SD of the estimator
    across simulations
  * (b) coverage of the estimator's expected value
  * (c) coverage of the truth model's probability, which includes ridge and
    calibration bias

**Selection rule:**
* Choose the scheme whose SD ratio is closest to 1 among those with ratio
  ≥ 0.9, so the method does not understate.
* If none reaches 0.9, choose the one with the highest ratio and report the
  under-coverage.
* Ties go to the simpler scheme.

### Real data (development seasons)

These are reported for the chosen scheme, with baselines on identical games:
* Brier and log loss of the forecast win probability. Baselines: the closing
  moneyline no-vig, the market-only forecast (model edge set to zero), the
  ratings-only forecast, the home base rate, and a coin flip.
* Calibration tables with counts per bin, and Brier by probability range.
* Margin and total prediction-interval coverage and mean width at 50, 80, 90 and
  95%, against a no-key-number normal baseline with the same center and scale.

### Holdout

The chosen method's real-data metrics are run once on 2022 onward and appended
here. Nothing is re-tuned afterwards. Better-looking holdout numbers would not
establish an improvement, and worse ones are reported as they are.


## Results (appended 2026-10-09; the protocol above is unchanged)

Recorded in `logs/experiments.jsonl` (group `UNC`). Reports:
`reports/uncertainty_simulation.md`, `reports/forecast_validation_dev.md`,
`reports/forecast_validation_holdout.md`.

### Scheme selection (simulation)

Truth = the point model on 2011–2020, forecasting the real 2021 games, with 60
simulated histories and 50 replicates per scheme. The dependence estimated from
real residuals is small: the team-season SD is about 0, and the season home-field
SD is 0.14 points. The market already absorbs team strength.

| scheme | SD ratio (pooled) | 90% coverage of E[estimate] | 90% coverage of truth |
|---|---|---|---|
| game | 1.22 | 0.97 | 0.93 |
| week4 | 1.19 | 0.96 | 0.92 |
| **season (selected)** | **1.07** | **0.91** | **0.87** |
| season*week4 | 1.79 | 0.99 | 0.98 |

* **Selected scheme:** `season`. Its SD ratio, 1.07, is the closest to 1 among the
  schemes at or above 0.9, as the pre-registered rule requires.
* **Bias:** coverage of the truth is a few points lower than coverage of the
  estimator's expectation. The estimator has a small bias, from ridge shrinkage and
  calibration, and the bootstrap does not correct it.
* **Stress scenario** (sensitivity only; team and season shocks of 1–2 points): every
  scheme keeps an SD ratio of at least 1.06. `season` reaches 0.92 coverage of the
  expectation but only 0.77 of the truth, because bias grows with dependence.

### Real games, out of sample

| period | games | forecast − closing ML Brier (95% CI) | margin PI coverage 50/80/90/95% | total PI coverage 50/80/90/95% | median 90% width of P(win) |
|---|---|---|---|---|---|
| development 2015–2021 | 1,881 | −0.00018 (−0.00096 to +0.00051) | 55.7 / 83.4 / 91.4 / 95.0 | 52.1 / 81.9 / 90.5 / 95.1 | 2.4 pts |
| holdout 2022–2026 (run once) | 1,201 | −0.00019 (−0.00073 to +0.00032) | 57.1 / 82.1 / 91.2 / 95.5 | 54.8 / 83.2 / 91.1 / 94.8 | 1.8 pts |

* **Win probability:** the forecast is indistinguishable from the closing
  moneyline, with a CI that includes 0 in both periods.
* **Ratings-only forecasts are clearly worse.** Without market lines, the model's
  Brier score is 0.2186 in development, against 0.2136 for the closing moneyline.
* **80–95% intervals:** margin and total coverage is within about 1–3 points of
  the nominal level.
* **50% intervals over-cover** (52–57%). Scores are integers, and equal-tailed
  intervals on integer outcomes include whole key numbers.
* **Parameter uncertainty is negligible next to game-to-game variation.** The
  predictive-mixture intervals are almost the same as the plug-in ones.
* **The win-probability intervals are narrow, 1.8–2.4 points.** That is because the
  forecast stays close to the closing market. It is not evidence that games are
  predictable, and it does not cover model or market misspecification.
