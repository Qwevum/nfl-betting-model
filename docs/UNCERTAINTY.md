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
