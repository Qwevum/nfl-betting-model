# Validation 2015-2026

Out-of-sample validation against baselines and the market.

Chronology: every test season is predicted by a model whose ratings use only
earlier games, whose regressions are fit only on earlier seasons, and whose
calibration (k shrinkage, market-anchored logistic) is fit by an inner
walk-forward that is also restricted to earlier seasons.

Prediction horizon: live forecasts are meant for kickoff - 60 minutes
(settings.horizon_minutes). What each historical input knew relative to that
horizon:

  input                    | historical source                 | available by horizon?
  -------------------------|-----------------------------------|-----------------------------
  team/QB ratings          | play-by-play of earlier games     | yes
  rest, division, site     | schedule                          | yes
  starting QB              | actual starter (nflverse)         | yes in practice: inactives are
                           |                                   | published ~90 min before kickoff
  roof (dome/closed)       | schedule, observed roof state     | assumed: retractable-roof calls are
                           |                                   | usually announced before the horizon
  weather                  | not used (observed wind removed)  | n/a, no historical forecasts exist
  injury reports           | not used historically             | n/a, the feed has no publish times
  market prices            | nflverse CLOSING consensus        | NO: the close is after the horizon

Market prices are the binding limitation: no timestamped historical multi-book
snapshots exist in this repository, so model and market are both evaluated at the
CLOSING information set (the market's best case), on exactly the same games.
Horizon-matched data is collected from now on by `run.py collect` / `predict`
snapshots; nothing is backfilled.

Decision rules replayed historically: EV > min_edge at the consensus closing price,
robustness to a 0.5-point error, and the gap_points model-vs-market rule, all from
the Settings passed in (the report prints the values used). NOT
replayed (no data): quote timestamps/freshness, the multi-book leave-one-book-out
reference, starter-availability conditions, the outdoor-total weather rule, and the
executable/conditional distinction. Historical "bets" are therefore closer to
"conditional recommendations at the closing consensus price" than to live
executable bets.

Model design choices (features, hyperparameters, calibration method) were made
after looking at 2015-2025 results, so those seasons are not a pristine holdout.

Settings used: `max_odds_age_minutes=30, clock_skew_minutes=2, kickoff_match_minutes=10, qb_confirm_max_age_minutes=2880, weather_max_age_minutes=720, weather_valid_window_minutes=180, min_reference_books=2, horizon_minutes=60, horizon_tolerance_minutes=15, min_edge=0.02, gap_points=4, kelly_fraction=0.25, max_stake_units=2, odds_api_regions=us, probability_model=model, actionable_books=, watchlist_size=10, uncertainty_replicates=100, uncertainty_deep_replicates=1000, uncertainty_level=0.9, uncertainty_seed=20261009, uncertainty_scheme=season, uncertainty_workers=0, experimental_ev_lower_bound_rule=False`

## Probability quality (identical rows for every predictor)

`model minus market (Brier)`: negative means the model beat the closing market; the CI resamples whole games.

| target | predictor | n | brier | log_loss | ece | ci95 |
|---|---|---|---|---|---|---|
| home covers | coin flip | 3013 | 0.2500 | 0.6931 | nan | nan |
| home covers | market no-vig (closing) | 3013 | 0.2499 | 0.6930 | 0.0095 | nan |
| home covers | model | 3013 | 0.2500 | 0.6931 | 0.0103 | nan |
| home covers | model minus market (Brier) | 3013 | 0.0001 | nan | nan | -0.00013 to +0.00030 |
| home covers | model without QB feature | 3013 | 0.2500 | 0.6932 | 0.0107 | nan |
| over hits | coin flip | 3065 | 0.2500 | 0.6931 | nan | nan |
| over hits | market no-vig (closing) | 3065 | 0.2500 | 0.6931 | 0.0119 | nan |
| over hits | model | 3065 | 0.2501 | 0.6933 | 0.0093 | nan |
| over hits | model minus market (Brier) | 3065 | 0.0001 | nan | nan | -0.00022 to +0.00045 |
| over hits | model without QB feature | 3065 | 0.2501 | 0.6933 | 0.0093 | nan |
| winner | coin flip | 3082 | 0.2500 | 0.6931 | nan | nan |
| winner | home-team base rate | 3082 | 0.2479 | 0.6889 | 0.0178 | nan |
| winner | market no-vig (closing) | 3082 | 0.2127 | 0.6133 | 0.0163 | nan |
| winner | model | 3082 | 0.2128 | 0.6134 | 0.0169 | nan |
| winner | model minus market (Brier) | 3082 | 0.0001 | nan | nan | -0.00039 to +0.00049 |
| winner | model without QB feature | 3082 | 0.2129 | 0.6138 | 0.0153 | nan |
| winner | ratings only (no market) | 3082 | 0.2191 | 0.6278 | 0.0290 | nan |

## Model minus market Brier, by season

| season | home covers | over hits | winner |
|---|---|---|---|
| 2015.0000 | -0.0001 | 0.0005 | 0.0001 |
| 2016.0000 | -0.0002 | -0.0003 | -0.0003 |
| 2017.0000 | -0.0006 | -0.0010 | -0.0008 |
| 2018.0000 | 0.0002 | -0.0008 | -0.0009 |
| 2019.0000 | 0.0003 | 0.0009 | 0.0010 |
| 2020.0000 | 0.0007 | 0.0005 | 0.0005 |
| 2021.0000 | -0.0000 | 0.0007 | 0.0001 |
| 2022.0000 | 0.0003 | 0.0005 | -0.0001 |
| 2023.0000 | 0.0000 | 0.0000 | 0.0006 |
| 2024.0000 | 0.0001 | 0.0001 | 0.0000 |
| 2025.0000 | 0.0001 | -0.0001 | 0.0005 |
| 2026.0000 | 0.0001 | 0.0001 | -0.0011 |

## Calibration of win probabilities

| bin | n | model predicted | actual | n (market) | market predicted | actual (market bins) |
|---|---|---|---|---|---|---|
| (-0.001, 0.1] | 1 | 0.0899 | 0.0000 | 3 | 0.0943 | 0.0000 |
| (0.1, 0.2] | 73 | 0.1622 | 0.2055 | 62 | 0.1635 | 0.1935 |
| (0.2, 0.3] | 226 | 0.2545 | 0.2124 | 225 | 0.2543 | 0.2044 |
| (0.3, 0.4] | 389 | 0.3528 | 0.3676 | 397 | 0.3528 | 0.3602 |
| (0.4, 0.5] | 447 | 0.4473 | 0.4273 | 471 | 0.4473 | 0.4374 |
| (0.5, 0.6] | 542 | 0.5532 | 0.5498 | 522 | 0.5565 | 0.5536 |
| (0.6, 0.7] | 608 | 0.6496 | 0.6168 | 633 | 0.6484 | 0.6177 |
| (0.7, 0.8] | 541 | 0.7495 | 0.7486 | 548 | 0.7490 | 0.7555 |
| (0.8, 0.9] | 228 | 0.8419 | 0.8465 | 203 | 0.8464 | 0.8670 |
| (0.9, 1.0] | 27 | 0.9141 | 0.9630 | 18 | 0.9127 | 0.9444 |

## Margin error by season

| season | games | MAE market | MAE model own line | MAE fair line | MAE own line w/o QB |
|---|---|---|---|---|---|
| 2015.0000 | 267.0000 | 10.1180 | 10.3500 | 10.1179 | 10.4419 |
| 2016.0000 | 267.0000 | 9.0431 | 9.2010 | 9.0417 | 9.3283 |
| 2017.0000 | 267.0000 | 10.0805 | 10.2423 | 10.0690 | 10.4599 |
| 2018.0000 | 267.0000 | 9.8745 | 10.0255 | 9.8818 | 10.0520 |
| 2019.0000 | 267.0000 | 10.1760 | 10.5403 | 10.1924 | 10.5439 |
| 2020.0000 | 269.0000 | 9.7900 | 10.0664 | 9.8217 | 10.1019 |
| 2021.0000 | 285.0000 | 10.6667 | 11.0068 | 10.6676 | 11.1034 |
| 2022.0000 | 284.0000 | 8.7835 | 9.0580 | 8.8102 | 9.0359 |
| 2023.0000 | 285.0000 | 9.9842 | 10.4883 | 10.0030 | 10.4705 |
| 2024.0000 | 285.0000 | 9.7035 | 10.0415 | 9.6921 | 10.1729 |
| 2025.0000 | 285.0000 | 9.6702 | 10.0273 | 9.6871 | 10.1324 |
| 2026.0000 | 65.0000 | 9.7538 | 9.5708 | 9.7490 | 9.8149 |

## Historical replay at closing consensus prices (staking: flat 1u, and 0.25x Kelly capped at 2u)

Rules applied: EV above 2.0%, robustness to a 0.5-pt error, 4-pt gap. Not applied: quote freshness, multi-book reference, starter availability, weather. CIs resample whole games, so a spread, moneyline and total on the same game are not treated as independent.

| market | bets | games | W-L-P | flat ROI | flat ROI 95% (game-clustered) | flat units | flat max drawdown | qtr-Kelly ROI | qtr-Kelly max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| ml | 46 | 46 | 23-23-0 | +11.8% | -23.9% to +48.6% | +5.4 | 9.1 | +1.2% | 12.1 |
| spread | 2 | 2 | 1-1-0 | +0.0% | -100.0% to +100.0% | +0.0 | 1.0 | +0.0% | 0.8 |
| total | 28 | 28 | 12-16-0 | -16.3% | -51.2% to +19.5% | -4.6 | 6.2 | -17.3% | 6.6 |
| all | 76 | 72 | 36-40-0 | +1.1% | -26.0% to +31.3% | +0.9 | 13.5 | -5.0% | 16.7 |

### Same rules at standard retail juice (4.76% overround)

The recorded consensus prices carry about 2.4% overround through 2022 and about 4.7% from 2023 (see the season table). Most bettors pay the latter. Here every consensus pair is re-priced proportionally at 4.76% (the -110/-110 margin), keeping its no-vig probabilities, and the decision rules are re-applied at those prices.

| market | bets | games | W-L-P | flat ROI | flat ROI 95% (game-clustered) | flat units | flat max drawdown | qtr-Kelly ROI | qtr-Kelly max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| ml | 12 | 12 | 6-6-0 | +18.2% | -51.3% to +92.2% | +2.2 | 3.0 | +19.0% | 3.3 |
| total | 1 | 1 | 1-0-0 | +89.3% | +89.3% to +89.3% | +0.9 | 0.0 | +89.3% | 0.0 |
| all | 13 | 13 | 7-6-0 | +23.7% | -41.7% to +93.2% | +3.1 | 3.0 | +23.9% | 3.3 |

### By season

| season | bets | flat_units | flat_roi | spread_or | ml_or | total_or | bets @4.76% | units @4.76% | roi @4.76% |
|---|---|---|---|---|---|---|---|---|---|
| 2015.0000 | 24.0000 | -6.8 | -28.1% | 2.43% | 2.45% | 2.43% | 3.0000 | +0.4 | +12.2% |
| 2016.0000 | 3.0000 | +1.2 | +41.6% | 2.43% | 2.45% | 2.43% | 1.0000 | +1.5 | +148.0% |
| 2017.0000 | 2.0000 | -0.1 | -5.8% | 2.44% | 2.44% | 2.43% | nan | nan | nan |
| 2018.0000 | 12.0000 | -0.8 | -6.5% | 2.65% | 2.45% | 2.63% | 2.0000 | -2.0 | -100.0% |
| 2019.0000 | 31.0000 | +4.5 | +14.6% | 2.43% | 2.71% | 2.67% | 6.0000 | +0.5 | +7.6% |
| 2020.0000 | 3.0000 | +3.7 | +124.6% | 2.44% | 2.73% | 2.67% | 1.0000 | +2.8 | +278.0% |
| 2021.0000 | 1.0000 | -1.0 | -100.0% | 2.51% | 2.80% | 2.67% | nan | nan | nan |
| 2022.0000 | 0.0000 | +0.0 | nan | 2.43% | 2.65% | 2.90% | nan | nan | nan |
| 2023.0000 | 0.0000 | +0.0 | nan | 4.75% | 4.26% | 4.76% | nan | nan | nan |
| 2024.0000 | 0.0000 | +0.0 | nan | 4.75% | 4.26% | 4.75% | nan | nan | nan |
| 2025.0000 | 0.0000 | +0.0 | nan | 4.71% | 4.26% | 4.71% | nan | nan | nan |
| 2026.0000 | 0.0000 | +0.0 | nan | 4.71% | 4.26% | 4.71% | nan | nan | nan |

## Value of the model's edge: offset calibration c, per walk-forward season

c is the change in log-odds per point of model edge (blend minus line) in the live probability sigmoid(logit p_market + c * edge). Each row is the model used for that test season (fit on earlier seasons only); the 95% CI resamples whole games of its inner out-of-sample rows (1000 resamples). c = 0 means the model's disagreement with the market adds nothing.

| test season | market | games | c | lo | hi | includes 0 |
|---|---|---|---|---|---|---|
| 2015 | spread | 261 | +0.0083 | -0.1095 | +0.1002 | True |
| 2015 | total | 264 | -0.0587 | -0.2285 | +0.1016 | True |
| 2015 | ml | 266 | +0.0572 | -0.0465 | +0.1769 | True |
| 2016 | spread | 518 | +0.0151 | -0.0600 | +0.1011 | True |
| 2016 | total | 526 | -0.0282 | -0.1564 | +0.0936 | True |
| 2016 | ml | 533 | +0.0414 | -0.0419 | +0.1301 | True |
| 2017 | spread | 780 | +0.0234 | -0.0439 | +0.0974 | True |
| 2017 | total | 792 | -0.0389 | -0.1480 | +0.0718 | True |
| 2017 | ml | 798 | +0.0485 | -0.0234 | +0.1307 | True |
| 2018 | spread | 1039 | +0.0353 | -0.0285 | +0.1002 | True |
| 2018 | total | 1059 | -0.0683 | -0.1676 | +0.0241 | True |
| 2018 | ml | 1064 | +0.0597 | -0.0084 | +0.1415 | True |
| 2019 | spread | 1297 | +0.0303 | -0.0307 | +0.0893 | True |
| 2019 | total | 1323 | -0.0779 | -0.1646 | +0.0182 | True |
| 2019 | ml | 1329 | +0.0688 | +0.0016 | +0.1361 | False |
| 2020 | spread | 1554 | +0.0241 | -0.0321 | +0.0820 | True |
| 2020 | total | 1589 | -0.0622 | -0.1527 | +0.0241 | True |
| 2020 | ml | 1595 | +0.0552 | -0.0080 | +0.1237 | True |
| 2021 | spread | 1823 | +0.0103 | -0.0451 | +0.0662 | True |
| 2021 | total | 1853 | -0.0527 | -0.1361 | +0.0360 | True |
| 2021 | ml | 1863 | +0.0477 | -0.0079 | +0.1113 | True |
| 2022 | spread | 2104 | +0.0115 | -0.0433 | +0.0637 | True |
| 2022 | total | 2135 | -0.0369 | -0.1201 | +0.0504 | True |
| 2022 | ml | 2147 | +0.0448 | -0.0113 | +0.1045 | True |
| 2023 | spread | 2378 | +0.0021 | -0.0486 | +0.0531 | True |
| 2023 | total | 2416 | -0.0211 | -0.0988 | +0.0592 | True |
| 2023 | ml | 2429 | +0.0450 | -0.0096 | +0.1038 | True |
| 2024 | spread | 2649 | -0.0044 | -0.0583 | +0.0460 | True |
| 2024 | total | 2697 | -0.0182 | -0.0956 | +0.0576 | True |
| 2024 | ml | 2714 | +0.0369 | -0.0151 | +0.0911 | True |
| 2025 | spread | 2930 | +0.0027 | -0.0485 | +0.0477 | True |
| 2025 | total | 2979 | -0.0112 | -0.0824 | +0.0583 | True |
| 2025 | ml | 2999 | +0.0361 | -0.0157 | +0.0924 | True |
| 2026 | spread | 3214 | -0.0047 | -0.0577 | +0.0460 | True |
| 2026 | total | 3264 | -0.0172 | -0.0898 | +0.0461 | True |
| 2026 | ml | 3283 | +0.0292 | -0.0243 | +0.0785 | True |

* **spread**: the 95% CI includes 0 in 12 of 12 seasons (every season): the model's edge is not distinguishable from no value.
* **total**: the 95% CI includes 0 in 12 of 12 seasons (every season): the model's edge is not distinguishable from no value.
* **ml**: the 95% CI includes 0 in 11 of 12 seasons.
