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
robustness to a 0.5-point error, and the 4-point model-vs-market gap rule. NOT
replayed (no data): quote timestamps/freshness, the multi-book leave-one-book-out
reference, starter-availability conditions, the outdoor-total weather rule, and the
executable/conditional distinction. Historical "bets" are therefore closer to
"conditional recommendations at the closing consensus price" than to live
executable bets.

Model design choices (features, hyperparameters, calibration method) were made
after looking at 2015-2025 results, so those seasons are not a pristine holdout.

## Probability quality (identical rows for every predictor)

`model minus market (Brier)`: negative means the model beat the closing market; the CI resamples whole games.

| target | predictor | n | brier | log_loss | ece | ci95 |
|---|---|---|---|---|---|---|
| home covers | coin flip | 3013 | 0.2500 | 0.6931 | nan | nan |
| home covers | market no-vig (closing) | 3013 | 0.2499 | 0.6930 | 0.0095 | nan |
| home covers | model | 3013 | 0.2502 | 0.6935 | 0.0126 | nan |
| home covers | model minus market (Brier) | 3013 | 0.0003 | nan | nan | -0.00064 to +0.00118 |
| home covers | model without QB feature | 3013 | 0.2502 | 0.6936 | 0.0163 | nan |
| over hits | coin flip | 3065 | 0.2500 | 0.6931 | nan | nan |
| over hits | market no-vig (closing) | 3065 | 0.2500 | 0.6931 | 0.0119 | nan |
| over hits | model | 3065 | 0.2503 | 0.6937 | 0.0044 | nan |
| over hits | model minus market (Brier) | 3065 | 0.0003 | nan | nan | -0.00047 to +0.00111 |
| over hits | model without QB feature | 3065 | 0.2503 | 0.6937 | 0.0044 | nan |
| winner | coin flip | 3082 | 0.2500 | 0.6931 | nan | nan |
| winner | home-team base rate | 3082 | 0.2479 | 0.6889 | 0.0178 | nan |
| winner | market no-vig (closing) | 3082 | 0.2127 | 0.6133 | 0.0163 | nan |
| winner | model | 3082 | 0.2128 | 0.6137 | 0.0172 | nan |
| winner | model minus market (Brier) | 3082 | 0.0001 | nan | nan | -0.00051 to +0.00068 |
| winner | model without QB feature | 3082 | 0.2129 | 0.6140 | 0.0192 | nan |
| winner | ratings only (no market) | 3082 | 0.2191 | 0.6278 | 0.0290 | nan |

## Model minus market Brier, by season

| season | home covers | over hits | winner |
|---|---|---|---|
| 2015.0000 | -0.0027 | -0.0007 | 0.0002 |
| 2016.0000 | -0.0004 | 0.0030 | 0.0004 |
| 2017.0000 | 0.0011 | -0.0005 | -0.0005 |
| 2018.0000 | -0.0015 | -0.0015 | -0.0006 |
| 2019.0000 | 0.0001 | 0.0010 | 0.0009 |
| 2020.0000 | 0.0031 | 0.0001 | 0.0001 |
| 2021.0000 | -0.0005 | 0.0014 | -0.0007 |
| 2022.0000 | 0.0018 | -0.0010 | 0.0003 |
| 2023.0000 | 0.0003 | -0.0013 | 0.0011 |
| 2024.0000 | 0.0002 | 0.0023 | -0.0002 |
| 2025.0000 | 0.0007 | 0.0010 | 0.0005 |
| 2026.0000 | 0.0013 | 0.0001 | -0.0004 |

## Calibration of win probabilities

| bin | n | model predicted | actual | n (market) | market predicted | actual (market bins) |
|---|---|---|---|---|---|---|
| (-0.001, 0.1] | 2 | 0.0879 | 0.5000 | 3 | 0.0943 | 0.0000 |
| (0.1, 0.2] | 91 | 0.1617 | 0.1868 | 62 | 0.1635 | 0.1935 |
| (0.2, 0.3] | 252 | 0.2570 | 0.2341 | 225 | 0.2543 | 0.2044 |
| (0.3, 0.4] | 405 | 0.3540 | 0.3605 | 397 | 0.3528 | 0.3602 |
| (0.4, 0.5] | 444 | 0.4486 | 0.4617 | 471 | 0.4473 | 0.4374 |
| (0.5, 0.6] | 557 | 0.5533 | 0.5583 | 522 | 0.5565 | 0.5536 |
| (0.6, 0.7] | 581 | 0.6491 | 0.6145 | 633 | 0.6484 | 0.6177 |
| (0.7, 0.8] | 516 | 0.7485 | 0.7655 | 548 | 0.7490 | 0.7555 |
| (0.8, 0.9] | 210 | 0.8432 | 0.8571 | 203 | 0.8464 | 0.8670 |
| (0.9, 1.0] | 24 | 0.9151 | 0.9583 | 18 | 0.9127 | 0.9444 |

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

## Historical replay at closing consensus prices (staking: flat 1u, and quarter Kelly capped at 2u)

Rules applied: EV above the threshold, robustness to a 0.5-pt error, 4-pt gap. Not applied: quote freshness, multi-book reference, starter availability, weather. CIs resample whole games, so a spread, moneyline and total on the same game are not treated as independent.

| market | bets | games | W-L-P | flat ROI | flat ROI 95% (game-clustered) | flat units | flat max drawdown | qtr-Kelly ROI | qtr-Kelly max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| ml | 219 | 219 | 101-117-1 | +10.7% | -7.1% to +28.8% | +23.3 | 20.6 | +9.6% | 18.9 |
| spread | 669 | 669 | 332-319-18 | +3.4% | -4.6% to +11.2% | +22.5 | 26.1 | +2.5% | 39.1 |
| total | 376 | 376 | 182-190-4 | -2.3% | -12.0% to +7.3% | -8.5 | 23.4 | -3.2% | 28.2 |
| all | 1264 | 1008 | 615-626-23 | +3.0% | -3.1% to +8.9% | +37.4 | 30.5 | +2.0% | 47.2 |

### Same rules at standard retail juice (4.76% overround)

The recorded consensus prices carry about 2.4% overround through 2022 and about 4.7% from 2023 (see the season table). Most bettors pay the latter. Here every consensus pair is re-priced proportionally at 4.76% (the -110/-110 margin), keeping its no-vig probabilities, and the decision rules are re-applied at those prices.

| market | bets | games | W-L-P | flat ROI | flat ROI 95% (game-clustered) | flat units | flat max drawdown | qtr-Kelly ROI | qtr-Kelly max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| ml | 74 | 74 | 29-44-1 | +16.0% | -19.8% to +51.6% | +11.9 | 8.0 | +16.3% | 7.2 |
| spread | 342 | 342 | 165-166-11 | +0.3% | -9.7% to +11.1% | +0.9 | 19.4 | +0.1% | 24.7 |
| total | 121 | 121 | 58-62-1 | -4.4% | -20.7% to +12.8% | -5.3 | 13.1 | -4.6% | 14.8 |
| all | 537 | 486 | 252-272-13 | +1.4% | -7.6% to +10.7% | +7.5 | 27.6 | +0.9% | 35.3 |

### By season

| season | bets | flat_units | flat_roi | spread_or | ml_or | total_or | bets @4.76% | units @4.76% | roi @4.76% |
|---|---|---|---|---|---|---|---|---|---|
| 2015.0000 | 212.0000 | +26.1 | +12.3% | 2.43% | 2.45% | 2.43% | 77.0000 | +13.1 | +17.0% |
| 2016.0000 | 319.0000 | -11.9 | -3.7% | 2.43% | 2.45% | 2.43% | 170.0000 | +0.0 | +0.0% |
| 2017.0000 | 138.0000 | +9.5 | +6.9% | 2.44% | 2.44% | 2.43% | 63.0000 | +13.9 | +22.1% |
| 2018.0000 | 118.0000 | +13.5 | +11.4% | 2.65% | 2.45% | 2.63% | 51.0000 | +2.5 | +5.0% |
| 2019.0000 | 152.0000 | -1.6 | -1.1% | 2.43% | 2.71% | 2.67% | 71.0000 | -2.4 | -3.4% |
| 2020.0000 | 104.0000 | -18.6 | -17.9% | 2.44% | 2.73% | 2.67% | 44.0000 | -19.1 | -43.5% |
| 2021.0000 | 105.0000 | +10.8 | +10.3% | 2.51% | 2.80% | 2.67% | 31.0000 | -0.7 | -2.4% |
| 2022.0000 | 115.0000 | +8.6 | +7.5% | 2.43% | 2.65% | 2.90% | 30.0000 | +0.2 | +0.5% |
| 2023.0000 | 0.0000 | +0.0 | nan | 4.75% | 4.26% | 4.76% | nan | nan | nan |
| 2024.0000 | 0.0000 | +0.0 | nan | 4.75% | 4.26% | 4.75% | nan | nan | nan |
| 2025.0000 | 1.0000 | +1.0 | +105.0% | 4.71% | 4.26% | 4.71% | nan | nan | nan |
| 2026.0000 | 0.0000 | +0.0 | nan | 4.71% | 4.26% | 4.71% | nan | nan | nan |
