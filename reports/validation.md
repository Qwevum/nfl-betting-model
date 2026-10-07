# Validation 2015-2026

Out-of-sample validation against baselines and the market.

Walk-forward: every test season is predicted by a model fit (and calibrated) only
on earlier seasons. Ratings feeding each game use only games before it.

Information-timing notes (what a historical prediction "knew"):
  * Prices are nflverse consensus closing lines: available just before kickoff.
  * Starting QBs are the actual starters. Those are normally known ~90 minutes
    before kickoff (inactives); live predictions use projected starters instead.
  * Game-time wind/temperature are observed values; live predictions would need a
    forecast. This makes historical totals slightly easier than live ones.
  * Model design choices (features, hyperparameters, calibration method) were made
    after looking at 2015-2025 results, so those seasons are not a pristine
    holdout. The pristine test is the prediction log written from now on.

## Probability quality

| target | predictor | n | brier | log_loss |
|---|---|---|---|---|
| home covers | market no-vig (closing) | 3012 | 0.2499 | 0.6930 |
| home covers | coin flip | 3012 | 0.2500 | 0.6931 |
| home covers | model | 3012 | 0.2502 | 0.6935 |
| home covers | model without QB feature | 3012 | 0.2502 | 0.6936 |
| over hits | market no-vig (closing) | 3064 | 0.2500 | 0.6931 |
| over hits | coin flip | 3064 | 0.2500 | 0.6931 |
| over hits | model | 3064 | 0.2502 | 0.6936 |
| over hits | model without QB feature | 3064 | 0.2502 | 0.6936 |
| winner | market no-vig (closing) | 3081 | 0.2126 | 0.6130 |
| winner | model | 3082 | 0.2126 | 0.6132 |
| winner | model without QB feature | 3082 | 0.2128 | 0.6136 |
| winner | ratings only (no market) | 3082 | 0.2189 | 0.6274 |
| winner | home-team base rate | 3082 | 0.2479 | 0.6889 |
| winner | coin flip | 3082 | 0.2500 | 0.6931 |

## Calibration (model win probability)

| bin | n | predicted | actual |
|---|---|---|---|
| (0.0, 0.1] | 2 | 0.0879 | 0.5000 |
| (0.1, 0.2] | 91 | 0.1617 | 0.1868 |
| (0.2, 0.3] | 252 | 0.2570 | 0.2341 |
| (0.3, 0.4] | 405 | 0.3540 | 0.3605 |
| (0.4, 0.5] | 444 | 0.4486 | 0.4617 |
| (0.5, 0.6] | 557 | 0.5533 | 0.5583 |
| (0.6, 0.7] | 581 | 0.6491 | 0.6145 |
| (0.7, 0.8] | 517 | 0.7485 | 0.7660 |
| (0.8, 0.9] | 209 | 0.8433 | 0.8612 |
| (0.9, 1.0] | 24 | 0.9151 | 0.9583 |

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
| 2026.0000 | 64.0000 | 9.6328 | 9.4745 | 9.6252 | 9.7751 |

## Betting results (staking: flat 1u, and quarter Kelly capped at 2u)

| market | bets | W-L-P | win% | flat ROI | flat ROI 95% | qtr-Kelly units | qtr-Kelly ROI |
|---|---|---|---|---|---|---|---|
| ml | 219 | 101-117-1 | 46.3% | +10.7% | -7.5% to +28.8% | +23.0 | +9.6% |
| spread | 669 | 332-319-18 | 51.0% | +3.4% | -4.2% to +11.0% | +22.5 | +2.5% |
| total | 445 | 213-227-5 | 48.4% | -3.5% | -12.7% to +5.7% | -22.6 | -4.2% |
| all | 1333 | 646-663-24 | 49.4% | +2.3% | -3.5% to +8.0% | +22.9 | +1.4% |
