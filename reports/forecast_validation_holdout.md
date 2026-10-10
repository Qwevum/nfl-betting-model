# Game-forecast validation (holdout, seasons 2022-2026)

Chronological out of sample: each season is forecast by a point model and bootstrap replicates fit only on earlier seasons. Market inputs are the nflverse CLOSING consensus lines (the only historical market data), so the market-informed forecasts are evaluated at the market's best information set.

Uncertainty: `time-blocked Bayesian bootstrap of the fitting stage` (ubb-1), scheme `season`, 100 replicates, seed 20261009, level 90%. Runtime 4801s.

## Win probability: scoring on identical games

P(home wins | not tied), decided games where every predictor has a value.

| predictor | games | brier | log_loss | ece |
|---|---|---|---|---|
| forecast (market-informed + model) | 1201 | 0.2112 | 0.6101 | 0.0179 |
| market-only forecast (model edge 0) | 1201 | 0.2112 | 0.6100 | 0.0181 |
| closing moneyline no-vig | 1201 | 0.2114 | 0.6102 | 0.0206 |
| team ratings only | 1201 | 0.2192 | 0.6275 | 0.0359 |
| home-team base rate | 1201 | 0.2471 | 0.6874 | 0.0056 |
| coin flip | 1201 | 0.2500 | 0.6931 | nan |

Forecast minus closing moneyline Brier: -0.00019 (95% CI -0.00073 to +0.00032, game bootstrap); negative = forecast better.

## By season (Brier)

| season | games | Brier forecast | Brier market-only | Brier ML no-vig | Brier ratings only |
|---|---|---|---|---|---|
| 2022 | 282 | 0.2093 | 0.2093 | 0.2093 | 0.2173 |
| 2023 | 285 | 0.2176 | 0.2176 | 0.2186 | 0.2285 |
| 2024 | 285 | 0.2011 | 0.2010 | 0.2010 | 0.2135 |
| 2025 | 284 | 0.2114 | 0.2113 | 0.2109 | 0.2182 |
| 2026 | 65 | 0.2345 | 0.2343 | 0.2359 | 0.2152 |

## By forecast favorite probability

| favorite prob | games | mean forecast | favorite won | Brier forecast | Brier ML no-vig |
|---|---|---|---|---|---|
| (0.499, 0.6] | 416 | 55.5% | 57.7% | 0.2418 | 0.2413 |
| (0.6, 0.7] | 409 | 64.4% | 65.5% | 0.2254 | 0.2268 |
| (0.7, 0.8] | 276 | 74.9% | 75.4% | 0.1875 | 0.1876 |
| (0.8, 1.0] | 100 | 84.2% | 90.0% | 0.0908 | 0.0893 |

## Calibration (forecast, with counts)

| bin | n | predicted | actual |
|---|---|---|---|
| (0.1, 0.2] | 20 | 0.166 | 0.150 |
| (0.2, 0.3] | 86 | 0.252 | 0.256 |
| (0.3, 0.4] | 164 | 0.357 | 0.329 |
| (0.4, 0.5] | 199 | 0.447 | 0.427 |
| (0.5, 0.6] | 217 | 0.556 | 0.581 |
| (0.6, 0.7] | 245 | 0.645 | 0.645 |
| (0.7, 0.8] | 190 | 0.750 | 0.758 |
| (0.8, 0.9] | 77 | 0.842 | 0.909 |
| (0.9, 1.0] | 3 | 0.917 | 1.000 |

## Actual margin and total: prediction-interval coverage and width

Each interval is evaluated on the same games within an outcome/level group. Width in points.

| outcome | level | interval | games | coverage | mean width |
|---|---|---|---|---|---|
| margin | 50% | predictive mixture (method) | 1204 | 57.1% | 16.797 |
| margin | 50% | plug-in key-number | 1204 | 57.1% | 16.763 |
| margin | 50% | normal, no key numbers | 1204 | 57.7% | 17.510 |
| margin | 50% | team ratings only | 1204 | 56.6% | 16.854 |
| margin | 80% | predictive mixture (method) | 1204 | 82.1% | 32.795 |
| margin | 80% | plug-in key-number | 1204 | 82.3% | 32.737 |
| margin | 80% | normal, no key numbers | 1204 | 82.1% | 33.248 |
| margin | 80% | team ratings only | 1204 | 82.6% | 33.361 |
| margin | 90% | predictive mixture (method) | 1204 | 91.2% | 42.612 |
| margin | 90% | plug-in key-number | 1204 | 91.2% | 42.560 |
| margin | 90% | normal, no key numbers | 1204 | 90.9% | 42.629 |
| margin | 90% | team ratings only | 1204 | 90.3% | 43.387 |
| margin | 95% | predictive mixture (method) | 1204 | 95.5% | 51.102 |
| margin | 95% | plug-in key-number | 1204 | 95.5% | 51.023 |
| margin | 95% | normal, no key numbers | 1204 | 95.6% | 50.830 |
| margin | 95% | team ratings only | 1204 | 96.0% | 52.070 |
| total | 50% | predictive mixture (method) | 1204 | 54.8% | 17.692 |
| total | 50% | plug-in key-number | 1204 | 54.7% | 17.669 |
| total | 50% | normal, no key numbers | 1204 | 55.0% | 17.872 |
| total | 50% | team ratings only | 1204 | 53.1% | 18.084 |
| total | 80% | predictive mixture (method) | 1204 | 83.2% | 33.802 |
| total | 80% | plug-in key-number | 1204 | 83.2% | 33.768 |
| total | 80% | normal, no key numbers | 1204 | 83.3% | 33.941 |
| total | 80% | team ratings only | 1204 | 83.1% | 34.677 |
| total | 90% | predictive mixture (method) | 1204 | 91.1% | 43.403 |
| total | 90% | plug-in key-number | 1204 | 91.1% | 43.360 |
| total | 90% | normal, no key numbers | 1204 | 91.2% | 43.537 |
| total | 90% | team ratings only | 1204 | 91.5% | 44.423 |
| total | 95% | predictive mixture (method) | 1204 | 94.8% | 51.658 |
| total | 95% | plug-in key-number | 1204 | 94.5% | 51.619 |
| total | 95% | normal, no key numbers | 1204 | 94.9% | 51.846 |
| total | 95% | team ratings only | 1204 | 95.7% | 52.857 |

## Width of the win-probability intervals

These describe uncertainty in the ESTIMATED probability under this method (conditional on the closing market). They are narrow because the forecast stays close to the market; that does not make the game predictable and does not cover market or model misspecification. A single win or loss cannot test them; see the simulation report.

| games | level | median width (pct points) | 90th pct width | median SD (logit) |
|---|---|---|---|---|
| 1204 | 90% | 1.784 | 2.500 | 0.0253 |

