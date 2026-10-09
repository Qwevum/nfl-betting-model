# Game-forecast validation (development, seasons 2015-2021)

Chronological out of sample: each season is forecast by a point model and bootstrap replicates fit only on earlier seasons. Market inputs are the nflverse CLOSING consensus lines (the only historical market data), so the market-informed forecasts are evaluated at the market's best information set.

Uncertainty: `time-blocked Bayesian bootstrap of the fitting stage` (ubb-1), scheme `season`, 100 replicates, seed 20261009, level 90%. Runtime 186s.

## Win probability: scoring on identical games

P(home wins | not tied), decided games where every predictor has a value.

| predictor | games | brier | log_loss | ece |
|---|---|---|---|---|
| forecast (market-informed + model) | 1881 | 0.2134 | 0.6149 | 0.0356 |
| market-only forecast (model edge 0) | 1881 | 0.2136 | 0.6153 | 0.0305 |
| closing moneyline no-vig | 1881 | 0.2136 | 0.6153 | 0.0286 |
| team ratings only | 1881 | 0.2186 | 0.6264 | 0.0300 |
| home-team base rate | 1881 | 0.2484 | 0.6899 | 0.0256 |
| coin flip | 1881 | 0.2500 | 0.6931 | nan |

Forecast minus closing moneyline Brier: -0.00018 (95% CI -0.00096 to +0.00051, game bootstrap); negative = forecast better.

## By season (Brier)

| season | games | Brier forecast | Brier market-only | Brier ML no-vig | Brier ratings only |
|---|---|---|---|---|---|
| 2015 | 267 | 0.2292 | 0.2293 | 0.2287 | 0.2262 |
| 2016 | 265 | 0.2175 | 0.2178 | 0.2170 | 0.2184 |
| 2017 | 266 | 0.2029 | 0.2034 | 0.2031 | 0.2083 |
| 2018 | 265 | 0.2116 | 0.2123 | 0.2124 | 0.2118 |
| 2019 | 266 | 0.2135 | 0.2132 | 0.2141 | 0.2239 |
| 2020 | 268 | 0.2020 | 0.2018 | 0.2019 | 0.2151 |
| 2021 | 284 | 0.2169 | 0.2169 | 0.2177 | 0.2257 |

## By forecast favorite probability

| favorite prob | games | mean forecast | favorite won | Brier forecast | Brier ML no-vig |
|---|---|---|---|---|---|
| (0.499, 0.6] | 628 | 55.6% | 53.3% | 0.2483 | 0.2481 |
| (0.6, 0.7] | 648 | 64.5% | 61.9% | 0.2338 | 0.2343 |
| (0.7, 0.8] | 443 | 74.3% | 78.8% | 0.1690 | 0.1699 |
| (0.8, 1.0] | 162 | 84.5% | 86.4% | 0.1179 | 0.1164 |

## Calibration (forecast, with counts)

| bin | n | predicted | actual |
|---|---|---|---|
| (-0.001, 0.1] | 1 | 0.098 | 0.000 |
| (0.1, 0.2] | 34 | 0.160 | 0.206 |
| (0.2, 0.3] | 129 | 0.259 | 0.186 |
| (0.3, 0.4] | 227 | 0.355 | 0.339 |
| (0.4, 0.5] | 291 | 0.445 | 0.464 |
| (0.5, 0.6] | 337 | 0.557 | 0.531 |
| (0.6, 0.7] | 421 | 0.645 | 0.596 |
| (0.7, 0.8] | 314 | 0.744 | 0.777 |
| (0.8, 0.9] | 117 | 0.840 | 0.889 |
| (0.9, 1.0] | 10 | 0.921 | 0.800 |

## Actual margin and total: prediction-interval coverage and width

Each interval is evaluated on the same games within an outcome/level group. Width in points.

| outcome | level | interval | games | coverage | mean width |
|---|---|---|---|---|---|
| margin | 50% | predictive mixture (method) | 1889 | 55.7% | 17.718 |
| margin | 50% | plug-in key-number | 1889 | 55.4% | 17.630 |
| margin | 50% | normal, no key numbers | 1889 | 56.2% | 18.025 |
| margin | 50% | team ratings only | 1889 | 56.5% | 17.808 |
| margin | 80% | predictive mixture (method) | 1889 | 83.4% | 34.100 |
| margin | 80% | plug-in key-number | 1889 | 83.1% | 33.953 |
| margin | 80% | normal, no key numbers | 1889 | 83.4% | 34.228 |
| margin | 80% | team ratings only | 1889 | 83.4% | 34.587 |
| margin | 90% | predictive mixture (method) | 1889 | 91.4% | 44.098 |
| margin | 90% | plug-in key-number | 1889 | 91.3% | 43.913 |
| margin | 90% | normal, no key numbers | 1889 | 91.4% | 43.934 |
| margin | 90% | team ratings only | 1889 | 91.1% | 44.769 |
| margin | 95% | predictive mixture (method) | 1889 | 95.0% | 52.672 |
| margin | 95% | plug-in key-number | 1889 | 95.0% | 52.439 |
| margin | 95% | normal, no key numbers | 1889 | 95.0% | 52.337 |
| margin | 95% | team ratings only | 1889 | 95.6% | 53.529 |
| total | 50% | predictive mixture (method) | 1889 | 52.1% | 17.785 |
| total | 50% | plug-in key-number | 1889 | 52.2% | 17.778 |
| total | 50% | normal, no key numbers | 1889 | 51.8% | 17.706 |
| total | 50% | team ratings only | 1889 | 52.4% | 18.069 |
| total | 80% | predictive mixture (method) | 1889 | 81.9% | 33.830 |
| total | 80% | plug-in key-number | 1889 | 81.8% | 33.801 |
| total | 80% | normal, no key numbers | 1889 | 81.4% | 33.658 |
| total | 80% | team ratings only | 1889 | 81.0% | 34.445 |
| total | 90% | predictive mixture (method) | 1889 | 90.5% | 43.329 |
| total | 90% | plug-in key-number | 1889 | 90.5% | 43.308 |
| total | 90% | normal, no key numbers | 1889 | 90.5% | 43.176 |
| total | 90% | team ratings only | 1889 | 90.9% | 44.123 |
| total | 95% | predictive mixture (method) | 1889 | 95.1% | 51.481 |
| total | 95% | plug-in key-number | 1889 | 95.1% | 51.438 |
| total | 95% | normal, no key numbers | 1889 | 95.0% | 51.431 |
| total | 95% | team ratings only | 1889 | 94.9% | 52.399 |

## Width of the win-probability intervals

These describe uncertainty in the ESTIMATED probability under this method (conditional on the closing market). They are narrow because the forecast stays close to the market; that does not make the game predictable and does not cover market or model misspecification. A single win or loss cannot test them; see the simulation report.

| games | level | median width (pct points) | 90th pct width | median SD (logit) |
|---|---|---|---|---|
| 1889 | 90% | 2.398 | 4.495 | 0.0339 |

