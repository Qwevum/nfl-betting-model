# Probability model comparison (development seasons)

Seasons 2015-2021 (development), commit `0aa741b`.

Probabilities on identical rows (lower log loss / Brier is better):

| target | n | raw market | legacy | model | market-only | model - legacy (95% CI) | legacy - raw (95% CI) | mean shift from raw: legacy / model |
|---|---|---|---|---|---|---|---|---|
| winner | 1881 | 0.61529 | 0.61531 | 0.61515 | 0.61529 | -0.00016 (-0.00131, +0.00102) | +0.00001 (-0.00195, +0.00202) | 1.53% / 1.20% |
| home covers | 1843 | 0.69330 | 0.69312 | 0.69340 | 0.69330 | +0.00028 (-0.00240, +0.00283) | -0.00018 (-0.00282, +0.00260) | 2.39% / 0.56% |
| over hits | 1871 | 0.69299 | 0.69378 | 0.69318 | 0.69299 | -0.00060 (-0.00235, +0.00106) | +0.00079 (-0.00116, +0.00286) | 1.82% / 0.88% |

Brier and calibration error (ECE):

| target | arm | Brier | ECE |
|---|---|---|---|
| winner | raw | 0.21358 | 0.0286 |
| winner | legacy | 0.21353 | 0.0293 |
| winner | model | 0.21355 | 0.0261 |
| winner | market | 0.21358 | 0.0286 |
| home covers | raw | 0.25007 | 0.0181 |
| home covers | legacy | 0.24998 | 0.0082 |
| home covers | model | 0.25012 | 0.0196 |
| home covers | market | 0.25007 | 0.0181 |
| over hits | raw | 0.24992 | 0.0141 |
| over hits | legacy | 0.25032 | 0.0009 |
| over hits | model | 0.25002 | 0.0114 |
| over hits | market | 0.24992 | 0.0141 |

Historical replay at closing consensus prices (one price per side; no price shopping possible):

| arm | bets | bets/game | flat ROI (95% CI, game-clustered) | units | max drawdown |
|---|---|---|---|---|---|
| legacy | 1148 | 0.608 | +2.4% (-4.0%, +8.8%) | +27.8 | 30.5 |
| model | 76 | 0.040 | +1.1% (-26.0%, +30.2%) | +0.9 | 13.5 |
| market | 0 | 0 | n/a | 0 | 0 |

Historical replay at every pair re-priced at -110/-110 juice (4.76%) (one price per side; no price shopping possible):

| arm | bets | bets/game | flat ROI (95% CI, game-clustered) | units | max drawdown |
|---|---|---|---|---|---|
| legacy | 507 | 0.268 | +1.4% (-8.1%, +11.6%) | +7.3 | 24.6 |
| model | 13 | 0.007 | +23.7% (-42.5%, +91.1%) | +3.1 | 3.0 |
| market | 0 | 0 | n/a | 0 | 0 |

Market coverage of the 1889 scored games: spread 100%, total 100%, moneyline 100% (same rows for every arm).

Verdict: keep probability_model=model (not significantly worse than legacy on any target).
