# Spread calibration diagnosis (Step 1, read-only)

Model fit for 2026 week 5 (training rows before 2026-10-08), inner walk-forward rows: 3273 games, 2014-2026. edge = blend margin - spread line (points); p = proportional no-vig consensus probability.

## 1a. Fitted calibration and game-bootstrap 95% CIs (2000 resamples of whole games)

Null values: a = 0, b = 1 (market unchanged), c = 0 (model adds nothing).

| calibration | param | estimate | lo | hi | games | CI includes null |
|---|---|---|---|---|---|---|
| legacy | a | -0.0383 | -0.1077 | +0.0286 | 3273 | True |
| legacy | b | +0.3565 | -0.4975 | +1.1934 | 3273 | True |
| legacy | c | -0.0020 | -0.0545 | +0.0467 | 3273 | True |
| offset (a=0, b=1) | c | -0.0030 | -0.0547 | +0.0461 | 3273 | True |

## 1b. Refit by era

| calibration | param | estimate | lo | hi | games | CI includes null | era |
|---|---|---|---|---|---|---|---|
| legacy | a | -0.0681 | -0.1597 | +0.0160 | 2117 | True | 2015-2022 (~2.4% overround) |
| legacy | b | +0.5278 | -0.3735 | +1.3188 | 2117 | True | 2015-2022 (~2.4% overround) |
| legacy | c | +0.0050 | -0.0590 | +0.0687 | 2117 | True | 2015-2022 (~2.4% overround) |
| offset (a=0, b=1) | c | -0.0006 | -0.0631 | +0.0611 | 2117 | True | 2015-2022 (~2.4% overround) |
| legacy | a | +0.0393 | -0.0834 | +0.1720 | 895 | True | 2023-2026 (~4.7% overround) |
| legacy | b | +0.1500 | -0.7545 | +1.0476 | 895 | True | 2023-2026 (~4.7% overround) |
| legacy | c | -0.0511 | -0.2001 | +0.1050 | 895 | True | 2023-2026 (~4.7% overround) |
| offset (a=0, b=1) | c | -0.0449 | -0.1942 | +0.1105 | 895 | True | 2023-2026 (~4.7% overround) |

## 1c. Current slate (2026 week 5, games not yet played; consensus lines, no bookmaker quotes)

Model probability minus proportional no-vig, P(side covers | no push), split into the legacy calibration's market remapping (a, b with edge 0) and its model part (c x edge); last column: the LIVE offset calibration (a = 0, b = 1), whose difference is all model.

| game | side | edge (pts) | no-vig | legacy: remap (a,b) | legacy: model (c*edge) | legacy: total | offset (live): model (c*edge) |
|---|---|---|---|---|---|---|---|
| BAL @ ATL | ATL -3.5 | +1.5 | 48.3% | +0.16% | -0.08% | +0.09% | -0.12% |
| BAL @ ATL | BAL +3.5 | -1.5 | 51.7% | -0.16% | +0.08% | -0.09% | +0.12% |
| CHI @ GB | GB +1.5 | -0.2 | 50.0% | -0.96% | +0.01% | -0.95% | +0.01% |
| CHI @ GB | CHI -1.5 | +0.2 | 50.0% | +0.96% | -0.01% | +0.95% | -0.01% |
| CIN @ MIA | MIA +6.5 | -0.3 | 47.8% | +0.44% | +0.02% | +0.46% | +0.02% |
| CIN @ MIA | CIN -6.5 | +0.3 | 52.2% | -0.44% | -0.02% | -0.46% | -0.02% |
| CLE @ NYJ | NYJ -2.5 | -0.4 | 50.0% | -0.96% | +0.02% | -0.94% | +0.03% |
| CLE @ NYJ | CLE +2.5 | +0.4 | 50.0% | +0.96% | -0.02% | +0.94% | -0.03% |
| DEN @ LAC | LAC +3.5 | -0.1 | 51.7% | -2.07% | +0.00% | -2.07% | +0.01% |
| DEN @ LAC | DEN -3.5 | +0.1 | 48.3% | +2.07% | -0.00% | +2.07% | -0.01% |
| DET @ ARI | ARI +5.5 | -0.0 | 50.0% | -0.96% | +0.00% | -0.95% | +0.00% |
| DET @ ARI | DET -5.5 | +0.0 | 50.0% | +0.96% | -0.00% | +0.95% | -0.00% |
| HOU @ TEN | TEN +7.5 | +0.2 | 51.1% | -1.65% | -0.01% | -1.66% | -0.01% |
| HOU @ TEN | HOU -7.5 | -0.2 | 48.9% | +1.65% | +0.01% | +1.66% | +0.01% |
| IND @ PIT | PIT -3 | -0.3 | 48.9% | -0.26% | +0.02% | -0.24% | +0.02% |
| IND @ PIT | IND +3 | +0.3 | 51.1% | +0.26% | -0.02% | +0.24% | -0.02% |
| LV @ NE | NE -3.5 | +0.4 | 50.0% | -0.96% | -0.02% | -0.97% | -0.03% |
| LV @ NE | LV +3.5 | -0.4 | 50.0% | +0.96% | +0.02% | +0.97% | +0.03% |
| MIN @ NO | NO +2.5 | -1.0 | 49.6% | -0.68% | +0.05% | -0.63% | +0.08% |
| MIN @ NO | MIN -2.5 | +1.0 | 50.4% | +0.68% | -0.05% | +0.63% | -0.08% |
| NYG @ WAS | WAS -4.5 | -1.0 | 48.9% | -0.26% | +0.05% | -0.21% | +0.08% |
| NYG @ WAS | NYG +4.5 | +1.0 | 51.1% | +0.26% | -0.05% | +0.21% | -0.08% |
| PHI @ JAX | JAX -7.5 | +0.5 | 49.6% | -0.68% | -0.03% | -0.70% | -0.04% |
| PHI @ JAX | PHI +7.5 | -0.5 | 50.4% | +0.68% | +0.03% | +0.70% | +0.04% |
| SF @ SEA | SEA -3 | -1.0 | 50.0% | -0.96% | +0.05% | -0.91% | +0.07% |
| SF @ SEA | SF +3 | +1.0 | 50.0% | +0.96% | -0.05% | +0.91% | -0.07% |
| BUF @ LA | LA -3 | -0.1 | 52.2% | -2.35% | +0.00% | -2.35% | +0.01% |
| BUF @ LA | BUF +3 | +0.1 | 47.8% | +2.35% | -0.00% | +2.35% | -0.01% |

| component | mean |shift| (pct pts) | max |shift| (pct pts) |
|---|---|---|
| legacy: remap (a,b) | 0.95 | 2.35 |
| legacy: model (c*edge) | 0.02 | 0.08 |
| legacy: total | 0.94 | 2.35 |
| offset (live): model (c*edge) | 0.04 | 0.12 |

## 1d. Legacy (a, b, c) vs constrained offset (a = 0, b = 1, c) on home covers

Same walk-forward as `validate`, development seasons 2015-2021 only; paired game-bootstrap CIs; negative = offset better.

| metric | games | legacy | offset (a=0,b=1) | offset - legacy | 95% CI | CI includes 0 |
|---|---|---|---|---|---|---|
| log loss | 1843 | 0.6931 | 0.6934 | 0.0003 | -0.00240 to +0.00283 | True |
| Brier | 1843 | 0.2500 | 0.2501 | 0.0001 | -0.00120 to +0.00141 | True |

