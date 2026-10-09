# Probability-interval simulation

Protocol: `docs/UNCERTAINTY.md`. Truth = the point model fit on real training seasons; training outcomes simulated from its predictive distribution with team-season and season shocks; each simulation refits the point model and the bootstrap replicates under every block scheme and forecasts the real test-season games (closing consensus markets held fixed).

* **SD ratio**: mean bootstrap SD / true sampling SD of the estimator (logit scale). 1 = right size, < 1 = intervals too narrow.
* **cover E[est]**: coverage of the estimator's own expected value, i.e. of sampling variability only.
* **cover truth**: coverage of the truth model's probability. Lower than cover E[est] when the estimator is biased (ridge shrinkage, calibration); the bootstrap does not correct bias.

Limits: the truth is the model itself (well specified apart from estimation bias); market inputs are fixed and do not react to the shocks; the dependence is a simple random-effects approximation; real games also carry information neither the model nor this simulation contains.

## Scenario A: dependence estimated from real residuals (used for selection)

Train 2011-2020, test 2021 (285 games), 60 simulations x 50 replicates per scheme, seed 20261009, 1861s.

Shock SDs (points): `{'sd_margin_resid': 13.133, 'sd_total_resid': 13.292, 'tau_team_margin': 0.0, 'tau_season_hfa': np.float64(0.143), 'tau_team_total': 0.0, 'tau_season_total': 0.0}`

True sampling SD of the estimator, median game: 0.0387 (logit); median bias of its expectation vs truth: -0.0043 (logit).

| scheme | sd_ratio (pooled) | sd_ratio (median game) | cover E[est] 50% | cover truth 50% | cover E[est] 80% | cover truth 80% | cover E[est] 90% | cover truth 90% | cover E[est] 95% | cover truth 95% |
|---|---|---|---|---|---|---|---|---|---|---|
| game | 1.224 | 1.187 | 0.651 | 0.554 | 0.909 | 0.847 | 0.966 | 0.928 | 0.986 | 0.968 |
| week4 | 1.193 | 1.145 | 0.622 | 0.535 | 0.897 | 0.830 | 0.963 | 0.922 | 0.984 | 0.963 |
| season | 1.074 | 1.012 | 0.565 | 0.479 | 0.836 | 0.766 | 0.914 | 0.868 | 0.954 | 0.923 |
| season*week4 | 1.787 | 1.742 | 0.811 | 0.740 | 0.973 | 0.948 | 0.991 | 0.981 | 0.996 | 0.993 |

## Scenario B (sensitivity only, not used for selection): larger fixed shocks

Train 2011-2020, test 2021 (285 games), 60 simulations x 50 replicates per scheme, seed 20261010, 8513s.

Shock SDs (points): `{'sd_margin_resid': 13.133, 'sd_total_resid': 13.292, 'tau_team_margin': 2.0, 'tau_season_hfa': 1.0, 'tau_team_total': 2.0, 'tau_season_total': 2.0, 'override': True}`

True sampling SD of the estimator, median game: 0.0403 (logit); median bias of its expectation vs truth: -0.0098 (logit).

| scheme | sd_ratio (pooled) | sd_ratio (median game) | cover E[est] 50% | cover truth 50% | cover E[est] 80% | cover truth 80% | cover E[est] 90% | cover truth 90% | cover E[est] 95% | cover truth 95% |
|---|---|---|---|---|---|---|---|---|---|---|
| game | 1.214 | 1.143 | 0.604 | 0.442 | 0.891 | 0.723 | 0.961 | 0.829 | 0.983 | 0.896 |
| week4 | 1.165 | 1.090 | 0.581 | 0.426 | 0.863 | 0.694 | 0.942 | 0.802 | 0.974 | 0.872 |
| season | 1.147 | 1.059 | 0.541 | 0.407 | 0.832 | 0.672 | 0.919 | 0.774 | 0.960 | 0.842 |
| season*week4 | 1.825 | 1.734 | 0.775 | 0.626 | 0.974 | 0.885 | 0.995 | 0.952 | 0.999 | 0.980 |

**Selected scheme (pre-registered rule, estimated-dependence scenario): `season`**, closest to 1 among ratios >= 0.9 (ratio 1.07).

