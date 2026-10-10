# OL1: bet the opening spread toward the model (development, seasons 2015-2021)

Protocol: docs/EXPERIMENTS.md, section OL1. Model `c9f56b5`. **Historical only: nothing changes live predictions or decisions.**

> **Timing flag.** The file does not say when its 'open' line was recorded (no timestamps; one bookmaker per era: Pinnacle to Feb 2018, bet365 Sep 2018 - Feb 2025, betr from Sep 2025). It cannot be confirmed to fall in the live early window (168-84 h before kickoff). If it was posted earlier (look-ahead lines), the model knew more than the opener and these results are flattered.

Data: `historical_lines.xlsx`, 5495 of 5495 rows matched; games with an opening line in these seasons: 1889. Expected move a + b*g fit on earlier seasons only: 2016: a=-0.12, b=0.178, 2017: a=-0.12, b=0.191, 2018: a=-0.18, b=0.190, 2019: a=-0.20, b=0.184, 2020: a=-0.26, b=0.180, 2021: a=-0.30, b=0.173.

## Thresholds (|own line - open| >= T); break-even 52.4% at -110, 51.2% at -105

| T | bets | W | L | P | cover | cover_lo | cover_hi | pred_cover | roi_110 | roi_110_lo | roi_110_hi | roi_105 | roi_105_lo | roi_105_hi | roi_open | clv_points | clv_points_lo | clv_points_hi | mean_exp_move |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0000 | 786.0000 | 404.0000 | 369.0000 | 13.0000 | 52.3% | 48.6% | 55.8% | 52.2% | -0.2% | -6.7% | +6.2% | +2.0% | -4.6% | +8.6% | +0.9% | +0.54 | +0.41 | +0.68 | +0.60 |
| 3.0000 | 439.0000 | 233.0000 | 196.0000 | 10.0000 | 54.3% | 49.4% | 59.0% | 52.7% | +3.6% | -5.3% | +12.3% | +5.9% | -3.2% | +14.8% | +4.8% | +0.77 | +0.58 | +0.95 | +0.77 |
| 4.0000 | 237.0000 | 127.0000 | 104.0000 | 6.0000 | 55.0% | 48.5% | 61.5% | 53.2% | +4.8% | -6.9% | +16.5% | +7.2% | -4.8% | +19.1% | +6.0% | +0.92 | +0.65 | +1.17 | +0.93 |

**Chosen T = 4.** T=4: ROI at -110 +4.8% (95% CI -6.9% to +16.5%) - fails (CI includes 0).

## Breakdown at T = 4 (descriptive)

`pred_cover` = mean predicted P(cover | no push) from the expected move through the key-number margin distribution (expected close treated as fair); `cover` = actual.

| split | group | bets | W | L | P | cover | cover_lo | cover_hi | pred_cover | roi_110 | roi_110_lo | roi_110_hi | roi_105 | clv_points |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crosses 3 or 7 | False | 180 | 95 | 80 | 5 | 54.3% | 46.8% | 61.7% | 52.9% | +3.5% | -10.3% | +16.7% | +5.8% | +0.99 |
| crosses 3 or 7 | True | 17 | 9 | 8 | 0 | 52.9% | 29.4% | 76.5% | 56.7% | +1.1% | -43.9% | +46.0% | +3.4% | +0.74 |
| era | Pinnacle | 96 | 55 | 39 | 2 | 58.5% | 48.9% | 68.1% | 53.1% | +11.5% | -7.4% | +29.4% | +13.9% | +1.04 |
| era | bet365 | 141 | 72 | 65 | 4 | 52.6% | 43.8% | 60.6% | 53.3% | +0.3% | -15.4% | +15.3% | +2.5% | +0.84 |
| season | 2015 | 40 | 23 | 16 | 1 | 59.0% | 43.6% | 74.4% | nan | +12.3% | -16.4% | +40.9% | +14.8% | +0.70 |
| season | 2016 | 25 | 12 | 12 | 1 | 50.0% | 29.2% | 70.8% | 52.6% | -4.4% | -38.9% | +30.2% | -2.3% | +1.34 |
| season | 2017 | 31 | 20 | 11 | 0 | 64.5% | 48.3% | 80.6% | 53.4% | +23.2% | -7.8% | +54.0% | +26.0% | +1.23 |
| season | 2018 | 29 | 13 | 15 | 1 | 46.4% | 28.6% | 64.3% | 53.9% | -11.0% | -43.9% | +25.1% | -9.0% | +0.83 |
| season | 2019 | 43 | 25 | 17 | 1 | 59.5% | 42.9% | 73.8% | 52.9% | +13.3% | -15.4% | +40.0% | +15.8% | +1.03 |
| season | 2020 | 39 | 20 | 18 | 1 | 52.6% | 36.8% | 68.4% | 53.2% | +0.5% | -28.9% | +29.8% | +2.7% | +0.26 |
| season | 2021 | 30 | 14 | 15 | 1 | 48.3% | 31.0% | 65.5% | 53.2% | -7.6% | -42.4% | +27.3% | -5.6% | +1.33 |
| both usual QBs | False | 187 | 97 | 85 | 5 | 53.3% | 45.6% | 60.4% | 53.2% | +1.7% | -12.1% | +15.9% | +3.9% | +1.01 |
| both usual QBs | True | 50 | 30 | 19 | 1 | 61.2% | 46.9% | 75.5% | 53.1% | +16.5% | -8.4% | +41.3% | +19.1% | +0.60 |

