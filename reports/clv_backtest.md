# Opening lines and closing-line value (2015-2021)

Opening lines and closing-line value (backtest). Needs histlines.load() output.

Walk-forward: each test season uses a model fit on earlier seasons only, and the game's
pre-game ratings (built from games already played; the opener is posted after the previous
week's games). Caveats, stated in the report: the starting QB used is the actual starter
(possibly unknown when the line opened), and the closing line used for CLV is the file's.

1. Does the line move toward the model? OLS of (close - open) on (model - open), where
   "model" is (a) the ratings-only own line and (b) the blend recomputed AT THE OPENING LINE.
   Slope 0: the market ignores what the model knows; slope 1: the close moves all the way to
   the model. Game-bootstrap CI on the slope.
2. Bet at the open when EV > min_edge: probabilities from the live offset calibration anchored
   at the opening line and opening no-vig pair (-110/-110 when the file has no line odds),
   edge = blend at the open line minus the open line. Reported: count, CLV in points and in
   no-vig probability (closing distribution evaluated at the line bet), and flat ROI at the
   opening price and at -110, each with a game-bootstrap CI.

## Data

File `historical_lines.xlsx`: 5495 rows, 5495 matched to schedule games (3 with home/away swapped), 0 unmatched. Sign convention: handicap form (negated); correlation with nflverse closing line -0.994 on 3281 games. Median |file close - nflverse close|: 0.0.

Column mapping: `{'date': 'Date', 'home_team': 'Home Team', 'away_team': 'Away Team', 'home_line_open': 'Home Line Open', 'home_line_close': 'Home Line Close', 'home_line_odds_open': 'Home Line Odds Open', 'away_line_odds_open': 'Away Line Odds Open', 'home_line_odds_close': 'Home Line Odds Close', 'away_line_odds_close': 'Away Line Odds Close', 'total_open': 'Total Score Open', 'total_close': 'Total Score Close'}`

Games used: 1889; opening line odds from the file for 1889 (otherwise -110/-110).

## 1. Does the line move toward the model?

| model | games | slope | lo | hi | CI includes 0 | mean |model - open| |
|---|---|---|---|---|---|---|
| ratings-only own line | 1889 | 0.1743 | 0.1401 | 0.2069 | False | 2.0735 |
| blend at the opening line | 1889 | 0.1812 | 0.1213 | 0.2402 | False | 1.0845 |

## 2. Bet at the open when EV > 2% (probability_model `model`)

| metric | value | 95% CI |
|---|---|---|
| bets | 6 |  |
| W-L-P | 6-0-0 |  |
| mean CLV (points) | 0.5833 | -0.5000 to +1.6667 |
| mean CLV (no-vig probability) | 0.0086 | -0.0214 to +0.0412 |
| ROI at the opening price | 0.9598 | +0.8843 to +1.0354 |
| ROI at -110 | 0.9091 | +0.9091 to +0.9091 |

