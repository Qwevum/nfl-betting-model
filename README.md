# NFL spread / moneyline / total model

Power-ratings model that prices every NFL game, compares its numbers with the
sportsbooks, and flags bets where the expected value clears a threshold. Built
on free [nflverse](https://github.com/nflverse) data (scores, closing lines and
play-by-play EPA since 2010).

## Setup

```bash
cd nfl-model
pip install -r requirements.txt
```

## Weekly use

```bash
python run.py predict            # next week's games: lines, fair numbers, bets
python run.py predict --week 6   # a specific week
python run.py ratings            # current team power ratings
python run.py grade              # after the games: record, units, closing-line value
python run.py backtest           # walk-forward test against closing lines, 2015-now
```

`predict` prints a sheet like this, and saves it under `picks/`:

| column | meaning |
|---|---|
| market | consensus sportsbook spread |
| model | our own line from team ratings alone |
| fair | ratings blended with the market (what edges are measured from) |
| home_win% / mkt_home% | our win probability vs the market's no-vig probability |
| SPREAD / ML / TOTAL | recommended bet, best price found, EV, stake in units (1u = 1% of bankroll) |

Rows marked ⚠ **check news** are not bets: the model disagrees with the market by
4+ points, which nearly always means an injury or QB change the ratings can't see.

## Getting the best line (line shopping)

By default the model only sees the nflverse consensus line. To compare every book:

* **Automatic:** get a free key at [the-odds-api.com](https://the-odds-api.com)
  (500 requests/month; one `predict` run uses 1), then
  `export ODDS_API_KEY=yourkey` before running `predict`.
* **Manual:** type lines from your own apps into `odds_manual.csv`.

For each side the model scores every book's line and price and keeps the one with
the highest expected value, so +3.5 at -115 vs +3 at -105 is decided correctly,
including the chance of a push on 3.

## News adjustments

Put injury/QB adjustments in `adjustments.csv`, e.g. `CIN,-6,backup QB`. The
model's own line moves the full amount; the fair line only moves the share the
market isn't already pricing.

## How it works

1. **Ratings** (`nflmodel/ratings.py`): games are replayed in date order. Each team
   carries opponent-adjusted ratings for points margin, EPA/play, success rate and
   pass EPA on offense and defense, plus pace and points. Features for a game only
   use games before it. 60% of each rating carries over to the next season.
2. **Model** (`nflmodel/model.py`): ridge regressions fit on all past seasons
   predict the home margin and the total. The *blend* model also sees the market
   line; its weight on the market shows how much the ratings add.
3. **Probabilities**: margins are not bell-shaped (3, 7, 10, 6, 14 happen far more
   often), so the outcome distribution is re-weighted by how often each final
   margin and total has actually happened. That gets pushes and key numbers right.
4. **Bets** (`nflmodel/evaluate.py`): win/push/lose probabilities for every offer
   give an expected value. Minimum EV: 2% spreads/totals, 3% moneylines. Stakes are
   quarter-Kelly, capped at 2 units.

## Reading the results honestly

* Break-even at -110 is **52.4%**. Over a season (~50-100 bets) the 95% range
  around any win rate is roughly ±10%, so a single season proves very little.
* The backtest bets **closing** lines, the sharpest numbers there are. Betting
  earlier in the week and shopping books is where real-world edge usually comes from.
* Track **closing-line value** with `grade`: consistently getting better numbers
  than the close is the earliest reliable sign the model is finding something.

Bet only what you can afford to lose.
