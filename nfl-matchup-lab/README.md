# NFL Matchup Lab

Builds a one-page matchup report for an NFL game: unit-vs-unit ranks, pass coverage comparison,
1-on-1 matchups, a props board with season hit rates, PrizePicks payout break-evens, and 3 to 6 pick lineups.
Started from the Bills at Rams MNF report (Week 5, 2026).

## Run it
```bash
cd nfl-matchup-lab
python3 src/ranks.py     # league ranks for both teams -> src/stats.json
python3 src/build.py     # unit tables -> output/bills-rams-mnf.html
python3 src/props.py     # props board + 1-on-1 section
python3 src/ev.py        # hit estimates, break-evens, lineup value -> src/ev.json
python3 src/parlay.py    # payout check + lineups
open output/bills-rams-mnf.html
```
No dependencies beyond Python 3.

## How the numbers work
- **Ranks**: per-game or per-play stats across all 32 teams (FootballDB, Oct 12 2026). Ties marked T.
- **Hit estimate per prop**: half from this season's game log (shrunk toward 50% with a 5-game prior)
  plus a small matchup adjustment, half from the betting market's fair probability.
- **Break-even**: PrizePicks Power payouts 3x/5x/10x/20x/37.5x for 2-6 picks; Flex 2.25x/5x/10x/25x top tiers.
  Same-game lineups can get lower multipliers, so each lineup lists a minimum multiplier.
- **Game score model**: each team's points = average of its points scored per game and the opponent's points allowed per game.

## Layout
```
src/        pipeline scripts + HTML template (data is currently hardcoded inside the scripts)
output/     generated page
backtests/  SF at SEA prediction script, locked predictions, graded results
```

## Known limits
- All data is hand-entered from web pages for one game. Nothing fetches automatically yet.
- Four games of data per team; estimates are rough.
- Not betting advice.
