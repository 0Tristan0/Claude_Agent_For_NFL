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
python3 src/site.py      # website: site/index.html + site/games/<slug>.html from games.json
```

## Website
Live at <https://claude.ai/artifact/758ZYBxwk8dX14ZXeM9Db1> (private to the owner until shared from its Share menu).
The home page lists every game in `games.json` (upcoming with locked picks, graded results) plus a track record table;
each game with a `report` gets its own page under `games/`.

To add a game or grade one: edit `games.json`, run `python3 src/site.py`, then ask Claude Code to republish
`nfl-matchup-lab/site/index.html` to that link with every file in `site/games/` attached under `games/`.
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
output/     generated report pages
site/       generated website (home page + one page per game)
games.json  list of games shown on the website, with lock times and grades
backtests/  SF at SEA prediction script, locked predictions, graded results
```

## Known limits
- All data is hand-entered from web pages for one game. Nothing fetches automatically yet.
- Four games of data per team; estimates are rough.
- Not betting advice.
