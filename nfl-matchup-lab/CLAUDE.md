# CLAUDE.md

## What this is
A Python pipeline that renders a static HTML matchup report for one NFL game (see README.md).
Owner is a CS + Finance student; prefers iterative work, one section at a time, plain casual copy with no em dashes.

## Current state
- Data is hardcoded in `src/ranks.py`, `src/props.py`, `src/ev.py` for Bills at Rams (Oct 12 2026).
- `src/template.html` holds the page shell and CSS tokens; scripts inject sections via `{{PLACEHOLDER}}`
  strings and by replacing the `<footer class="notes">` marker.
- `backtests/` holds one graded backtest (SF at SEA). Its game-level picks were contaminated; see results file.

## Roadmap (do these in order)
1. **Data layer**: `src/fetch.py` that pulls team offense/defense, passing, rushing tables and player game logs
   into `data/<season>/<week>/*.json`. FootballDB URLs used so far are in the page footer. Cache responses.
2. **Generalize**: replace hardcoded team keys (BUF, LAR) with a `--home/--away` CLI. Matchup pairs, coverage stats
   and lineup builder should work for any two teams.
3. **Schedule + picker**: fetch the week's schedule, generate one report per game plus an `index.html` game picker.
4. **Props input**: a `data/props.csv` (player, team, stat, line, side, market_odds, source) the user can paste lines into,
   since PrizePicks lines live in the app.
5. **Backtest loop**: after each game, grade every pick automatically and append to `backtests/log.csv`;
   report hit rate and units by prop type. Freeze predictions before kickoff (write them with a timestamp).
6. **Model fixes**: add turnover and pace terms to the total model (missed SF-SEA total by 15).

## Conventions
- Keep the pipeline dependency-free unless a library does real work (requests/pandas are fine for step 1).
- Every number on the page must trace to a source listed in the footer.
- Never present estimates as certainties; keep the "not betting advice" caveat.
