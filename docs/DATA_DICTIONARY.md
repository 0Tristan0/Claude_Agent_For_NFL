# Data dictionary and provenance

All data come from public **nflverse** release files on GitHub
(`https://github.com/nflverse/nflverse-data/releases`). nflverse distributes its data under CC-BY 4.0
(attribution: nflverse); play-level statistics originate from the NFL, snap counts from Pro-Football-Reference,
and Next Gen Stats from the NFL. This project downloads release files only (no scraping of the original
providers), caches them locally, and pauses between requests.

**How the sources were verified (2026-10-04).** The rendered documentation sites (`nflverse.nflverse.com`,
`nflreadr.nflverse.com`) were blocked by this environment's network proxy, so field definitions and the update
schedule were read from the same documents' source in the `nflverse/nflreadr` repository
(`data-raw/dictionary_*.csv|json`, `vignettes/articles/nflverse_data_schedule.Rmd`). Every claim about
coverage below was then checked against the downloaded files. `nextgenstats.nfl.com` was not accessed directly;
NGS weekly data were taken from the nflverse release. The live per-source inventory (rows, seasons, retrieval
time, source timestamp, missingness) is generated in [DATA_INVENTORY.md](DATA_INVENTORY.md) and on the
dashboard's *Data health* page.

## Pregame-availability rules

| Source | Used as model input? | Available before a historical kickoff? | Why |
|---|---|---|---|
| Weekly player stats | Yes, **prior games only** | Yes, shifted | Published nightly after each game day |
| Snap counts (PFR) | Yes, **prior games only** | Yes, shifted | Published within hours to a day of each game |
| Schedule: teams, date, kickoff, week, game type | Yes | Yes | Fixed in advance |
| Schedule: `roof == "dome"` | Ablation only (context group) | Yes | Fixed dome is structural |
| Schedule: rest days | Ablation only (context group) | Yes | Derived from the schedule |
| Schedule: `temp`, `wind` | **No** | **No** | Blank for upcoming games, filled in after: observed weather, not a forecast |
| Schedule: `home_qb_name`, `away_qb_name` | **No** (context only) | **No** for completed games | The actual starter, known only afterwards |
| Schedule: betting lines | **No** | n/a | Deliberately excluded (no sportsbook data in this project) |
| Players table: draft pick, rookie season | Yes | Yes | Static facts |
| Players table: position, team | **No** | No | Current snapshot, not historical |
| Injury reports | **No** (context only) | Unverified | The files carry no publication timestamp (documented `date_modified` is absent) |
| Next Gen Stats weekly | **No** (context only) | Yes, shifted | Rows only for qualifying receivers, so missingness is not random |
| FTN/NGS participation (routes) | **No** (not ingested) | **No** in-season | 2023+ released only after the postseason; 2026 file returns 404 |

## Identifiers and joins

* **Player**: GSIS id (`player_id`, e.g. `00-0036900`). Snap counts use PFR ids and are mapped to GSIS ids with
  the nflverse players table (ambiguous PFR ids are dropped; about 0.2% of snap rows are unmapped). Names are
  never used as join keys; two players with the same name stay separate, and name lookups that match more than
  one id raise an error asking for the id.
* **Game**: nflverse `game_id` = `{season}_{week:02d}_{away}_{home}`. All stats and snap rows joined to the
  schedule with no orphans.
* **Teams**: relocated franchises are normalized to the current franchise code (`OAK→LV`, `SD→LAC`,
  `STL→LA`). Player stats already use current codes for every season, while schedules and snap counts use
  historical codes; without this step, 2012–2019 team totals failed to join for those franchises.
* **Season/week**: `season` is the season's starting year (January playoff games belong to the previous year).
  `season_type` is `POST` for WC/DIV/CON/SB; postseason weeks are 19–22 (2021+) and are kept, flagged by
  `is_post`.
* **Kickoff**: schedule `gameday` + `gametime` are US Eastern; converted to UTC with DST handled.
* **Traded/released players**: a player's team is the team of each game; the first game with a new team sets
  `team_changed = 1` and resets `games_with_team_prior`. History from the old team is still used.

## Participation spine (the most important join)

nflverse weekly player stats contain **no row for a player who played but recorded no statistic**. In
2013–2025, **15.1%** of WR/TE player-games with offensive snaps have no stats row. Building from stats alone
would drop real zero-target games and inflate every forecast.

The modelling table (`player_games`) therefore unions snap-count rows (offensive snaps > 0) with stats rows:

* In snap counts and missing from stats → `targets = receptions = yards = 0`, `stats_zero_filled = True`. This
  records an observed fact (he played and no target was logged); it is not imputation.
* In stats and missing from snap counts → kept, with `snap_share` missing (`snaps_missing = True`).
* If a game's snap counts are not yet published, nothing is zero-filled for that game.
* Absent from both → did not participate (inactive, scratch, or no offensive snap). **Not** a zero; such games
  are excluded from history and evaluation, and forecasts are conditional on participating.
* The 2012 snap-count file is published but contains **0 rows**, so 2012 has no spine. 2012 is used only as
  warm-up history, and its zero-target games are unrecoverable (so 2012 history slightly overstates usage).

## Modelling table fields (`data/processed/player_games.parquet`)

| Field | Definition | Denominator / unit | Source |
|---|---|---|---|
| `targets` | Passes thrown to the player (official attribution) | count | player stats |
| `receptions` | Catches | count | player stats |
| `receiving_yards` | Receiving yards (can be negative) | yards | player stats |
| `receiving_air_yards` | Air yards on all targets, including incompletions | yards | player stats |
| `receiving_yards_after_catch` | Yards after catch | yards | player stats |
| `receiving_20` | Receptions of 20+ yards (*inferred*: not in the official dictionary) | count | player stats |
| `team_targets` | Sum of all players' targets for the team in the game | count | derived |
| `target_share` | `targets / team_targets` (recomputed from team totals) | team targets | derived |
| `air_yards_share` | `receiving_air_yards / team air yards` | team air yards | derived |
| `snap_share` | Offensive snaps / team offensive snaps (PFR `offense_pct`) | team offensive snaps, **all play types** | snap counts |
| `team_dropbacks` | Team pass attempts + sacks | count | derived |
| `team_offense_plays` | Maximum offensive snaps of any player on the team (≈ team plays) | count | derived |
| `stats_zero_filled` | Played per snap counts, no stats row → zeros recorded | flag | derived |
| `snaps_missing` | No snap-count row for this player-game | flag | derived |
| `draft_pick_filled` | Overall pick; undrafted = 260 | pick | players |
| `is_rookie` | `season == rookie_season` | flag | players |

**Snap share is not route participation.** It counts run plays and pass-blocking snaps. Wherever it appears it
is labelled "snap share". No route data are used or imputed.

## Pregame features (`nfl_agent/features/pregame.py`)

All features are computed from the player's (or team's) *earlier* games only (the current game is excluded by
construction; verified by a perturbation test). Weights decay by game (half-life 10 games for players, 8 for
teams), with an extra ×0.6 at each season boundary. Rates are shrunk toward the position-level value from
earlier seasons by adding pseudo-observations.

| Feature | Definition | Shrinkage prior (pseudo-count) |
|---|---|---|
| `tpg_ew` | Weighted targets per game | position targets/game (3 games) |
| `tshare_ew` | Weighted Σtargets / Σteam targets (ratio of sums, not mean of ratios) | position target share (3 games) |
| `tshare_recent` | Same, half-life 1.5 games ("last ~2 games") | `tshare_ew` (1 game) |
| `catch_rate_ew` | Σreceptions / Σtargets | position catch rate (25 targets) |
| `ypr_ew` | Σyards / Σreceptions | position YPR (15 receptions) |
| `adot_ew` | Σair yards / Σtargets (average depth of target) | position aDOT (20 targets) |
| `yac_per_rec_ew` | ΣYAC / Σreceptions | position (15 receptions) |
| `explosive_rate_ew` | Σ20+ receptions / Σreceptions | position (20 receptions) |
| `ypt_ew` | `catch_rate_ew × ypr_ew` (yards per target) | derived |
| `snap_share_ew`, `snap_share_recent` | Weighted snap share over games with snap data | position mean (2 games / 1 game) |
| `eff_games`, `n_games_prior` | Weighted and raw count of prior games played | — |
| `games_with_team_prior`, `team_changed` | Tenure with current team; first game with a new team | — |
| `days_since_last_game` | Days since the player's previous game (long gaps suggest injury or role change) | — |
| `team_tpg_ew`, `team_dbpg_ew` | Team targets / dropbacks per game | league (4 games) |
| `opp_def_tpg_rel`, `opp_def_ypt_rel`, `opp_def_catch_rel` | Opponent WR/TE targets, yards per target and catch rate allowed, relative to league | league (4 / 8 games) |

Closely related metrics were not stacked blindly: target share and targets per game both enter the target model,
but air-yard share and WOPR (linear combinations of target and air-yard share) were left out to avoid
double-counting. The ablation in [EVALUATION.md](EVALUATION.md) shows which groups improve held-out scores.

## Context shown but not used as model input

Injury-report status, schedule-listed QB, roof, and NGS separation/cushion appear on the forecast page with their
provenance and the reason they are not inputs.

## User-supplied data

`Data health → Import user-supplied data` accepts a CSV with `player_id, game_id` and `routes_run` and/or
`route_participation` (optional `team_dropbacks`, `notes`). Rows are validated (ids, ranges, duplicates,
`routes_run ≤ team_dropbacks`), blanks stay missing, a source label is required, and the file hash and import time
are stored. Imported values are displayed as **user-supplied** on the Usage history page and are **not** model
inputs.

## Demo fixture

`python -m nfl_agent.cli build --demo` generates synthetic data (teams `DMA`–`DMH`, players `DEMO …`) in the same
raw schemas. It exists for tests and for offline use; the dashboard shows a persistent red banner whenever it is
loaded. It is never mixed with real data (separate directory).
