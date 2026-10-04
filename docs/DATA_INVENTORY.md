# Data inventory

*Generated 2026-10-04T18:13:43+00:00 from the local cache and ingestion log. Regenerate with `python -m nfl_agent.cli inventory`.*

## schedules

* URL: `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet`
* Seasons in cache: 1999-2026; rows: 7,548; fields: 46
* Update frequency (per nflverse): Every 5 minutes in season.
* Last retrieved (UTC): 2026-10-04T18:13:13+00:00; source timestamp: nan
* Usable before a historical kickoff: **Yes for facts fixed before kickoff (exceptions listed under limitations)**; used in model: True
* Limitations: temp/wind are blank for upcoming games and filled after games: treated as retrospectively observed weather, never as a pregame forecast. home_qb_name/away_qb_name for completed games are the actual starters (retrospective); for upcoming games they are nflverse's projection with unknown publication timing. roof 'open'/'closed' for retractable roofs is decided near kickoff; only 'dome' is treated as known in advance. Betting-market columns exist in this file; this project deliberately does not use them.

| field | present | missing share |
|---|---|---|
| gameday | True | 0.0% |
| gametime | True | 3.4% |
| roof | True | 0.0% |
| temp | True | 30.6% |
| wind | True | 30.6% |
| away_rest | True | 0.0% |
| home_qb_name | True | 2.6% |

## players

* URL: `https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet`
* Seasons in cache: n/a; rows: 24,844; fields: 39
* Update frequency (per nflverse): Daily.
* Last retrieved (UTC): 2026-10-04T17:37:34+00:00; source timestamp: 2026-10-04 11:06:29 EDT
* Usable before a historical kickoff: **Yes for facts fixed before kickoff (exceptions listed under limitations)**; used in model: True
* Limitations: Current snapshot: position and team reflect today, not history. We use it only for the PFR->GSIS id crosswalk and static draft information.

| field | present | missing share |
|---|---|---|
| gsis_id | True | 0.0% |
| pfr_id | True | 8.7% |
| draft_pick | True | 49.7% |
| rookie_season | True | 0.0% |

## player_stats

* URL: `https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet`
* Seasons in cache: 2012-2026; rows: 255,564; fields: 150
* Update frequency (per nflverse): Nightly after each game day, plus game-day refreshes; stat corrections land Mon-Wed (Thursday data is cleanest).
* Last retrieved (UTC): 2026-10-04T17:37:50+00:00; source timestamp: 2026-10-04 10:49:17 EDT
* Usable before a historical kickoff: **Yes, as prior-game history only (shifted)**; used in model: True
* Limitations: A player who played but recorded no statistic has NO row. Zero-target games must be recovered from snap counts (about 15% of WR/TE games with offensive snaps in 2025). receiving_10/20/40 are not in the official dictionary; we infer they count receptions of 10+/20+/40+ yards. Values can change after stat corrections, so a re-download may differ slightly from what was public the night after a game.

| field | present | missing share |
|---|---|---|
| targets | True | 0.0% |
| receptions | True | 0.0% |
| receiving_yards | True | 0.0% |
| receiving_air_yards | True | 0.0% |
| receiving_yards_after_catch | True | 0.0% |
| target_share | True | 0.0% |
| air_yards_share | True | 0.0% |
| receiving_20 | True | 0.0% |

## snap_counts

* URL: `https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet`
* Seasons in cache: 2013-2026; rows: 329,191; fields: 16
* Update frequency (per nflverse): Every 6 hours in season (0/6/12/18 UTC); depends on PFR publication.
* Last retrieved (UTC): 2026-10-04T17:38:04+00:00; source timestamp: 2026-10-02 07:01:49 EDT
* Usable before a historical kickoff: **Yes, as prior-game history only (shifted)**; used in model: True
* Limitations: Snap share counts every offensive snap, including run plays. It is NOT route participation and is never labelled as such. Uses PFR ids; ~0.2% of rows do not map to a GSIS id via the players table. The 2012 file is published but contains 0 rows (checked 2026-10-04): usable snap data start in 2013. 2012 is used only as feature warm-up history, and its zero-target games cannot be recovered (no snap spine), so 2012 history slightly overstates usage.

| field | present | missing share |
|---|---|---|
| offense_snaps | True | 0.0% |
| offense_pct | True | 0.0% |

## injuries

* URL: `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet`
* Seasons in cache: 2012-2026; rows: 77,521; fields: 17
* Update frequency (per nflverse): Daily at 7AM UTC in season.
* Last retrieved (UTC): 2026-10-04T17:38:19+00:00; source timestamp: 2026-10-04 09:08:56 EDT
* Usable before a historical kickoff: **Unverified: no publication timestamp; context only**; used in model: False
* Limitations: The current files contain no publication timestamp (the documented date_modified field is absent), so we cannot prove a row was public before kickoff. Excluded from the strict backtest; shown as labelled context only.

| field | present | missing share |
|---|---|---|
| report_status | True | 38.8% |
| practice_status | True | 0.1% |
| date_modified | True | 9.2% |

## nextgen_receiving

* URL: `https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_receiving.parquet`
* Seasons in cache: 2016-2026; rows: 15,064; fields: 23
* Update frequency (per nflverse): Nightly (3-5am ET) in season; depends on NGS publication.
* Last retrieved (UTC): 2026-10-04T17:38:21+00:00; source timestamp: 2026-10-04 08:58:41 EDT
* Usable before a historical kickoff: **Yes, as prior-game history only (shifted)**; used in model: False
* Limitations: Weekly rows exist only for qualifying receivers (~70-80 per week), so missingness is NOT random: low-volume players are systematically absent. Shown as context only. week 0 rows are season aggregates and must never be joined to individual games.

| field | present | missing share |
|---|---|---|
| avg_separation | True | 0.0% |
| avg_cushion | True | 0.0% |
| avg_intended_air_yards | True | 0.0% |
| avg_yac_above_expectation | True | 0.4% |
