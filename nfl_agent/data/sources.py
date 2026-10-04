"""Registry of public data sources and what we verified about them.

All URLs are nflverse GitHub release assets (publicly downloadable, CC-BY 4.0
licensed data as stated by nflverse; play-level data originates from the NFL
and other providers — see docs/DATA_DICTIONARY.md). Update schedules come from
the nflreadr article "nflverse Data Update and Availability Schedule"
(source: github.com/nflverse/nflreadr/blob/main/vignettes/articles/
nflverse_data_schedule.Rmd), read on 2026-10-04 because the rendered docs site
was blocked by this environment's network proxy.

`pregame_status` is our classification of whether a dataset's values can be
used as *pregame* inputs in a strict historical backtest:

* "pregame_derivable": only prior games' values are used (shifted), so they
  were public before kickoff (nightly updates after each game day).
* "known_in_advance": scheduling facts fixed before the game (teams, date,
  stadium, rest days, fixed dome).
* "retrospective": values filled in after the game (observed weather,
  actual starting QB). Never used as model inputs.
* "timing_unverified": plausibly pregame (e.g. official injury report) but the
  file carries no publication timestamp, so excluded from the strict backtest
  and shown only as labelled context.
* "postseason_release": published only after the season ends (FTN
  participation 2023+). Not available in-season.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"


@dataclass(frozen=True)
class Source:
    key: str
    tag: str
    url_template: str            # may contain {season}
    per_season: bool
    first_season: int
    description: str
    update_frequency: str
    pregame_status: str
    used_in_model: bool
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def url(self, season: int | None = None) -> str:
        return self.url_template.format(season=season) if self.per_season else self.url_template

    @property
    def timestamp_url(self) -> str:
        return f"{RELEASE_BASE}/{self.tag}/timestamp.json"


SOURCES: dict[str, Source] = {
    "player_stats": Source(
        key="player_stats",
        tag="stats_player",
        url_template=RELEASE_BASE + "/stats_player/stats_player_week_{season}.parquet",
        per_season=True,
        first_season=1999,
        description="Weekly player box-score stats computed by nflverse from NFL play-by-play "
                    "(targets, receptions, receiving yards, air yards, YAC, target share).",
        update_frequency="Nightly after each game day, plus game-day refreshes; stat corrections "
                         "land Mon-Wed (Thursday data is cleanest).",
        pregame_status="pregame_derivable",
        used_in_model=True,
        limitations=(
            "A player who played but recorded no statistic has NO row. Zero-target games must be "
            "recovered from snap counts (about 15% of WR/TE games with offensive snaps in 2025).",
            "receiving_10/20/40 are not in the official dictionary; we infer they count receptions "
            "of 10+/20+/40+ yards.",
            "Values can change after stat corrections, so a re-download may differ slightly from "
            "what was public the night after a game.",
        ),
    ),
    "snap_counts": Source(
        key="snap_counts",
        tag="snap_counts",
        url_template=RELEASE_BASE + "/snap_counts/snap_counts_{season}.parquet",
        per_season=True,
        first_season=2012,
        description="Pro-Football-Reference offensive/defensive/special-teams snap counts per "
                    "player-game (PFR player id).",
        update_frequency="Every 6 hours in season (0/6/12/18 UTC); depends on PFR publication.",
        pregame_status="pregame_derivable",
        used_in_model=True,
        limitations=(
            "Snap share counts every offensive snap, including run plays. It is NOT route "
            "participation and is never labelled as such.",
            "Uses PFR ids; ~0.2% of rows do not map to a GSIS id via the players table.",
            "The 2012 file is published but contains 0 rows (checked 2026-10-04): usable snap data start "
            "in 2013. 2012 is used only as feature warm-up history, and its zero-target games cannot be "
            "recovered (no snap spine), so 2012 history slightly overstates usage.",
        ),
    ),
    "schedules": Source(
        key="schedules",
        tag="schedules",
        url_template=RELEASE_BASE + "/schedules/games.parquet",
        per_season=False,
        first_season=1999,
        description="Game schedule/results: game_id, season, week, game_type, kickoff date/time "
                    "(Eastern), teams, scores, roof, rest days, weather, listed QBs.",
        update_frequency="Every 5 minutes in season.",
        pregame_status="known_in_advance",
        used_in_model=True,
        limitations=(
            "temp/wind are blank for upcoming games and filled after games: treated as "
            "retrospectively observed weather, never as a pregame forecast.",
            "home_qb_name/away_qb_name for completed games are the actual starters (retrospective); "
            "for upcoming games they are nflverse's projection with unknown publication timing.",
            "roof 'open'/'closed' for retractable roofs is decided near kickoff; only 'dome' is "
            "treated as known in advance.",
            "Betting-market columns exist in this file; this project deliberately does not use them.",
        ),
    ),
    "players": Source(
        key="players",
        tag="players",
        url_template=RELEASE_BASE + "/players/players.parquet",
        per_season=False,
        first_season=1920,
        description="Player master table: GSIS id, PFR id, names, position, draft info, rookie season.",
        update_frequency="Daily.",
        pregame_status="known_in_advance",
        used_in_model=True,
        limitations=(
            "Current snapshot: position and team reflect today, not history. We use it only for "
            "the PFR->GSIS id crosswalk and static draft information.",
        ),
    ),
    "nextgen_receiving": Source(
        key="nextgen_receiving",
        tag="nextgen_stats",
        url_template=RELEASE_BASE + "/nextgen_stats/ngs_receiving.parquet",
        per_season=False,
        first_season=2016,
        description="NFL Next Gen Stats weekly receiving: separation, cushion, intended air yards, "
                    "expected YAC.",
        update_frequency="Nightly (3-5am ET) in season; depends on NGS publication.",
        pregame_status="pregame_derivable",
        used_in_model=False,
        limitations=(
            "Weekly rows exist only for qualifying receivers (~70-80 per week), so missingness is "
            "NOT random: low-volume players are systematically absent. Shown as context only.",
            "week 0 rows are season aggregates and must never be joined to individual games.",
        ),
    ),
    "injuries": Source(
        key="injuries",
        tag="injuries",
        url_template=RELEASE_BASE + "/injuries/injuries_{season}.parquet",
        per_season=True,
        first_season=2009,
        description="Official NFL injury report game status (Out/Doubtful/Questionable) and practice "
                    "participation.",
        update_frequency="Daily at 7AM UTC in season.",
        pregame_status="timing_unverified",
        used_in_model=False,
        limitations=(
            "The current files contain no publication timestamp (the documented date_modified field "
            "is absent), so we cannot prove a row was public before kickoff. Excluded from the "
            "strict backtest; shown as labelled context only.",
        ),
    ),
    "participation": Source(
        key="participation",
        tag="pbp_participation",
        url_template=RELEASE_BASE + "/pbp_participation/pbp_participation_{season}.parquet",
        per_season=True,
        first_season=2016,
        description="Play-level on-field personnel; 'route' is the route of the targeted receiver only.",
        update_frequency="2016-2022 from NGS; 2023+ from FTN, released only after the postseason.",
        pregame_status="postseason_release",
        used_in_model=False,
        limitations=(
            "Not available during the season (2026 file returns 404 as of 2026-10-04).",
            "Does not directly give routes run per receiver; deriving them needs play-by-play pass "
            "flags and still counts pass-blocking snaps. Not ingested in v1.",
        ),
    ),
}

# Sources the v1 pipeline downloads.
INGESTED = ("schedules", "players", "player_stats", "snap_counts", "injuries", "nextgen_receiving")
