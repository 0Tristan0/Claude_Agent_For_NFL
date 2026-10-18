"""Registry of nflverse release files and the rule for when each is usable before kickoff.

Ported from nfl_agent/data/sources.py @90f63d6 and extended. Facts were checked live on 2026-10-12
(see the plan's research notes). `availability` documents how the no-lookahead rules treat a source:

  game_lag     rows for a game become usable `config/model.toml [lags_hours]` hours after its kickoff
  prior_season usable only for predictions in later seasons (published after the postseason)
  static       facts fixed before a game (crosswalk ids, stadiums); snapshot fields are never history
  grading      outcomes; only track/ and backtest/ scoring may read them
  live_only    timing cannot be proven historically; used live (snapshotted at freeze), never in backtests
"""
from __future__ import annotations

from dataclasses import dataclass, field

RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"


@dataclass(frozen=True)
class Source:
    key: str
    tag: str
    filename: str                 # may contain {season}
    per_season: bool
    first_season: int
    description: str
    availability: str
    lag: str | None = None        # key into [lags_hours] for game_lag sources
    last_season: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def url(self, season: int | None = None) -> str:
        name = self.filename.format(season=season) if self.per_season else self.filename
        return f"{RELEASE_BASE}/{self.tag}/{name}"

    @property
    def timestamp_url(self) -> str:
        return f"{RELEASE_BASE}/{self.tag}/timestamp.json"


def _pfr(kind: str) -> Source:
    return Source(
        key=f"pfr_{kind}", tag="pfr_advstats", filename=f"advstats_week_{kind}_{{season}}.parquet",
        per_season=True, first_season=2018, availability="game_lag", lag="pfr",
        description=f"Pro-Football-Reference weekly advanced {kind} stats (pfr_player_id).",
        notes=("Published 1-3 days after games; keyed on pfr_player_id, mapped to GSIS via players.",),
    )


SOURCES: dict[str, Source] = {s.key: s for s in [
    Source("schedules", "schedules", "games.parquet", False, 1999,
           "Schedule, results and closing game lines (spread_line, total_line, moneylines).",
           "static",
           notes=("Lines are a single snapshot overwritten until kickoff, so historical values are "
                  "closing lines.", "temp, wind and QB names are filled after games: never features.",
                  "Only roof == 'dome' is known in advance.")),
    Source("players", "players", "players.parquet", False, 1920,
           "Player master: gsis_id, pfr_id, names, draft info.", "static",
           notes=("Current snapshot: position/team/status reflect today, never used as history.",)),
    Source("pbp", "pbp", "play_by_play_{season}.parquet", True, 1999,
           "nflfastR play-by-play (EPA, WP, xpass, air yards, participants).", "game_lag", lag="pbp"),
    Source("snap_counts", "snap_counts", "snap_counts_{season}.parquet", True, 2013,
           "PFR snap counts per player-game.", "game_lag", lag="snaps",
           notes=("Snap share counts run plays too: it is not route participation.",)),
    Source("stats_player", "stats_player", "stats_player_week_{season}.parquet", True, 1999,
           "Official weekly player stats (grading and data checks only).", "grading"),
    Source("participation", "pbp_participation", "pbp_participation_{season}.parquet", True, 2016,
           "Players on the field per play; used for dropback participation of earlier seasons.",
           "prior_season", last_season=2025,
           notes=("Published after the postseason; no in-season file exists.",)),
    Source("ftn", "ftn_charting", "ftn_charting_{season}.parquet", True, 2022,
           "FTN charting: play action, screens, catchable ball, drops (CC-BY-SA 4.0, attribute FTN).",
           "game_lag", lag="ftn"),
    Source("rosters", "weekly_rosters", "roster_weekly_{season}.parquet", True, 2002,
           "Weekly rosters: position by week.", "game_lag", lag="pbp"),
    Source("injuries", "injuries", "injuries_{season}.parquet", True, 2009,
           "Official injury report final status per week.", "live_only",
           notes=("date_modified exists only through 2024, so historical timing is unprovable for 2025+.",)),
    Source("depth", "depth_charts", "depth_charts_{season}.parquet", True, 2001,
           "Depth charts (weekly through 2024, timestamped dt from 2025).", "game_lag", lag="depth"),
    _pfr("pass"), _pfr("rush"), _pfr("rec"), _pfr("def"),
]}

# Sources and season ranges the pipeline downloads (participation needs S-1 and S-2 for route ratios).
INGEST_PLAN: dict[str, int] = {
    "schedules": 0, "players": 0,
    "pbp": 2018, "snap_counts": 2018, "stats_player": 2018, "participation": 2018,
    "ftn": 2022, "rosters": 2018, "injuries": 2018, "depth": 2022,
    "pfr_pass": 2018, "pfr_rush": 2018, "pfr_rec": 2018, "pfr_def": 2018,
}
