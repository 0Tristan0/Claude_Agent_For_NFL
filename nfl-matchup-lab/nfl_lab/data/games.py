"""Games table from the nflverse schedule.

`games` holds only what is known before kickoff (an allowlist). Outcomes (scores, result, total,
overtime) go to `game_results`, which only track/ and backtest/ scoring may read. temp, wind, QB
names, coaches and referee are dropped entirely: they are filled in after games.

Lines in the schedule file are one snapshot that is overwritten until kickoff, so for completed
games they are closing lines. Live runs snapshot them at freeze time (data/game_lines.csv).
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import teams

ET = ZoneInfo("America/New_York")
POST_TYPES = {"WC", "DIV", "CON", "SB"}

PREGAME_COLUMNS = [
    "game_id", "season", "week", "game_type", "season_type", "kickoff_utc", "home_team", "away_team",
    "location", "div_game", "home_rest", "away_rest", "stadium_id", "stadium", "roof", "roof_pregame",
    "surface", "spread_line", "total_line", "home_moneyline", "away_moneyline", "home_spread_odds",
    "away_spread_odds", "over_odds", "under_odds",
]
RESULT_COLUMNS = ["game_id", "home_score", "away_score", "result", "total", "overtime"]
# Columns that must never reach features, projections, sims or picks.
BLOCKED_COLUMNS = {"home_score", "away_score", "result", "total", "overtime", "temp", "wind",
                   "home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id", "home_coach",
                   "away_coach", "referee"}


def to_utc(s: pd.Series) -> pd.Series:
    """Normalize any datetime-like series to datetime64[us, UTC] (pandas 3 defaults vary by source)."""
    out = pd.to_datetime(s, utc=True)
    return out.astype("datetime64[us, UTC]")


def kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    local = pd.to_datetime(gameday.astype(str) + " " + gametime.fillna("13:00").astype(str),
                           format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").astype(
        "datetime64[us, UTC]")


def build_games(schedules: pd.DataFrame, first_season: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = teams.normalize(schedules, ["home_team", "away_team"])
    if first_season is not None:
        g = g[g["season"] >= first_season]
    g = g.copy()
    g["season_type"] = np.where(g["game_type"].isin(POST_TYPES), "POST", "REG")
    g["kickoff_utc"] = kickoff_utc(g["gameday"], g["gametime"])
    # Only a fixed dome is known well before kickoff; retractable open/closed is a game-day call.
    g["roof_pregame"] = np.where(g["roof"].eq("dome"), "dome",
                                 np.where(g["roof"].eq("outdoors"), "outdoors",
                                          np.where(g["roof"].isin(["open", "closed"]), "retractable", "unknown")))
    games = g[[c for c in PREGAME_COLUMNS if c in g.columns]].sort_values(["kickoff_utc", "game_id"],
                                                                         ignore_index=True)
    results = g[[c for c in RESULT_COLUMNS if c in g.columns]].copy()
    results["completed"] = results["home_score"].notna() & results["away_score"].notna()
    return games, results.reset_index(drop=True)


def team_view(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) with opponent, home flag, rest and the market-implied points.

    nflverse spread_line is positive when the home team is favored, so
    home implied = (total + spread)/2 and away implied = (total - spread)/2.
    """
    base = ["game_id", "season", "week", "season_type", "kickoff_utc", "stadium_id", "roof_pregame",
            "div_game", "spread_line", "total_line"]
    home = games[base].assign(team=games["home_team"], opponent=games["away_team"], is_home=1.0,
                              rest_days=games["home_rest"], opp_rest_days=games["away_rest"],
                              team_spread=games["spread_line"],
                              implied_points=(games["total_line"] + games["spread_line"]) / 2)
    away = games[base].assign(team=games["away_team"], opponent=games["home_team"], is_home=0.0,
                              rest_days=games["away_rest"], opp_rest_days=games["home_rest"],
                              team_spread=-games["spread_line"],
                              implied_points=(games["total_line"] - games["spread_line"]) / 2)
    tv = pd.concat([home, away], ignore_index=True)
    neutral = games.set_index("game_id")["location"].eq("Neutral")
    tv.loc[tv["game_id"].map(neutral).fillna(False).astype(bool), "is_home"] = 0.5
    return tv.sort_values(["kickoff_utc", "game_id", "team"], ignore_index=True)
