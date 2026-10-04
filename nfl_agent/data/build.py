"""Build the modelling tables from raw source tables.

Outputs
-------
games         one row per game_id, with kickoff in UTC and pregame-known context
team_games    one row per (game_id, team): team passing volume and what the
              opponent defence allowed to WR/TE
player_games  one row per (player_id, game_id) for WR/TE who *participated*
              (took >= 1 offensive snap, or recorded a receiving stat)

Participation spine
-------------------
nflverse weekly player stats omit players who played but recorded no stat.
Using stats rows alone would silently drop zero-target games and bias every
forecast upward. We therefore build the spine from PFR snap counts, mapped to
GSIS ids, and union stats rows. A player-game present in snap counts with
offensive snaps but absent from stats gets 0 targets / 0 receptions / 0 yards
and ``stats_zero_filled = True`` -- a recorded fact (he played, no target was
logged), not an imputation. If snap counts for a game are not yet published,
nothing is zero-filled for that game.

Players absent from both sources did not participate (inactive, healthy
scratch, or no offensive snap). Forecasts are conditional on participation.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import config

ET = ZoneInfo(config.KICKOFF_TZ)
POST_TYPES = {"WC", "DIV", "CON", "SB"}

# Relocated franchises: player stats use the current franchise code for every
# season, schedules and snap counts use the historical code. We normalise all
# team columns to the franchise code (game_id strings are left untouched; they
# are the stable game identifier).
FRANCHISE_CODES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}


def normalize_teams(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(FRANCHISE_CODES)
    return df


def build_games(schedules: pd.DataFrame) -> pd.DataFrame:
    g = schedules.copy()
    g["season_type"] = np.where(g["game_type"].isin(POST_TYPES), "POST", "REG")
    gametime = g["gametime"].fillna("13:00")
    local = pd.to_datetime(g["gameday"] + " " + gametime, format="%Y-%m-%d %H:%M", errors="coerce")
    g["kickoff_utc"] = local.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    # Only a fixed dome is known well in advance; retractable open/closed is a game-day call.
    g["fixed_dome"] = (g["roof"] == "dome").astype(float)
    g.loc[g["roof"].isna(), "fixed_dome"] = np.nan
    keep = ["game_id", "season", "week", "game_type", "season_type", "gameday", "gametime",
            "kickoff_utc", "away_team", "home_team", "away_score", "home_score", "completed",
            "roof", "fixed_dome", "away_rest", "home_rest", "location", "stadium",
            "temp", "wind", "away_qb_name", "home_qb_name"]
    return g[[c for c in keep if c in g.columns]].sort_values("kickoff_utc", ignore_index=True)


def team_view(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team, with opponent, home flag and rest."""
    home = games.assign(team=games["home_team"], opponent=games["away_team"], is_home=1.0,
                        rest_days=games["home_rest"], opp_rest_days=games["away_rest"],
                        listed_qb=games.get("home_qb_name"))
    away = games.assign(team=games["away_team"], opponent=games["home_team"], is_home=0.0,
                        rest_days=games["away_rest"], opp_rest_days=games["home_rest"],
                        listed_qb=games.get("away_qb_name"))
    tv = pd.concat([home, away], ignore_index=True)
    tv.loc[tv["location"].eq("Neutral"), "is_home"] = 0.5
    cols = ["game_id", "season", "week", "season_type", "kickoff_utc", "completed", "team",
            "opponent", "is_home", "rest_days", "opp_rest_days", "fixed_dome", "listed_qb"]
    return tv[cols].sort_values(["kickoff_utc", "team"], ignore_index=True)


def pfr_to_gsis(players: pd.DataFrame) -> dict[str, str]:
    m = players.dropna(subset=["pfr_id", "gsis_id"])
    m = m[~m["pfr_id"].duplicated(keep=False)]
    return dict(zip(m["pfr_id"], m["gsis_id"]))


def build_team_games(stats: pd.DataFrame, snaps: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    s = stats
    team = s.groupby(["game_id", "team"], as_index=False).agg(
        team_targets=("targets", "sum"),
        team_pass_attempts=("attempts", "sum"),
        team_sacks=("sacks_suffered", "sum"),
        team_air_yards=("receiving_air_yards", "sum"),
        team_rec_yards=("receiving_yards", "sum"),
    )
    team["team_dropbacks"] = team["team_pass_attempts"] + team["team_sacks"]
    wrte = s[s["position"].isin(config.POSITIONS)]
    wr = wrte.groupby(["game_id", "team"], as_index=False).agg(
        wrte_targets=("targets", "sum"), wrte_receptions=("receptions", "sum"),
        wrte_rec_yards=("receiving_yards", "sum"))
    team = team.merge(wr, on=["game_id", "team"], how="left")
    if not snaps.empty:
        plays = snaps.groupby(["game_id", "team"], as_index=False)["offense_snaps"].max()
        plays = plays.rename(columns={"offense_snaps": "team_offense_plays"})
        team = team.merge(plays, on=["game_id", "team"], how="left")
    else:
        team["team_offense_plays"] = np.nan
    out = tv.merge(team, on=["game_id", "team"], how="left")
    # Opponent defence view: what this team's *opponent* allowed in this game
    # is the offence numbers of this team. We store, per (game, defense team),
    # what it allowed.
    allowed = team.rename(columns={
        "team": "opponent", "wrte_targets": "def_wrte_targets_allowed",
        "wrte_receptions": "def_wrte_receptions_allowed", "wrte_rec_yards": "def_wrte_yards_allowed",
        "team_dropbacks": "def_dropbacks_faced"})[["game_id", "opponent", "def_wrte_targets_allowed",
                                                  "def_wrte_receptions_allowed", "def_wrte_yards_allowed",
                                                  "def_dropbacks_faced"]]
    # 'opponent' here is the offence; the defence is the row's team.
    allowed = allowed.rename(columns={"opponent": "offense_team"})
    out = out.merge(allowed, left_on=["game_id", "opponent"], right_on=["game_id", "offense_team"], how="left")
    out = out.drop(columns=["offense_team"])
    # Team-game rows exist for completed games only when stats exist.
    out["has_stats"] = out["team_targets"].notna()
    return out


def build_player_games(stats: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame,
                       games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    xwalk = pfr_to_gsis(players)
    sn = snaps.copy()
    sn["player_id"] = sn["pfr_player_id"].map(xwalk)
    sn = sn[sn["offense_snaps"].fillna(0) > 0]
    snap_games = set(snaps["game_id"].unique())

    st = stats[stats["position"].isin(config.POSITIONS) | stats["player_id"].isin(
        sn.loc[sn["position"].isin(config.POSITIONS), "player_id"].dropna())].copy()
    st_cols = ["player_id", "game_id", "player_display_name", "position", "team", "opponent_team",
               "targets", "receptions", "receiving_yards", "receiving_air_yards",
               "receiving_yards_after_catch", "receiving_tds", "receiving_20", "receiving_40"]
    st = st[[c for c in st_cols if c in st.columns]]

    sn_wr = sn[sn["position"].isin(config.POSITIONS) & sn["player_id"].notna()]
    sn_wr = sn_wr[["player_id", "game_id", "player", "position", "team", "opponent",
                   "offense_snaps", "offense_pct"]].rename(
        columns={"player": "snap_name", "position": "snap_position", "team": "snap_team",
                 "opponent": "snap_opponent"})
    # Snap rows for stat-row players listed at another position (e.g. a RB/WR hybrid)
    sn_other = sn[sn["player_id"].isin(st["player_id"]) & ~sn.index.isin(sn_wr.index)]
    sn_other = sn_other[["player_id", "game_id", "offense_snaps", "offense_pct"]]

    pg = st.merge(sn_wr, on=["player_id", "game_id"], how="outer")
    pg = pg.merge(sn_other, on=["player_id", "game_id"], how="left", suffixes=("", "_alt"))
    for c in ("offense_snaps", "offense_pct"):
        pg[c] = pg[c].fillna(pg.pop(c + "_alt"))

    pg["player_name"] = pg["player_display_name"].fillna(pg["snap_name"])
    pg["position"] = pg["position"].fillna(pg["snap_position"])
    pg["team"] = pg["team"].fillna(pg["snap_team"])
    pg["opponent"] = pg["opponent_team"].fillna(pg["snap_opponent"])
    pg = pg[pg["position"].isin(config.POSITIONS)]

    has_stats_row = pg["targets"].notna()
    in_snap_game = pg["game_id"].isin(snap_games)
    zero_fill = ~has_stats_row & in_snap_game & (pg["offense_snaps"].fillna(0) > 0)
    for c in ("targets", "receptions", "receiving_yards", "receiving_air_yards",
              "receiving_yards_after_catch", "receiving_tds", "receiving_20", "receiving_40"):
        if c in pg.columns:
            pg.loc[zero_fill, c] = 0.0
    pg["stats_zero_filled"] = zero_fill
    pg["snaps_missing"] = pg["offense_snaps"].isna()
    # Source quirk: a handful of rows with receptions > targets. Keep yards, raise targets.
    pg["targets"] = np.where(pg["receptions"] > pg["targets"], pg["receptions"], pg["targets"])

    pg = pg.merge(games[["game_id", "season", "week", "season_type", "game_type", "kickoff_utc"]],
                  on="game_id", how="left")
    tg = team_games[["game_id", "team", "team_targets", "team_air_yards", "team_dropbacks",
                     "team_offense_plays", "is_home", "rest_days", "fixed_dome"]]
    pg = pg.merge(tg, on=["game_id", "team"], how="left")
    pg["target_share"] = np.where(pg["team_targets"] > 0, pg["targets"] / pg["team_targets"], np.nan)
    pg["air_yards_share"] = np.where(pg["team_air_yards"].abs() > 0,
                                     pg["receiving_air_yards"] / pg["team_air_yards"], np.nan)
    pg["snap_share"] = pg["offense_pct"]

    dp = players[["gsis_id", "draft_pick", "draft_round", "rookie_season", "birth_date"]].rename(
        columns={"gsis_id": "player_id"})
    pg = pg.merge(dp, on="player_id", how="left")
    pg["draft_pick_filled"] = pg["draft_pick"].fillna(260.0)   # undrafted ~ after last pick
    pg["is_rookie"] = (pg["season"] == pg["rookie_season"]).astype(float)
    pg.loc[pg["rookie_season"].isna(), "is_rookie"] = np.nan

    keep = ["player_id", "player_name", "position", "game_id", "season", "week", "season_type",
            "game_type", "kickoff_utc", "team", "opponent", "targets", "receptions",
            "receiving_yards", "receiving_air_yards", "receiving_yards_after_catch", "receiving_tds",
            "receiving_20", "receiving_40", "offense_snaps", "snap_share", "target_share",
            "air_yards_share", "team_targets", "team_dropbacks", "team_offense_plays",
            "is_home", "rest_days", "fixed_dome", "draft_pick", "draft_pick_filled", "draft_round",
            "rookie_season", "is_rookie", "stats_zero_filled", "snaps_missing"]
    pg = pg[keep].sort_values(["kickoff_utc", "player_id"], ignore_index=True)
    assert not pg.duplicated(["player_id", "game_id"]).any(), "duplicate player-game rows"
    return pg


def build_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    raw = dict(raw)
    raw["schedules"] = normalize_teams(raw["schedules"], ["home_team", "away_team"])
    raw["player_stats"] = normalize_teams(raw["player_stats"], ["team", "opponent_team"])
    raw["snap_counts"] = normalize_teams(raw["snap_counts"], ["team", "opponent"])
    first = int(raw["player_stats"]["season"].min())
    games = build_games(raw["schedules"][raw["schedules"]["season"] >= first])
    tv = team_view(games)
    team_games = build_team_games(raw["player_stats"], raw["snap_counts"], tv)
    player_games = build_player_games(raw["player_stats"], raw["snap_counts"], raw["players"],
                                      games, team_games)
    return {"games": games, "team_games": team_games, "player_games": player_games}
