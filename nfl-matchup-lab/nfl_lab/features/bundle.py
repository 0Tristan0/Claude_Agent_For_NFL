"""All pregame features for one game, computed at `asof` (default: kickoff).

This is the function the leak tests run on full and on truncated data: everything it returns must be
identical whether or not rows that were unavailable at `asof` exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import lines as L
from ..data.games import BLOCKED_COLUMNS, LINE_COLUMNS
from . import opponent
from .asof import visible
from .player import PlayerFeatures
from .team import TeamFeatures


@dataclass
class Engines:
    player: PlayerFeatures
    team: TeamFeatures


def engines(tables: dict[str, pd.DataFrame], player_half_life: float = 4.0, team_half_life: float = 8.0,
            season_discount: float = 0.5) -> Engines:
    return Engines(
        PlayerFeatures(tables["player_games"], tables.get("player_pfr_rush"), player_half_life, season_discount),
        TeamFeatures(tables["team_games"], tables.get("team_pressure"), team_half_life, season_discount),
    )


def candidates(player_games: pd.DataFrame, team: str, asof: pd.Timestamp, season: int) -> pd.DataFrame:
    """Players who appeared for `team` this season (or last season if none yet) before `asof`."""
    pg = visible(player_games, asof)
    pg = pg[(pg["team"] == team) & (pg["any_snaps"] > 0)]
    cur = pg[pg["season"] == season]
    use = cur if len(cur) else pg[pg["season"] == season - 1]
    last = use.sort_values("available_at").groupby("player_id").tail(1)
    return last[["player_id", "player_name", "position", "team"]].reset_index(drop=True)


def game_features(tables: dict[str, pd.DataFrame], game_id: str, asof: pd.Timestamp | None = None,
                  eng: Engines | None = None) -> dict[str, pd.DataFrame]:
    games = tables["games"]
    g = games[games["game_id"] == game_id]
    if g.empty:
        raise KeyError(f"unknown game {game_id}")
    g = g.iloc[0]
    asof = pd.Timestamp(asof) if asof is not None else g["kickoff_utc"]
    asof = min(asof, g["kickoff_utc"])
    season = int(g["season"])
    eng = eng or engines(tables)
    sides = [g["home_team"], g["away_team"]]
    tv = tables["team_view"]
    view = tv[tv["game_id"] == game_id].drop(columns=[c for c in BLOCKED_COLUMNS if c in tv.columns]).copy()
    game_row = g.drop(labels=[c for c in BLOCKED_COLUMNS if c in g.index]).copy()
    line_source = "schedule (closing)"
    if asof < g["kickoff_utc"]:
        # Closing lines are not known yet: use the latest manually captured line before asof, else nothing.
        snap = L.latest_game_line(tables.get("game_lines", pd.DataFrame()), game_id, asof)
        for c in LINE_COLUMNS:
            if c in game_row.index:
                game_row[c] = snap[c] if snap is not None and c in snap.index else float("nan")
        line_source = f"captured {snap['captured_at']}" if snap is not None else "none before asof"
        spread, total = game_row.get("spread_line", float("nan")), game_row.get("total_line", float("nan"))
        home = view["team"] == g["home_team"]
        view["spread_line"], view["total_line"] = spread, total
        view["team_spread"] = [spread if h else -spread for h in home]
        view["implied_points"] = [(total + spread) / 2 if h else (total - spread) / 2 for h in home]
    game_row["line_source"] = line_source
    ctx = tables["context"][tables["context"]["game_id"] == game_id]
    team_f = eng.team.features(sides, asof, season)
    adj = opponent.ratings(tables["team_games"], asof, season)
    adj = adj[adj["team"].isin(sides)] if adj is not None else pd.DataFrame()
    cands = pd.concat([candidates(tables["player_games"], t, asof, season) for t in sides], ignore_index=True)
    pf = eng.player.features(cands["player_id"].tolist(), asof, season)
    players = cands.merge(pf, on="player_id", how="left")
    return {"game": game_row.to_frame().T.reset_index(drop=True),
            "team_view": view.reset_index(drop=True), "context": ctx.reset_index(drop=True),
            "team": team_f, "opponent_adjusted": adj.reset_index(drop=True), "players": players,
            "asof": pd.DataFrame({"asof": [asof]})}
