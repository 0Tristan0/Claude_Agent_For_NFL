"""Data checks for the Gate 1 report: pbp-derived counts vs official stats, and route-proxy quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import games as G, ingest, pbp, player_games, routes

PAIRS = [("targets", "targets"), ("receptions", "receptions"), ("rec_yds", "receiving_yards"),
         ("carries", "carries"), ("rush_yds", "rushing_yards"), ("pass_att", "attempts"),
         ("completions", "completions"), ("pass_yds", "passing_yards"), ("pass_tds", "passing_tds"),
         ("ints", "passing_interceptions"), ("rush_tds", "rushing_tds"), ("rec_tds", "receiving_tds")]


def official_match(pg: pd.DataFrame, seasons) -> pd.DataFrame:
    """Share of player-games where the pbp count equals the official stat exactly (QB/RB/WR/TE)."""
    st = ingest.load("stats_player", seasons=seasons)
    st = st[st["position"].isin(["QB", "RB", "FB", "WR", "TE"])]
    cols = ["game_id", "player_id"] + [o for _, o in PAIRS]
    m = pg[pg["season"].isin(seasons)].merge(st[cols], on=["game_id", "player_id"], how="inner",
                                             suffixes=("", "_official"))
    rows = []
    for season, g in m.groupby("season"):
        r = {"season": season, "player_games": len(g)}
        for a, o in PAIRS:
            oc = o + "_official" if o in pg.columns else o
            r[a] = float(((g[a].fillna(0) - g[oc].fillna(0)).abs() < 0.5).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def route_proxy_quality(test_season: int) -> pd.DataFrame:
    """Estimated routes (ratios from seasons before test_season) vs actual dropbacks on the field."""
    seasons = [test_season - 2, test_season - 1, test_season]
    p = pbp.prepare(ingest.load("pbp", seasons=seasons, columns=pbp.COLUMNS))
    games, _ = G.build_games(ingest.load("schedules"), min(seasons))
    tv = G.team_view(games)
    pg = player_games.build(p, ingest.load("snap_counts", seasons=seasons), ingest.load("players"),
                            ingest.load("rosters", seasons=seasons), tv)
    part = ingest.load("participation", seasons=seasons, columns=["nflverse_game_id", "play_id", "offense_players"])
    dbp = routes.dropback_participation(part, p)
    rr = routes.ratios(pg[pg["season"] < test_season], dbp[dbp["game_id"].str[:4].astype(int) < test_season],
                       [test_season])
    est = routes.attach_routes(pg[pg["season"] == test_season], rr)
    x = est.merge(dbp, on=["game_id", "player_id"], how="inner")
    x = x[x["position"].isin(["WR", "TE", "RB"]) & (x["offense_snaps"] > 0)]
    rows = []
    for pos, g in x.groupby("position"):
        rows.append(dict(position=pos, player_games=len(g),
                         corr=float(np.corrcoef(g["routes_est"], g["db_on_field"])[0, 1]),
                         mae=float((g["routes_est"] - g["db_on_field"]).abs().mean()),
                         mean_actual=float(g["db_on_field"].mean()), mean_est=float(g["routes_est"].mean()),
                         share_with_player_ratio=float((g["route_ratio_source"] == "player").mean())))
    return pd.DataFrame(rows)
