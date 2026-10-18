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
    """Exact-match rate of pbp-derived counts vs nflverse weekly stats, per season.

    Rows are taken from the official side (QB/RB/FB/WR/TE player-games with any target, carry or pass
    attempt) and left-joined to player_games, so a player-game missing from our table counts as a miss.
    Note: nflverse builds these stats from the same play-by-play, so this checks our filters and joins;
    see pfr_carry_check for an independent source."""
    st = ingest.load("stats_player", seasons=seasons)
    st = st[st["position"].isin(["QB", "RB", "FB", "WR", "TE"])]
    st = st[(st["targets"].fillna(0) > 0) | (st["carries"].fillna(0) > 0) | (st["attempts"].fillna(0) > 0)]
    cols = ["game_id", "player_id", "season"] + [o for _, o in PAIRS]
    ours = pg[["game_id", "player_id"] + [a for a, _ in PAIRS]].rename(columns={a: f"{a}__ours" for a, _ in PAIRS})
    m = st[cols].merge(ours, on=["game_id", "player_id"], how="left")
    rows = []
    for season, g in m.groupby("season"):
        r = {"season": season, "official_player_games": len(g), "missing_from_ours": int(g["targets__ours"].isna().sum())}
        for a, o in PAIRS:
            r[a] = float(((g[f"{a}__ours"] - g[o].fillna(0)).abs() < 0.5).mean())   # NaN (missing) counts as a miss
        rows.append(r)
    return pd.DataFrame(rows)


def pfr_carry_check(pg: pd.DataFrame, seasons) -> pd.DataFrame:
    """Independent source: PFR rushing carries vs our pbp carries for non-QBs (exact-match rate)."""
    from .player_games import pfr_to_gsis
    pr = ingest.load("pfr_rush", seasons=seasons)
    xw = pfr_to_gsis(ingest.load("players"), ingest.load("rosters", seasons=seasons))
    pr = pr.assign(player_id=pr["pfr_player_id"].map(xw)).dropna(subset=["player_id"])
    pr = pr.groupby(["game_id", "player_id"], as_index=False)["carries"].sum()
    x = pg[(pg["position"] != "QB")][["game_id", "player_id", "season", "carries"]].merge(
        pr, on=["game_id", "player_id"], how="inner", suffixes=("", "_pfr"))
    x = x[x["carries_pfr"] > 0]
    return (x.assign(match=(x["carries"] - x["carries_pfr"]).abs() < 0.5, within1=(x["carries"] - x["carries_pfr"]).abs() <= 1)
            .groupby("season").agg(player_games=("match", "size"), exact=("match", "mean"), within_one=("within1", "mean"))
            .reset_index())


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
