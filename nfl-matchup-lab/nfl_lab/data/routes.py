"""Estimated route participation.

No nflverse file gives routes per player during a season, and play-level participation is published
only after the postseason. So:

1. From participation seasons we can see (S-1 and S-2), measure each player's dropback participation
   (share of team dropbacks he was on the field for) and divide it by his snap share. That ratio says
   how much of his playing time comes on pass plays.
2. Shrink the ratio toward the position mean with ~300 pseudo-dropbacks.
3. In season S, estimated routes in a game = min(1, snap share x ratio) x team dropbacks.

It counts pass-blocking snaps as routes, so TE and RB numbers run high. Labelled "est." everywhere.
Rule: participation from season S never feeds a season-S estimate (it was not public yet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_PSEUDO_DROPBACKS = 300.0
PRIOR_SEASON_WEIGHT = 0.5      # S-2 counts half as much as S-1
LOOKBACK = (1, 2)              # seasons back used for a season-S estimate (never 0: tests plant 0)


def dropback_participation(participation: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Per player-game count of team dropbacks the player was on the field for."""
    if participation.empty:
        return pd.DataFrame(columns=["game_id", "player_id", "db_on_field"])
    part = participation.rename(columns={"nflverse_game_id": "game_id"})[["game_id", "play_id", "offense_players"]]
    db = p.loc[p["dropback"], ["game_id", "play_id"]]
    x = part.merge(db, on=["game_id", "play_id"], how="inner")
    x = x[x["offense_players"].notna() & (x["offense_players"].astype(str).str.len() > 0)]
    ids = x["offense_players"].astype(str).str.split(";")
    long = pd.DataFrame({"game_id": np.repeat(x["game_id"].to_numpy(), ids.str.len().to_numpy()),
                         "player_id": np.concatenate(ids.to_numpy()) if len(ids) else []})
    return long.groupby(["game_id", "player_id"]).size().rename("db_on_field").reset_index()


def ratios(pg: pd.DataFrame, dbp: pd.DataFrame, valid_seasons) -> pd.DataFrame:
    """Route ratio per (player_id, valid_season) using participation seasons < valid_season only."""
    x = pg.merge(dbp, on=["game_id", "player_id"], how="inner")
    x = x[(x["offense_snaps"] > 0) & (x["team_dropbacks"] > 0) & (x["team_offense_snaps"] > 0)]
    per = x.groupby(["player_id", "season"]).agg(
        position=("position", "last"), db_on=("db_on_field", "sum"), team_db=("team_dropbacks", "sum"),
        snaps=("offense_snaps", "sum"), team_snaps=("team_offense_snaps", "sum")).reset_index()
    seasons_avail = set(per["season"])
    rows = []
    for S in valid_seasons:
        prev = [S - k for k in LOOKBACK if S - k in seasons_avail]
        if not prev:
            continue
        w = per[per["season"].isin(prev)].copy()
        w["w"] = np.where(w["season"] == max(prev), 1.0, PRIOR_SEASON_WEIGHT)
        for c in ("db_on", "team_db", "snaps", "team_snaps"):
            w[c] = w[c] * w["w"]
        a = w.groupby("player_id").agg(position=("position", "last"), db_on=("db_on", "sum"),
                                       team_db=("team_db", "sum"), snaps=("snaps", "sum"),
                                       team_snaps=("team_snaps", "sum"))
        dbp_rate = a["db_on"] / a["team_db"]
        snap_rate = a["snaps"] / a["team_snaps"]
        raw = dbp_rate / snap_rate
        pos_mean = (a["db_on"].groupby(a["position"]).sum() / a["team_db"].groupby(a["position"]).sum()) / (
            a["snaps"].groupby(a["position"]).sum() / a["team_snaps"].groupby(a["position"]).sum())
        n_eff = snap_rate * a["team_db"]                 # dropbacks he'd be on for at ratio 1
        prior = a["position"].map(pos_mean)
        shrunk = (n_eff * raw + K_PSEUDO_DROPBACKS * prior) / (n_eff + K_PSEUDO_DROPBACKS)
        rows.append(pd.DataFrame({"player_id": a.index, "valid_season": S, "position": a["position"].to_numpy(),
                                  "route_ratio": shrunk.to_numpy(), "route_ratio_raw": raw.to_numpy(),
                                  "ratio_n_eff": n_eff.to_numpy()}))
        for pos, v in pos_mean.items():
            rows.append(pd.DataFrame({"player_id": [f"__POS__{pos}"], "valid_season": [S], "position": [pos],
                                      "route_ratio": [v], "route_ratio_raw": [v], "ratio_n_eff": [np.nan]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["player_id", "valid_season", "position", "route_ratio", "route_ratio_raw", "ratio_n_eff"])


def attach_routes(pg: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    """Add route_ratio and routes_est to player-games (QB ratio is not meaningful and set to 0)."""
    pg = pg.copy()
    key = rr[~rr["player_id"].str.startswith("__POS__")][["player_id", "valid_season", "route_ratio"]]
    pg = pg.merge(key.rename(columns={"valid_season": "season"}), on=["player_id", "season"], how="left")
    pos = rr[rr["player_id"].str.startswith("__POS__")].set_index(["valid_season", "position"])["route_ratio"]
    fill = pd.Series([pos.get((s, p), np.nan) for s, p in zip(pg["season"], pg["position"])], index=pg.index)
    pg["route_ratio_source"] = np.where(pg["route_ratio"].notna(), "player", "position")
    pg["route_ratio"] = pg["route_ratio"].fillna(fill)
    pg.loc[pg["position"] == "QB", "route_ratio"] = 0.0
    part = np.minimum(1.0, pg["snap_share"].fillna(0) * pg["route_ratio"].fillna(0))
    pg["routes_est"] = part * pg["team_dropbacks"].fillna(0)
    return pg
