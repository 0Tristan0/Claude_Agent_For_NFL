"""Slim play-by-play with derived flags, and a drives table.

How nflfastR codes the plays we care about (checked on 2024 data):
  scramble   play_type 'run', qb_scramble 1, qb_dropback 1, rush_attempt 1 (counts as an official rush)
  kneel      play_type 'qb_kneel', rush_attempt 1, about -1 yard (counts as an official rush)
  spike      play_type 'qb_spike', pass_attempt 1 (counts as an official incomplete pass)
  sack       play_type 'pass', sack 1, pass_attempt 1 (NOT an official pass attempt)
  no_play    penalties that wipe out the play; never counted
  two-point  two_point_attempt 1; excluded from every stat
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, teams

COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining",
    "fixed_drive", "fixed_drive_result", "play_type", "pass", "rush", "qb_dropback", "qb_scramble",
    "qb_kneel", "qb_spike", "pass_attempt", "rush_attempt", "sack", "qb_hit", "interception", "fumble_lost",
    "complete_pass", "air_yards", "yards_after_catch", "passing_yards", "rushing_yards", "receiving_yards",
    "yards_gained", "receiver_player_id", "rusher_player_id", "passer_player_id", "td_player_id", "td_team",
    "touchdown", "pass_touchdown", "rush_touchdown", "epa", "success", "wp", "xpass", "pass_oe",
    "posteam_score", "defteam_score", "posteam_score_post", "score_differential", "two_point_attempt",
    "field_goal_attempt", "field_goal_result", "punt_attempt", "extra_point_attempt", "drive_inside20",
]


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(0).to_numpy(dtype=float) if col in df.columns else np.zeros(len(df))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the needed columns and add flags. Input: raw nflverse pbp (one or more seasons)."""
    raw = raw.reindex(columns=list(dict.fromkeys(list(raw.columns) + COLUMNS)))
    p = raw[COLUMNS].sort_values(["game_id", "play_id"], kind="mergesort")
    p = p.drop_duplicates(["game_id", "play_id"]).reset_index(drop=True)
    p = teams.normalize(p, ["posteam", "defteam", "home_team", "away_team", "td_team"])
    m = config.model()
    pt = p["play_type"].astype("object")
    two_pt = _num(p, "two_point_attempt") == 1
    p["kneel"] = (pt == "qb_kneel").to_numpy()
    p["spike"] = (pt == "qb_spike").to_numpy()
    p["scramble"] = ((pt == "run") & (_num(p, "qb_scramble") == 1)).to_numpy()
    p["designed_run"] = ((pt == "run") & (_num(p, "qb_scramble") == 0)).to_numpy() & ~two_pt
    p["scrimmage"] = pt.isin(["pass", "run", "qb_kneel", "qb_spike"]).to_numpy() & ~two_pt
    p["dropback"] = p["scrimmage"] & (_num(p, "qb_dropback") == 1) & ~p["spike"]
    p["is_sack"] = p["scrimmage"] & (_num(p, "sack") == 1)
    p["pass_att"] = p["scrimmage"] & (_num(p, "pass_attempt") == 1) & ~p["is_sack"]
    p["target"] = p["pass_att"] & p["receiver_player_id"].notna() & ~p["spike"]
    p["rush_att"] = p["scrimmage"] & (_num(p, "rush_attempt") == 1) & p["rusher_player_id"].notna()
    wp = p["wp"].astype(float)
    p["neutral"] = (p["scrimmage"] & ~p["kneel"] & ~p["spike"] & wp.between(m["neutral"]["wp_low"],
                                                                            m["neutral"]["wp_high"])).to_numpy()
    yl = p["yardline_100"].astype(float)
    p["rz"] = (yl <= 20).to_numpy() & p["scrimmage"]
    p["i10"] = (yl <= 10).to_numpy() & p["scrimmage"]
    yg = _num(p, "yards_gained")
    ex = m["explosive"]
    p["explosive"] = (p["dropback"] & (yg >= ex["pass_yards"])) | (p["designed_run"] & (yg >= ex["rush_yards"]))
    g = m["garbage"]
    p["garbage"] = (_num(p, "qtr") >= g["quarter"]) & (np.abs(_num(p, "score_differential")) >= g["score_diff"])
    p["giveaway"] = p["scrimmage"] & ((_num(p, "interception") == 1) | (_num(p, "fumble_lost") == 1))
    p["season"] = p["season"].astype(int)
    return p


def pace_seconds(p: pd.DataFrame) -> pd.Series:
    """Seconds from each neutral snap to the team's next snap (any snap, penalties included) on the same drive.

    Snaps with 2 minutes or less left in the half, at either end, are excluded (clock rules change), and gaps
    over pace_max_gap_seconds are dropped as timeouts, reviews or injuries rather than clipped."""
    m = config.model()["neutral"]
    snaps = p[p["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike", "no_play"]) &
              p["posteam"].notna()].sort_values(["game_id", "play_id"])
    nxt = snaps.groupby(["game_id", "fixed_drive", "posteam"])[["game_seconds_remaining", "qtr",
                                                               "half_seconds_remaining"]].shift(-1)
    gap = snaps["game_seconds_remaining"] - nxt["game_seconds_remaining"]
    lim = m["pace_min_half_seconds"]
    ok = (snaps["neutral"] & (snaps["half_seconds_remaining"] > lim) & (nxt["half_seconds_remaining"] > lim)
          & nxt["qtr"].eq(snaps["qtr"]) & gap.gt(0) & gap.le(m["pace_max_gap_seconds"]))
    return gap.where(ok).reindex(p.index)


def drives(p: pd.DataFrame) -> pd.DataFrame:
    """One row per possession. A red-zone trip is any snap (field goal tries included) at the 20 or closer;
    possessions with no offensive snap at all (return TDs, kneel-only end of half) are flagged has_snap = False."""
    s = p[p["posteam"].notna() & p["fixed_drive"].notna()].sort_values(["game_id", "play_id"])
    key = ["game_id", "fixed_drive", "posteam"]
    two_pt = s["two_point_attempt"].fillna(0).eq(1)
    snap = (s["scrimmage"] | s["play_type"].eq("field_goal")) & ~two_pt
    rz_snap = snap & (s["yardline_100"] <= 20)
    s = s.assign(_snap=snap, _rz_snap=rz_snap)
    first = s.groupby(key).head(1).set_index(key)
    agg = s.groupby(key).agg(
        n_plays=("scrimmage", "sum"), has_snap=("_snap", "max"), result=("fixed_drive_result", "first"),
        reached_rz=("_rz_snap", "max"), score_pre=("posteam_score", "min"), score_post=("posteam_score_post", "max"))
    agg["start_yardline"] = first["yardline_100"]
    agg["start_wp"] = first["wp"]
    agg["rz_entry_wp"] = s[s["_rz_snap"]].groupby(key)["wp"].first()
    agg["points"] = (agg["score_post"] - agg["score_pre"]).clip(lower=0)
    agg["td"] = agg["result"].eq("Touchdown")
    return agg.reset_index()
