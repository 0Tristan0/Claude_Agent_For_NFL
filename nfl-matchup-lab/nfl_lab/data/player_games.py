"""Player-game table: one row per (game_id, player_id) for QB, RB, FB, WR and TE who were active.

Spine (ported from nfl_agent/data/build.py @90f63d6, widened to QB/RB): PFR snap counts mapped to
GSIS ids, unioned with every player who appears in play-by-play. A player with snaps but no target
is a recorded zero, not a missing row; a player with no snaps in any phase did not play.

Counts come from play-by-play so that shares and team denominators use one consistent source.
Official stats (stats_player) are used only for grading and for the data check in Gate 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import teams

SKILL = {"QB", "RB", "FB", "WR", "TE"}


def pfr_to_gsis(players: pd.DataFrame) -> dict[str, str]:
    m = players.dropna(subset=["pfr_id", "gsis_id"])
    m = m[~m["pfr_id"].duplicated(keep=False)]
    return dict(zip(m["pfr_id"], m["gsis_id"]))


def _player_sums(p: pd.DataFrame) -> pd.DataFrame:
    s = p[p["scrimmage"]]
    t = s[s["target"]]
    rec = t.assign(
        targets=1.0, receptions=t["complete_pass"].fillna(0).astype(float),
        rec_yds=t["receiving_yards"].fillna(0).astype(float), air_yards=t["air_yards"].fillna(0).astype(float),
        yac=np.where(t["complete_pass"].fillna(0) == 1, t["yards_after_catch"].fillna(0), 0.0),
        rec_tds=((t["pass_touchdown"].fillna(0) == 1) & (t["td_player_id"] == t["receiver_player_id"])).astype(float),
        rz_targets=t["rz"].astype(float), i10_targets=t["i10"].astype(float),
        deep_targets=(t["air_yards"].fillna(0) >= 20).astype(float),
        garbage_targets=t["garbage"].astype(float),
    ).groupby(["game_id", "receiver_player_id"]).agg(
        team=("posteam", "first"), targets=("targets", "sum"), receptions=("receptions", "sum"),
        rec_yds=("rec_yds", "sum"), air_yards=("air_yards", "sum"), yac=("yac", "sum"),
        rec_tds=("rec_tds", "sum"), rz_targets=("rz_targets", "sum"), i10_targets=("i10_targets", "sum"),
        deep_targets=("deep_targets", "sum"), garbage_targets=("garbage_targets", "sum"))
    rec.index = rec.index.set_names(["game_id", "player_id"])

    r = s[s["rush_att"]]
    rush = r.assign(
        carries=1.0, designed_carries=r["designed_run"].astype(float), scrambles=r["scramble"].astype(float),
        kneels=r["kneel"].astype(float), rush_yds=r["rushing_yards"].fillna(0).astype(float),
        rush_tds=((r["rush_touchdown"].fillna(0) == 1) & (r["td_player_id"] == r["rusher_player_id"])).astype(float),
        rz_carries=(r["rz"] & r["designed_run"]).astype(float), i10_carries=(r["i10"] & r["designed_run"]).astype(float),
        garbage_carries=(r["garbage"] & r["designed_run"]).astype(float),
    ).groupby(["game_id", "rusher_player_id"]).agg(
        team_r=("posteam", "first"), carries=("carries", "sum"), designed_carries=("designed_carries", "sum"),
        scrambles=("scrambles", "sum"), kneels=("kneels", "sum"), rush_yds=("rush_yds", "sum"),
        rush_tds=("rush_tds", "sum"), rz_carries=("rz_carries", "sum"), i10_carries=("i10_carries", "sum"),
        garbage_carries=("garbage_carries", "sum"))
    rush.index = rush.index.set_names(["game_id", "player_id"])

    q = s[s["passer_player_id"].notna() & (s["dropback"] | s["spike"])]
    pas = q.assign(
        dropbacks=q["dropback"].astype(float), pass_att=q["pass_att"].astype(float),
        completions=(q["pass_att"] & (q["complete_pass"].fillna(0) == 1)).astype(float),
        pass_yds=np.where(q["pass_att"], q["passing_yards"].fillna(0), 0.0),
        pass_tds=(q["pass_att"] & (q["pass_touchdown"].fillna(0) == 1)).astype(float),
        ints=(q["pass_att"] & (q["interception"].fillna(0) == 1)).astype(float),
        sacks_taken=q["is_sack"].astype(float),
    ).groupby(["game_id", "passer_player_id"]).agg(
        team_q=("posteam", "first"), dropbacks=("dropbacks", "sum"), pass_att=("pass_att", "sum"),
        completions=("completions", "sum"), pass_yds=("pass_yds", "sum"), pass_tds=("pass_tds", "sum"),
        ints=("ints", "sum"), sacks_taken=("sacks_taken", "sum"))
    pas.index = pas.index.set_names(["game_id", "player_id"])

    out = rec.join(rush, how="outer").join(pas, how="outer")
    out["team"] = out["team"].fillna(out.pop("team_r")).fillna(out.pop("team_q"))
    return out.reset_index()


def _team_denoms(p: pd.DataFrame) -> pd.DataFrame:
    s = p[p["scrimmage"] & p["posteam"].notna()]
    d = s.assign(
        team_targets=s["target"].astype(float),
        team_air_yards=np.where(s["target"], s["air_yards"].fillna(0), 0.0),
        team_designed_carries=s["designed_run"].astype(float) * s["rusher_player_id"].notna(),
        team_rz_targets=(s["target"] & s["rz"]).astype(float),
        team_i10_carries=(s["designed_run"] & s["i10"]).astype(float) * s["rusher_player_id"].notna(),
        team_rz_carries=(s["designed_run"] & s["rz"]).astype(float) * s["rusher_player_id"].notna(),
        team_dropbacks=s["dropback"].astype(float),
        team_plays=(~s["kneel"] & ~s["spike"]).astype(float),
    )
    cols = ["team_targets", "team_air_yards", "team_designed_carries", "team_rz_targets", "team_i10_carries",
            "team_rz_carries", "team_dropbacks", "team_plays"]
    return d.groupby(["game_id", "posteam"], as_index=False)[cols].sum().rename(columns={"posteam": "team"})


def _positions(rosters: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    r = rosters.dropna(subset=["gsis_id"])
    r = r[["season", "week", "gsis_id", "position"]].drop_duplicates(["season", "week", "gsis_id"])
    static = players.dropna(subset=["gsis_id"]).set_index("gsis_id")["position"].to_dict()
    return r.rename(columns={"gsis_id": "player_id", "position": "roster_position"}), static


def build(p: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame, rosters: pd.DataFrame,
          tv: pd.DataFrame) -> pd.DataFrame:
    xwalk = pfr_to_gsis(players)
    sn = teams.normalize(snaps, ["team", "opponent"]).copy()
    sn["player_id"] = sn["pfr_player_id"].map(xwalk)
    for c in ("offense_snaps", "defense_snaps", "st_snaps", "offense_pct"):
        sn[c] = sn[c].fillna(0).astype(float)
    sn["any_snaps"] = sn["offense_snaps"] + sn["defense_snaps"] + sn["st_snaps"]
    sn = sn[sn["player_id"].notna() & (sn["any_snaps"] > 0)]
    sn = sn.sort_values("any_snaps", ascending=False).drop_duplicates(["game_id", "player_id"])
    snap_cols = sn[["game_id", "player_id", "team", "position", "player", "offense_snaps", "offense_pct",
                    "any_snaps"]].rename(columns={"team": "snap_team", "position": "snap_position",
                                                  "player": "snap_name"})
    team_snaps = sn.groupby(["game_id", "team"])["offense_snaps"].max().rename("team_offense_snaps")

    sums = _player_sums(p)
    pg = sums.merge(snap_cols, on=["game_id", "player_id"], how="outer")
    pg["team"] = pg["team"].fillna(pg["snap_team"])
    pg = pg[pg["team"].notna()]

    pos, static = _positions(rosters, players)
    pg = pg.merge(tv[["game_id", "team", "opponent", "season", "week", "season_type", "kickoff_utc", "is_home"]],
                  on=["game_id", "team"], how="inner")
    pg = pg.merge(pos, on=["season", "week", "player_id"], how="left")
    pg["position"] = pg["roster_position"].fillna(pg["snap_position"]).fillna(pg["player_id"].map(static))
    pg = pg[pg["position"].isin(SKILL)].copy()
    pg.loc[pg["position"] == "FB", "position"] = "RB"

    count_cols = ["targets", "receptions", "rec_yds", "air_yards", "yac", "rec_tds", "rz_targets", "i10_targets",
                  "deep_targets", "garbage_targets", "carries", "designed_carries", "scrambles", "kneels",
                  "rush_yds", "rush_tds", "rz_carries", "i10_carries", "garbage_carries", "dropbacks", "pass_att",
                  "completions", "pass_yds", "pass_tds", "ints", "sacks_taken"]
    snap_games = set(sn["game_id"])
    pg["snaps_published"] = pg["game_id"].isin(snap_games)
    for c in count_cols:
        pg[c] = pg[c].fillna(0.0)
    pg = pg.merge(_team_denoms(p), on=["game_id", "team"], how="left")
    pg = pg.merge(team_snaps.reset_index(), on=["game_id", "team"], how="left")
    pg["snap_share"] = np.where(pg["team_offense_snaps"] > 0, pg["offense_snaps"] / pg["team_offense_snaps"], np.nan)
    pg["player_name"] = pg["snap_name"]
    keep = ["game_id", "player_id", "player_name", "position", "team", "opponent", "season", "week", "season_type",
            "kickoff_utc", "is_home", "offense_snaps", "any_snaps", "team_offense_snaps", "snap_share",
            "snaps_published"] + count_cols + [
            "team_targets", "team_air_yards", "team_designed_carries", "team_rz_targets", "team_i10_carries",
            "team_rz_carries", "team_dropbacks", "team_plays"]
    pg = pg[keep].sort_values(["kickoff_utc", "game_id", "player_id"], ignore_index=True)
    assert not pg.duplicated(["game_id", "player_id"]).any(), "duplicate player-game rows"
    return pg


def attach_names(pg: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    names = players.dropna(subset=["gsis_id"]).set_index("gsis_id")["display_name"]
    pg = pg.copy()
    pg["player_name"] = pg["player_name"].fillna(pg["player_id"].map(names))
    return pg
