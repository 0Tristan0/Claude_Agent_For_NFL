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


def pfr_to_gsis(players: pd.DataFrame, rosters: pd.DataFrame | None = None) -> dict[str, str]:
    """PFR id -> GSIS id from the players table, plus pairs from weekly rosters for ids the table lacks."""
    frames = [players[["pfr_id", "gsis_id"]]]
    if rosters is not None and {"pfr_id", "gsis_id"} <= set(rosters.columns):
        frames.append(rosters[["pfr_id", "gsis_id"]])
    m = pd.concat(frames).dropna().drop_duplicates()
    m = m[~m["pfr_id"].duplicated(keep=False)]
    return dict(zip(m["pfr_id"], m["gsis_id"]))


def _name_key(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.replace(r"[.'`\-]", "", regex=True)
            .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True).str.split().str.join(" "))


def _match_by_name(sn: pd.DataFrame, rosters: pd.DataFrame) -> pd.Series:
    """For snap rows whose PFR id is unknown, find the GSIS id on that team's same-week roster by name."""
    need = sn["player_id"].isna()
    if not need.any() or rosters is None or rosters.empty or "full_name" not in rosters.columns:
        return sn["player_id"]
    r = teams.normalize(rosters.dropna(subset=["gsis_id", "full_name"]), ["team"])
    r = r.assign(key=_name_key(r["full_name"]))[["season", "week", "team", "key", "gsis_id"]]
    r = r.drop_duplicates(["season", "week", "team", "key"], keep=False)
    q = sn.loc[need, ["season", "week", "team", "player"]].assign(key=lambda d: _name_key(d["player"]))
    hit = q.reset_index().merge(r, on=["season", "week", "team", "key"], how="left").set_index("index")["gsis_id"]
    out = sn["player_id"].copy()
    out.loc[hit.index] = hit
    return out


def assign_position(pg: pd.DataFrame) -> pd.Series:
    """Position as of that week: the weekly roster if it is a skill position, else the snap-count position,
    else the role the stats show. The players-table snapshot (today's position) is never used."""
    def skill(x):
        return x.where(x.isin(SKILL))
    role = pd.Series(np.select([pg["pass_att"].fillna(0) > 0, pg["designed_carries"].fillna(0) > pg["targets"].fillna(0),
                                pg["targets"].fillna(0) > 0], ["QB", "RB", "WR"], default=""), index=pg.index)
    return skill(pg["roster_position"]).fillna(skill(pg["snap_position"])).fillna(role.replace("", np.nan))


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
        designed_rush_yds=np.where(r["designed_run"], r["rushing_yards"].fillna(0), 0.0),
        scramble_yds=np.where(r["scramble"], r["rushing_yards"].fillna(0), 0.0),
        rush_tds=((r["rush_touchdown"].fillna(0) == 1) & (r["td_player_id"] == r["rusher_player_id"])).astype(float),
        rz_carries=(r["rz"] & r["designed_run"]).astype(float), i10_carries=(r["i10"] & r["designed_run"]).astype(float),
        garbage_carries=(r["garbage"] & r["designed_run"]).astype(float),
    ).groupby(["game_id", "rusher_player_id"]).agg(
        team_r=("posteam", "first"), carries=("carries", "sum"), designed_carries=("designed_carries", "sum"),
        scrambles=("scrambles", "sum"), kneels=("kneels", "sum"), rush_yds=("rush_yds", "sum"),
        designed_rush_yds=("designed_rush_yds", "sum"), scramble_yds=("scramble_yds", "sum"),
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
    # A scramble is a dropback (nflfastR credits it to the rusher, not the passer).
    out["dropbacks"] = out["dropbacks"].fillna(0) + out["scrambles"].fillna(0)
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


def _positions(rosters: pd.DataFrame) -> pd.DataFrame:
    if rosters is None or rosters.empty:
        return pd.DataFrame(columns=["season", "week", "player_id", "roster_position"])
    r = rosters.dropna(subset=["gsis_id"])
    r = r[["season", "week", "gsis_id", "position"]].drop_duplicates(["season", "week", "gsis_id"])
    return r.rename(columns={"gsis_id": "player_id", "position": "roster_position"})


def build(p: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame, rosters: pd.DataFrame,
          tv: pd.DataFrame) -> pd.DataFrame:
    xwalk = pfr_to_gsis(players, rosters)
    sn = teams.normalize(snaps, ["team", "opponent"]).copy()
    sn["player_id"] = sn["pfr_player_id"].map(xwalk)
    sn["player_id"] = _match_by_name(sn, rosters)
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

    pg = pg.merge(tv[["game_id", "team", "opponent", "season", "week", "season_type", "kickoff_utc", "is_home"]],
                  on=["game_id", "team"], how="inner")
    pg = pg.merge(_positions(rosters), on=["season", "week", "player_id"], how="left")
    pg["position"] = assign_position(pg)
    pg = pg[pg["position"].isin(SKILL)].copy()
    pg.loc[pg["position"] == "FB", "position"] = "RB"

    count_cols = ["targets", "receptions", "rec_yds", "air_yards", "yac", "rec_tds", "rz_targets", "i10_targets",
                  "deep_targets", "garbage_targets", "carries", "designed_carries", "scrambles", "kneels",
                  "rush_yds", "designed_rush_yds", "scramble_yds", "rush_tds", "rz_carries", "i10_carries", "garbage_carries", "dropbacks", "pass_att",
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
