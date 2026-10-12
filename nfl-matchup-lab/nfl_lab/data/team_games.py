"""Team-game tables: one row per (game_id, team) with offense sums and the matching defense-allowed sums.

Everything is stored as numerators and denominators (sums and counts), never as per-game rates, so
recency-weighted aggregation later is a ratio of weighted sums. Efficiency stats use neutral plays
only (win probability 10-90%, no kneels or spikes). Volume counts (plays, dropbacks, drives) use all
scrimmage plays because the sim needs them; neutral versions are stored alongside where they matter.

Offense columns start with `off_`; `def_` columns are what the team's defense allowed (the
opponent's offense columns), so a defense's "EPA allowed" is def_pass_epa_sum / def_n_pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import teams
from .pbp import pace_seconds


def _sum(mask, values=None):
    return mask.astype(float) if values is None else np.where(mask, values, 0.0)


def offense(p: pd.DataFrame, drives: pd.DataFrame) -> pd.DataFrame:
    s = p[p["scrimmage"] & p["posteam"].notna()].copy()
    n, db, run = s["neutral"], s["dropback"], s["designed_run"]
    epa = s["epa"].fillna(0).to_numpy(float)
    suc = s["success"].fillna(0).to_numpy(float)
    has_x = n & s["xpass"].notna()
    s["pace"] = pace_seconds(p).reindex(s.index)
    cols = {
        # volume, all scrimmage plays
        "plays": _sum(~s["kneel"] & ~s["spike"]),
        "dropbacks": _sum(db), "pass_att": _sum(s["pass_att"]), "targets": _sum(s["target"]),
        "sacks": _sum(s["is_sack"]), "scrambles": _sum(s["scramble"]), "designed_runs": _sum(run),
        "kneels": _sum(s["kneel"]), "giveaways": _sum(s["giveaway"]),
        "garbage_plays": _sum(s["garbage"] & ~s["kneel"] & ~s["spike"]),
        "air_yards": _sum(s["target"], s["air_yards"].fillna(0)),
        # neutral efficiency
        "n_neutral": _sum(n), "n_pass": _sum(n & db), "n_run": _sum(n & run),
        "pass_epa_sum": _sum(n & db, epa), "run_epa_sum": _sum(n & run, epa),
        "pass_sr_sum": _sum(n & db, suc), "run_sr_sum": _sum(n & run, suc),
        "expl_pass": _sum(n & db & s["explosive"]), "expl_run": _sum(n & run & s["explosive"]),
        "n_sacks": _sum(n & s["is_sack"]),
        "n_hit_or_sack": _sum(n & db & (s["is_sack"] | (s["qb_hit"].fillna(0) == 1))),
        "n_giveaways": _sum(n & s["giveaway"]),
        "n_x": _sum(has_x), "xpass_sum": _sum(has_x, s["xpass"].fillna(0)),
        "pass_oe_sum": _sum(has_x, s["pass_oe"].fillna(0) / 100.0),
        "pass_on_x": _sum(has_x & db),
        "n_pace": s["pace"].notna().astype(float), "pace_sum": s["pace"].fillna(0).to_numpy(float),
    }
    df = pd.DataFrame(cols, index=s.index)
    df["game_id"], df["team"] = s["game_id"].to_numpy(), s["posteam"].to_numpy()
    out = df.groupby(["game_id", "team"], as_index=False).sum()

    d = drives[drives["posteam"].notna()].copy()
    rz_n = d["reached_rz"].astype(bool) & d["rz_entry_wp"].between(0.10, 0.90)
    d = d.assign(drives=1.0, td_drives=d["td"].astype(float), rz_trips=d["reached_rz"].astype(float),
                 rz_tds=(d["reached_rz"].astype(bool) & d["td"]).astype(float),
                 rz_trips_n=rz_n.astype(float), rz_tds_n=(rz_n & d["td"]).astype(float),
                 drive_plays=d["n_plays"].astype(float))
    dagg = d.groupby(["game_id", "posteam"], as_index=False)[
        ["drives", "td_drives", "rz_trips", "rz_tds", "rz_trips_n", "rz_tds_n", "drive_plays"]].sum()
    out = out.merge(dagg.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    return out


def pfr_pressure(pfr_pass: pd.DataFrame) -> pd.DataFrame:
    """Pressures allowed per team-game from PFR passer rows (one row per QB, so no double counting)."""
    if pfr_pass.empty:
        return pd.DataFrame(columns=["game_id", "team", "pressures", "pfr_dropbacks"])
    q = teams.normalize(pfr_pass, ["team", "opponent"])
    q = q.assign(pressures=q["times_pressured"].fillna(0),
                 pfr_dropbacks=(q["times_pressured"] / (q["times_pressured_pct"].replace(0, np.nan))).fillna(0))
    return q.groupby(["game_id", "team"], as_index=False)[["pressures", "pfr_dropbacks"]].sum()


def ftn_team(ftn: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Screen, play-action and motion counts per team-game (FTN charting, CC-BY-SA 4.0)."""
    if ftn.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    f = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f = f.merge(p[["game_id", "play_id", "posteam", "dropback", "scrimmage"]], on=["game_id", "play_id"], how="inner")
    f = f[f["scrimmage"]]
    agg = f.assign(
        ftn_plays=1.0, ftn_dropbacks=f["dropback"].astype(float),
        screens=(f["is_screen_pass"].fillna(False).astype(bool) & f["dropback"]).astype(float),
        play_action=(f["is_play_action"].fillna(False).astype(bool) & f["dropback"]).astype(float),
        motion=f["is_motion"].fillna(False).astype(bool).astype(float),
    ).groupby(["game_id", "posteam"], as_index=False)[["ftn_plays", "ftn_dropbacks", "screens", "play_action",
                                                       "motion"]].sum()
    return agg.rename(columns={"posteam": "team"})


def with_defense(off: pd.DataFrame, tv: pd.DataFrame, value_cols) -> pd.DataFrame:
    """Attach `def_<col>` = what this team's defense allowed (the opponent's offense values)."""
    o = off.rename(columns={c: f"off_{c}" for c in value_cols})
    d = off.rename(columns={c: f"def_{c}" for c in value_cols}).rename(columns={"team": "opponent"})
    out = tv.merge(o, on=["game_id", "team"], how="inner")
    return out.merge(d, on=["game_id", "opponent"], how="left")


def build(p: pd.DataFrame, drives: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    off = offense(p, drives)
    cols = [c for c in off.columns if c not in ("game_id", "team")]
    tg = with_defense(off, tv[["game_id", "team", "opponent", "season", "week", "season_type", "kickoff_utc", "is_home"]],
                      cols)
    return tg.sort_values(["kickoff_utc", "game_id", "team"], ignore_index=True)


def build_side(table: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    """Same offense/defense mirroring for a side table (PFR pressure, FTN)."""
    if table.empty:
        return table
    cols = [c for c in table.columns if c not in ("game_id", "team")]
    return with_defense(table, tv[["game_id", "team", "opponent", "season", "week", "season_type", "kickoff_utc"]],
                        cols).sort_values(["kickoff_utc", "game_id", "team"], ignore_index=True)
