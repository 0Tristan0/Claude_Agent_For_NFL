"""Pregame player opportunity and efficiency features (recency-weighted, before any shrinkage).

Every rate is a ratio of weighted sums over the player's earlier games, read through AsofIndex, so a
target game never contributes to its own features. Phase 2 turns these sums into Bayesian posteriors;
the raw rates here are what the report shows next to the shrunk values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import AsofIndex

SUMS = ["offense_snaps", "team_offense_snaps", "routes_est", "rk_targets", "rk_rec_yds", "rk_team_dropbacks",
        "team_dropbacks", "targets", "team_targets",
        "air_yards", "team_air_yards", "receptions", "rec_yds", "yac", "rec_tds", "rz_targets", "team_rz_targets",
        "i10_targets", "deep_targets", "designed_carries", "team_designed_carries", "i10_carries",
        "team_i10_carries", "rz_carries", "team_rz_carries", "carries", "scrambles", "kneels", "rush_yds", "designed_rush_yds",
        "scramble_yds",
        "rush_tds", "dropbacks", "pass_att", "completions", "pass_yds", "pass_tds", "ints", "sacks_taken",
        "team_plays"]
PFR_SUMS = ["pfr_carries", "ybc", "yaco", "broken_tackles"]


def _ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


class PlayerFeatures:
    def __init__(self, player_games: pd.DataFrame, pfr_rush: pd.DataFrame | None = None,
                 half_life: float | None = 4.0, season_discount: float = 0.5):
        pg = player_games[player_games["any_snaps"].fillna(0) > 0]
        self.idx = AsofIndex(pg, "player_id", SUMS, half_life=half_life, season_discount=season_discount)
        self.season_idx = AsofIndex(pg, "player_id", SUMS, half_life=None, season_discount=0.0)
        self.pfr = None
        if pfr_rush is not None and len(pfr_rush):
            self.pfr = AsofIndex(pfr_rush, "player_id", PFR_SUMS, half_life=half_life, season_discount=season_discount)

    def features(self, player_ids, asof: pd.Timestamp, season: int) -> pd.DataFrame:
        s = self.idx.frame(player_ids, asof, season)
        cur = self.season_idx.frame(player_ids, asof, season)    # this season only (discount 0 drops older)
        f = pd.DataFrame({"player_id": list(player_ids)})
        f["eff_games"] = s["eff_games"].to_numpy()
        f["games_this_season"] = cur["eff_games"].to_numpy()
        f["snap_share"] = _ratio(s.offense_snaps, s.team_offense_snaps)
        # route-based rates use only games where routes could be estimated (rk_ = routes known)
        f["route_part_est"] = _ratio(s.routes_est, s.rk_team_dropbacks)
        f["target_share"] = _ratio(s.targets, s.team_targets)
        f["tprr_est"] = _ratio(s.rk_targets, s.routes_est)
        f["air_yards_share"] = _ratio(s.air_yards, s.team_air_yards)
        f["adot"] = _ratio(s.air_yards, s.targets)
        f["rz_target_share"] = _ratio(s.rz_targets, s.team_rz_targets)
        f["carry_share"] = _ratio(s.designed_carries, s.team_designed_carries)
        f["i10_carry_share"] = _ratio(s.i10_carries, s.team_i10_carries)
        f["rz_carry_share"] = _ratio(s.rz_carries, s.team_rz_carries)
        f["yprr_est"] = _ratio(s.rk_rec_yds, s.routes_est)
        f["catch_rate"] = _ratio(s.receptions, s.targets)
        f["yds_per_catch"] = _ratio(s.rec_yds, s.receptions)
        f["yac_per_catch"] = _ratio(s.yac, s.receptions)
        f["yds_per_carry"] = _ratio(s.designed_rush_yds, s.designed_carries)     # same carry set as carries_pg
        f["scrambles_pg"] = _ratio(s.scrambles, s.eff_games)
        f["yds_per_scramble"] = _ratio(s.scramble_yds, s.scrambles)
        f["targets_pg"] = _ratio(s.targets, s.eff_games)
        f["carries_pg"] = _ratio(s.designed_carries, s.eff_games)
        f["dropbacks_pg"] = _ratio(s.dropbacks, s.eff_games)
        f["pass_yds_per_att"] = _ratio(s.pass_yds, s.pass_att)
        f["completion_rate"] = _ratio(s.completions, s.pass_att)
        f["int_rate"] = _ratio(s.ints, s.pass_att)
        f["scramble_rate"] = _ratio(s.scrambles, s.dropbacks)
        if self.pfr is not None:
            q = self.pfr.frame(player_ids, asof, season)
            f["yaco_per_att"] = _ratio(q.yaco, q.pfr_carries)
            f["ybc_per_att"] = _ratio(q.ybc, q.pfr_carries)
        for c in SUMS:
            f[f"sum_{c}"] = s[c].to_numpy()
        return f
