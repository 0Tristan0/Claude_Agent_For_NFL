"""Pregame team features from neutral-situation play-by-play (win probability 10-90%).

Each is a ratio of recency-weighted sums over the team's earlier games, for its offense (`off_`) and
for what its defense allowed (`def_`). Raw, not opponent-adjusted; see features/opponent.py for that.
"""
from __future__ import annotations

import pandas as pd

from .asof import AsofIndex
from .player import _ratio

BASE = ["plays", "dropbacks", "pass_att", "targets", "sacks", "scrambles", "designed_runs", "kneels", "giveaways",
        "n_neutral", "n_pass", "n_run", "pass_epa_sum", "run_epa_sum", "pass_sr_sum", "run_sr_sum", "expl_pass",
        "expl_run", "n_sacks", "n_hit_or_sack", "n_giveaways", "n_x", "xpass_sum", "pass_oe_sum", "pass_on_x",
        "n_pace", "pace_sum", "drives", "td_drives", "rz_trips", "rz_tds", "rz_trips_n", "rz_tds_n"]


def rates(s: pd.DataFrame, side: str) -> dict:
    g = lambda c: s[f"{side}_{c}"]
    return {
        f"{side}_pass_epa": _ratio(g("pass_epa_sum"), g("n_pass")),
        f"{side}_rush_epa": _ratio(g("run_epa_sum"), g("n_run")),
        f"{side}_pass_sr": _ratio(g("pass_sr_sum"), g("n_pass")),
        f"{side}_rush_sr": _ratio(g("run_sr_sum"), g("n_run")),
        f"{side}_expl_pass_rate": _ratio(g("expl_pass"), g("n_pass")),
        f"{side}_expl_rush_rate": _ratio(g("expl_run"), g("n_run")),
        f"{side}_neutral_pass_rate": _ratio(g("pass_on_x"), g("n_x")),
        f"{side}_proe": _ratio(g("pass_oe_sum"), g("n_x")),
        f"{side}_sec_per_play": _ratio(g("pace_sum"), g("n_pace")),
        f"{side}_sack_rate": _ratio(g("n_sacks"), g("n_pass")),
        f"{side}_hit_sack_rate": _ratio(g("n_hit_or_sack"), g("n_pass")),
        f"{side}_giveaway_rate": _ratio(g("n_giveaways"), g("n_neutral")),
        f"{side}_rz_trips_pd": _ratio(g("rz_trips"), g("drives")),
        f"{side}_rz_td_rate": _ratio(g("rz_tds"), g("rz_trips")),
        f"{side}_rz_td_rate_neutral": _ratio(g("rz_tds_n"), g("rz_trips_n")),
        f"{side}_plays_pg": _ratio(g("plays"), s["eff_games"]),
        f"{side}_drives_pg": _ratio(g("drives"), s["eff_games"]),
    }


class TeamFeatures:
    def __init__(self, team_games: pd.DataFrame, pressure: pd.DataFrame | None = None,
                 half_life: float | None = 8.0, season_discount: float = 0.5):
        cols = [f"{side}_{c}" for side in ("off", "def") for c in BASE]
        self.idx = AsofIndex(team_games, "team", cols, half_life=half_life, season_discount=season_discount)
        self.p_idx = None
        if pressure is not None and len(pressure):
            pc = ["off_pressures", "off_pressure_dropbacks", "def_pressures", "def_pressure_dropbacks"]
            self.p_idx = AsofIndex(pressure, "team", pc, half_life=half_life, season_discount=season_discount)

    def features(self, teams_, asof: pd.Timestamp, season: int) -> pd.DataFrame:
        s = self.idx.frame(teams_, asof, season)
        f = pd.DataFrame({"team": list(teams_), "eff_games": s["eff_games"].to_numpy()})
        for side in ("off", "def"):
            for k, v in rates(s, side).items():
                f[k] = v
        if self.p_idx is not None:
            q = self.p_idx.frame(teams_, asof, season)
            # all situations (PFR game totals); the 10-90% win-probability pressure metric is *_hit_sack_rate
            f["off_pressure_rate_pfr"] = _ratio(q.off_pressures, q.off_pressure_dropbacks)
            f["def_pressure_rate_pfr"] = _ratio(q.def_pressures, q.def_pressure_dropbacks)
        return f
