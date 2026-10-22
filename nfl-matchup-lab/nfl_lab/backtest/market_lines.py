"""Calibration at market-style lines.

PrizePicks and sportsbook lines sit near the model's mean outcome (Week 6 2026 board: receiving yards +0.5 yd,
passing yards +0.4 yd from the v0 mean), while the first calibration was fitted at stand-in lines near a player's
recent median. Yardage distributions are skewed, so the model's chance of going under a line set at its own mean
needs its own check. For each player-game, line = model mean rounded to .5; compare the simulated P(under) with
what happened. Fit per stat on one season (2023), check on another (2024); 2025 is never touched.

    python -m nfl_lab market-cal          # runs both seasons (about 10 minutes each at 1,500 sims), fits, checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..features import bundle
from ..picks import calibration
from ..project import v0 as P
from ..sim import v0 as S
from . import quick

STATS = ["pass_yds", "rec_yds", "receptions", "rush_yds"]
MIN_MEAN = {"pass_yds": 15, "rec_yds": 15, "rush_yds": 15, "receptions": 1.5}


def run(tables: dict, season: int, weeks=range(3, 13), n_sims: int = 1500, progress=None) -> pd.DataFrame:
    if season >= config.TEST_SEASON:
        raise ValueError(f"{config.TEST_SEASON} is the holdout season")
    fit_ = P.fit(tables, season)
    eng = bundle.engines(tables)
    games = tables["games"]
    gl = games[(games["season"] == season) & (games["season_type"] == "REG") & games["week"].isin(list(weeks))]
    pg = tables["player_games"]
    out = []
    for i, (_, g) in enumerate(gl.iterrows()):
        played = pg[(pg["game_id"] == g["game_id"]) & (pg["any_snaps"] > 0) & pg["position"].isin(P.POSITIONS)]
        act = {t: played[played["team"] == t][["player_id", "player_name", "position"]].reset_index(drop=True)
               for t in (g["home_team"], g["away_team"])}
        proj = P.project_game(tables, g["game_id"], fit_, eng=eng, active_override=act)
        sim = S.simulate(proj, n_sims)
        for _, r in played.iterrows():
            pid = r["player_id"]
            if pid not in sim.players:
                continue
            for stat in quick.GRADED.get(r["position"], []):
                if stat not in STATS or (r["position"] == "QB" and not sim.meta[pid]["starter_qb"]):
                    continue
                x = sim.stat(pid, stat)
                m = float(x.mean())
                if m < MIN_MEAN[stat]:
                    continue
                line = np.floor(m) + 0.5
                out.append(dict(season=season, game_id=g["game_id"], stat=stat, position=r["position"], line=line,
                                raw_under=float(np.clip((x < line).mean(), 0.01, 0.99)), under=float(r[stat] < line)))
        if progress and i % 20 == 0:
            progress(f"{season}: {i + 1}/{len(gl)} games")
    return pd.DataFrame(out)


def fit(df: pd.DataFrame) -> dict:
    """Shift only (slope fixed at 1): the fit sees lines at the model mean, where chances sit in a narrow band,
    so a fitted slope would extrapolate badly to lines away from the mean. A shift fixes the skew bias at the
    center without making far-off lines look more certain."""
    from scipy.optimize import minimize_scalar
    by = {}
    for stat, g in df.groupby("stat"):
        x = calibration._logit(g["raw_under"])
        y = g["under"].to_numpy(float)

        def nll(b):
            q = np.clip(1 / (1 + np.exp(-(x + b))), 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        by[stat] = dict(a=1.0, b=float(minimize_scalar(nll, bounds=(-2, 2), method="bounded").x), n=int(len(g)))
    return {"by_stat": by}


def check(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    rows = []
    for stat, g in df.groupby("stat"):
        p = np.array([calibration.market_under(r, stat, params) for r in g["raw_under"]])
        y = g["under"].to_numpy()
        rows.append(dict(stat=stat, n=len(g), raw_says=g["raw_under"].mean(), fitted_says=p.mean(), happened=y.mean(),
                         brier_raw=float(np.mean((g["raw_under"] - y) ** 2)), brier_fitted=float(np.mean((p - y) ** 2))))
    return pd.DataFrame(rows)
