"""Quick calibration check of the v0 model on a validation season (never 2025: holdout).

For every regular-season game from week 3 on, the model projects and simulates the game as of kickoff
(fits from earlier seasons only), then probabilities are graded against what happened.

Lines: there are no free historical prop lines, so each player's line is a STAND-IN: the median of his
last 6 games, or half a unit above it when the median is a whole number (anytime TD at 0.5). These are not
PrizePicks lines; real lines are sharper, so hit rates here overstate what real picks would do. The point
is calibration (do 60% calls hit about 60% of the time?) and a comparison with a simple baseline: the
player's hit rate at that line over his last 8 games, pulled toward 50% (the old lab's game-log half,
without its market half).

Actives: players who took a snap in the game (the inactive list is public ~90 minutes before kickoff).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import bundle
from ..features.asof import visible
from ..project import v0 as P
from ..sim import v0 as S

STAT_COLS = {"rec_yds": "rec_yds", "receptions": "receptions", "rush_yds": "rush_yds", "pass_yds": "pass_yds",
             "anytime_td": None}
GRADED = {"WR": ["rec_yds", "receptions", "anytime_td"], "TE": ["rec_yds", "receptions", "anytime_td"],
          "RB": ["rush_yds", "rec_yds", "receptions", "anytime_td"], "QB": ["pass_yds", "rush_yds", "anytime_td"]}


def _actual(row, stat):
    if stat == "anytime_td":
        return float(row["rec_tds"] + row["rush_tds"] >= 1)
    return float(row[STAT_COLS[stat]])


def run(tables: dict, season: int, weeks=range(3, 19), n_sims: int = 2000, progress=None,
        calibrate: bool = False) -> pd.DataFrame:
    if season >= 2025:
        raise ValueError("2025 is the holdout season: the quick check runs on 2024 or earlier only")
    fit_ = P.fit(tables, season)
    eng = bundle.engines(tables)
    games = tables["games"]
    gl = games[(games["season"] == season) & (games["season_type"] == "REG") & games["week"].isin(list(weeks))]
    pg_all = tables["player_games"]
    out = []
    for i, (_, g) in enumerate(gl.iterrows()):
        gid, ko = g["game_id"], g["kickoff_utc"]
        played = pg_all[(pg_all["game_id"] == gid) & (pg_all["any_snaps"] > 0) & pg_all["position"].isin(P.POSITIONS)]
        act = {t: played[played["team"] == t][["player_id", "player_name", "position"]].reset_index(drop=True)
               for t in (g["home_team"], g["away_team"])}
        proj = P.project_game(tables, gid, fit_, eng=eng, active_override=act)
        sim = S.simulate(proj, n_sims)
        hist = visible(pg_all, ko)
        for _, r in played.iterrows():
            pid = r["player_id"]
            if pid not in sim.players:
                continue
            h = hist[(hist["player_id"] == pid) & (hist["any_snaps"] > 0)].sort_values("kickoff_utc")
            if len(h) < 3:
                continue
            is_starter = sim.meta[pid]["starter_qb"]
            for stat in GRADED.get(r["position"], []):
                if r["position"] == "QB" and not is_starter:
                    continue
                if stat == "anytime_td":
                    line, past = 0.5, (h["rec_tds"] + h["rush_tds"] >= 1).astype(float).to_numpy()
                else:
                    past = h[STAT_COLS[stat]].astype(float).to_numpy()
                    med = np.median(past[-6:])
                    line = np.floor(med) + 0.5          # the median when it ends in .5, else half a unit above
                    if stat in ("rec_yds", "rush_yds", "pass_yds") and line < 4.5:
                        continue                    # skip near-zero usage lines
                x = sim.stat(pid, stat)
                p = float(np.clip((x > line).mean(), 0.01, 0.99))
                if calibrate:
                    from ..picks import calibration
                    p = float(calibration.apply(p, calibration.load(), stat))
                last8 = past[-8:]
                b0 = (float((last8 > line).sum()) + 2.5) / (len(last8) + 5)
                out.append(dict(game_id=gid, week=g["week"], player_id=pid, player=r["player_name"],
                                position=r["position"], team=r["team"], stat=stat, line=line, p_model=p, p_b0=b0,
                                proj_mean=float(x.mean()), actual=_actual(r, stat),
                                hit=float(_actual(r, stat) > line), n_hist=len(h)))
        if progress and i % 25 == 0:
            progress(f"{i + 1}/{len(gl)} games")
    return pd.DataFrame(out)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cluster_ci(df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of a per-row metric's mean by resampling whole games."""
    g = df.groupby("game_id")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    k = len(g)
    w = rng.multinomial(k, np.full(k, 1 / k), size=n_boot)
    vals = (w @ g["sum"].to_numpy()) / (w @ g["count"].to_numpy())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def report(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["d_brier"] = (df["p_model"] - df["hit"]) ** 2 - (df["p_b0"] - df["hit"]) ** 2
    by = []
    for stat, g in df.groupby("stat"):
        lo, hi = cluster_ci(g, "d_brier")
        by.append(dict(stat=stat, n=len(g), hit_rate=g["hit"].mean(), mean_p_model=g["p_model"].mean(),
                       brier_model=brier(g["p_model"], g["hit"]), brier_old=brier(g["p_b0"], g["hit"]),
                       logloss_model=log_loss(g["p_model"], g["hit"]), logloss_old=log_loss(g["p_b0"], g["hit"]),
                       d_brier=g["d_brier"].mean(), d_brier_lo=lo, d_brier_hi=hi,
                       proj_bias=(g["proj_mean"] - g["actual"]).mean()))
    bins = np.array([0, .3, .4, .45, .5, .55, .6, .65, .7, 1.0001])
    df["bin"] = np.digitize(df["p_model"], bins[1:-1])
    cal = df.groupby("bin").agg(n=("hit", "size"), mean_p=("p_model", "mean"), hit_rate=("hit", "mean")).reset_index()
    cal["lo"] = cal["mean_p"]
    se = np.sqrt(cal["hit_rate"] * (1 - cal["hit_rate"]) / cal["n"])
    cal["ci_lo"], cal["ci_hi"] = cal["hit_rate"] - 1.96 * se, cal["hit_rate"] + 1.96 * se
    lo, hi = cluster_ci(df, "d_brier")
    overall = dict(n=len(df), games=df["game_id"].nunique(), brier_model=brier(df["p_model"], df["hit"]),
                   brier_old=brier(df["p_b0"], df["hit"]), d_brier=df["d_brier"].mean(), d_brier_lo=lo, d_brier_hi=hi)
    # bias by role: compare projected mean to actual for receivers ranked by projected targets within team
    return dict(overall=overall, by_stat=pd.DataFrame(by), calibration=cal.drop(columns=["lo"]))
