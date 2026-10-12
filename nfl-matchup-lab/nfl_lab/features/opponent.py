"""Opponent-adjusted team ratings (weighted ridge, refit for every kickoff slot).

For a per-play metric m (EPA per dropback, success rate, ...) on team-game rows visible at `asof`:

    m_g = mu + off[team] + def[opp] + h * (is_home - 0.5) + noise

weights = neutral plays x recency (half-life in days, extra discount per season back). The ridge
penalty pulls every rating toward 0 (league average), so teams with few games stay near average.
def[opp] > 0 means the defense allows more than average. "Opponent-adjusted defense" = mu + def.

Only rows with available_at < asof enter the fit (test: a planted `<=` must be caught).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import visible

METRICS = {  # name: (numerator column, denominator column) on the offense side of team_games
    "pass_epa": ("off_pass_epa_sum", "off_n_pass"),
    "rush_epa": ("off_run_epa_sum", "off_n_run"),
    "pass_sr": ("off_pass_sr_sum", "off_n_pass"),
    "rush_sr": ("off_run_sr_sum", "off_n_run"),
    "hit_sack_rate": ("off_n_hit_or_sack", "off_n_pass"),
    "expl_pass_rate": ("off_expl_pass", "off_n_pass"),
}


def fit(team_games: pd.DataFrame, asof: pd.Timestamp, season: int, metric: str, lam: float = 400.0,
        half_life_days: float = 56.0, season_discount: float = 0.5) -> pd.DataFrame:
    num, den = METRICS[metric]
    d = visible(team_games, asof)
    d = d[(d[den] > 0) & (d["season"] >= season - 1)]
    if d.empty:
        return pd.DataFrame(columns=["team", f"adj_off_{metric}", f"adj_def_{metric}", "mu"])
    asof = pd.Timestamp(asof)
    age_days = (asof - d["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
    w = d[den].to_numpy(float) * 0.5 ** (age_days / half_life_days) * season_discount ** (season - d["season"].to_numpy())
    y = d[num].to_numpy(float) / d[den].to_numpy(float)
    teams_ = sorted(set(d["team"]) | set(d["opponent"]))
    ix = {t: i for i, t in enumerate(teams_)}
    n, k = len(d), len(teams_)
    X = np.zeros((n, 2 + 2 * k))
    X[:, 0] = 1.0
    # centered so mu is the neutral-site level (home +h/2, away -h/2)
    X[:, 1] = (d["is_home"].to_numpy(float) - 0.5) if "is_home" in d else 0.0
    X[np.arange(n), 2 + d["team"].map(ix).to_numpy()] = 1.0
    X[np.arange(n), 2 + k + d["opponent"].map(ix).to_numpy()] = 1.0
    # ridge in "per-play" units: lam pseudo-plays at 0 for each rating, intercept/home unpenalized
    P = np.diag([0.0, 0.0] + [lam] * (2 * k))
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + P, XtW @ y)
    mu = beta[0]
    out = pd.DataFrame({"team": teams_, f"adj_off_{metric}": mu + beta[2:2 + k], f"adj_def_{metric}": mu + beta[2 + k:],
                        "mu": mu})
    return out


def ratings(team_games: pd.DataFrame, asof: pd.Timestamp, season: int, metrics=None, **kw) -> pd.DataFrame:
    out = None
    for m in metrics or METRICS:
        r = fit(team_games, asof, season, m, **kw).drop(columns="mu")
        out = r if out is None else out.merge(r, on="team", how="outer")
    return out
