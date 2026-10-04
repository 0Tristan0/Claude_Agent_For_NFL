"""Simple baselines that every advanced model must beat.

B1  rolling      : point forecast = mean of the player's last 8 games played;
                   interval = empirical quantiles of his last 16 games.
B2  empirical    : milestone probability = share of his last 16 games reaching
                   the milestone, shrunk toward the position-wide rate with k
                   pseudo-games (Beta-binomial posterior mean). Rookies/no
                   history -> the position rate.
B3  workload     : mean = (shrunk targets per game) x (shrunk yards per target).
                   Distribution = mean x R, where R is the empirical ratio
                   outcome/mean observed in training rows with a similar mean
                   (5 quantile bins). Non-parametric, so tails are learned.
B3n normal       : same mean, Normal errors with bin-specific SD. Included only
                   to show what a normal-distribution assumption does to tails.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from .simulate import QUANTILES

N_HIST = 16


def lagged_outcomes(df: pd.DataFrame, col: str, n: int = N_HIST) -> np.ndarray:
    """(rows, n) matrix of the player's previous n *played* outcomes (most recent first, NaN padded)."""
    d = df.sort_values(["player_id", "kickoff_utc"])
    out = np.full((len(df), n), np.nan)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for _, grp in d.groupby("player_id", sort=False):
        hist: deque = deque(maxlen=n)
        vals = grp[col].to_numpy(float)
        played = grp["played"].to_numpy()
        for idx, v, pl in zip(grp.index, vals, played):
            if hist:
                h = list(hist)[::-1]
                out[pos[idx], :len(h)] = h
            if pl:
                hist.append(v)
    return out


class RollingEmpiricalBaseline:
    """B1 (point + intervals) and B2 (shrunk milestone frequencies)."""

    name = "B1/B2 rolling-empirical"

    def __init__(self, k: float = 6.0):
        self.k = k

    def fit(self, train: pd.DataFrame, outcome: str, thresholds) -> "RollingEmpiricalBaseline":
        tr = train[train["played"]]
        self.outcome, self.thresholds = outcome, tuple(thresholds)
        self.prior = {(pos, t): float((g[outcome] >= t).mean())
                      for pos, g in tr.groupby("position") for t in self.thresholds}
        self.prior_mean = tr.groupby("position")[outcome].mean().to_dict()
        self.prior_q = {pos: np.quantile(g[outcome], QUANTILES) for pos, g in tr.groupby("position")}
        return self

    def predict(self, df: pd.DataFrame, hist: np.ndarray) -> dict[str, np.ndarray]:
        n_hist = np.sum(~np.isnan(hist), axis=1)
        pos = df["position"].to_numpy()
        res: dict[str, np.ndarray] = {}
        last8 = hist[:, :8]
        with np.errstate(all="ignore"), _quiet():
            m8 = np.nanmean(last8, axis=1)
        pm = np.array([self.prior_mean.get(p, np.nan) for p in pos])
        res["mean"] = np.where(n_hist > 0, m8, pm)
        with np.errstate(all="ignore"), _quiet():
            q = np.nanquantile(hist, QUANTILES, axis=1)
        pq = np.array([self.prior_q[p] for p in pos]).T
        q = np.where(n_hist[None, :] >= 4, q, pq)       # too few games: position quantiles
        for i, x in enumerate(QUANTILES):
            res[f"q{int(round(x * 1000)):03d}"] = q[i]
        res["median"] = res["q500"]
        for t in self.thresholds:
            hits = np.nansum(hist >= t, axis=1)
            p0 = np.array([self.prior[(p, t)] for p in pos])
            res[f"p_ge_{t}"] = (hits + self.k * p0) / (n_hist + self.k)
        return res


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._w.__exit__(*a)


class WorkloadBaseline:
    """B3: workload mean x empirical ratio distribution (and B3n normal variant)."""

    name = "B3 workload"

    def __init__(self, outcome: str = "receiving_yards", n_bins: int = 5, normal: bool = False):
        self.outcome, self.n_bins, self.normal = outcome, n_bins, normal

    def mean_of(self, df: pd.DataFrame) -> np.ndarray:
        if self.outcome == "receiving_yards":
            return (df["tpg_ew"] * df["ypt_ew"]).to_numpy(float)
        if self.outcome == "receptions":
            return (df["tpg_ew"] * df["catch_rate_ew"]).to_numpy(float)
        return df["tpg_ew"].to_numpy(float)

    def fit(self, train: pd.DataFrame, thresholds) -> "WorkloadBaseline":
        tr = train[train["played"]]
        mu = self.mean_of(tr)
        y = tr[self.outcome].to_numpy(float)
        # rescale so the mean is unbiased on training data
        self.scale = float(np.sum(y) / np.sum(mu))
        mu = mu * self.scale
        self.edges = np.quantile(mu, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        b = np.digitize(mu, self.edges)
        self.ratios = [np.sort(y[b == i] / mu[b == i]) for i in range(self.n_bins)]
        self.sds = [float(np.std(y[b == i] - mu[b == i])) for i in range(self.n_bins)]
        self.thresholds = tuple(thresholds)
        return self

    def pit(self, df: pd.DataFrame, y: np.ndarray, seed: int = 0) -> np.ndarray:
        """Randomized PIT of observed outcomes under this baseline's distribution."""
        mu = np.maximum(self.mean_of(df) * self.scale, 1e-9)
        b = np.digitize(mu, self.edges)
        u = np.random.default_rng(seed).random(len(mu))
        out = np.zeros(len(mu))
        for i in range(self.n_bins):
            idx = b == i
            if not idx.any():
                continue
            if self.normal:
                lo = stats.norm.cdf((y[idx] - 0.5 - mu[idx]) / self.sds[i])
                hi = stats.norm.cdf((y[idx] + 0.5 - mu[idx]) / self.sds[i])
            else:
                r = self.ratios[i]
                lo = np.searchsorted(r, y[idx] / mu[idx] - 1e-9, side="left") / len(r)
                hi = np.searchsorted(r, y[idx] / mu[idx] + 1e-9, side="right") / len(r)
            out[idx] = lo + u[idx] * (hi - lo)
        return out

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        mu = self.mean_of(df) * self.scale
        b = np.digitize(mu, self.edges)
        res: dict[str, np.ndarray] = {"mean": mu}
        qs = np.zeros((len(QUANTILES), len(mu)))
        ps = {t: np.zeros(len(mu)) for t in self.thresholds}
        for i in range(self.n_bins):
            idx = b == i
            if not idx.any():
                continue
            if self.normal:
                sd = self.sds[i]
                qs[:, idx] = mu[idx][None, :] + sd * stats.norm.ppf(QUANTILES)[:, None]
                for t in self.thresholds:   # continuity correction for integer outcomes
                    ps[t][idx] = stats.norm.sf((t - 0.5 - mu[idx]) / sd)
            else:
                r = self.ratios[i]
                qs[:, idx] = mu[idx][None, :] * np.quantile(r, QUANTILES)[:, None]
                for t in self.thresholds:
                    ps[t][idx] = 1.0 - np.searchsorted(r, t / np.maximum(mu[idx], 1e-9), side="left") / len(r)
        for j, x in enumerate(QUANTILES):
            res[f"q{int(round(x * 1000)):03d}"] = qs[j]
        res["median"] = res["q500"]
        for t in self.thresholds:
            res[f"p_ge_{t}"] = ps[t]
        return res
