"""Scoring rules and cluster-bootstrap uncertainty.

All metrics are computed per forecast row; uncertainty is estimated with a
cluster bootstrap that resamples whole *games* (all player rows of a game stay
together), because teammates' outcomes in one game are correlated. Rows of the
same player across weeks are also correlated; the game-cluster bootstrap does
not capture that, so its intervals are, if anything, too narrow (documented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def interval_hit(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((y >= lo) & (y <= hi)).astype(float)


def cluster_bootstrap(values: pd.DataFrame, cluster: pd.Series, n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Mean of each column with 95% CI from a cluster (game) bootstrap.

    values: per-row metric columns (e.g. a model-minus-baseline loss difference).
    """
    codes, uniq = pd.factorize(cluster)
    k = len(uniq)
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, codes, values.to_numpy(float))
    counts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)    # (B, k)
    boot = (w @ sums) / (w @ counts)[:, None]
    est = sums.sum(0) / counts.sum()
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return pd.DataFrame({"metric": values.columns, "estimate": est, "ci_low": lo, "ci_high": hi})


def calibration_table(p: np.ndarray, y: np.ndarray, bins=(0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)):
    b = np.digitize(p, bins[1:-1])
    rows = []
    for i in range(len(bins) - 1):
        idx = b == i
        if idx.sum() == 0:
            continue
        n = int(idx.sum())
        obs = float(y[idx].mean())
        se = np.sqrt(max(obs * (1 - obs), 1e-9) / n)
        rows.append(dict(bin_low=bins[i], bin_high=min(bins[i + 1], 1.0), n=n, mean_pred=float(p[idx].mean()),
                         observed=obs, obs_ci_low=max(0.0, obs - 1.96 * se), obs_ci_high=min(1.0, obs + 1.96 * se)))
    return pd.DataFrame(rows)


def crps_from_quantiles(qdict: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    """Approximate CRPS via the average pinball loss over the stored quantile levels (x2)."""
    total = np.zeros(len(y))
    levels = []
    for k, q in qdict.items():
        tau = int(k[1:]) / 1000.0
        levels.append(tau)
        d = y - q
        total += np.maximum(tau * d, (tau - 1) * d)
    return 2.0 * total / max(len(levels), 1)
