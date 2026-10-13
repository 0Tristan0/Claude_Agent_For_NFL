"""Probability recalibration for v0: p_cal = logistic(a * logit(p) + b).

Fitted on one validation season's quick check (2023) and judged on another (2024), never on 2025.
v0's raw sim probabilities are too confident at the extremes (its 79% calls hit about 72% in 2024), so
this layer pulls them in, per stat type where there is enough data. Lineup hit chances keep the sim's
correlation by reweighting (raking) the simulated games to the calibrated leg probabilities.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from .. import config

PATH = config.BACKTESTS_DIR / "eval" / "v0_calibration.json"
# A second map for market-style lines (PrizePicks and sportsbook lines sit near the model's mean, not near a
# player's recent median): fitted on P(under) at a line equal to the model's own mean, 2023, checked on 2024.
MARKET_PATH = config.BACKTESTS_DIR / "eval" / "v0_calibration_market.json"
MARKET_FAMILY = {"pass_rush_yds": "pass_yds"}      # stats with no fit of their own borrow a close one


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(p, y) -> dict:
    x, y = _logit(p), np.asarray(y, float)

    def nll(theta):
        a, b = theta
        q = 1 / (1 + np.exp(-(a * x + b)))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return dict(a=float(r.x[0]), b=float(r.x[1]), n=int(len(y)))


def apply(p, params: dict | None, stat: str | None = None):
    """Calibrated probability. Per-stat parameters (params['by_stat'][stat]) when present, else pooled."""
    if not params:
        return np.asarray(p, float)
    prm = (params.get("by_stat") or {}).get(stat, params) if stat else params
    return 1 / (1 + np.exp(-(prm["a"] * _logit(p) + prm.get("b", 0.0))))


def load() -> dict | None:
    return json.loads(PATH.read_text()) if PATH.exists() else None


def load_market() -> dict | None:
    return json.loads(MARKET_PATH.read_text()) if MARKET_PATH.exists() else None


def market_under(raw_under: float, stat: str, market: dict | None) -> float | None:
    """Calibrated P(under) at a market line, or None when there is no market fit for this stat."""
    if not market:
        return None
    key = stat if stat in market.get("by_stat", {}) else MARKET_FAMILY.get(stat)
    if key not in market.get("by_stat", {}):
        return None
    prm = market["by_stat"][key]
    return float(1 / (1 + np.exp(-(prm["a"] * _logit(raw_under) + prm.get("b", 0.0)))))


def save(params: dict, **meta) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({**params, **meta}, indent=1))
