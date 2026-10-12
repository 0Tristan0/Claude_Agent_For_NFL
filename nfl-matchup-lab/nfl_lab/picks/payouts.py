"""PrizePicks payout tables (config/payouts.toml), break-evens and entry expected value."""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import brentq

from .. import config


def tables() -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    c = config.load("payouts")
    power = {int(k): float(v) for k, v in c["power"].items()}
    flex = {int(n): {int(k): float(v) for k, v in t.items()} for n, t in c["flex"].items()}
    return power, flex


def power_breakeven(n: int) -> float:
    """Per-leg hit rate at which an n-pick Power entry breaks even (all legs equally likely)."""
    power, _ = tables()
    return (1.0 / power[n]) ** (1.0 / n)


def flex_ev_equal(n: int, p: float) -> float:
    _, flex = tables()
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) * m for k, m in flex[n].items())


def flex_breakeven(n: int) -> float:
    return brentq(lambda p: flex_ev_equal(n, p) - 1.0, 0.3, 0.95)


def breakevens() -> dict[str, float]:
    power, flex = tables()
    out = {f"power{n}": power_breakeven(n) for n in power}
    out.update({f"flex{n}": flex_breakeven(n) for n in flex})
    return out


def rake(hits: np.ndarray, target_p, iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Sim weights (summing to 1) under which each leg hits with its target probability.

    Iterative proportional fitting over the legs: the joint structure of the sims (which legs hit
    together) is kept as much as possible while the marginals move to the calibrated chances."""
    m, n = hits.shape
    w = np.full(n, 1.0 / n)
    tp = np.clip(np.asarray(target_p, float), 1e-4, 1 - 1e-4)
    for _ in range(iters):
        worst = 0.0
        for i in range(m):
            h = hits[i]
            p = w[h].sum()
            if p <= 0 or p >= 1:
                continue
            w = np.where(h, w * tp[i] / p, w * (1 - tp[i]) / (1 - p))
            worst = max(worst, abs(p - tp[i]))
        if worst < tol:
            break
    return w / w.sum()


def rake_error(hits: np.ndarray, w: np.ndarray, target_p) -> float:
    """Largest gap between a leg's weighted hit rate and its target after raking."""
    return float(np.max(np.abs((w[None, :] * hits).sum(1) - np.asarray(target_p, float))))


def entry_ev(hits: np.ndarray, scale: float = 1.0, weights: np.ndarray | None = None) -> dict:
    """Expected return per $1 from a (legs x sims) boolean hit matrix taken from the same sims.

    Power pays only when every leg hits; Flex pays by the number of hits. Both come from the joint
    outcomes, never from multiplying leg probabilities. `scale` multiplies the all-hit probability
    (calibration of the legs; 1.0 = raw sims); Flex hit-count probabilities are left as simulated."""
    power, flex = tables()
    n = hits.shape[0]
    k = hits.sum(0)
    w = np.full(hits.shape[1], 1.0 / hits.shape[1]) if weights is None else weights
    p_all = float(min(1.0, w[k == n].sum() * scale))
    out = dict(n=n, p_all=p_all, power_mult=power.get(n), power_ev=p_all * power[n] if n in power else None)
    if n in flex:
        dist = np.bincount(k, weights=w, minlength=n + 1)
        out["flex_ev"] = float(sum(dist[j] * m for j, m in flex[n].items()))
        out["flex_dist"] = dist
    else:
        out["flex_ev"] = None
    out["p_independent"] = float(np.prod((w[None, :] * hits).sum(1)) * scale)
    out["min_power_mult"] = (1.0 / p_all) if p_all > 0 else float("inf")
    if out["flex_ev"]:
        out["min_flex_scale"] = 1.0 / out["flex_ev"]           # payouts must be at least this x the table
    se = np.sqrt(p_all * (1 - p_all) / hits.shape[1]) * (power[n] if n in power else 0)
    out["power_ev_se"] = float(se)
    return out
