"""Monte-Carlo simulation of the targets -> receptions -> yards chain.

Per player-game, with parameters estimated by models.structured:

  targets     T ~ NegativeBinomial(mean mu_t, size r(mu_t)) (gamma-Poisson mixture;
                  dispersion varies with the mean)
  receptions  C | T ~ BetaBinomial(T, p, rho)                (game-level catch-rate shock)
  yards       Y | C = s * sum_{i<=C} X_i,
              X_i ~ Gamma(shape a, mean m)  iid per-catch yards
              s   ~ Gamma(mean 1, var tau2) game-level efficiency shock shared by all
                    catches in the game (so catches are NOT treated as independent)

C <= T holds by construction. Y = 0 when C = 0. Simulated yards are rounded to
whole yards because recorded yards are integers.
"""
from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary identifiers (same inputs -> same draws)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def simulate(params: dict[str, np.ndarray], n_sims: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    mu_t = np.asarray(params["mu_t"], float)[:, None]
    n = mu_t.shape[0]
    r = np.broadcast_to(np.asarray(params["nb_size"], float).reshape(-1, 1) if np.ndim(params["nb_size"]) else
                        np.asarray(params["nb_size"], float), (n, 1))
    lam = mu_t * rng.gamma(r, 1.0 / r, size=(n, n_sims))
    T = rng.poisson(lam)
    p = np.clip(np.asarray(params["p_catch"], float)[:, None], 1e-4, 1 - 1e-4)
    rho = float(params["bb_rho"])
    if rho > 1e-6:
        conc = 1.0 / rho - 1.0
        pg = rng.beta(p * conc, (1 - p) * conc, size=(n, n_sims))
    else:
        pg = np.broadcast_to(p, (n, n_sims))
    C = rng.binomial(T, pg)
    m = np.asarray(params["m_ypr"], float)[:, None]
    a = float(params["gamma_shape"])
    tau2 = float(params["tau2"])
    base = rng.gamma(np.maximum(C * a, 1e-12), m / a)
    base = np.where(C > 0, base, 0.0)
    if tau2 > 1e-8:
        s = rng.gamma(1.0 / tau2, tau2, size=(n, n_sims))
        base = base * s
    Y = np.round(base)
    return {"targets": T, "receptions": C, "yards": Y}


QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


def summarize(samples: np.ndarray, thresholds) -> dict[str, np.ndarray]:
    """Row-wise summary of simulated draws (rows = forecasts, cols = draws)."""
    q = np.quantile(samples, QUANTILES, axis=1)
    out = {f"q{int(round(x * 1000)):03d}": q[i] for i, x in enumerate(QUANTILES)}
    out["mean"] = samples.mean(axis=1)
    out["median"] = q[QUANTILES.index(0.50)]
    for t in thresholds:
        out[f"p_ge_{t}"] = (samples >= t).mean(axis=1)
    return out
