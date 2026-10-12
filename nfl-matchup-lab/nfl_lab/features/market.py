"""Odds parsing and vig removal.

Two-way markets are devigged with four methods. Qualification uses the smallest edge across the
methods in config/picks.toml (multiplicative, power, Shin), so a pick never qualifies just because of
which method was chosen. One-sided prices use an assumed, stat-specific overround and are flagged.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.optimize import brentq


def american_to_prob(odds: float) -> float:
    """Implied probability (with vig) of American odds: -175 -> 0.6364, +145 -> 0.4082."""
    o = float(odds)
    if o == 0 or abs(o) < 100:
        raise ValueError(f"not American odds: {odds}")
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def prob_to_american(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("probability must be in (0, 1)")
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def parse_price(x) -> tuple[float | None, str]:
    """Parse a price cell. Returns (implied probability, kind).

    "-110", "+145", "145"  -> American odds          kind "american"
    "52c", "52¢"           -> exchange price, cents   kind "exchange"
    "0.52", "52%"          -> probability             kind "prob"
    blank / None           -> (None, "none")
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None, "none"
    s = str(x).strip().lower().replace(" ", "")
    if not s:
        return None, "none"
    if s.endswith("c") or s.endswith("¢"):
        return float(s[:-1]) / 100.0, "exchange"
    if s.endswith("%"):
        return float(s[:-1]) / 100.0, "prob"
    v = float(s)
    if 0 < v < 1:
        return v, "prob"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", s) and abs(v) >= 100:
        return american_to_prob(v), "american"
    raise ValueError(f"cannot parse price {x!r}")


def devig(probs, method: str = "power") -> np.ndarray:
    """Remove the vig from implied probabilities of mutually exclusive outcomes (they sum to > 1)."""
    q = np.asarray(probs, dtype=float)
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("implied probabilities must be in (0, 1)")
    total = q.sum()
    if method == "multiplicative":
        return q / total
    if method == "additive":
        out = q - (total - 1.0) / len(q)
        if np.any(out <= 0):
            raise ValueError("additive devig produced a non-positive probability")
        return out
    if method == "power":
        if abs(total - 1.0) < 1e-12:
            return q.copy()
        k = brentq(lambda k: np.sum(q ** k) - 1.0, 1e-6, 50.0)
        return q ** k
    if method == "shin":
        if abs(total - 1.0) < 1e-12:
            return q.copy()

        def shin_probs(z):
            return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / total) - z) / (2 * (1 - z))

        z = brentq(lambda z: shin_probs(z).sum() - 1.0, 0.0, 0.999)
        return shin_probs(z)
    raise ValueError(f"unknown devig method {method!r}")


def _none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def fair_two_way(p_side: float | None, p_other: float | None, method: str = "power",
                 one_sided_overround: float = 0.045) -> tuple[float | None, str]:
    """Fair probability of one side. Returns (prob, note).

    Both sides priced: devig. One side only: divide out an assumed overround (note "one-sided").
    Exchange prices that already sum to about 1 pass through unchanged.
    """
    p_side, p_other = _none(p_side), _none(p_other)
    if p_side is None and p_other is None:
        return None, "no price"
    if p_side is not None and p_other is not None:
        if p_side + p_other <= 1.0 + 1e-9:
            return p_side / (p_side + p_other), "two-way (no vig)"
        return float(devig([p_side, p_other], method)[0]), "two-way"
    if p_side is not None:
        return p_side / (1.0 + one_sided_overround), "one-sided"
    return 1.0 - p_other / (1.0 + one_sided_overround), "one-sided (other side)"


def min_edge(p_model: float, p_side, p_other, methods=("multiplicative", "power", "shin"),
             one_sided_overround: float = 0.045) -> float | None:
    """Smallest model-minus-market edge across devig methods (None when there is no market price)."""
    edges = []
    for m in methods:
        fair, _ = fair_two_way(p_side, p_other, m, one_sided_overround)
        if fair is None:
            return None
        edges.append(p_model - fair)
    return min(edges)
