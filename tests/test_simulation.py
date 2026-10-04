"""Simulation invariants: receptions <= targets, valid & monotone probabilities, reproducibility."""
import numpy as np
import pytest

from nfl_agent import config
from nfl_agent.models import simulate as sim


def params(n=50, mu=6.0, size=8.0, p=0.65, m=12.0):
    return dict(mu_t=np.full(n, mu), nb_size=np.full(n, size), p_catch=np.full(n, p), m_ypr=np.full(n, m),
                bb_rho=0.02, gamma_shape=1.4, tau2=0.01)


def test_receptions_never_exceed_targets_and_yards_consistent():
    d = sim.simulate(params(), 5000, np.random.default_rng(1))
    assert (d["receptions"] <= d["targets"]).all()
    assert (d["receptions"] >= 0).all()
    assert (d["yards"][d["receptions"] == 0] == 0).all()
    assert (d["yards"] >= 0).all()


def test_probabilities_valid_and_monotone():
    d = sim.simulate(params(n=5), 20000, np.random.default_rng(2))
    s = sim.summarize(d["yards"], config.RECEIVING_YARD_MILESTONES)
    ps = np.array([s[f"p_ge_{t}"] for t in config.RECEIVING_YARD_MILESTONES])
    assert ((ps >= 0) & (ps <= 1)).all()
    assert (np.diff(ps, axis=0) <= 0).all(), "P(Y >= t) must not increase with t"
    q = np.array([s[k] for k in ("q025", "q100", "q250", "q500", "q750", "q900", "q975")])
    assert (np.diff(q, axis=0) >= 0).all()


def test_reproducible_with_fixed_seed():
    a = sim.simulate(params(), 2000, np.random.default_rng(42))
    b = sim.simulate(params(), 2000, np.random.default_rng(42))
    c = sim.simulate(params(), 2000, np.random.default_rng(43))
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["yards"], c["yards"])
    assert sim.stable_seed("x", "y", 1) == sim.stable_seed("x", "y", 1)
    assert sim.stable_seed("x", "y", 1) != sim.stable_seed("x", "y", 2)


def test_moments_match_specification():
    mu, size, p, m = 7.0, 6.0, 0.6, 11.0
    d = sim.simulate(params(n=1, mu=mu, size=size, p=p, m=m) | {"bb_rho": 0.0, "tau2": 0.0}, 400000,
                     np.random.default_rng(3))
    T, C, Y = d["targets"][0], d["receptions"][0], d["yards"][0]
    assert T.mean() == pytest.approx(mu, rel=0.01)
    assert T.var() == pytest.approx(mu + mu ** 2 / size, rel=0.03)      # negative binomial variance
    assert C.mean() == pytest.approx(mu * p, rel=0.01)
    assert Y.mean() == pytest.approx(mu * p * m, rel=0.01)


def test_service_forecast_reproducible_and_complete(demo_service):
    svc = demo_service
    q = svc.features[svc.features["is_query"]].iloc[0]
    a = svc.forecast(q["player_id"], q["game_id"], n_sims=4000, bootstrap=5)
    b = svc.forecast(q["player_id"], q["game_id"], n_sims=4000, bootstrap=5)
    assert a.yards == b.yards and a.milestone_probabilities == b.milestone_probabilities
    assert a.milestone_probability_ranges == b.milestone_probability_ranges
    ps = [a.milestone_probabilities[str(t)] for t in config.RECEIVING_YARD_MILESTONES]
    assert all(0 <= p <= 1 for p in ps) and all(x >= y for x, y in zip(ps, ps[1:]))
    for lo, hi in a.milestone_probability_ranges.values():
        assert 0 <= lo <= hi <= 1
    assert a.conditional_on_participation is True
    assert a.yards["interval_50"][0] <= a.yards["median"] <= a.yards["interval_50"][1]
    assert a.yards["interval_95"][0] <= a.yards["interval_80"][0] <= a.yards["interval_50"][0]
    assert sum(a.target_pmf.values()) == pytest.approx(1.0)
    assert a.data_mode == "demo"
    for key in ("model_version", "data_cutoff", "forecast_created_at", "game_id", "player_id"):
        assert getattr(a, key)


def test_scenario_scales_targets_and_is_labelled(demo_service):
    svc = demo_service
    q = svc.features[svc.features["is_query"]].iloc[0]
    base = svc.forecast(q["player_id"], q["game_id"], n_sims=4000, bootstrap=0)
    low = svc.forecast(q["player_id"], q["game_id"], playing_time_multiplier=0.5, n_sims=4000, bootstrap=0)
    assert low.model_params["expected_targets"] == pytest.approx(0.5 * base.model_params["expected_targets"])
    assert low.scenario["is_base"] is False and "HYPOTHETICAL" in low.scenario["assumption"]
    assert low.milestone_probabilities["50"] <= base.milestone_probabilities["50"]
