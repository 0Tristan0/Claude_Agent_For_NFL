"""Baselines, structured model, scoring rules, calibration and bootstrap."""
import numpy as np
import pandas as pd
import pytest

from nfl_agent import config
from nfl_agent.evaluation import metrics as M
from nfl_agent.evaluation import walkforward as W
from nfl_agent.features.pregame import build_feature_table
from nfl_agent.models.baselines import RollingEmpiricalBaseline, WorkloadBaseline
from nfl_agent.models.structured import StructuredModel, fit_dispersion_curve, nb_size_for

YT = config.RECEIVING_YARD_MILESTONES


@pytest.fixture(scope="module")
def feats(demo_tables):
    f = build_feature_table(demo_tables["player_games"], demo_tables["team_games"])
    f["played"] = f["targets"].notna()
    return f


def test_empirical_baseline_shrinkage():
    train = pd.DataFrame({"position": ["WR"] * 10, "receiving_yards": [0, 10, 20, 30, 40, 50, 60, 70, 80, 150.0],
                          "played": True})
    b = RollingEmpiricalBaseline(k=6).fit(train, "receiving_yards", (100,))
    p0 = 0.1
    hist = np.full((2, 16), np.nan)
    hist[1, :16] = 120.0                         # 16 straight 100+ games
    r = b.predict(pd.DataFrame({"position": ["WR", "WR"]}), hist)
    assert r["p_ge_100"][0] == pytest.approx(p0)                    # no history -> position rate
    assert r["p_ge_100"][1] == pytest.approx((16 + 6 * p0) / (16 + 6))


def test_workload_baseline_probabilities_and_pit(feats):
    seasons = sorted(feats["season"].unique())
    tr, te = feats[(feats["season"] < seasons[-1]) & feats["played"]], feats[(feats["season"] == seasons[-1]) & feats["played"]]
    for normal in (False, True):
        b = WorkloadBaseline(normal=normal).fit(tr, YT)
        r = b.predict(te)
        ps = np.array([r[f"p_ge_{t}"] for t in YT])
        assert ((ps >= 0) & (ps <= 1)).all() and (np.diff(ps, axis=0) <= 1e-12).all()
        pit = b.pit(te, te["receiving_yards"].to_numpy(float))
        assert ((pit >= 0) & (pit <= 1)).all()


def test_structured_model_outputs(feats):
    seasons = sorted(feats["season"].unique())
    tr, te = feats[(feats["season"] < seasons[-1]) & feats["played"]], feats[(feats["season"] == seasons[-1]) & feats["played"]]
    for learner in ("gbm", "glm"):
        m = StructuredModel(target_learner=learner).fit(tr)
        r = m.simulate(te.head(200), 1000, seed=1, actual={"yards": te.head(200)["receiving_yards"].to_numpy(float)})
        ps = np.array([r[f"yards_p_ge_{t}"] for t in YT])
        assert ((ps >= 0) & (ps <= 1)).all() and (np.diff(ps, axis=0) <= 0).all()
        assert ((r["yards_pit"] >= 0) & (r["yards_pit"] <= 1)).all()
        assert (r["mu_t"] > 0).all() and ((r["p_catch"] > 0) & (r["p_catch"] < 1)).all()
        c = m.target_contributions(te.head(1))
        assert len(c) == len(m.t_cols) and np.isfinite(c["contribution"]).all()


def test_dispersion_curve_recovers_mean_dependent_overdispersion():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 10, 200000)
    size = 1.0 / (0.5 * mu ** -1.0)               # 1/r = 0.5 / mu
    y = rng.poisson(mu * rng.gamma(size, 1 / size))
    c0, c1 = fit_dispersion_curve(mu, y)
    assert c1 == pytest.approx(-1.0, abs=0.2)
    assert nb_size_for(np.array([4.0]), (c0, c1))[0] == pytest.approx(8.0, rel=0.25)


def test_scores_and_calibration_basics():
    y = np.array([0, 1, 1, 0.0])
    assert M.brier(np.array([0, 1, 1, 0.0]), y).sum() == 0
    assert M.log_loss(np.array([0.5] * 4), y).mean() == pytest.approx(np.log(2))
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    o = (rng.uniform(0, 1, 20000) < p).astype(float)
    cal = M.calibration_table(p, o)
    assert cal["n"].sum() == 20000
    assert np.allclose(cal["mean_pred"], cal["observed"], atol=0.03)


def test_cluster_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(2)
    vals = pd.DataFrame({"d": rng.normal(-0.5, 1, 3000)})
    cl = pd.Series(np.repeat(np.arange(300), 10))
    r = M.cluster_bootstrap(vals, cl, n_boot=300)
    row = r.iloc[0]
    assert row["ci_low"] < row["estimate"] < row["ci_high"] < 0


def test_pit_coverage_is_nominal_for_correct_model():
    """A correctly specified discrete forecast should give nominal randomized-PIT coverage."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(0.2, 6, 4000)
    y = rng.poisson(lam)
    draws = rng.poisson(lam[:, None], size=(4000, 3000))
    pit = (draws < y[:, None]).mean(1) + rng.random(4000) * (draws == y[:, None]).mean(1)
    for lvl in (0.5, 0.8, 0.95):
        assert np.mean(np.abs(pit - 0.5) <= lvl / 2) == pytest.approx(lvl, abs=0.03)


def test_walkforward_end_to_end_on_demo(feats):
    seasons = sorted(feats["season"].unique())
    preds = W.run(feats, [seasons[-1]], W.DEFAULT_SPECS, n_sims=500, first_train=seasons[0])
    sc = W.score(preds, n_boot=50)
    s = sc["summary"]
    assert set(preds["model"]) == {sp.name for sp in W.DEFAULT_SPECS}
    assert s["value"].notna().any()
    assert {"MAE yards", "LogLoss >=100"} <= set(sc["diff_vs_B3"]["metric"])
