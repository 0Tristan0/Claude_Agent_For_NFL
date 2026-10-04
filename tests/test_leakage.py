"""Leakage prevention: shifted rolling windows, priors from earlier seasons only, train-only fitting."""
import numpy as np
import pandas as pd
import pytest

from nfl_agent.data import build
from nfl_agent.evaluation.walkforward import fold_masks
from nfl_agent.features.pregame import FeatureParams, _ew_state, build_feature_table, position_priors
from nfl_agent.models.baselines import lagged_outcomes
from nfl_agent.models.structured import StructuredModel, design

from .conftest import tiny_raw

FEATS = ["tpg_ew", "tshare_ew", "tshare_recent", "catch_rate_ew", "ypr_ew", "adot_ew", "snap_share_ew",
         "snap_share_recent", "explosive_rate_ew", "eff_games", "n_games_prior", "team_tpg_ew"]


def test_current_and_future_outcomes_never_change_pregame_features(demo_tables):
    pg, tg = demo_tables["player_games"], demo_tables["team_games"]
    base = build_feature_table(pg, tg).set_index(["player_id", "game_id"])
    pid = pg["player_id"].value_counts().index[0]
    rows = pg[pg["player_id"] == pid].sort_values("kickoff_utc")
    target = rows.iloc[len(rows) // 2]
    pert = pg.copy()
    later = (pert["player_id"] == pid) & (pert["kickoff_utc"] >= target["kickoff_utc"])
    pert.loc[later, ["targets", "receptions", "receiving_yards"]] = [30, 25, 400]   # absurd realised outcomes
    pert.loc[later, "snap_share"] = 1.0
    new = build_feature_table(pert, tg).set_index(["player_id", "game_id"])
    key = (pid, target["game_id"])
    for c in FEATS:
        assert base.loc[key, c] == pytest.approx(new.loc[key, c], nan_ok=True), f"{c} leaked the current game"
    earlier = rows[rows["kickoff_utc"] < target["kickoff_utc"]]["game_id"]
    for g in earlier:
        assert base.loc[(pid, g), "tpg_ew"] == pytest.approx(new.loc[(pid, g), "tpg_ew"])
    nxt = rows[rows["kickoff_utc"] > target["kickoff_utc"]]["game_id"].iloc[0]
    assert new.loc[(pid, nxt), "tpg_ew"] > base.loc[(pid, nxt), "tpg_ew"], "later games should see the change"


def test_rolling_window_is_shifted_exactly():
    t = build.build_all(tiny_raw())
    p = FeatureParams(player_half_life=2.0, k_tpg=0.0, season_discount=1.0)
    f = build_feature_table(t["player_games"], t["team_games"], p)
    p1 = f[f["player_id"] == "P1"].sort_values("week")
    d = 0.5 ** (1 / 2.0)
    # week 1: no history; week 2: only week 1 (4 targets); week 3: weeks 1-2 decayed
    assert p1["n_games_prior"].tolist() == [0, 1, 2]
    assert p1["tpg_ew"].iloc[1] == pytest.approx(4.0)
    assert p1["tpg_ew"].iloc[2] == pytest.approx((4 * d + 6) / (d + 1))
    assert p1["eff_games"].iloc[0] == 0.0


def test_season_discount_applied_once_per_boundary_even_with_unplayed_rows():
    vals = np.array([[1.0], [1.0], [1.0], [1.0]])
    valid = np.array([True, False, False, True])
    season = np.array([2020, 2021, 2021, 2021])
    out = _ew_state(vals, valid, season, decay=1.0, season_discount=0.5)
    assert out[:, 0].tolist() == [0.0, 0.5, 0.5, 0.5]


def test_lagged_outcomes_exclude_current_and_unplayed():
    df = pd.DataFrame({"player_id": ["a"] * 4, "kickoff_utc": pd.date_range("2030-01-01", periods=4, tz="UTC"),
                       "receiving_yards": [10.0, 20.0, np.nan, 40.0], "played": [True, True, False, True]})
    h = lagged_outcomes(df, "receiving_yards", n=3)
    assert np.isnan(h[0]).all()
    assert h[1, 0] == 10 and np.isnan(h[1, 1])
    assert h[3, :2].tolist() == [20.0, 10.0]           # most recent first, unplayed row skipped


def test_priors_use_only_earlier_seasons(demo_tables):
    pg = demo_tables["player_games"].copy()
    pg["played"] = pg["targets"].notna()
    s_last = pg["season"].max()
    a = position_priors(pg)
    pg.loc[pg["season"] == s_last, "receptions"] = pg.loc[pg["season"] == s_last, "targets"]  # 100% catch rate
    b = position_priors(pg)
    pa = a[a["season"] == s_last].set_index("position")["p_catch"]
    pb = b[b["season"] == s_last].set_index("position")["p_catch"]
    assert (pa == pb).all(), "season S priors must not use season S outcomes"


def test_walk_forward_folds_are_chronological():
    df = pd.DataFrame({"season": [2019, 2020, 2021, 2021, 2022], "played": [True, True, True, False, True]})
    tr, te = fold_masks(df, 2021, first_train=2020)
    assert df.loc[tr, "season"].tolist() == [2020]
    assert df.loc[te, "season"].tolist() == [2021]          # unplayed (inactive) row not scored
    assert not (tr & te).any()


def test_preprocessing_fitted_on_training_rows_only(demo_tables):
    f = build_feature_table(demo_tables["player_games"], demo_tables["team_games"])
    f["played"] = f["targets"].notna()
    seasons = sorted(f["season"].unique())
    train = f[f["season"] < seasons[-1]]
    m = StructuredModel(target_learner="glm").fit(train)
    imputer = m.t_model.named_steps["simpleimputer"]
    expected = design(train[train["played"]])[m.t_cols].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected, equal_nan=True)
    # Predicting on wildly different test data must not refit anything
    test = f[f["season"] == seasons[-1]].copy()
    test["tpg_ew"] = 50.0
    m.predict_params(test)
    assert np.allclose(imputer.statistics_, expected, equal_nan=True)
