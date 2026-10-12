"""v0 model: simulation sanity, payout and break-even math, qualification rules, holdout guard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_lab.features import bundle
from nfl_lab.picks import calibration, lineups, payouts
from nfl_lab.project import v0 as P
from nfl_lab.sim import v0 as S


@pytest.fixture(scope="module")
def v0_run(tables):
    T = dict(tables)
    fit_ = P.fit(T, 2023)
    eng = bundle.engines(T)
    g = T["games"][(T["games"]["season"] == 2023) & (T["games"]["week"] == 5)].iloc[3]
    proj = P.project_game(T, g["game_id"], fit_, eng=eng)
    sim = S.simulate(proj, 20000)
    return fit_, proj, sim


def test_fit_uses_only_earlier_seasons(tables):
    fit_ = P.fit(tables, 2023)
    assert max(fit_.seasons) <= 2022
    assert P.fit_seasons(2026)[-1] == 2024 and P.fit_seasons(2025)[-1] == 2024      # 2025 holdout never fitted


def test_sim_points_match_market(v0_run):
    _, proj, sim = v0_run
    for t, tp in proj["teams"].items():
        se = sim.teams[t]["points"].std() / np.sqrt(sim.n)
        assert abs(sim.teams[t]["points"].mean() - tp.implied_points) < 4 * se + 0.05


def test_sim_volume_matches_projection(v0_run):
    _, proj, sim = v0_run
    for t, tp in proj["teams"].items():
        assert abs(sim.teams[t]["plays"].mean() - tp.plays) < 0.6


def test_player_means_follow_shares(v0_run):
    fit_, proj, sim = v0_run
    for t, tp in proj["teams"].items():
        team_targets = sim.teams[t]["pass_att"].mean() * fit_.league["targets_per_att"]
        for _, r in tp.players.iterrows():
            if r["target_share"] > 0.05:
                got = sim.stat(r["player_id"], "targets").mean()
                assert got == pytest.approx(r["target_share"] * team_targets, rel=0.06)


def test_sim_identities_and_ranges(v0_run):
    _, proj, sim = v0_run
    for t in proj["teams"]:
        pids = [p for p, m in sim.meta.items() if m["team"] == t]
        qb = [p for p in pids if sim.meta[p]["starter_qb"]][0]
        rec = sum(sim.stat(p, "receptions") for p in pids)
        rtd = sum(sim.stat(p, "rec_tds") for p in pids)
        assert (rec <= sim.stat(qb, "completions")).all()        # completions = listed catches + other
        assert (rtd <= sim.stat(qb, "pass_tds")).all()
        for p in pids:
            assert (sim.stat(p, "receptions") <= sim.stat(p, "targets")).all()
        for p in pids:
            for s in ("rec_yds", "receptions", "rush_yds", "anytime_td"):
                for line in (0.5, 4.5, 30.5):
                    q = S.p_over(sim, p, s, line)
                    assert 0 <= q["over"] <= 1 and abs(q["over"] + q["under"] + q["push"] - 1) < 1e-9


def test_p_over_non_increasing_in_line(v0_run):
    _, _, sim = v0_run
    pid = max(sim.meta, key=lambda p: sim.stat(p, "rec_yds").mean())
    ps = [S.p_over(sim, pid, "rec_yds", l)["over"] for l in np.arange(0.5, 150, 5)]
    assert all(a >= b for a, b in zip(ps, ps[1:]))


def test_seeded_reproducible(v0_run):
    _, proj, sim = v0_run
    again = S.simulate(proj, 20000)
    pid = next(iter(sim.players))
    assert np.array_equal(sim.stat(pid, "rec_yds"), again.stat(pid, "rec_yds"))


def test_joint_never_above_weakest_leg_and_qb_wr_stack_positive(v0_run):
    _, proj, sim = v0_run
    t = next(iter(proj["teams"]))
    pids = [p for p, m in sim.meta.items() if m["team"] == t]
    qb = [p for p in pids if sim.meta[p]["starter_qb"]][0]
    wr = max((p for p in pids if p != qb), key=lambda p: sim.stat(p, "rec_yds").mean())
    a = sim.stat(qb, "pass_yds") > np.median(sim.stat(qb, "pass_yds"))
    b = sim.stat(wr, "rec_yds") > np.median(sim.stat(wr, "rec_yds"))
    joint = (a & b).mean()
    assert joint <= min(a.mean(), b.mean())
    assert joint > a.mean() * b.mean()                 # same-team QB and top receiver overs move together
    nested = (sim.stat(wr, "rec_yds") > 20.5) & (sim.stat(wr, "rec_yds") > 40.5)
    assert nested.mean() == (sim.stat(wr, "rec_yds") > 40.5).mean()


def test_breakevens_match_legacy_values():
    legacy = json.loads((Path(__file__).resolve().parents[1] / "src" / "ev.json").read_text())["be"]
    be = payouts.breakevens()
    for k, v in legacy.items():
        assert be[k] == pytest.approx(v, abs=1e-4), k


def test_entry_ev_math():
    rng = np.random.default_rng(0)
    H = rng.random((3, 100000)) < np.array([[0.6], [0.6], [0.6]])
    ev = payouts.entry_ev(H)
    assert ev["power_ev"] == pytest.approx(5 * ev["p_all"])
    assert ev["p_all"] == pytest.approx(0.216, abs=0.005)
    certain = payouts.entry_ev(np.ones((4, 10), bool))
    assert certain["power_ev"] == pytest.approx(10) and certain["flex_ev"] == pytest.approx(5)
    assert certain["min_power_mult"] == pytest.approx(1)


def test_rake_matches_targets_and_keeps_correlation():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(50000)
    H = np.vstack([z + rng.standard_normal(50000) * 0.8 > 0, z + rng.standard_normal(50000) * 0.8 > 0.2])
    w = payouts.rake(H, [0.55, 0.45])
    assert (w[H[0]].sum(), w[H[1]].sum()) == (pytest.approx(0.55, abs=1e-4), pytest.approx(0.45, abs=1e-4))
    joint = w[H[0] & H[1]].sum()
    assert joint > 0.55 * 0.45 + 0.05                  # correlation preserved


def test_calibration_shrinks_extremes():
    p = np.r_[np.full(2000, 0.8), np.full(2000, 0.2)]
    rng = np.random.default_rng(2)
    y = np.r_[rng.random(2000) < 0.7, rng.random(2000) < 0.3].astype(float)
    c = calibration.fit(p, y)
    assert 0.6 < calibration.apply(0.8, c) < 0.8 and 0.2 < calibration.apply(0.2, c) < 0.4


def test_qualification_needs_market_and_breakeven(v0_run):
    _, proj, sim = v0_run
    t = next(iter(proj["teams"]))
    pid = max((p for p, m in sim.meta.items() if m["team"] == t), key=lambda p: sim.stat(p, "rec_yds").mean())
    med = float(np.median(sim.stat(pid, "rec_yds")))
    base = dict(game_id=sim.game_id, player_id=pid, player="x", team=t, stat="rec_yds", line_type="standard")
    props = pd.DataFrame([
        dict(base, line=med - 25.5, source="book", odds_kind="american", p_over_implied=0.53, p_under_implied=0.52,
             fair_over=0.505, fair_under=0.495),                       # model far above market: big edge
        dict(base, line=med - 25.5, source="prizepicks", odds_kind="none", p_over_implied=np.nan,
             p_under_implied=np.nan, fair_over=np.nan, fair_under=np.nan),   # no market price
    ])
    legs = lineups.leg_table(props, {sim.game_id: sim}, None)
    over = legs[legs["side"] == "over"]
    book, pp = over[over["source"] == "book"].iloc[0], over[over["source"] == "prizepicks"].iloc[0]
    assert not pp["qualified_any"]                                     # never qualifies without a market price
    if not book["far_from_line"]:
        assert book["qualified_any"] == (book["p_model"] >= book["be3"] + 0.03 and book["market_edge"] >= 0.03)


def test_quick_check_refuses_holdout(tables):
    from nfl_lab.backtest import quick
    with pytest.raises(ValueError, match="holdout"):
        quick.run(tables, 2025)
