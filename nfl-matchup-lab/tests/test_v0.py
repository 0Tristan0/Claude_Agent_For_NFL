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
        tm = sim.teams[t]
        assert (rec <= tm["completions"]).all() and (rtd <= tm["pass_tds"]).all()   # team = listed + other
        assert (sim.stat(qb, "completions") <= tm["completions"]).all()            # starter may leave early
        assert (sim.stat(qb, "pass_tds") <= tm["pass_tds"]).all()
        assert (sum(sim.stat(p, "rush_tds") for p in pids) <= tm["rush_tds"]).all()
        assert (sim.stat(qb, "pass_yds") == np.round(tm["pass_yds"])).mean() > 0.9
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


def test_qualification_prizepicks_only(v0_run):
    _, proj, sim = v0_run
    t = next(iter(proj["teams"]))
    pid = max((p for p, m in sim.meta.items() if m["team"] == t), key=lambda p: sim.stat(p, "rec_yds").mean())
    line = float(np.floor(np.quantile(sim.stat(pid, "rec_yds"), 0.33))) + 0.5     # over about two-thirds of sims
    base = dict(game_id=sim.game_id, player_id=pid, player="x", team=t, stat="rec_yds", line_type="standard", line=line)
    priced = dict(odds_kind="american", p_over_implied=0.53, p_under_implied=0.52, fair_over=0.505, fair_under=0.495)
    unpriced = dict(odds_kind="none", p_over_implied=np.nan, p_under_implied=np.nan, fair_over=np.nan, fair_under=np.nan)
    props = pd.DataFrame([dict(base, source="prizepicks", **priced), dict(base, source="prizepicks", **unpriced),
                          dict(base, source="fanduel", **priced)])
    legs = lineups.leg_table(props, {sim.game_id: sim}, None)
    over = legs[legs["side"] == "over"].reset_index(drop=True)
    pp_mkt, pp_only, book = over.loc[0], over.loc[1], over.loc[2]
    assert pp_mkt["basis"] == "model + market" and pp_only["basis"] == "model only"
    for leg in (pp_mkt, pp_only):
        assert leg["qualified_any"] == (not leg["far_from_line"] and leg["p_model"] >= min(leg[f"be{n}"] for n in lineups.SIZES) + 0.03
                                        and leg["market_edge"] >= 0.03)
    assert pp_only["market_edge"] == pytest.approx(pp_only["p_model"] - 0.5)       # PrizePicks line taken as a coin flip
    assert not book["lineup_ok"] and not book["qualified_any"]                      # not a PrizePicks line: reference only


def test_lineups_use_prizepicks_lines_and_always_show_best(v0_run):
    _, proj, sim = v0_run
    picks = []
    for t in proj["teams"]:
        pid = max((p for p, m in sim.meta.items() if m["team"] == t), key=lambda p: sim.stat(p, "rec_yds").mean())
        picks.append((t, pid, float(np.floor(np.quantile(sim.stat(pid, "rec_yds"), 0.45))) + 0.5))
    def props(source):
        return pd.DataFrame([dict(game_id=sim.game_id, player_id=pid, player=pid, team=t, stat="rec_yds", line=line,
                                  line_type="standard", source=source, odds_kind="none", p_over_implied=np.nan,
                                  p_under_implied=np.nan, fair_over=np.nan, fair_under=np.nan) for t, pid, line in picks])
    sims = {sim.game_id: sim}
    best = lineups.best_lineups(lineups.leg_table(props("prizepicks"), sims, None), sims, sizes=(2,))
    assert len(best[2]) >= 1                                    # something to show even when nothing clears the bar
    L = best[2][0]
    assert isinstance(L["recommended"], bool) and all(x["prizepicks"] for x in L["legs"])
    assert len({x["team"] for x in L["legs"]}) == 2 and L["flex_ev"] is None   # 2-pick: Power only
    none = lineups.best_lineups(lineups.leg_table(props("legacy_ev_py"), sims, None), sims, sizes=(2,))
    assert none[2] == []                                        # other sources never go into lineups


def test_quick_check_refuses_holdout(tables):
    from nfl_lab.backtest import quick
    with pytest.raises(ValueError, match="holdout"):
        quick.run(tables, 2025)


# ---- review follow-ups -----------------------------------------------------------------------------

def test_v0_projection_and_sim_have_no_lookahead(raw):
    """Fit, projection and a seeded sim are identical when all data unavailable at kickoff is removed."""
    import leakcheck
    from nfl_lab.data import pipeline
    s = raw["schedules"]
    gid = s[(s["season"] == 2023) & (s["week"] == 6)].iloc[3]["game_id"]
    ko = leakcheck.kickoffs(s)[gid]
    full = pipeline.build_tables(raw, first_season=2021)
    trunc = pipeline.build_tables(leakcheck.truncate(raw, ko, 2023), first_season=2021)
    a = P.project_game(full, gid, P.fit(full, 2023))
    b = P.project_game(trunc, gid, P.fit(trunc, 2023))
    for t in a["teams"]:
        ta, tb = a["teams"][t], b["teams"][t]
        assert (ta.plays, ta.pass_rate, ta.sack_rate) == (tb.plays, tb.pass_rate, tb.sack_rate)
        pd.testing.assert_frame_equal(ta.players.reset_index(drop=True), tb.players.reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
    sa, sb = S.simulate(a, 2000), S.simulate(b, 2000)
    for pid in sa.players:
        for k in ("rec_yds", "rush_yds", "rec_tds", "pass_yds"):
            assert np.array_equal(sa.players[pid][k], sb.players[pid][k])


def test_fit_ignores_target_season_and_later(tables, raw):
    from nfl_lab.data import pipeline
    cut = {k: (v[v["season"] < 2023] if "season" in v.columns else v) for k, v in raw.items()}
    cut["schedules"] = raw["schedules"][raw["schedules"]["season"] < 2023]
    cut["pbp"] = raw["pbp"][raw["pbp"]["season"] < 2023]
    cut["participation"] = raw["participation"][raw["participation"]["nflverse_game_id"].str[:4].astype(int) < 2023]
    cut["ftn"] = raw["ftn"][raw["ftn"]["season"] < 2023]
    small = pipeline.build_tables(cut, first_season=2021)
    f1, f2 = P.fit(tables, 2023), P.fit(small, 2023)
    assert np.allclose(f1.plays_coef, f2.plays_coef) and np.allclose(f1.pr_coef, f2.pr_coef)
    assert f1.league == pytest.approx(f2.league)


def test_live_run_refuses_holdout_season(tables, monkeypatch):
    from nfl_lab import config as cfg, v0run
    monkeypatch.setattr(v0run, "load_tables", lambda: dict(tables, game_lines=pd.DataFrame()))
    monkeypatch.setattr(cfg, "TEST_SEASON", 2023)
    gid = tables["games"][tables["games"]["season"] == 2023]["game_id"].iloc[0]
    with pytest.raises(ValueError, match="holdout"):
        v0run.run_games([gid], 500, write_pages=False)


def test_td_weights_keep_expected_share():
    rng = np.random.default_rng(3)
    events = rng.poisson([[0.3], [3.0], [6.0]], size=(3, 50000)).astype(float)
    base = np.array([0.05, 0.25, 0.70])
    w = S._per_event_weights(events, base)
    assert w.mean(1) == pytest.approx(base, rel=0.02)
    assert (w[:, events[0] == 0][0] == 0).all()                     # no catch, no TD


def test_carry_yards_keep_mean_and_breakaways():
    rng = np.random.default_rng(4)
    emp = np.r_[np.full(900, 3.0), np.full(70, -2.0), np.full(30, 45.0)]
    counts = np.full(20000, 15)
    y = S._carry_yards(rng, counts, ypc=5.0, mult=np.ones(20000), emp=emp, ypc_pos=float(emp.mean()))
    assert y.mean() / 15 == pytest.approx(5.0, rel=0.02)
    assert (y > 100).mean() > 0.01                                    # big games still happen


def test_rake_error_reports_infeasible_targets():
    h = np.tile(np.array([True, False, True, False]), (2, 2500))     # two identical legs
    w = payouts.rake(h, [0.60, 0.65])
    assert payouts.rake_error(h, w, [0.60, 0.65]) > 1e-3


def test_pick_rules_exchange_goblin_integer(v0_run):
    _, proj, sim = v0_run
    t = next(iter(proj["teams"]))
    pid = max((p for p, m in sim.meta.items() if m["team"] == t), key=lambda p: sim.stat(p, "receptions").mean())
    base = dict(game_id=sim.game_id, player_id=pid, player="x", team=t, stat="receptions", source="prizepicks",
                p_over_implied=0.30, p_under_implied=np.nan, fair_over=0.30, fair_under=0.70)
    props = pd.DataFrame([dict(base, line=1.5, odds_kind="exchange", line_type="standard"),
                          dict(base, line=1.5, odds_kind="exchange", line_type="goblin"),
                          dict(base, line=2.0, odds_kind="exchange", line_type="standard")])
    legs = lineups.leg_table(props, {sim.game_id: sim}, None)
    over = legs[legs["side"] == "over"].reset_index(drop=True)
    assert over.loc[0, "market_edge"] == pytest.approx(over.loc[0, "p_model"] - 0.30)     # no overround on exchange
    assert not over.loc[1, "qualified_any"]                                               # goblin never qualifies
    assert over.loc[2, "integer_push"] and not over.loc[2, "qualified_any"]               # integer count line
