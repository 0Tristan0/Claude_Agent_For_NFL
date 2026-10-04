"""Export for the Receiving Line Checker page: curve math, staleness, and the exported structure."""
import numpy as np
import pandas as pd
import pytest

from nfl_agent.forecast import slate


def test_ge_curve_matches_direct_counts():
    rng = np.random.default_rng(0)
    x = rng.poisson(40, 5000)
    c = slate.ge_curve(x, kmax=60)
    assert len(c) == 62
    for k in (0, 10, 40, 61):
        assert c[k] == pytest.approx((x >= k).mean())
    assert c[0] == 1.0 and all(a >= b for a, b in zip(c, c[1:]))


def test_trim_drops_only_trailing_zeros():
    assert slate._r([1.0, 0.5, 0.0, 0.0]) == [1.0, 0.5]
    assert slate._r([0.0]) == [0.0]
    assert slate._r([0.31234, 0.0, 0.1]) == [0.312, 0.0, 0.1]


def test_line_probabilities_from_curve_sum_to_one():
    """The page reads P(more) for a 64.5 line as P(Y >= 65); check the identity on a known curve."""
    y = np.array([0, 10, 30, 65, 65, 70, 120])
    c = slate.ge_curve(y, kmax=200)
    p_more = c[65]                              # line 64.5
    assert p_more == pytest.approx(4 / 7)
    p_more_int, p_push = c[66], c[65] - c[66]   # integer line 65
    assert p_more_int + p_push + (1 - c[65]) == pytest.approx(1.0)


def test_stale_counts_unplayed_and_unscored_earlier_games():
    k = lambda d: pd.Timestamp(d, tz="UTC")  # noqa: E731
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3", "g4"], "home_team": ["A", "B", "A", "A"],
                          "away_team": ["X", "A", "Y", "Z"],
                          "kickoff_utc": [k("2030-09-07"), k("2030-09-14"), k("2030-09-21"), k("2030-09-28")]})
    tg = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "has_stats": [True, False, False]})
    assert slate.stale_games(games, tg, "A", k("2030-09-28")) == 2     # g2 (played, no stats) + g3 (not yet played)
    assert slate.stale_games(games, tg, "A", k("2030-09-14")) == 0
    assert slate.stale_games(games, tg, "X", k("2030-09-28")) == 0


def test_build_slate_structure(demo_service):
    svc = demo_service
    q = svc.features[svc.features["is_query"]]
    now = q["kickoff_utc"].min() - pd.Timedelta(hours=1)
    d = slate.build_slate(svc, days=8, bootstrap=3, n_sims=2000, boot_sims=500, now=now)
    assert d["data_mode"] == "demo" and d["players"] and d["games"]
    keys = {(p["id"], p["game"]) for p in d["players"]}
    assert len(keys) == len(d["players"]), "player+game must be unique"
    for p in d["players"][:10]:
        for stat in ("yds", "rec"):
            c = p[stat]
            for name in ("ge", "lo", "hi", "base"):
                v = c[name]
                assert all(0 <= x <= 1 for x in v)
                assert all(a >= b - 1e-9 for a, b in zip(v, v[1:])), f"{stat}.{name} must not increase"
            assert c["ge"][0] == pytest.approx(1.0)
        assert p["evidence"] in ("limited", "moderate", "substantial")
        assert pd.Timestamp(p["kickoff"]) > now
