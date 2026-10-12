"""Checks on the real nflverse cache. Skipped when data/raw or data/tables is missing."""
import numpy as np
import pandas as pd
import pytest

import leakcheck
from nfl_lab import config

pytestmark = [pytest.mark.data]


@pytest.fixture(scope="module")
def real_pg(has_cache):
    if not has_cache:
        pytest.skip("no nflverse cache: run python -m nfl_lab ingest && python -m nfl_lab build")
    from nfl_lab.data import pipeline
    return pipeline.load_table("player_games")


def test_pbp_counts_match_official_stats(real_pg):
    from nfl_lab.data import qa
    m = qa.official_match(real_pg, [2023, 2024, 2025])
    assert len(m) == 3
    for col in ("targets", "carries", "receptions", "pass_att", "completions", "pass_tds"):
        assert (m[col] >= 0.995).all(), f"{col}: {m[col].tolist()}"
    assert (m["rec_yds"] >= 0.99).all() and (m["rush_yds"] >= 0.99).all()


@pytest.mark.slow
def test_route_proxy_quality_2024():
    from nfl_lab.data import qa
    q = qa.route_proxy_quality(2024).set_index("position")
    assert (q["corr"] > 0.9).all() and (q["mae"] < 4).all()


def _real_raw(seasons):
    from nfl_lab.data import pipeline
    return pipeline.load_raw(seasons)


@pytest.mark.slow
def test_raw_truncation_on_20_random_real_games(has_cache):
    """The oracle on real data: 20 random 2023-2024 regular-season games."""
    if not has_cache:
        pytest.skip("no cache")
    rng = np.random.default_rng(20261012)
    from nfl_lab.data import ingest
    sched = ingest.load("schedules")
    pool = sched[sched["season"].isin([2023, 2024]) & (sched["game_type"] == "REG") & (sched["week"] >= 2)]
    picks = pool.sample(20, random_state=int(rng.integers(1 << 30)))
    for season, grp in picks.groupby("season"):
        raw = _real_raw([season - 2, season - 1, season])
        raw["rosters"] = raw["rosters"][["season", "week", "gsis_id", "position", "team"]]
        for gid in grp["game_id"]:
            leakcheck.assert_no_lookahead(raw, gid, first_season=season - 2)
