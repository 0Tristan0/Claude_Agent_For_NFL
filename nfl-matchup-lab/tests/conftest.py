import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synthetic  # noqa: E402

from nfl_lab import config  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return synthetic.build_raw()


@pytest.fixture(scope="session")
def tables(raw):
    from nfl_lab.data import pipeline
    return pipeline.build_tables(raw, first_season=2021)


@pytest.fixture(scope="session")
def has_cache():
    return config.MANIFEST.exists() and (config.TABLES_DIR / "player_games.parquet").exists()


def game(raw, season, week, k=3):
    """The k-th game of a synthetic week (k=3 is the Sunday 4:25 slot)."""
    s = raw["schedules"]
    return s[(s["season"] == season) & (s["week"] == week)].iloc[k]


def kickoff(raw, gid):
    from leakcheck import kickoffs
    return kickoffs(raw["schedules"])[gid]


def ts(x):
    return pd.Timestamp(x)
