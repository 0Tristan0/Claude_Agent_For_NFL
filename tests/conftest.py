"""Shared fixtures. Tests use the synthetic DEMO fixture or tiny hand-made tables: no network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_agent import config
from nfl_agent.data import build, demo


@pytest.fixture(scope="session")
def demo_raw():
    # fixed calendar so tests are deterministic regardless of the date they run
    return demo.make_demo_raw(seasons=(2021, 2022, 2023, 2024), completed_weeks_last=4)


@pytest.fixture(scope="session")
def demo_tables(demo_raw):
    return build.build_all(demo_raw)


@pytest.fixture()
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.setattr(config, "IMPORTS_DIR", tmp_path / "imports")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite")
    return tmp_path


@pytest.fixture(scope="session")
def demo_service(demo_tables, tmp_path_factory):
    from nfl_agent.forecast.service import ForecastService
    d = tmp_path_factory.mktemp("svc")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DEMO_DIR", d / "demo")
    mp.setattr(config, "PROCESSED_DIR", d / "processed")
    meta = {"data_mode": "demo", "built_at": "2024-01-01T00:00:00+00:00", "data_cutoff": None, "n_player_games": 0}
    svc = ForecastService(dict(demo_tables, injuries=pd.DataFrame(), nextgen_receiving=pd.DataFrame()), meta)
    yield svc
    mp.undo()


def tiny_raw():
    """Two teams, one season, hand-made: used where exact numbers must be checked."""
    games = pd.DataFrame([
        dict(game_id=f"2030_{w:02d}_AAA_BBB", season=2030, game_type="REG", week=w, gameday=f"2030-09-{9 + 7 * (w - 1):02d}",
             gametime="13:00", away_team="AAA", home_team="BBB", away_score=20, home_score=17, roof="outdoors",
             away_rest=7.0, home_rest=7.0, location="Home", stadium="X", temp=np.nan, wind=np.nan,
             away_qb_name=None, home_qb_name=None) for w in (1, 2, 3)])
    players = pd.DataFrame([dict(gsis_id="P1", pfr_id="pf1", display_name="Rec One", position="WR", draft_round=1.0,
                                 draft_pick=10.0, rookie_season=2028, birth_date="2000-01-01"),
                            dict(gsis_id="P2", pfr_id="pf2", display_name="Rec Two", position="WR", draft_round=np.nan,
                                 draft_pick=np.nan, rookie_season=2030, birth_date="2000-01-01"),
                            dict(gsis_id="Q1", pfr_id="pq1", display_name="Qb One", position="QB", draft_round=1.0,
                                 draft_pick=1.0, rookie_season=2025, birth_date="1995-01-01")])
    stat = lambda pid, name, pos, w, tg, rc, yd, att=0: dict(  # noqa: E731
        player_id=pid, player_display_name=name, position=pos, season=2030, week=w, season_type="REG",
        game_id=f"2030_{w:02d}_AAA_BBB", team="AAA", opponent_team="BBB", targets=tg, receptions=rc,
        receiving_yards=yd, receiving_air_yards=tg * 10, receiving_yards_after_catch=yd // 3, receiving_tds=0,
        receiving_20=0, receiving_40=0, attempts=att, sacks_suffered=0)
    stats = pd.DataFrame([stat("Q1", "Qb One", "QB", w, 0, 0, 0, att=30) for w in (1, 2, 3)] +
                         [stat("P1", "Rec One", "WR", 1, 4, 3, 40), stat("P1", "Rec One", "WR", 2, 6, 4, 55),
                          stat("P1", "Rec One", "WR", 3, 8, 6, 90)])
    snap = lambda pfr, name, pos, w, n, pct: dict(  # noqa: E731
        game_id=f"2030_{w:02d}_AAA_BBB", season=2030, game_type="REG", week=w, player=name, pfr_player_id=pfr,
        position=pos, team="AAA", opponent="BBB", offense_snaps=n, offense_pct=pct)
    # P2 plays weeks 1 and 3 with no stat row (zero-target games); inactive in week 2.
    snaps = pd.DataFrame([snap("pq1", "Qb One", "QB", w, 60, 1.0) for w in (1, 2, 3)] +
                         [snap("pf1", "Rec One", "WR", w, 50, 0.83) for w in (1, 2, 3)] +
                         [snap("pf2", "Rec Two", "WR", 1, 10, 0.17), snap("pf2", "Rec Two", "WR", 3, 12, 0.2)])
    return {"schedules": games, "players": players, "player_stats": stats, "snap_counts": snaps}
