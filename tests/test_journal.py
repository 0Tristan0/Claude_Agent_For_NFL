"""Forecast journal: immutable snapshots, revisions stored separately, results attached without edits."""
import hashlib
import sqlite3

import pandas as pd
import pytest

from nfl_agent import db
from nfl_agent.forecast import service as S


@pytest.fixture()
def fc_and_db(demo_service, tmp_path):
    svc = demo_service
    q = svc.features[svc.features["is_query"]].iloc[0]
    fc = svc.forecast(q["player_id"], q["game_id"], n_sims=2000, bootstrap=0)
    return svc, fc, tmp_path / "journal.sqlite"


def test_snapshot_cannot_be_updated_or_deleted(fc_and_db):
    _, fc, path = fc_and_db
    fid = S.save_forecast(fc, path)
    with db.session(path) as con:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            con.execute("UPDATE forecasts SET payload_json='{}' WHERE forecast_id=?", (fid,))
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            con.execute("DELETE FROM forecasts WHERE forecast_id=?", (fid,))


def test_refresh_stored_as_revision(fc_and_db):
    _, fc, path = fc_and_db
    a = S.save_forecast(fc, path)
    b = S.save_forecast(fc, path)
    with db.session(path) as con:
        rows = con.execute("SELECT forecast_id, revision, supersedes FROM forecasts ORDER BY revision").fetchall()
    assert [(r["forecast_id"], r["revision"], r["supersedes"]) for r in rows] == [(a, 0, None), (b, 1, a)]


def test_snapshot_unchanged_after_results_arrive(fc_and_db):
    svc, fc, path = fc_and_db
    fid = S.save_forecast(fc, path)
    with db.session(path) as con:
        before = con.execute("SELECT payload_json, payload_sha256 FROM forecasts WHERE forecast_id=?", (fid,)).fetchone()
    # pretend the game was played: inject the player's realised result into the service tables
    games = svc.games.copy()
    games.loc[games["game_id"] == fc.game_id, "completed"] = True
    tg = svc.tg.copy()
    tg.loc[tg["game_id"] == fc.game_id, "has_stats"] = True
    played = pd.DataFrame([{"player_id": fc.player_id, "game_id": fc.game_id, "targets": 7, "receptions": 5,
                            "receiving_yards": 66.0}])
    fake = type("Svc", (), {"games": games, "tg": tg, "pg": pd.concat([svc.pg, played], ignore_index=True),
                            "data_mode": "demo"})()
    assert S.record_results(fake, path) == 1
    with db.session(path) as con:
        after = con.execute("SELECT payload_json, payload_sha256 FROM forecasts WHERE forecast_id=?", (fid,)).fetchone()
        res = con.execute("SELECT * FROM results").fetchone()
    assert before["payload_json"] == after["payload_json"]
    assert after["payload_sha256"] == hashlib.sha256(after["payload_json"].encode()).hexdigest()
    assert res["receiving_yards"] == 66.0 and res["played"] == 1


def test_inactive_player_result_is_not_zero_yards(fc_and_db):
    svc, fc, path = fc_and_db
    S.save_forecast(fc, path)
    games = svc.games.copy()
    games.loc[games["game_id"] == fc.game_id, "completed"] = True
    tg = svc.tg.copy()
    tg.loc[tg["game_id"] == fc.game_id, "has_stats"] = True
    fake = type("Svc", (), {"games": games, "tg": tg, "pg": svc.pg, "data_mode": "demo"})()
    S.record_results(fake, path)
    with db.session(path) as con:
        res = con.execute("SELECT played, receiving_yards FROM results").fetchone()
    assert res["played"] == 0 and res["receiving_yards"] is None


def test_pregame_flag(fc_and_db):
    _, fc, path = fc_and_db
    S.save_forecast(fc, path)
    late = S.Forecast(**{**fc.__dict__, "forecast_created_at": "2099-01-01T00:00:00+00:00"})
    S.save_forecast(late, path)
    with db.session(path) as con:
        flags = [r["is_pregame"] for r in con.execute("SELECT is_pregame FROM forecasts ORDER BY revision")]
    early_expected = int(pd.Timestamp(fc.forecast_created_at) < pd.Timestamp(fc.kickoff_utc))
    assert flags == [early_expected, 0]
