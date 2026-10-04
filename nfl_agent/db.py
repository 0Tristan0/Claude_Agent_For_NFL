"""SQLite storage for metadata: ingestion log and the forecast journal.

Forecast snapshots are immutable: SQL triggers reject UPDATE and DELETE on the
`forecasts` table. Revised forecasts are new rows that reference the row they
supersede. Actual results live in a separate table.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL,
    season INTEGER,
    url TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,          -- UTC ISO-8601
    status TEXT NOT NULL,                -- ok | cached | failed | not_published
    source_last_updated TEXT,            -- from nflverse timestamp.json, if available
    n_rows INTEGER,
    sha256 TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,            -- UTC ISO-8601
    player_id TEXT NOT NULL,
    player_name TEXT,
    game_id TEXT NOT NULL,
    kickoff_utc TEXT,
    is_pregame INTEGER NOT NULL,         -- created before kickoff?
    model_version TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    revision INTEGER NOT NULL,           -- 0 = first snapshot for player/game/model
    supersedes TEXT,                     -- forecast_id of the previous revision
    scenario TEXT NOT NULL DEFAULT 'base',
    data_mode TEXT NOT NULL,             -- real | demo
    payload_json TEXT NOT NULL,          -- full numeric forecast
    payload_sha256 TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS forecasts_no_update
BEFORE UPDATE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecast snapshots are immutable; store a revision instead');
END;

CREATE TRIGGER IF NOT EXISTS forecasts_no_delete
BEFORE DELETE ON forecasts
BEGIN
    SELECT RAISE(ABORT, 'forecast snapshots are immutable and cannot be deleted');
END;

CREATE TABLE IF NOT EXISTS results (
    player_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    played INTEGER NOT NULL,             -- 1 if player took an offensive snap
    targets INTEGER,
    receptions INTEGER,
    receiving_yards REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (player_id, game_id)
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


@contextmanager
def session(path: Path | str | None = None):
    con = connect(path)
    try:
        yield con
        con.commit()
    finally:
        con.close()
