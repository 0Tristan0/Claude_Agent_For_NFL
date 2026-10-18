"""Paths, season ranges and the TOML config files under config/."""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # nfl-matchup-lab/
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("NFL_LAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
TABLES_DIR = DATA_DIR / "tables"
MANIFEST = RAW_DIR / "manifest.jsonl"
BACKTESTS_DIR = ROOT / "backtests"
EXPERIMENTS_DIR = ROOT / "experiments"


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Read config/<name>.toml."""
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def model() -> dict:
    return load("model")


def seasons() -> dict:
    return model()["seasons"]


FIRST_SEASON: int = seasons()["first"]
VALIDATE_SEASONS: tuple[int, ...] = tuple(seasons()["validate"])
TEST_SEASON: int = seasons()["test"]
LIVE_SEASON: int = int(os.environ.get("NFL_LAB_LIVE_SEASON", seasons()["live"]))


def lag_hours(source: str) -> float:
    return float(model()["lags_hours"][source])


def config_hash(names=("model", "picks", "payouts", "stats")) -> str:
    blob = json.dumps({n: load(n) for n in names}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def ensure_dirs() -> None:
    for d in (RAW_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
