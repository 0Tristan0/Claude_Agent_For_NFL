"""Central configuration: paths, seasons, model version, milestone thresholds.

Nothing in here is a secret. The application needs no API keys.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NFL_AGENT_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache" / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMPORTS_DIR = DATA_DIR / "user_imports"
DEMO_DIR = DATA_DIR / "cache" / "demo"
DB_PATH = DATA_DIR / "nfl_agent.sqlite"
DOCS_DIR = ROOT / "docs"

# Seasons ingested. Snap counts (our participation spine) start in 2012.
FIRST_SEASON = 2012
# The current season is ingested too; it is the "live" season.
LAST_SEASON = int(os.environ.get("NFL_AGENT_LAST_SEASON", "2026"))

# Evaluation design (see docs/METHODOLOGY.md).
FIRST_TRAIN_SEASON = 2014          # 2012-2013 serve only as feature warm-up history
VALIDATION_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]  # walk-forward model selection
FINAL_TEST_SEASON = 2025           # untouched until model choices were frozen

POSITIONS = ("WR", "TE")

RECEIVING_YARD_MILESTONES = (25, 50, 75, 100, 125, 150)
RECEPTION_MILESTONES = (2, 4, 6, 8)

MODEL_VERSION = "structured-gbm-v1.0"   # targets GBM + catch/yards GLMs, history+snap features
BASELINE_VERSION = "workload-baseline-v1.0"

# Monte-Carlo sizes. Evaluation uses fewer draws per forecast because it scores
# tens of thousands of forecasts; Monte-Carlo error at 2,000 draws is ~1
# percentage point at worst (p=0.5) and far smaller in the tails.
N_SIMS_APP = 20_000
N_SIMS_EVAL = 2_000
GLOBAL_SEED = 20240907

KICKOFF_TZ = "America/New_York"   # nflverse schedule gametime is Eastern


def ensure_dirs() -> None:
    for d in (CACHE_DIR, PROCESSED_DIR, IMPORTS_DIR, DEMO_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
