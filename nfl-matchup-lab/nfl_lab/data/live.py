"""Live-only inputs: who is out (official injury report + your overrides). Never used in backtests,
because the injury file's publication times can't be verified for 2025 onward.

data/overrides.csv (optional): player,team,status  with status out | in. "in" forces a player onto the
active list (e.g. a call-up), "out" removes him.
"""
from __future__ import annotations

import pandas as pd

from .. import config, teams
from . import ingest

OVERRIDES = config.DATA_DIR / "overrides.csv"
OUT_STATUSES = {"Out", "Doubtful"}


def injury_report(season: int, week: int) -> pd.DataFrame:
    inj = ingest.load("injuries", seasons=[season])
    if inj.empty:
        return inj
    inj = teams.normalize(inj, ["team"])
    return inj[(inj["week"] == week)][["team", "gsis_id", "full_name", "position", "report_status",
                                       "report_primary_injury", "practice_status"]]


def out_ids(season: int, week: int, team_codes) -> tuple[set, pd.DataFrame]:
    rep = injury_report(season, week)
    out = set()
    if not rep.empty:
        rep = rep[rep["team"].isin(team_codes)]
        out |= set(rep.loc[rep["report_status"].isin(OUT_STATUSES), "gsis_id"].dropna())
    return out, rep


def overrides() -> pd.DataFrame:
    if not OVERRIDES.exists():
        return pd.DataFrame(columns=["player", "team", "status"])
    o = pd.read_csv(OVERRIDES, dtype=str, keep_default_na=False)
    o["team"] = o["team"].map(teams.code)
    o["status"] = o["status"].str.lower().str.strip()
    return o
