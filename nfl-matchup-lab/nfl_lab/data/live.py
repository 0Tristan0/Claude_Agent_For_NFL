"""Live-only inputs: who is out (official injury report + your overrides). Never used in backtests,
because the injury file's publication times can't be verified for 2025 onward.

data/overrides.csv (optional): player,team,status  with status out | in. "in" forces a player onto the
active list (e.g. a call-up), "out" removes him. Optional columns: game_id (the row applies to that game only),
source, note.
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


def overrides(game_id: str | None = None) -> pd.DataFrame:
    """Your overrides; rows with a game_id apply to that game only."""
    if not OVERRIDES.exists():
        return pd.DataFrame(columns=["player", "team", "status"])
    o = pd.read_csv(OVERRIDES, dtype=str, keep_default_na=False)
    o["team"] = o["team"].map(teams.code)
    o["status"] = o["status"].str.lower().str.strip()
    if "game_id" in o.columns:
        o = o[o["game_id"].isin(["", game_id or ""])]
    return o.reset_index(drop=True)


NEWS = config.DATA_DIR / "news.csv"


def news(game_id: str, before: pd.Timestamp) -> pd.DataFrame:
    """data/news.csv: hand nudges from injury reports and beat news, like the original report's matchup nudges.
    Columns: game_id, player, team, stat ('*' = all his props), more_adj (added to his chance of More before the
    estimate is halved with the market; negative favors Less), reason, source, captured_at (must be before kickoff)."""
    cols = ["game_id", "player", "team", "stat", "more_adj", "reason", "source", "captured_at"]
    if not NEWS.exists():
        return pd.DataFrame(columns=cols)
    n = pd.read_csv(NEWS, dtype=str, keep_default_na=False)
    n = n[n["game_id"] == game_id].copy()
    n["captured_at"] = pd.to_datetime(n["captured_at"], utc=True)
    n = n[n["captured_at"] < before]
    n["team"] = n["team"].map(teams.code)
    n["more_adj"] = n["more_adj"].astype(float)
    return n.reset_index(drop=True)
