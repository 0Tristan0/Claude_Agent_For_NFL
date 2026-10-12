"""Game context known before kickoff: home/away, rest, short week, time-zone travel, dome, surface.

Time zones come from config/stadiums.csv (stadium coordinates and tz from the greerreNFL/Stadiums
dataset, hand-checked; attribution in the report footer). A team's home time zone is the tz of the
stadium it hosted most games at that season (schedule facts, known in advance).
"""
from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import CONFIG_DIR


@lru_cache(maxsize=1)
def stadiums() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "stadiums.csv").set_index("stadium_id")


def _utc_offset_hours(tz: str, when: pd.Timestamp) -> float:
    try:
        return when.tz_convert(ZoneInfo(tz)).utcoffset().total_seconds() / 3600
    except Exception:
        return np.nan


def build(tv: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    st = stadiums()
    home_games = games[games["location"].ne("Neutral")]
    home_stadium = (home_games.groupby(["season", "home_team"])["stadium_id"]
                    .agg(lambda s: s.value_counts().index[0]).rename("home_stadium_id").reset_index()
                    .rename(columns={"home_team": "team"}))
    c = tv[["game_id", "season", "team", "opponent", "kickoff_utc", "is_home", "rest_days", "stadium_id",
            "roof_pregame"]].merge(home_stadium, on=["season", "team"], how="left")
    c["short_week"] = (c["rest_days"] <= 5).astype(float)
    c.loc[c["rest_days"].isna(), "short_week"] = np.nan
    tz_game = c["stadium_id"].map(st["tz"])
    tz_home = c["home_stadium_id"].map(st["tz"])
    c["tz_shift_hours"] = [(_utc_offset_hours(g, k) - _utc_offset_hours(h, k)) if isinstance(g, str) and
                           isinstance(h, str) else np.nan for g, h, k in zip(tz_game, tz_home, c["kickoff_utc"])]
    c["body_clock_kickoff_hour"] = [k.tz_convert(ZoneInfo(h)).hour + k.tz_convert(ZoneInfo(h)).minute / 60
                                    if isinstance(h, str) else np.nan for h, k in zip(tz_home, c["kickoff_utc"])]
    c["dome"] = (c["roof_pregame"] == "dome").astype(float)
    return c.drop(columns=["home_stadium_id"])
