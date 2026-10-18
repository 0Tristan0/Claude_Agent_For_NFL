"""Independent no-lookahead oracle.

The availability rules here are written separately from the pipeline (they do not import nfl_lab's
lag config): a game's play-by-play is public 16 h after kickoff, snap counts 28 h, PFR advanced stats
and FTN charting 76 h, participation only after the season, and scores once the game ends (4 h).

`assert_no_lookahead` rebuilds everything from raw data truncated to what existed at `asof` and
requires the game's features to be identical to those built from the full data. `poison` replaces
every not-yet-available row with absurd values; features must not move.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_lab.data import pipeline
from nfl_lab.features import bundle

PUBLIC_AFTER_H = {"pbp": 16, "snap_counts": 28, "rosters": 16, "ftn": 76, "pfr_pass": 76, "pfr_rush": 76}
GAME_LENGTH_H = 4
OUTCOMES = ["home_score", "away_score", "result", "total", "overtime"]


def kickoffs(schedules: pd.DataFrame) -> pd.Series:
    et = ZoneInfo("America/New_York")
    ts = [pd.Timestamp(f"{d} {t or '13:00'}").tz_localize(et).tz_convert("UTC")
          for d, t in zip(schedules["gameday"], schedules["gametime"].fillna("13:00"))]
    return pd.Series(ts, index=schedules["game_id"].to_numpy())


def _game_col(key: str) -> str:
    return "nflverse_game_id" if key in ("participation", "ftn") else "game_id"


def unavailable_mask(raw: dict, key: str, asof: pd.Timestamp, query_season: int) -> pd.Series:
    df = raw[key]
    if key == "participation":
        season = df["nflverse_game_id"].str[:4].astype(int)
        return season >= query_season
    if key == "rosters":
        ko = kickoffs(raw["schedules"])
        wk = raw["schedules"].set_index(["season", "week"])["game_id"]
        first = {k: ko[v].min() if isinstance(v, pd.Series) else ko[v] for k, v in wk.groupby(level=[0, 1])}
        when = pd.Series([first.get((s, w), pd.Timestamp.max.tz_localize("UTC")) for s, w in
                          zip(df["season"], df["week"])], index=df.index)
        return when + pd.Timedelta(hours=PUBLIC_AFTER_H[key]) >= asof
    ko = kickoffs(raw["schedules"])
    when = df[_game_col(key)].map(ko)
    return when + pd.Timedelta(hours=PUBLIC_AFTER_H[key]) >= asof


def truncate(raw: dict, asof: pd.Timestamp, query_season: int) -> dict:
    out = {}
    for k, df in raw.items():
        if k in PUBLIC_AFTER_H or k == "participation":
            out[k] = df[~unavailable_mask(raw, k, asof, query_season)].copy()
        elif k == "schedules":
            s = df.copy()
            ended = kickoffs(df) + pd.Timedelta(hours=GAME_LENGTH_H) < asof
            s.loc[~s["game_id"].map(ended).to_numpy(), [c for c in OUTCOMES if c in s]] = np.nan
            out[k] = s
        else:
            out[k] = df.copy()
    return out


def poison(raw: dict, asof: pd.Timestamp, query_season: int, rng=None) -> dict:
    """Absurd finite values in every unavailable row, rows duplicated and shuffled."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for k, df in raw.items():
        d = df.copy()
        if k in PUBLIC_AFTER_H or k == "participation":
            m = unavailable_mask(raw, k, asof, query_season).to_numpy()
            keep = ("season", "week", "play_id", "nflverse_play_id", "fixed_drive", "qtr")
            bools = [c for c in d.columns if pd.api.types.is_bool_dtype(d[c]) and c not in keep]
            num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c not in keep and c not in bools]
            d[num] = d[num].astype(float)
            d.loc[m, num] = d.loc[m, num].fillna(0) * 37 + 911
            for c in bools:
                d.loc[m, c] = ~d.loc[m, c]
            if k == "participation":
                d.loc[m, "offense_players"] = ";".join(sorted(set(raw["players"]["gsis_id"]))[:11])
            if k == "pbp":
                d.loc[m, "play_type"] = "pass"
            d = pd.concat([d, d[m]], ignore_index=True) if k == "pbp" else d
            d = d.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
        elif k == "schedules":
            ended = kickoffs(d) + pd.Timedelta(hours=GAME_LENGTH_H) < asof
            d.loc[~d["game_id"].map(ended).to_numpy(), [c for c in OUTCOMES if c in d]] = 999
        out[k] = d
    return out


def features(raw: dict, game_id: str, asof=None, first_season=None) -> dict:
    tables = pipeline.build_tables(raw, first_season=first_season)
    return bundle.game_features(tables, game_id, asof)


def assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        x = a[k].sort_index(axis=1).reset_index(drop=True)
        y = b[k].sort_index(axis=1).reset_index(drop=True)
        pd.testing.assert_frame_equal(x, y, check_exact=True, check_dtype=False, obj=k)


def assert_no_lookahead(raw: dict, game_id: str, asof=None, first_season=None) -> None:
    ko = kickoffs(raw["schedules"])[game_id]
    asof = pd.Timestamp(asof) if asof is not None else ko
    season = int(game_id[:4])
    full = features(raw, game_id, asof, first_season)
    trunc = features(truncate(raw, asof, season), game_id, asof, first_season)
    assert_same(full, trunc)
