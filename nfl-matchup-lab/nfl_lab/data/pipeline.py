"""Build every table from raw sources and stamp each history row with `available_at`.

`build_tables(raw)` is a pure function of the raw source frames, so the leak tests can run it on a
synthetic league and on truncated copies. `build()` loads the cache, runs it, and writes
data/tables/*.parquet plus tables/meta.json.

available_at = kickoff + config/model.toml [lags_hours][source]. A row is invisible to any query whose
asof is at or before that time (features/asof.py uses a strict `<`).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .. import config
from . import context, games as G, ingest, pbp, player_games, routes, team_games

PBP_SEASONS_NEEDED = None   # all ingested


def _stamp(df: pd.DataFrame, lag_key: str) -> pd.DataFrame:
    df = df.copy()
    df["available_at"] = (df["kickoff_utc"] + pd.to_timedelta(config.lag_hours(lag_key), unit="h")).astype(
        "datetime64[us, UTC]")
    return df


def build_tables(raw: dict[str, pd.DataFrame], first_season: int | None = None) -> dict[str, pd.DataFrame]:
    first_season = first_season or config.FIRST_SEASON
    games, results = G.build_games(raw["schedules"], first_season)
    tv = G.team_view(games)
    p = pbp.prepare(raw["pbp"])
    p = p[p["game_id"].isin(set(games["game_id"]))]
    dr = pbp.drives(p)

    tg = _stamp(team_games.build(p, dr, tv), "pbp")
    pressure = team_games.pfr_pressure(raw.get("pfr_pass", pd.DataFrame()))
    if not pressure.empty:
        dbk = tg[["game_id", "team", "off_dropbacks"]].rename(columns={"off_dropbacks": "pressure_dropbacks"})
        pressure = pressure.merge(dbk, on=["game_id", "team"], how="inner")
        pressure = pressure[pressure["charted"]].drop(columns="charted")
    pressure = team_games.build_side(pressure, tv)
    ftn = team_games.build_side(team_games.ftn_team(raw.get("ftn", pd.DataFrame()), p), tv)

    pg = player_games.build(p, raw["snap_counts"], raw["players"], raw.get("rosters", pd.DataFrame()), tv)
    pg = player_games.attach_names(pg, raw["players"])
    seasons = sorted(games["season"].unique())
    part = raw.get("participation", pd.DataFrame())
    dbp = routes.dropback_participation(part, p) if not part.empty else pd.DataFrame(
        columns=["game_id", "player_id", "db_on_field"])
    rr = routes.ratios(pg, dbp, [s for s in seasons] + [max(seasons) + 1])
    pg = routes.attach_routes(pg, rr) if not rr.empty else pg.assign(
        route_ratio=float("nan"), routes_est=float("nan"), route_ratio_source="none", routes_known=False,
        rk_targets=0.0, rk_rec_yds=0.0, rk_team_dropbacks=0.0)
    pg = _stamp(pg, "snaps")

    out = {
        "games": games, "game_results": results, "team_view": tv,
        "context": context.build(tv, games),
        "plays": p, "drives": dr, "team_games": tg,
        "team_pressure": _stamp(pressure, "pfr") if not pressure.empty else pressure,
        "team_ftn": _stamp(ftn, "ftn") if not ftn.empty else ftn,
        "player_games": pg, "route_ratios": rr,
    }
    pfr_rush = raw.get("pfr_rush", pd.DataFrame())
    if not pfr_rush.empty:
        out["player_pfr_rush"] = _stamp(_pfr_rush(pfr_rush, raw["players"], raw.get("rosters"), tv), "pfr")
    return out


def _pfr_rush(pfr_rush: pd.DataFrame, players: pd.DataFrame, rosters, tv: pd.DataFrame) -> pd.DataFrame:
    x = pfr_rush.copy()
    x["player_id"] = x["pfr_player_id"].map(player_games.pfr_to_gsis(players, rosters))
    x = x[x["player_id"].notna()]
    keep = x[["game_id", "player_id", "carries", "rushing_yards_before_contact", "rushing_yards_after_contact",
              "rushing_broken_tackles"]].rename(columns={"carries": "pfr_carries", "rushing_yards_before_contact": "ybc",
                                                         "rushing_yards_after_contact": "yaco",
                                                         "rushing_broken_tackles": "broken_tackles"})
    keep = keep.groupby(["game_id", "player_id"], as_index=False).sum()
    return keep.merge(tv[["game_id", "kickoff_utc", "season", "week"]].drop_duplicates("game_id"), on="game_id",
                      how="inner")


RAW_KEYS = ["schedules", "players", "pbp", "snap_counts", "rosters", "participation", "ftn", "pfr_pass", "pfr_rush"]


def load_raw(seasons=None) -> dict[str, pd.DataFrame]:
    raw = {}
    for k in RAW_KEYS:
        cols = pbp.COLUMNS if k == "pbp" else (
            ["nflverse_game_id", "play_id", "offense_players"] if k == "participation" else None)
        raw[k] = ingest.load(k, seasons=seasons, columns=cols)
    return raw


def build(write: bool = True) -> dict[str, pd.DataFrame]:
    config.ensure_dirs()
    tables = build_tables(load_raw())
    if write:
        for name, df in tables.items():
            if name == "plays":
                for s, part in df.groupby("season"):
                    part.to_parquet(config.TABLES_DIR / f"plays_{s}.parquet", index=False)
            else:
                df.to_parquet(config.TABLES_DIR / f"{name}.parquet", index=False)
        meta = dict(built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    config_hash=config.config_hash(), data_hashes=ingest.data_hashes(),
                    rows={k: int(len(v)) for k, v in tables.items()})
        (config.TABLES_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    return tables


def load_table(name: str) -> pd.DataFrame:
    if name == "plays":
        files = sorted(config.TABLES_DIR.glob("plays_*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(config.TABLES_DIR / f"{name}.parquet")
