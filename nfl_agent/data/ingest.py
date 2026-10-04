"""Download nflverse release files with local Parquet caching.

Policy
------
* Completed seasons are cached indefinitely (use ``refresh=True`` to pull stat
  corrections).
* The current season and whole-history files (schedules, players, NGS) are
  re-downloaded when the cached copy is older than ``max_age_hours``.
* A failed download never deletes a cached file. Failures are written to the
  ingestion log and surfaced on the Data Health page.
* No credentials are used or logged.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .. import config, db
from .sources import INGESTED, SOURCES, Source

log = logging.getLogger(__name__)

USER_AGENT = "nfl-agent-educational-research/1.0 (+local, non-commercial)"
REQUEST_PAUSE_S = 0.5     # be polite to GitHub release hosting


@dataclass
class FetchResult:
    source_key: str
    season: int | None
    path: Path | None
    status: str
    message: str = ""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cache_path(src: Source, season: int | None) -> Path:
    name = f"{src.key}_{season}.parquet" if src.per_season else f"{src.key}.parquet"
    return config.CACHE_DIR / name


def _get(url: str, timeout: int = 60, retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # network failure: back off
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {type(last_exc).__name__}")


def fetch_source_timestamp(src: Source) -> str | None:
    try:
        resp = _get(src.timestamp_url, timeout=20, retries=1)
        if resp.ok:
            return resp.json().get("last_updated")
    except Exception:  # timestamp is informational only
        return None
    return None


def _log(con, src: Source, season, url, status, ts=None, n_rows=None, sha=None, msg=""):
    con.execute(
        "INSERT INTO ingestion_log (source_key, season, url, retrieved_at, status, "
        "source_last_updated, n_rows, sha256, message) VALUES (?,?,?,?,?,?,?,?,?)",
        (src.key, season, url, utcnow_iso(), status, ts, n_rows, sha, msg[:500]),
    )


def fetch(src: Source, season: int | None, *, refresh: bool = False, max_age_hours: float = 6,
          current_season: int = config.LAST_SEASON, con=None, source_ts: str | None = None) -> FetchResult:
    path = cache_path(src, season)
    url = src.url(season)
    is_live = (not src.per_season) or season == current_season
    if path.exists() and not refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if not is_live or age_h < max_age_hours:
            return FetchResult(src.key, season, path, "cached")
    own = con is None
    con = con or db.connect()
    try:
        try:
            resp = _get(url)
        except RuntimeError as exc:
            _log(con, src, season, url, "failed", source_ts, msg=str(exc) + (" (using cache)" if path.exists() else ""))
            return FetchResult(src.key, season, path if path.exists() else None, "failed", str(exc))
        if resp.status_code == 404:
            _log(con, src, season, url, "not_published", source_ts, msg="HTTP 404: file not published")
            return FetchResult(src.key, season, None, "not_published", "HTTP 404")
        content = resp.content
        try:
            df = pd.read_parquet(io.BytesIO(content))
        except Exception as exc:
            _log(con, src, season, url, "failed", source_ts, msg=f"unreadable parquet: {type(exc).__name__}")
            return FetchResult(src.key, season, path if path.exists() else None, "failed", "unreadable parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
        status = "ok" if len(df) else "empty"
        _log(con, src, season, url, status, source_ts, len(df), hashlib.sha256(content).hexdigest(),
             msg="" if len(df) else "file is published but contains 0 rows")
        time.sleep(REQUEST_PAUSE_S)
        return FetchResult(src.key, season, path, status, "" if len(df) else "0 rows")
    finally:
        if own:
            con.commit()
            con.close()


def ingest_all(first_season: int = config.FIRST_SEASON, last_season: int = config.LAST_SEASON,
               refresh: bool = False, keys: tuple[str, ...] = INGESTED) -> list[FetchResult]:
    config.ensure_dirs()
    results: list[FetchResult] = []
    with db.session() as con:
        for key in keys:
            src = SOURCES[key]
            ts = fetch_source_timestamp(src)
            seasons = range(max(first_season, src.first_season), last_season + 1) if src.per_season else [None]
            for season in seasons:
                r = fetch(src, season, refresh=refresh, con=con, source_ts=ts, current_season=last_season)
                log.info("%s %s: %s %s", key, season, r.status, r.message)
                results.append(r)
    return results


def load_cached(key: str, seasons: list[int] | None = None) -> pd.DataFrame:
    """Concatenate cached files for a source. Missing seasons are skipped."""
    src = SOURCES[key]
    if not src.per_season:
        p = cache_path(src, None)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    frames = []
    for season in seasons or range(src.first_season, config.LAST_SEASON + 1):
        p = cache_path(src, season)
        if p.exists():
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def has_real_data() -> bool:
    return all(cache_path(SOURCES[k], None).exists() for k in ("schedules", "players")) and any(
        cache_path(SOURCES["player_stats"], s).exists() for s in range(config.FIRST_SEASON, config.LAST_SEASON + 1)
    )
