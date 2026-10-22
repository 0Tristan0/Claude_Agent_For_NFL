"""Download nflverse release files into a content-addressed parquet cache.

Ported from nfl_agent/data/ingest.py @90f63d6 (fetch/_get/backoff, cache kept on failure) with changes:
files are stored as data/raw/<key>/<key>_<season>.<sha8>.parquet so a refresh never overwrites a
version that a LOCK or a frozen prediction recorded, and every attempt is appended to
data/raw/manifest.jsonl (key, season, url, retrieved_at, source_last_updated, sha256, rows, status).

Policy: completed seasons are cached until --refresh; the live season and whole-history files are
refetched when the newest copy is older than max_age_hours. A failed download never deletes a file.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from .. import config
from .sources import INGEST_PLAN, SOURCES, Source

log = logging.getLogger(__name__)
USER_AGENT = "nfl-lab-educational-research/0.1 (+local, non-commercial)"
REQUEST_PAUSE_S = 0.3


@dataclass
class FetchResult:
    key: str
    season: int | None
    path: Path | None
    status: str          # cached | ok | empty | not_published | failed
    message: str = ""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get(url: str, timeout: int = 90, retries: int = 4) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {type(last).__name__}")


def source_timestamp(src: Source) -> str | None:
    try:
        resp = _get(src.timestamp_url, timeout=20, retries=1)
        return resp.json().get("last_updated") if resp.ok else None
    except Exception:
        return None


def read_manifest(manifest: Path | None = None) -> list[dict]:
    m = manifest or config.MANIFEST
    if not m.exists():
        return []
    return [json.loads(line) for line in m.read_text().splitlines() if line.strip()]


def _append(row: dict, manifest: Path | None = None) -> None:
    m = manifest or config.MANIFEST
    m.parent.mkdir(parents=True, exist_ok=True)
    with open(m, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def current_files(manifest: Path | None = None) -> dict[tuple[str, int | None], dict]:
    """Newest successful manifest row per (key, season) whose file still exists."""
    out: dict[tuple[str, int | None], dict] = {}
    base = (manifest or config.MANIFEST).parent
    for row in read_manifest(manifest):
        if row["status"] in ("ok", "empty") and (base / row["path"]).exists():
            out[(row["key"], row["season"])] = row
    return out


def fetch(src: Source, season: int | None, *, refresh: bool = False, max_age_hours: float = 6.0,
          live_season: int | None = None, source_ts: str | None = None,
          raw_dir: Path | None = None) -> FetchResult:
    raw_dir = raw_dir or config.RAW_DIR
    manifest = raw_dir / "manifest.jsonl"
    live_season = live_season or config.LIVE_SEASON
    url = src.url(season)
    have = current_files(manifest).get((src.key, season))
    is_live = (not src.per_season) or season == live_season
    if have and not refresh:
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(have["retrieved_at"])).total_seconds() / 3600
        if not is_live or age_h < max_age_hours:
            return FetchResult(src.key, season, raw_dir / have["path"], "cached")
    row = dict(key=src.key, season=season, url=url, retrieved_at=utcnow_iso(), source_last_updated=source_ts,
               sha256=None, n_rows=None, path=None, status="failed", message="")
    try:
        resp = _get(url)
    except RuntimeError as exc:
        row["message"] = str(exc) + (" (kept cached copy)" if have else "")
        _append(row, manifest)
        return FetchResult(src.key, season, raw_dir / have["path"] if have else None, "failed", str(exc))
    if resp.status_code == 404:
        row.update(status="not_published", message="HTTP 404")
        _append(row, manifest)
        return FetchResult(src.key, season, None, "not_published", "HTTP 404")
    content = resp.content
    try:
        n_rows = pq.ParquetFile(io.BytesIO(content)).metadata.num_rows
    except Exception as exc:
        row["message"] = f"unreadable parquet: {type(exc).__name__}"
        _append(row, manifest)
        return FetchResult(src.key, season, raw_dir / have["path"] if have else None, "failed", row["message"])
    sha = hashlib.sha256(content).hexdigest()
    name = f"{src.key}_{season}.{sha[:8]}.parquet" if src.per_season else f"{src.key}.{sha[:8]}.parquet"
    rel = Path(src.key) / name
    path = raw_dir / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
    row.update(sha256=sha, n_rows=n_rows, path=str(rel), status="ok" if n_rows else "empty",
               message="" if n_rows else "published with 0 rows")
    _append(row, manifest)
    time.sleep(REQUEST_PAUSE_S)
    return FetchResult(src.key, season, path, row["status"], row["message"])


def ingest(keys=None, last_season: int | None = None, refresh: bool = False,
           raw_dir: Path | None = None) -> list[FetchResult]:
    last_season = last_season or config.LIVE_SEASON
    results = []
    for key in keys or INGEST_PLAN:
        src = SOURCES[key]
        ts = source_timestamp(src)
        if src.per_season:
            first = max(INGEST_PLAN.get(key, src.first_season), src.first_season)
            last = min(last_season, src.last_season or last_season)
            seasons = range(first, last + 1)
        else:
            seasons = [None]
        for season in seasons:
            r = fetch(src, season, refresh=refresh, source_ts=ts, raw_dir=raw_dir, live_season=last_season)
            log.info("%s %s: %s %s", key, season, r.status, r.message)
            results.append(r)
    return results


def load(key: str, seasons=None, columns=None, raw_dir: Path | None = None) -> pd.DataFrame:
    """Concatenate the current cached files of a source (missing seasons are skipped)."""
    raw_dir = raw_dir or config.RAW_DIR
    files = current_files(raw_dir / "manifest.jsonl")
    src = SOURCES[key]
    if not src.per_season:
        row = files.get((key, None))
        return _read(raw_dir / row["path"], columns) if row else pd.DataFrame()
    frames = []
    wanted = seasons if seasons is not None else sorted(s for (k, s) in files if k == key)
    for s in wanted:
        row = files.get((key, s))
        if row:
            frames.append(_read(raw_dir / row["path"], columns))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read(path: Path, columns=None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    have = set(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_parquet(path, columns=[c for c in columns if c in have])


def data_hashes(raw_dir: Path | None = None) -> dict[str, str]:
    """sha256 of every current raw file, for LOCK files and frozen predictions."""
    raw_dir = raw_dir or config.RAW_DIR
    return {f"{k}:{s}": r["sha256"] for (k, s), r in sorted(current_files(raw_dir / "manifest.jsonl").items(),
                                                            key=lambda kv: (kv[0][0], kv[0][1] or 0))}
