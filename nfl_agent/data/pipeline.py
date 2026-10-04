"""Load raw tables (real cache or demo fixture), validate, build, and persist."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from .. import config
from . import build, demo, ingest, validate

log = logging.getLogger(__name__)

META_PATH = lambda: config.PROCESSED_DIR / "metadata.json"  # noqa: E731


def load_raw_real() -> dict[str, pd.DataFrame]:
    raw = {
        "schedules": ingest.load_cached("schedules"),
        "players": ingest.load_cached("players"),
        "player_stats": ingest.load_cached("player_stats", list(range(config.FIRST_SEASON, config.LAST_SEASON + 1))),
        "snap_counts": ingest.load_cached("snap_counts"),
        "injuries": ingest.load_cached("injuries", list(range(config.FIRST_SEASON, config.LAST_SEASON + 1))),
        "nextgen_receiving": ingest.load_cached("nextgen_receiving"),
    }
    return raw


def run(mode: str = "auto", download: bool = True, refresh: bool = False) -> dict:
    """Build processed tables. mode: 'real', 'demo', or 'auto' (real if reachable)."""
    config.ensure_dirs()
    fetch_failures: list[str] = []
    if mode in ("real", "auto") and download:
        results = ingest.ingest_all(refresh=refresh)
        fetch_failures = [f"{r.source_key} {r.season}: {r.status} {r.message}" for r in results
                          if r.status == "failed"]
    if mode == "real" or (mode == "auto" and ingest.has_real_data()):
        raw = load_raw_real()
        data_mode = "real"
    else:
        log.warning("Real data unavailable: building clearly-labelled DEMO fixture data.")
        raw = demo.make_demo_raw()
        data_mode = "demo"
    issues = {k: validate.validate(k, v) for k, v in raw.items()}
    fatal = [i for k in ("schedules", "players", "player_stats") for i in issues[k] if "missing required" in i or "empty" in i]
    if fatal:
        raise RuntimeError("Cannot build tables: " + "; ".join(fatal))
    tables = build.build_all(raw)
    out_dir = config.PROCESSED_DIR if data_mode == "real" else config.DEMO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    # Context-only tables (never model inputs): injuries, NGS weekly.
    for name in ("injuries", "nextgen_receiving"):
        if not raw[name].empty:
            raw[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    pg = tables["player_games"]
    completed = tables["games"][tables["games"]["completed"]]
    meta = {
        "data_mode": data_mode,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_cutoff": str(completed["kickoff_utc"].max()) if len(completed) else None,
        "seasons": sorted(int(s) for s in pg["season"].dropna().unique()),
        "n_player_games": int(len(pg)),
        "zero_filled_share": float(pg["stats_zero_filled"].mean()) if len(pg) else None,
        "validation_issues": issues,
        "fetch_failures": fetch_failures,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_tables(prefer: str = "real") -> tuple[dict[str, pd.DataFrame], dict]:
    """Load processed tables. Falls back to demo tables (labelled) if real ones are absent."""
    for d in ((config.PROCESSED_DIR, config.DEMO_DIR) if prefer == "real" else (config.DEMO_DIR, config.PROCESSED_DIR)):
        meta_p = d / "metadata.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            tables = {}
            for name in ("games", "team_games", "player_games", "injuries", "nextgen_receiving"):
                p = d / f"{name}.parquet"
                tables[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            return tables, meta
    raise FileNotFoundError("No processed data. Run: python -m nfl_agent.cli build")
