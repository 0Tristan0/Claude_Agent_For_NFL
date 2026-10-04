"""User-supplied CSV imports for data that public sources do not provide (e.g. routes run).

Imported values are stored with provenance (file name, hash, import time, the
user's source label) and are displayed with a "user-supplied" label. They are
NOT used as model inputs: the model was validated without them, and their
pregame availability cannot be verified by this application.

Schema (CSV, header row required):

    player_id       GSIS id, e.g. 00-0036900                 (required)
    game_id         nflverse game id, e.g. 2025_01_CIN_CLE    (required)
    routes_run      integer >= 0                               (optional)
    route_participation  share of team dropbacks with a route, 0-1 (optional)
    team_dropbacks  integer >= 0                               (optional)
    notes           free text                                  (optional)

At least one of routes_run / route_participation is required. Empty cells stay
missing; they are never converted to zero.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone

import pandas as pd

from .. import config

REQUIRED = ["player_id", "game_id"]
VALUE_COLS = ["routes_run", "route_participation", "team_dropbacks"]
GAME_ID_RE = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")


def validate_routes_csv(raw: bytes, known_players: set[str] | None = None,
                        known_games: set[str] | None = None) -> tuple[pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype={"player_id": str, "game_id": str})
    except Exception as exc:
        return None, [f"Could not parse CSV: {type(exc).__name__}"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {missing}"]
    if not any(c in df.columns for c in ("routes_run", "route_participation")):
        return None, ["Need at least one of routes_run or route_participation"]
    for c in VALUE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")       # blanks stay NaN
            if (df[c] < 0).any():
                errors.append(f"{c}: negative values")
    if "route_participation" in df.columns and (df["route_participation"] > 1).any():
        errors.append("route_participation must be a share between 0 and 1")
    if "routes_run" in df.columns and "team_dropbacks" in df.columns:
        bad = (df["routes_run"] > df["team_dropbacks"]).sum()
        if bad:
            errors.append(f"{bad} rows with routes_run > team_dropbacks")
    bad_ids = ~df["game_id"].astype(str).str.match(GAME_ID_RE)
    if bad_ids.any():
        errors.append(f"{int(bad_ids.sum())} rows with malformed game_id (expected e.g. 2025_01_CIN_CLE)")
    if df.duplicated(["player_id", "game_id"]).any():
        errors.append("duplicate player_id/game_id rows")
    if known_players is not None:
        unk = set(df["player_id"]) - known_players
        if unk:
            errors.append(f"{len(unk)} unknown player_id values (e.g. {sorted(unk)[:3]})")
    if known_games is not None:
        unk = set(df["game_id"]) - known_games
        if unk:
            errors.append(f"{len(unk)} unknown game_id values (e.g. {sorted(unk)[:3]})")
    return (None if errors else df), errors


def save_import(df: pd.DataFrame, raw: bytes, filename: str, source_label: str) -> dict:
    if not source_label.strip():
        raise ValueError("A source label is required (where did these numbers come from?)")
    config.IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(raw).hexdigest()
    meta = {"filename": filename, "sha256": sha, "source_label": source_label.strip(),
            "imported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "rows": int(len(df)), "used_in_model": False,
            "provenance": "USER-SUPPLIED - not verified by this application"}
    out = df.copy()
    out["provenance"] = f"user-supplied: {meta['source_label']}"
    out["import_sha256"] = sha
    out.to_parquet(config.IMPORTS_DIR / f"routes_{sha[:12]}.parquet", index=False)
    (config.IMPORTS_DIR / f"routes_{sha[:12]}.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_imports() -> tuple[pd.DataFrame, list[dict]]:
    if not config.IMPORTS_DIR.exists():
        return pd.DataFrame(), []
    frames = [pd.read_parquet(p) for p in sorted(config.IMPORTS_DIR.glob("routes_*.parquet"))]
    metas = [json.loads(p.read_text()) for p in sorted(config.IMPORTS_DIR.glob("routes_*.json"))]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), metas
