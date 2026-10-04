"""Data inventory: what each source provides, coverage, freshness, missingness, pregame status.

Written to docs/DATA_INVENTORY.md and shown on the Data Health page. Everything
is computed from the local cache and the ingestion log, so it reflects what was
actually retrieved (not what documentation promises).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .. import config, db
from . import ingest
from .sources import INGESTED, SOURCES

PREGAME_LABEL = {
    "pregame_derivable": "Yes, as prior-game history only (shifted)",
    "known_in_advance": "Yes for facts fixed before kickoff (exceptions listed under limitations)",
    "retrospective": "No (filled after games)",
    "timing_unverified": "Unverified: no publication timestamp; context only",
    "postseason_release": "No: released after the season",
}

KEY_FIELDS = {
    "player_stats": ["targets", "receptions", "receiving_yards", "receiving_air_yards", "receiving_yards_after_catch",
                     "target_share", "air_yards_share", "receiving_20"],
    "snap_counts": ["offense_snaps", "offense_pct"],
    "schedules": ["gameday", "gametime", "roof", "temp", "wind", "away_rest", "home_qb_name"],
    "players": ["gsis_id", "pfr_id", "draft_pick", "rookie_season"],
    "injuries": ["report_status", "practice_status", "date_modified"],
    "nextgen_receiving": ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"],
}


def ingestion_log(limit: int = 500) -> pd.DataFrame:
    with db.session() as con:
        return pd.read_sql_query("SELECT * FROM ingestion_log ORDER BY id DESC LIMIT ?", con, params=(limit,))


def latest_status() -> pd.DataFrame:
    log = ingestion_log(5000)
    if log.empty:
        return log
    return log.sort_values("id").groupby(["source_key", "season"], dropna=False).tail(1)


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, miss_rows = [], []
    log = latest_status()
    for key in INGESTED:
        src = SOURCES[key]
        df = ingest.load_cached(key)
        seasons = sorted(int(s) for s in df["season"].dropna().unique()) if "season" in df else []
        lg = log[log["source_key"] == key] if len(log) else pd.DataFrame()
        rows.append(dict(
            source=key, url=src.url("{season}") if src.per_season else src.url(),
            seasons=f"{seasons[0]}-{seasons[-1]}" if seasons else "n/a", rows=len(df),
            fields=len(df.columns), update_frequency=src.update_frequency,
            last_retrieved_utc=lg["retrieved_at"].max() if len(lg) else None,
            source_last_updated=lg["source_last_updated"].dropna().max() if len(lg) and lg["source_last_updated"].notna().any() else None,
            failures=int((lg["status"] == "failed").sum()) if len(lg) else 0,
            pregame_available=PREGAME_LABEL[src.pregame_status], used_in_model=src.used_in_model,
            limitations=" ".join(src.limitations)))
        for f in KEY_FIELDS.get(key, []):
            miss_rows.append(dict(source=key, field=f, present=f in df.columns,
                                  missing_share=float(df[f].isna().mean()) if f in df.columns else 1.0))
    return pd.DataFrame(rows), pd.DataFrame(miss_rows)


def write_inventory() -> str:
    inv, miss = build_inventory()
    lines = [f"# Data inventory\n\n*Generated {datetime.now(timezone.utc).replace(microsecond=0).isoformat()} from the "
             "local cache and ingestion log. Regenerate with `python -m nfl_agent.cli inventory`.*\n"]
    for _, r in inv.iterrows():
        lines.append(f"## {r['source']}\n")
        lines.append(f"* URL: `{r['url']}`")
        lines.append(f"* Seasons in cache: {r['seasons']}; rows: {r['rows']:,}; fields: {r['fields']}")
        lines.append(f"* Update frequency (per nflverse): {r['update_frequency']}")
        lines.append(f"* Last retrieved (UTC): {r['last_retrieved_utc']}; source timestamp: {r['source_last_updated']}")
        lines.append(f"* Usable before a historical kickoff: **{r['pregame_available']}**; used in model: {r['used_in_model']}")
        lines.append(f"* Limitations: {r['limitations']}\n")
        m = miss[miss["source"] == r["source"]]
        if len(m):
            lines.append("| field | present | missing share |\n|---|---|---|")
            for _, x in m.iterrows():
                lines.append(f"| {x['field']} | {x['present']} | {x['missing_share']:.1%} |")
            lines.append("")
    path = config.DOCS_DIR / "DATA_INVENTORY.md"
    path.write_text("\n".join(lines))
    return str(path)
