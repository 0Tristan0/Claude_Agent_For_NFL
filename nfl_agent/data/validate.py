"""Schema validation for downloaded and user-supplied tables.

Validation never silently repairs data. It returns a list of human-readable
issues; callers decide whether an issue is fatal.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "schedules": ["game_id", "season", "game_type", "week", "gameday", "gametime",
                  "away_team", "home_team", "away_score", "home_score", "roof",
                  "away_rest", "home_rest"],
    "players": ["gsis_id", "pfr_id", "display_name", "position", "draft_round", "draft_pick",
                "rookie_season"],
    "player_stats": ["player_id", "player_display_name", "position", "season", "week",
                     "season_type", "game_id", "team", "opponent_team", "targets", "receptions",
                     "receiving_yards", "receiving_air_yards", "receiving_yards_after_catch",
                     "attempts", "sacks_suffered"],
    "snap_counts": ["game_id", "season", "game_type", "week", "player", "pfr_player_id",
                    "position", "team", "opponent", "offense_snaps", "offense_pct"],
    "injuries": ["season", "week", "team", "gsis_id", "full_name", "report_status"],
    "nextgen_receiving": ["season", "week", "player_gsis_id", "avg_separation", "avg_cushion",
                          "avg_intended_air_yards"],
}

NON_NEGATIVE = {
    "player_stats": ["targets", "receptions"],
    "snap_counts": ["offense_snaps"],
}


def validate(key: str, df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if df is None or df.empty:
        return [f"{key}: table is empty or missing"]
    missing = [c for c in REQUIRED_COLUMNS.get(key, []) if c not in df.columns]
    if missing:
        issues.append(f"{key}: missing required columns {missing}")
    for col in NON_NEGATIVE.get(key, []):
        if col in df.columns and (pd.to_numeric(df[col], errors="coerce") < 0).any():
            issues.append(f"{key}: negative values in {col}")
    if key == "player_stats" and {"targets", "receptions"} <= set(df.columns):
        bad = (df["receptions"] > df["targets"]).sum()
        if bad:
            issues.append(f"{key}: {bad} rows with receptions > targets (source quirk; kept as-is, "
                          "targets raised to receptions in the modelling table)")
    if key == "player_stats" and {"player_id", "game_id"} <= set(df.columns):
        dup = df.duplicated(["player_id", "game_id"]).sum()
        if dup:
            issues.append(f"{key}: {dup} duplicate player-game rows")
    if key == "snap_counts" and {"pfr_player_id", "game_id"} <= set(df.columns):
        dup = df.duplicated(["pfr_player_id", "game_id"]).sum()
        if dup:
            issues.append(f"{key}: {dup} duplicate player-game rows")
    if key == "schedules" and "game_id" in df.columns and df["game_id"].duplicated().any():
        issues.append("schedules: duplicate game_id values")
    return issues


def missingness(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    out = pd.DataFrame({
        "field": cols,
        "missing_share": [float(df[c].isna().mean()) for c in cols],
        "n_rows": len(df),
    })
    return out.sort_values("missing_share", ascending=False, ignore_index=True)
