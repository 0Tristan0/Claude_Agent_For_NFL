"""Prop lines you paste into data/props.csv, and game lines in data/game_lines.csv.

props.csv columns
  required: player, team, stat, line, source, over_odds, under_odds
  optional: captured_at (UTC ISO; defaults to the file's modified time and is flagged),
            game_id, odds_kind, line_type (standard | goblin | demon), payout_multiplier, notes

Prices accept American odds (-115, +145), exchange cents (52c) or probabilities (0.52, 52%).
PrizePicks rows usually have no odds: leave both odds blank and set source = prizepicks.
odds_kind = legacy_blend marks the hand-blended probabilities from the old src/ev.py: they are shown
for reference only and never used for CLV or for the "beats the market" test.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, teams
from ..features import market

REQUIRED = ["player", "team", "stat", "line", "source", "over_odds", "under_odds"]
OPTIONAL = ["captured_at", "game_id", "odds_kind", "line_type", "payout_multiplier", "notes"]
PROPS_PATH = config.DATA_DIR / "props.csv"
GAME_LINES_PATH = config.DATA_DIR / "game_lines.csv"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def stat_key(s: str) -> str:
    aliases = config.load("stats")["aliases"]
    k = str(s).strip().lower()
    if k in aliases.values():
        return k
    if k in aliases:
        return aliases[k]
    raise ValueError(f"unknown stat {s!r}; add it to config/stats.toml [aliases]")


def parse_times(s: pd.Series) -> pd.Series:
    """UTC timestamps from ISO strings in any mix of formats; blanks become NaT."""
    return pd.to_datetime(s.replace("", np.nan), utc=True, format="ISO8601")


def load_props(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or PROPS_PATH)
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED + OPTIONAL)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""
    df["team"] = df["team"].map(teams.code)
    df["stat"] = df["stat"].map(stat_key)
    df["line"] = df["line"].astype(float)
    df["line_type"] = df["line_type"].replace("", "standard").str.lower()
    df["odds_kind"] = df["odds_kind"].str.lower()
    mtime = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    df["captured_at_assumed"] = df["captured_at"].eq("")
    df["captured_at"] = parse_times(df["captured_at"]).fillna(mtime).astype("datetime64[us, UTC]")
    over, under = [], []
    for i, r in df.iterrows():
        for col, out in (("over_odds", over), ("under_odds", under)):
            try:
                out.append(market.parse_price(r[col]))
            except ValueError as exc:
                raise ValueError(f"{path.name} row {i + 2}, column {col}: {exc}") from None
    df["p_over_implied"] = [o[0] for o in over]
    df["p_under_implied"] = [u[0] for u in under]
    kinds = [o[1] if o[1] != "none" else u[1] for o, u in zip(over, under)]
    df["odds_kind"] = np.where(df["odds_kind"].eq(""), kinds, df["odds_kind"])
    df["payout_multiplier"] = pd.to_numeric(df["payout_multiplier"], errors="coerce")
    return df


def fair_probs(props: pd.DataFrame) -> pd.DataFrame:
    """Add fair_over / fair_under / market_note using the display devig method and stat overrounds."""
    cfg = config.load("picks")
    over_round = cfg["one_sided_overround"]
    out = props.copy()
    fo, fu, note = [], [], []
    for _, r in out.iterrows():
        if r["odds_kind"] == "legacy_blend":
            po, pu = market._none(r["p_over_implied"]), market._none(r["p_under_implied"])
            fo.append(po if po is not None else (1 - pu if pu is not None else np.nan))
            fu.append(pu if pu is not None else (1 - po if po is not None else np.nan))
            note.append("legacy blended estimate (reference only)")
            continue
        orr = 0.0 if r["odds_kind"] in ("exchange", "prob") else over_round.get(r["stat"], over_round["default"])
        f, n = market.fair_two_way(r["p_over_implied"], r["p_under_implied"], cfg["display_devig"], orr)
        fo.append(np.nan if f is None else f)
        fu.append(np.nan if f is None else 1 - f)
        note.append(n)
    out["fair_over"], out["fair_under"], out["market_note"] = fo, fu, note
    return out


def resolve_players(props: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Match prop player names to player_id among candidates (columns player_id, player_name, team).

    Exact normalized match within the team first, then 'first initial + last name'. Unmatched rows
    keep player_id = NaN and are reported, never guessed."""
    c = candidates.dropna(subset=["player_name"]).drop_duplicates(["player_id", "team"]).copy()
    c["key"] = c["player_name"].map(norm_name)
    c["short"] = c["key"].map(lambda s: (s.split()[0][:1] + " " + s.split()[-1]) if " " in s else s)
    out = props.copy()
    ids = []
    for _, r in out.iterrows():
        k = norm_name(r["player"])
        pool = c[c["team"] == r["team"]]
        hit = pool[pool["key"] == k]
        if len(hit) != 1:
            short = (k.split()[0][:1] + " " + k.split()[-1]) if " " in k else k
            hit = pool[pool["short"] == short]
        ids.append(hit["player_id"].iloc[0] if len(hit) == 1 else np.nan)
    out["player_id"] = ids
    return out


def load_game_lines(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or GAME_LINES_PATH)
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_csv(path)
    g["captured_at"] = parse_times(g["captured_at"].astype(str)).astype("datetime64[us, UTC]")
    return g


def latest_game_line(game_lines: pd.DataFrame, game_id: str, asof: pd.Timestamp) -> pd.Series | None:
    """Latest manually captured game line strictly before `asof` (same strict rule as AsofIndex)."""
    if game_lines.empty:
        return None
    g = game_lines[(game_lines["game_id"] == game_id) & (game_lines["captured_at"] < pd.Timestamp(asof))]
    return None if g.empty else g.sort_values("captured_at").iloc[-1]
