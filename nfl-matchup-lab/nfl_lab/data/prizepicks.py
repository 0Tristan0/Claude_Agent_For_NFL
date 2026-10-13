"""Import the PrizePicks NFL board into data/props.csv.

Source: the JSON feed the PrizePicks web app itself reads (FEED_URL). It is not an official API, so it can change
shape or refuse requests, and PrizePicks' terms limit automated access: this pulls the board once, when you ask.
If this machine can't reach it, open FEED_URL in your own browser, save the page as a .json file, and import that
file instead:  python -m nfl_lab prizepicks --file board.json

What gets imported: single-player NFL projections that are still pre-game, for stats the simulator models.
Goblin and demon lines are kept but marked (they never qualify for lineups; their payouts differ). Each import
replaces earlier PrizePicks rows that have no game_id (the live board), keeps rows tied to a game_id (past
games, kept for the record), and saves a copy of the raw board under data/prizepicks/.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .. import config, teams
from .lines import OPTIONAL, PROPS_PATH, REQUIRED

FEED_URL = "https://api.prizepicks.com/projections?league_id=9&per_page=1000&single_stat=true"
ARCHIVE_DIR = config.DATA_DIR / "prizepicks"

# PrizePicks stat names -> stat keys the simulator produces
PP_STATS = {"Pass Yards": "pass_yds", "Rush Yards": "rush_yds", "Receiving Yards": "rec_yds", "Receptions": "receptions",
            "Rush+Rec TDs": "rush_rec_tds", "Rush+Rec Yds": "rush_rec_yds", "Pass TDs": "pass_tds",
            "Pass Attempts": "pass_att", "Pass Completions": "completions", "Rush Attempts": "rush_att",
            "INT": "ints", "Interceptions": "ints"}


def fetch(timeout: float = 30) -> dict:
    """Download the board. Raises with a plain message when the network or PrizePicks refuses."""
    import requests
    try:
        r = requests.get(FEED_URL, timeout=timeout, headers={
            "Accept": "application/json", "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nfl-matchup-lab"})
    except requests.RequestException as exc:
        raise RuntimeError(f"could not reach PrizePicks ({exc.__class__.__name__}). If this machine's network blocks "
                           f"api.prizepicks.com, open {FEED_URL} in a browser, save it as a .json file and use --file.") from None
    if r.status_code != 200:
        raise RuntimeError(f"PrizePicks answered HTTP {r.status_code}; it may be blocking automated requests. "
                           f"Open {FEED_URL} in a browser, save it as a .json file and use --file.")
    return r.json()


def parse(feed: dict, captured_at: pd.Timestamp) -> tuple[pd.DataFrame, Counter]:
    """Board JSON (JSON:API shape: data = projections, included = players) -> props.csv rows, plus skip counts."""
    players = {str(x.get("id")): x.get("attributes", {}) for x in feed.get("included", []) if x.get("type") == "new_player"}
    rows, skipped = [], Counter()
    for d in feed.get("data", []):
        if d.get("type") != "projection":
            continue
        a = d.get("attributes", {})
        if a.get("status") not in (None, "", "pre_game"):
            skipped["not pre-game"] += 1
            continue
        if a.get("is_promo"):
            skipped["promo"] += 1
            continue
        stat = PP_STATS.get(str(a.get("stat_type", "")).strip())
        if stat is None:
            skipped[f"stat not modeled: {a.get('stat_type')}"] += 1
            continue
        rel = (d.get("relationships", {}).get("new_player") or {}).get("data") or {}
        p = players.get(str(rel.get("id")))
        if not p:
            skipped["player missing from feed"] += 1
            continue
        if p.get("league") not in (None, "", "NFL") or p.get("combo"):
            skipped["not a single NFL player"] += 1
            continue
        name = p.get("display_name") or p.get("name") or ""
        if " + " in name:
            skipped["not a single NFL player"] += 1
            continue
        try:
            team = teams.code(p.get("team") or p.get("team_name") or "")
        except ValueError:
            skipped[f"unknown team: {p.get('team')}"] += 1
            continue
        line = float(a["line_score"])
        if stat == "rush_rec_tds" and line == 0.5:
            stat = "anytime_td"                       # 0.5 Rush+Rec TDs is an anytime-TD prop
        rows.append(dict(player=name, team=teams.display(team), stat=stat, line=line, source="prizepicks",
                         over_odds="", under_odds="", captured_at=captured_at.strftime("%Y-%m-%dT%H:%M:%SZ"), game_id="",
                         odds_kind="", line_type=str(a.get("odds_type") or "standard").lower(), payout_multiplier="",
                         notes=f"PrizePicks board, starts {a.get('start_time', '')}"))
    out = pd.DataFrame(rows, columns=REQUIRED + OPTIONAL)
    return out.drop_duplicates(["player", "team", "stat", "line", "line_type"]), skipped


def write(rows: pd.DataFrame, path: Path | None = None, raw: dict | None = None) -> Path:
    """Replace the live PrizePicks board in props.csv (rows without a game_id) with `rows`."""
    path = Path(path or PROPS_PATH)
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=REQUIRED + OPTIONAL)
    for c in REQUIRED + OPTIONAL:
        if c not in old.columns:
            old[c] = ""
    live = old["source"].str.lower().eq("prizepicks") & old["game_id"].eq("")
    new = pd.concat([old[~live], rows.astype(str).replace("nan", "")], ignore_index=True)[old.columns.tolist()]
    new.to_csv(path, index=False)
    if raw is not None and len(rows):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = rows["captured_at"].iloc[0].replace(":", "").replace("-", "")
        (ARCHIVE_DIR / f"board-{stamp}.json").write_text(json.dumps(raw))
    return path
