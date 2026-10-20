"""Graded before-kickoff test for one finished game.

The model is run exactly as it would have been before kickoff: projections and simulations use only stats that
were available before the game (the game's own stats must not be in data/tables; this is checked), and the lines
are the ones on file in data/props.csv for that game, captured before kickoff, all treated as standard PrizePicks
lines without a sportsbook price. The top lineups are then graded against nflverse's official weekly stats.

    python -m nfl_lab retro --game 2026_05_BUF_LA --final "BUF 37, LAR 34 (OT)"
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .. import config
from ..data import lines
from ..picks import payouts

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"

COLS = {"rec_yds": ["receiving_yards"], "receptions": ["receptions"], "rush_yds": ["rushing_yards"],
        "rush_att": ["carries"], "pass_yds": ["passing_yards"], "pass_att": ["attempts"], "completions": ["completions"],
        "pass_tds": ["passing_tds"], "ints": ["passing_interceptions"], "rush_rec_yds": ["rushing_yards", "receiving_yards"],
        "pass_rush_yds": ["passing_yards", "rushing_yards"], "anytime_td": ["rushing_tds", "receiving_tds"],
        "rush_rec_tds": ["rushing_tds", "receiving_tds"]}


def fetch_stats(season: int) -> pd.DataFrame:
    import requests
    r = requests.get(STATS_URL.format(season=season), timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        f.write(r.content)
        f.flush()
        return pd.read_parquet(f.name)


def fetch_finals(season: int, week: int) -> dict[str, str]:
    """{game_id: "DEN 17, SEA 14 (OT)"} for that week's finished games, winner first, from nflverse's schedule."""
    import io
    import requests
    from ..data import sources
    r = requests.get(sources.SOURCES["schedules"].url(), timeout=60)
    r.raise_for_status()
    g = pd.read_parquet(io.BytesIO(r.content))
    g = g[(g["season"] == season) & (g["week"] == week) & g["home_score"].notna() & g["away_score"].notna()]
    out = {}
    for x in g.itertuples():
        a, h = ("LAR" if x.away_team == "LA" else x.away_team), ("LAR" if x.home_team == "LA" else x.home_team)
        (w, ws), (l, ls) = sorted([(a, int(x.away_score)), (h, int(x.home_score))], key=lambda t: -t[1])
        out[x.game_id] = f"{w} {ws}, {l} {ls}" + (" (OT)" if x.overtime == 1 else "")
    return out


def fetch_snaps(season: int) -> pd.DataFrame:
    import io
    import requests
    from ..data import sources
    r = requests.get(sources.SOURCES["snap_counts"].url(season), timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def _name(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def played(snaps: pd.DataFrame | None, week: int, player: str, team: str) -> bool:
    """Did the player take at least one offense, defense or special-teams snap that week (nflverse/PFR snap counts)?"""
    if snaps is None or snaps.empty:
        return False
    t = "LA" if team == "LAR" else team
    row = snaps[(snaps["week"] == week) & (snaps["team"] == t) & (snaps["player"].map(_name) == _name(player))]
    n = sum(row[c].fillna(0) for c in ("offense_snaps", "defense_snaps", "st_snaps") if c in row)
    return bool((n > 0).any()) if len(row) else False


def sat_out(stats: pd.DataFrame, snaps: pd.DataFrame | None, week: int, player: str, team: str) -> bool:
    """True once both the stats and the snap counts for the player's team that week are posted and he took no snap:
    on PrizePicks that pick is void."""
    if snaps is None or snaps.empty or "team" not in stats:
        return False
    t = "LA" if team == "LAR" else team
    posted = ((stats["week"] == week) & (stats["team"] == t)).any() and ((snaps["week"] == week) & (snaps["team"] == t)).any()
    return bool(posted) and not played(snaps, week, player, team)


def actual(stats: pd.DataFrame, player_id: str, week: int, stat: str):
    row = stats[(stats["player_id"] == player_id) & (stats["week"] == week)]
    if row.empty:
        return None                                   # did not play: PrizePicks would void the pick
    v = float(row[COLS[stat]].fillna(0).sum(axis=1).iloc[0])
    return float(v >= 1) if stat == "anytime_td" else v


def grade(best: dict, stats: pd.DataFrame, week: int) -> dict:
    """Mark each leg hit/missed and each lineup's return per $1 at its suggested play (Power or Flex)."""
    power, flex = payouts.tables()
    out = {}
    for n, lst in best.items():
        graded = []
        for L in lst:
            L = dict(L)
            legs, hits, graded_legs = [], 0, 0
            for x in L["legs"]:
                x = dict(x)
                v = actual(stats, x["player_id"], week, x["stat"])
                if v is not None:
                    hit = v > x["line"] if x["side"] == "over" else v < x["line"]
                    shown = ("scored" if v else "no TD") if x["stat"] == "anytime_td" else f"{v:g}"
                    x["result"] = dict(actual=shown, hit=bool(hit))
                    hits += hit
                    graded_legs += 1
                legs.append(x)
            L["legs"] = legs
            k = int(n)
            if graded_legs == k:
                ret = (power[k] if hits == k else 0.0) if L["play_as"] == "Power" else float(flex.get(k, {}).get(hits, 0.0))
                L["outcome"] = dict(hits=hits, of=k, returned=ret, play_as=L["play_as"])
            graded.append(L)
        out[n] = graded
    return out


def board_for(game_id: str) -> pd.DataFrame:
    """The lines on file for this game, as standard PrizePicks lines without a sportsbook price."""
    raw = pd.read_csv(lines.PROPS_PATH, dtype=str, keep_default_na=False)
    g = raw[raw["game_id"] == game_id].drop_duplicates(["player", "team", "stat", "line"]).copy()
    g["source"], g["over_odds"], g["under_odds"], g["odds_kind"], g["line_type"] = "prizepicks", "", "", "", "standard"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "props.csv"
        g.to_csv(p, index=False)
        return lines.load_props(p)


def run(game_id: str, final: str = "", n_sims: int = 20000) -> dict:
    from .. import v0run
    T = v0run.load_tables()
    if (T["player_games"]["game_id"] == game_id).any():
        raise RuntimeError(f"{game_id} is already in data/tables: the test would see the result. Use tables built "
                           "before the game.")
    season = int(T["games"].set_index("game_id").loc[game_id, "season"])
    stats = fetch_stats(season)
    board = board_for(game_id)
    res = v0run.run_games([game_id], n_sims=n_sims, props_override=board,
                          retro=dict(stats=stats, final=final))
    return res["results"][game_id].get("graded_lineups", {})
