"""Run the v0 model for one game or a whole week: projections, prop probabilities, lineups and pages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config, teams
from .data import lines, live, pipeline
from .features import bundle
from .picks import lineups
from .project import v0 as P
from .sim import v0 as S

TABLES = ["games", "team_view", "context", "team_games", "team_pressure", "player_games", "player_pfr_rush", "plays"]
CHECK_PATH = config.BACKTESTS_DIR / "eval" / "v0_check_2024.json"


def load_tables() -> dict:
    T = {n: pipeline.load_table(n) for n in TABLES}
    T["game_lines"] = lines.load_game_lines()
    return T


def slug(g) -> str:
    return f"{int(g['season'])}-w{int(g['week']):02d}-{teams.display(g['away_team']).lower()}-{teams.display(g['home_team']).lower()}"


def _active_with_overrides(T, g, asof, out, ov):
    act = {}
    for team in (g["home_team"], g["away_team"]):
        a = P.active_players(T, team, asof, int(g["season"]), out)
        o = ov[ov["team"] == team] if len(ov) else ov
        if len(o):
            keyed = a.assign(k=a["player_name"].map(lines.norm_name))
            drop = set(o.loc[o["status"] == "out", "player"].map(lines.norm_name))
            a = keyed[~keyed["k"].isin(drop)].drop(columns="k").reset_index(drop=True)
        act[team] = a
    return act


def run_games(game_ids: list[str], n_sims: int = 20000, write_pages: bool = True) -> dict:
    T = load_tables()
    games = T["games"].set_index("game_id")
    seasons = {int(games.loc[g, "season"]) for g in game_ids}
    fits = {s: P.fit(T, s) for s in seasons}
    eng = bundle.engines(T)
    props = lines.load_props()
    props = lines.fair_probs(props) if len(props) else props
    ov = live.overrides()
    results, sims = {}, {}
    for gid in game_ids:
        g = games.loc[gid].copy()
        g["game_id"] = gid
        week, season = int(g["week"]), int(g["season"])
        out, report = live.out_ids(season, week, [g["home_team"], g["away_team"]])
        act = _active_with_overrides(T, g, min(pd.Timestamp.now(tz="UTC"), g["kickoff_utc"]), out, ov)
        proj = P.project_game(T, gid, fits[season], eng=eng, active_override=act)
        sim = S.simulate(proj, n_sims)
        sims[gid] = sim
        results[gid] = dict(proj=proj, sim=sim, injuries=report)
    # props for these games (matched by game_id, or by team for rows without one)
    legs = pd.DataFrame()
    if len(props):
        rows = []
        for gid, r in results.items():
            g = r["proj"]["game"]
            mine = props[(props["game_id"] == gid) | ((props["game_id"] == "") & props["team"].isin([g["home_team"], g["away_team"]]))]
            if mine.empty:
                continue
            mine = mine.assign(game_id=gid)
            cands = pd.concat([tp.players[["player_id", "player_name"]].assign(team=t) for t, tp in r["proj"]["teams"].items()])
            rows.append(lines.resolve_players(mine, cands))
        if rows:
            legs = lineups.leg_table(pd.concat(rows, ignore_index=True), sims, None)
    best = lineups.best_lineups(legs, sims) if len(legs) else {}
    check = load_check()
    pages = []
    if write_pages:
        from .report import v0page
        for gid, r in results.items():
            gl = legs[legs["game_id"] == gid] if len(legs) else legs
            gbest = lineups.best_lineups(gl, {gid: sims[gid]}) if len(gl) else {}
            path = config.ROOT / "output" / "v0" / f"{slug(r['proj']['game'])}.html"
            pages.append(v0page.render(r["proj"], r["sim"], gl, gbest, check, path))
    return dict(results=results, legs=legs, best=best, pages=pages, fits=fits)


def load_check() -> dict | None:
    if not CHECK_PATH.exists():
        return None
    d = json.loads(CHECK_PATH.read_text())
    d["calibration"] = pd.DataFrame(d["calibration"])
    return d


def save_check(rep: dict) -> None:
    CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = dict(overall={k: float(v) for k, v in rep["overall"].items()},
               by_stat=rep["by_stat"].to_dict("records"), calibration=rep["calibration"].to_dict("records"))
    CHECK_PATH.write_text(json.dumps(out, indent=1, default=float))


def week_games(season: int, week: int) -> list[str]:
    T = pipeline.load_table("games")
    g = T[(T["season"] == season) & (T["week"] == week) & T["spread_line"].notna()]
    return g.sort_values("kickoff_utc")["game_id"].tolist()
