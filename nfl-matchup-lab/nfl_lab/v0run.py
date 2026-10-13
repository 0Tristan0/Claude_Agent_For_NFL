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
CHECK_DIR = config.BACKTESTS_DIR / "eval"
CHECK_SEASON = 2024           # season whose check is shown on pages


def load_tables() -> dict:
    T = {n: pipeline.load_table(n) for n in TABLES}
    T["game_lines"] = lines.load_game_lines()
    return T


def slug(g) -> str:
    return f"{int(g['season'])}-w{int(g['week']):02d}-{teams.display(g['away_team']).lower()}-{teams.display(g['home_team']).lower()}"


def _active_with_overrides(T, g, asof, out, ov):
    """Live actives: a snap in the team's last 2 games, minus Out/Doubtful, with data/overrides.csv applied.
    'out' removes a player; 'in' adds him (from his latest game) and, for a QB, makes him the starter."""
    from .features.asof import visible
    act = {}
    hist = visible(T["player_games"], asof)
    for team in (g["home_team"], g["away_team"]):
        a = P.active_players(T, team, asof, int(g["season"]), out).assign(force_starter=False)
        o = ov[ov["team"] == team] if len(ov) else ov
        if len(o):
            key = a["player_name"].map(lines.norm_name)
            drop = set(o.loc[o["status"] == "out", "player"].map(lines.norm_name))
            a = a[~key.isin(drop)].reset_index(drop=True)
            for _, r in o[o["status"] == "in"].iterrows():
                h = hist[hist["player_name"].map(lines.norm_name) == lines.norm_name(r["player"])]
                if h.empty:
                    print(f"override: no past games found for {r['player']} ({team}); not added")
                    continue
                last = h.sort_values("kickoff_utc").iloc[-1]
                a = a[a["player_id"] != last["player_id"]]
                a = pd.concat([a, pd.DataFrame([dict(player_id=last["player_id"], player_name=last["player_name"],
                                                      position=last["position"], force_starter=last["position"] == "QB")])],
                              ignore_index=True)
        act[team] = a.reset_index(drop=True)
    return act


def _outs(T, g, asof) -> tuple[set, str]:
    """Out/Doubtful from this week's injury report; if it isn't published yet, last week's Out/Doubtful
    players who also took no snap in their team's last game (with a warning on the page)."""
    from .features.asof import visible
    season, week = int(g["season"]), int(g["week"])
    out, rep = live.out_ids(season, week, [g["home_team"], g["away_team"]])
    if rep is not None and len(rep):
        return out, f"injury report week {week}"
    prev, rep_prev = live.out_ids(season, week - 1, [g["home_team"], g["away_team"]]) if week > 1 else (set(), None)
    hist = visible(T["player_games"], asof)
    keep = set()
    for team in (g["home_team"], g["away_team"]):
        h = hist[(hist["team"] == team) & (hist["season"] == season)]
        if h.empty:
            continue
        last = h[h["kickoff_utc"] == h["kickoff_utc"].max()]
        played = set(last.loc[last["any_snaps"] > 0, "player_id"])
        keep |= {p for p in prev if p not in played}
    return keep, (f"week {week} injury report not published yet: using week {week - 1} Out/Doubtful players who "
                  f"missed their last game" if week > 1 else "no injury report")


def run_games(game_ids: list[str], n_sims: int = 20000, write_pages: bool = True, allow_holdout: bool = False) -> dict:
    T = load_tables()
    games = T["games"].set_index("game_id")
    seasons = {int(games.loc[g, "season"]) for g in game_ids}
    if config.TEST_SEASON in seasons and not allow_holdout:
        raise ValueError(f"{config.TEST_SEASON} is the holdout season: it is projected only by the final test")
    fits = {s: P.fit(T, s) for s in seasons}
    eng = bundle.engines(T)
    props = lines.load_props()
    props = lines.fair_probs(props) if len(props) else props
    ov = live.overrides()
    now = pd.Timestamp.now(tz="UTC")
    results, sims = {}, {}
    for gid in game_ids:
        g = games.loc[gid].copy()
        g["game_id"] = gid
        asof = min(now, g["kickoff_utc"])
        out, injury_note = _outs(T, g, asof)
        act = _active_with_overrides(T, g, asof, out, ov)
        proj = P.project_game(T, gid, fits[season_of(g)], eng=eng, active_override=act)
        proj["injury_note"] = injury_note
        proj["run_at"] = now
        proj["jerseys"] = jerseys(int(g["season"]), int(g["week"]))
        sim = S.simulate(proj, n_sims)
        sims[gid] = sim
        results[gid] = dict(proj=proj, sim=sim)
    legs = pd.DataFrame()
    unresolved = []
    if len(props):
        rows = []
        for gid, r in results.items():
            g = r["proj"]["game"]
            ko = g["kickoff_utc"]
            blank = (props["game_id"] == "") & props["team"].isin([g["home_team"], g["away_team"]])
            # rows without a game_id count only if captured in the week before this kickoff
            blank &= (props["captured_at"] < ko) & (props["captured_at"] > ko - pd.Timedelta(days=7))
            mine = props[(props["game_id"] == gid) | blank]
            if mine.empty:
                continue
            mine = mine.assign(game_id=gid).sort_values("captured_at")
            mine = mine.drop_duplicates(["player", "team", "stat", "line", "source"], keep="last")
            cands = pd.concat([tp.players[["player_id", "player_name"]].assign(team=t) for t, tp in r["proj"]["teams"].items()])
            res = lines.resolve_players(mine, cands)
            unresolved += [f"{x.player} ({x.team})" for x in res[res["player_id"].isna()].itertuples()]
            rows.append(res)
        if rows:
            legs = lineups.leg_table(pd.concat(rows, ignore_index=True), sims, None)
    if unresolved:
        print("props rows not matched to an active player (check spelling, or he is listed out):", ", ".join(unresolved))
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
    return dict(results=results, legs=legs, best=best, pages=pages, fits=fits, unresolved=unresolved)


_JERSEYS: dict = {}


def jerseys(season: int, week: int) -> dict:
    """Jersey number per player id from the weekly rosters (latest week up to this one). Page display only."""
    if (season, week) not in _JERSEYS:
        from .data import ingest
        try:
            r = ingest.load("rosters", seasons=[season])
        except Exception:                       # no roster file yet: pages just omit numbers
            r = pd.DataFrame(columns=["week", "gsis_id", "jersey_number"])
        r = r[r["week"] <= week].dropna(subset=["gsis_id", "jersey_number"]).sort_values("week", kind="mergesort")
        _JERSEYS[(season, week)] = (r.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["jersey_number"]
                                    .astype(int).to_dict())
    return _JERSEYS[(season, week)]


def season_of(g) -> int:
    return int(g["season"])


def check_path(season: int):
    return CHECK_DIR / f"v0_check_{season}.json"


def load_check(season: int = CHECK_SEASON) -> dict | None:
    p = check_path(season)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["calibration"] = pd.DataFrame(d["calibration"])
    return d


def save_check(rep: dict, season: int, meta: dict) -> None:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    o = rep["overall"]
    overall = {k: (int(v) if k in ("n", "games") else float(v)) for k, v in o.items()}
    out = dict(overall=overall, by_stat=rep["by_stat"].to_dict("records"),
               calibration=rep["calibration"].to_dict("records"), meta=meta)
    check_path(season).write_text(json.dumps(out, indent=1, default=float))


def week_games(season: int, week: int) -> list[str]:
    T = pipeline.load_table("games")
    g = T[(T["season"] == season) & (T["week"] == week) & T["spread_line"].notna()]
    return g.sort_values("kickoff_utc")["game_id"].tolist()
