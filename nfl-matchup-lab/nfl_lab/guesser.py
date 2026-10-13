"""The Guesser method: how the Bills-Rams $20 entries were built, applied to every game. No game simulation.

For each standard PrizePicks line in a game:
  recent    games over (or under) the line in the player's last 4 games this season, pulled toward 50%:
            (hits + 2.5) / (games + 5)                                         (the original src/ev.py)
  market    the no-vig sportsbook chance if a price is on the row, else 50% (a PrizePicks line is set near even)
  matchup   this game's matchups (picks/matchups.py): pass rush vs offensive line, deep receivers under
            pressure, receiving yards the defense allows to the position, pass and run defense, game script,
            red-zone defense; capped at 8 points (the original's hand nudges ran -1 to +8), reasons written out
  estimate  (recent + matchup + market) / 2, exactly the original formula ("half our read, half the market");
            the side with the higher estimate is the pick
Confidence tiers come from the estimate. Players with fewer than 3 games this season are shown but kept out of
lineups (too small a sample, like Parkinson on the original board).

Lineups use one game only: the top picks by estimate (Lean, 52.5%, or better), one per player, both teams. Entry A is the
top 3 as a Flex (it still pays at 2 of 3), Entry B the top 4 as a Power, and the 5- and 6-pick add the next picks
(played as whichever of Power or Flex returns more). Like the
original, a lineup's chance multiplies the picks' estimates. The Guesser puts a pretend $20 on each game's
Entry A ($12) and Entry B ($8), frozen before kickoff, so the Lineups tab and the Guesser always match.
Only stats from before kickoff are used.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, teams
from .data import lines, live
from .features import bundle
from .features.asof import visible
from .picks import matchups, oldmodel, payouts
from .report.picks_html import GUESSER_STAKE

RECENT_N, PRIOR = 4, 5
MATCHUP_CAP = 0.08
MIN_GAMES = 3
MIN_PICK = 0.525                           # Lean or better: a coin flip is not a pick
TIERS = [(0.60, "High"), (0.575, "Med-High"), (0.55, "Medium"), (0.525, "Lean"), (0.0, "Coin flip")]
STAKES = GUESSER_STAKE                     # Entry A $12, Entry B $8
OUT_DIR = config.ROOT / "output" / "games"
LIVE_DIR = config.BACKTESTS_DIR / "live"
PP_STAT = {"rec_yds": "rec yds", "rush_yds": "rush yds", "pass_yds": "pass yds", "receptions": "receptions",
           "anytime_td": "rush+rec TDs", "rush_rec_tds": "rush+rec TDs", "rush_rec_yds": "rush+rec yds",
           "pass_tds": "pass TDs", "pass_att": "pass attempts", "completions": "completions", "rush_att": "rush attempts",
           "ints": "INTs", "pass_rush_yds": "pass+rush yds"}


def recent_rate(hits: int, games: int) -> float:
    """Hits at the line over his last games, pulled toward 50% (src/ev.py shrink)."""
    return (hits + PRIOR * 0.5) / (games + PRIOR)


def estimate(hits: int, games: int, matchup: float, market: float) -> float:
    """The original formula: (recent + matchup + market) / 2, "half our read, half the market"."""
    return (recent_rate(hits, games) + matchup + market) / 2


def tier(p: float) -> str:
    return next(name for cut, name in TIERS if p >= cut)


def game_label(g) -> str:
    return f"{teams.display(g['away_team'])} at {teams.display(g['home_team'])}"


def slug(g) -> str:
    return (f"{int(g['season'])}-w{int(g['week']):02d}-{teams.display(g['away_team']).lower()}-"
            f"{teams.display(g['home_team']).lower()}")


def _board(props: pd.DataFrame, g) -> pd.DataFrame:
    """Standard PrizePicks lines for this game, captured in the week before kickoff."""
    if props is None or props.empty:
        return pd.DataFrame()
    ko = g["kickoff_utc"]
    pp = props[props["source"].str.lower().eq("prizepicks") & props["line_type"].eq("standard")
               & props["team"].isin([g["home_team"], g["away_team"]]) & props["stat"].isin(list(oldmodel.STAT_COLS))
               & props["game_id"].isin(["", g["game_id"]])
               & (props["captured_at"] < ko) & (props["captured_at"] > ko - pd.Timedelta(days=7))]
    return pp.sort_values("captured_at").drop_duplicates(["player", "team", "stat", "line"], keep="last")


def rate_game(T: dict, eng, props: pd.DataFrame, game_id: str, ov: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """One row per line: the pick, its estimate and the reasons. Returns (picks, injury note)."""
    from . import v0run
    games = T["games"].set_index("game_id")
    g = games.loc[game_id].copy()
    g["game_id"] = game_id
    ko, season = g["kickoff_utc"], int(g["season"])
    board = _board(props, g)
    if board.empty:
        return pd.DataFrame(), ""
    out, note = v0run._outs(T, g, ko)
    ov = ov if ov is not None else live.overrides()
    bo, starters = v0run.board_overrides(props, g)
    act = v0run._active_with_overrides(T, g, ko, out, pd.concat([ov, bo], ignore_index=True), starters,
                                       quiet=set(bo["player"]))
    cands = pd.concat([a.assign(team=t) for t, a in act.items()], ignore_index=True)
    res = lines.resolve_players(board, cands)
    res = res.dropna(subset=["player_id"])
    pos = dict(zip(cands["player_id"], cands["position"]))
    pg = visible(T["player_games"], ko)
    pg = pg[(pg["season"] == season) & (pg["any_snaps"] > 0)]
    count_stats = set(config.load("stats")["count_stats"]["keys"])
    rows = []
    for _, r in res.iterrows():
        if r["stat"] in count_stats and float(r["line"]).is_integer():
            continue                                             # a push would void the leg; skip whole numbers
        h = pg[pg["player_id"] == r["player_id"]].sort_values("kickoff_utc").tail(RECENT_N)
        vals = oldmodel._value(h, r["stat"]).to_numpy() if len(h) else np.array([])
        n = len(vals)
        for side in ("over", "under"):
            if side == "under" and r["stat"] == "anytime_td":
                continue
            hits = int((vals > r["line"]).sum() if side == "over" else (vals < r["line"]).sum())
            recent = recent_rate(hits, n)
            fo = r.get("fair_over", np.nan)
            mkt = (float(fo) if side == "over" else 1 - float(fo)) if fo == fo and fo is not None else 0.5
            rows.append(dict(game_id=game_id, player_id=r["player_id"], player=r["player"], team=r["team"],
                             position=pos.get(r["player_id"], ""), stat=r["stat"], line=float(r["line"]), side=side,
                             hits=hits, games=n, log=[float(x) for x in vals], recent=recent, market=mkt,
                             p_model=recent, captured_at=str(r["captured_at"])))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, note
    df = matchups.adjust(df, matchups.context(T, game_id, eng), pos, cap=MATCHUP_CAP)   # p_model = recent + matchup
    df["p_model"] = (df["p_model"] + df["market"]) / 2                                   # half our read, half the market
    df = df.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates(["player_id", "stat", "line"])
    df["tier"] = df["p_model"].map(tier)
    return df.reset_index(drop=True), note


def _note(r) -> str:
    log = ", ".join(f"{v:g}" for v in r["log"]) if r["stat"] != "anytime_td" else \
        ", ".join("TD" if v else "no TD" for v in r["log"])
    word = "More" if r["side"] == "over" else "Less"
    return f"{r['tier']} · {word} in {r['hits']} of last {r['games']}" + (f" ({log})" if log else "")


def _flex_ev(ps: list[float]) -> float | None:
    _, flex = payouts.tables()
    n = len(ps)
    if n not in flex:
        return None
    return float(sum(np.prod([p if c else 1 - p for c, p in zip(combo, ps)]) * flex[n].get(sum(combo), 0)
                     for combo in itertools.product([0, 1], repeat=n)))


def lineups(picks: pd.DataFrame, label: str = "", sizes=(3, 4, 5, 6)) -> dict:
    """Entry A (3-pick Flex), Entry B (4-pick Power), then 5 and 6 picks: the top estimates, one per player."""
    power, _ = payouts.tables()
    out = {}
    if picks is None or picks.empty:
        return out
    pool = picks[(picks["games"] >= MIN_GAMES) & (picks["p_model"] >= MIN_PICK)]
    pool = pool.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates("player_id")
    for n in sizes:
        pick = pool.head(n)
        if len(pick) < n:
            continue
        if pick["team"].nunique() < 2:                       # PrizePicks needs both teams
            other = pool[~pool["team"].isin(pick["team"])].head(1)
            if other.empty:
                continue
            pick = pd.concat([pick.head(n - 1), other])
        ps = pick["p_model"].tolist()
        p_all = float(np.prod(ps))
        pw, fx = p_all * power[n], _flex_ev(ps)
        play = {3: "Flex", 4: "Power"}.get(n) or ("Power" if pw >= (fx or 0) else "Flex")   # A and B as on the original slip
        best = fx if play == "Flex" else pw
        entry = {3: "A", 4: "B"}.get(n)
        out[str(n)] = [dict(
            n=n, entry=entry, recommended=bool(best > 1.0), basis="Guesser method", play_as=play, game=label,
            p_all=p_all, p_independent=p_all, power_ev=pw, flex_ev=fx, best_ev=best, min_power_mult=1 / p_all,
            legs=[dict(player=r.player, player_id=r.player_id, team=teams.display(r.team), stat=r.stat, side=r.side,
                       line=r.line, p_model=float(r.p_model), game_id=r.game_id, note=_note(r._asdict()),
                       reasons=list(r.reasons)) for r in pick.itertuples()])]
    return out


def leg_text(x: dict) -> str:
    return f"{x['player']} {PP_STAT.get(x['stat'], x['stat'])} {'more' if x['side'] == 'over' else 'less'} {x['line']:g}"


def write_bankroll(g, lu: dict, now: pd.Timestamp) -> Path | None:
    """The $20 paper bet on Entry A ($12 Flex) and Entry B ($8 Power). Never rewritten once the game has started."""
    path = LIVE_DIR / f"{slug(g)}-bankroll.json"
    ko = pd.Timestamp(g["kickoff_utc"])
    if path.exists() and now >= ko:
        return None                                        # frozen: the game has started
    if "3" not in lu or "4" not in lu:
        path.unlink(missing_ok=True)                       # no entries any more (before kickoff): no bet
        return None
    _, flex = payouts.tables()
    power, _ = payouts.tables()
    A, B = lu["3"][0], lu["4"][0]
    entries = [dict(id="A", type="flex", stake=STAKES["A"], legs=[leg_text(x) for x in A["legs"]],
                    legs_detail=A["legs"], pays={str(k): v for k, v in flex[3].items()}),
               dict(id="B", type="power", stake=STAKES["B"], legs=[leg_text(x) for x in B["legs"]],
                    legs_detail=B["legs"], pays={"4": power[4]})]
    d = dict(game=g["game_id"], label=game_label(g), kickoff_utc=f"{ko:%Y-%m-%d %H:%M} UTC",
             frozen_at=f"{now:%Y-%m-%d %H:%M} UTC", bankroll=20.0, reserve_not_bet=0.0, paper_money=True,
             picks_source="Guesser method (nfl_lab/guesser.py): recent form + market + matchups, same game",
             entries=entries, expected_return_per_dollar={"A_flex3": dict(estimate=round(A["best_ev"], 3)),
                                                        "B_power4": dict(estimate=round(B["best_ev"], 3))},
             result=None)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, default=str))
    return path


def run_week(season: int, week: int, write: bool = True) -> dict:
    """Rate every game of the week that has PrizePicks lines; write per-game results and Guesser entries."""
    from . import v0run
    T = v0run.load_tables()
    eng = bundle.engines(T)
    props = lines.load_props()
    props = lines.fair_probs(props) if len(props) else props
    games = T["games"]
    gl = games[(games["season"] == season) & (games["week"] == week)].sort_values("kickoff_utc")
    now = pd.Timestamp.now(tz="UTC")
    out = {}
    for _, g in gl.iterrows():
        gid = g["game_id"]
        if now >= g["kickoff_utc"] and (OUT_DIR / f"{slug(g)}.json").exists():
            out[gid] = json.loads((OUT_DIR / f"{slug(g)}.json").read_text())   # started: keep what was frozen
            continue
        picks, note = rate_game(T, eng, props, gid)
        lu = lineups(picks, game_label(g))
        ctx = matchups.context(T, gid, eng)
        rec = dict(game_id=gid, season=season, week=week, label=game_label(g), away=teams.display(g["away_team"]),
                   home=teams.display(g["home_team"]), kickoff_utc=str(g["kickoff_utc"]),
                   home_spread=float(-g["spread_line"]) if g["spread_line"] == g["spread_line"] else None,
                   total=float(g["total_line"]) if g["total_line"] == g["total_line"] else None,
                   injury_note=note, built_at=str(now), units=unit_table(ctx, g),
                   picks=[] if picks.empty else json.loads(picks.drop(columns=["reasons"]).assign(
                       reasons=picks["reasons"]).to_json(orient="records")),
                   lineups=lu)
        out[gid] = rec
        if write:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{slug(g)}.json").write_text(json.dumps(rec, indent=1, default=str))
            write_bankroll(g, lu, now)
    return out


def unit_table(ctx: dict, g) -> list[dict]:
    """Unit vs unit: each offense against the defense it faces, ranks out of 32 (1 = best for that unit)."""
    tf = ctx["tf"]
    rows = []
    for team, opp in ((g["away_team"], g["home_team"]), (g["home_team"], g["away_team"])):
        if team not in tf.index or opp not in tf.index:
            continue
        o, d = tf.loc[team], tf.loc[opp]
        def rk(col, val, high_good):
            s = tf[col].dropna()
            return int((s > val).sum() + 1) if high_good else int((s < val).sum() + 1)
        for label, ocol, dcol, o_high_good in (
                ("EPA per dropback", "off_pass_epa", "def_pass_epa", True),
                ("EPA per rush", "off_rush_epa", "def_rush_epa", True),
                ("Explosive pass rate", "off_expl_pass_rate", "def_expl_pass_rate", True),
                ("Pressure rate (PFR)", "off_pressure_rate_pfr", "def_pressure_rate_pfr", False),
                ("Red-zone TD rate", "off_rz_td_rate", "def_rz_td_rate", True)):
            if ocol not in tf or dcol not in tf:
                continue
            # offense rank: 1 = best offense; defense rank: 1 = best defense (allows least / pressures most)
            orank = rk(ocol, o[ocol], o_high_good)
            drank = rk(dcol, d[dcol], not o_high_good) if label != "Pressure rate (PFR)" else rk(dcol, d[dcol], True)
            rows.append(dict(offense=teams.display(team), defense=teams.display(opp), stat=label,
                             off_value=float(o[ocol]), def_value=float(d[dcol]), off_rank=orank, def_rank=drank))
    return rows


def grade_week(season: int, week: int, stats: pd.DataFrame) -> list[str]:
    """Fill in results for this week's Guesser bets whose games have official stats. Returns graded game ids."""
    from .backtest import retro
    done = []
    for path in sorted(LIVE_DIR.glob(f"{season}-w{week:02d}-*-bankroll.json")):
        d = json.loads(path.read_text())
        if d.get("result") and not d["result"].get("provisional"):
            continue
        legs, missing = {}, False
        for en in d["entries"]:
            for x, text in zip(en.get("legs_detail", []), en["legs"]):
                v = retro.actual(stats, x["player_id"], week, x["stat"])
                if v is None:
                    missing = True
                    continue
                hit = v > x["line"] if x["side"] == "over" else v < x["line"]
                shown = ("scored" if v else "no TD") if x["stat"] == "anytime_td" else v
                legs[text] = dict(actual=shown, hit=bool(hit))
        if not legs or missing:
            continue
        d["result"] = dict(final="", graded_at=f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", provisional=False,
                           source=f"nflverse stats_player_week_{season}", legs=legs)
        path.write_text(json.dumps(d, indent=2, default=str))
        done.append(d["game"])
    return done
