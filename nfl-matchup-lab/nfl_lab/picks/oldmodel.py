"""The old model: the method behind the original Bills-Rams report (src/ev.py), applied to PrizePicks lines.

For each standard PrizePicks line:
  recent    = games over the line in the player's last 4 games this season, pulled toward 50%:
              (hits + 2.5) / (games + 5)                                  (src/ev.py shrink, prior 5)
  matchup   = +3 points if the opponent is among the 8 defenses allowing the most of that stat this season,
              -3 points if among the 8 allowing the least (the old report set these nudges by hand from the
              unit ranks; here a fixed rule applies the same idea)
  estimate  = average of (recent + matchup) and the market chance, which is the no-vig sportsbook price when
              one is on the row, else 50% (a PrizePicks line taken as a coin flip)
The side with the higher estimate is the pick. Lineups take the highest estimates (one pick per player, at
least two teams) and, like the old method, multiply the legs' chances as if they were independent.
Only data available before kickoff is used (rows with available_at before the game).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import teams
from ..features.asof import visible
from . import payouts

LAST_N = 4
PRIOR = 5
NUDGE = 0.03
TIER = 8

# stat -> (player_games columns summed for the stat, columns summed for "allowed by the defense")
STAT_COLS = {"rec_yds": ["rec_yds"], "receptions": ["receptions"], "rush_yds": ["rush_yds"], "rush_att": ["carries"],
             "pass_yds": ["pass_yds"], "pass_att": ["pass_att"], "completions": ["completions"], "pass_tds": ["pass_tds"],
             "ints": ["ints"], "rush_rec_yds": ["rush_yds", "rec_yds"], "pass_rush_yds": ["pass_yds", "rush_yds"],
             "anytime_td": ["rush_tds", "rec_tds"], "rush_rec_tds": ["rush_tds", "rec_tds"]}


def _value(df: pd.DataFrame, stat: str) -> pd.Series:
    v = df[STAT_COLS[stat]].sum(axis=1)
    return (v >= 1).astype(float) if stat == "anytime_td" else v


def defense_ranks(player_games: pd.DataFrame, asof: pd.Timestamp, season: int) -> dict:
    """stat -> {defense team: rank by amount allowed per game this season, 1 = allows the most}."""
    pg = visible(player_games, asof)
    pg = pg[(pg["season"] == season) & (pg["season_type"] == "REG")]
    out = {}
    if pg.empty:
        return out
    for stat in STAT_COLS:
        allowed = pg[STAT_COLS[stat]].sum(axis=1)          # for TD props: touchdowns allowed
        v = pg.assign(v=allowed)
        per_game = v.groupby(["opponent", "game_id"])["v"].sum().groupby("opponent").mean()
        out[stat] = per_game.rank(ascending=False, method="min").to_dict()
    return out


def estimates(legs: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """One row per standard PrizePicks line: the old model's pick and its estimate."""
    if legs is None or legs.empty:
        return pd.DataFrame()
    pp = legs[legs["prizepicks"] & legs["line_type"].eq("standard")]
    if pp.empty:
        return pd.DataFrame()
    games = tables["games"].set_index("game_id")
    pg = tables["player_games"]
    rows, rank_cache = [], {}
    for (gid, pid, stat, line), g in pp.groupby(["game_id", "player_id", "stat", "line"], sort=False):
        if stat not in STAT_COLS:
            continue
        game = games.loc[gid]
        ko, season = game["kickoff_utc"], int(game["season"])
        team = g["team"].iloc[0]
        opp = game["away_team"] if team == game["home_team"] else game["home_team"]
        if (season, ko) not in rank_cache:
            rank_cache[(season, ko)] = defense_ranks(pg, ko, season)
        rk = rank_cache[(season, ko)].get(stat, {})
        n_def = max(len(rk), 1)
        r = rk.get(opp)
        soft = r is not None and r <= TIER
        tough = r is not None and r > n_def - TIER
        hist = visible(pg[pg["player_id"] == pid], ko)
        hist = hist[(hist["season"] == season) & (hist["any_snaps"] > 0)].sort_values("kickoff_utc").tail(LAST_N)
        vals = _value(hist, stat).to_numpy() if len(hist) else np.array([])
        n = len(vals)
        sides = []
        for side in ("over", "under"):
            if side == "under" and stat == "anytime_td":
                continue
            hits = int((vals > line).sum() if side == "over" else (vals < line).sum())
            recent = (hits + PRIOR * 0.5) / (n + PRIOR)
            nudge = (NUDGE if soft else -NUDGE if tough else 0.0) * (1 if side == "over" else -1)
            leg = g[g["side"] == side]
            mkt = float(leg["p_market"].iloc[0]) if len(leg) and leg["p_market"].notna().iloc[0] else 0.5
            est = (recent + nudge + mkt) / 2
            sides.append(dict(game_id=gid, player_id=pid, player=g["player"].iloc[0], team=team, stat=stat, line=float(line),
                              side=side, p_model=float(est), hits=hits, games=n, nudge=nudge, market=mkt,
                              opp=opp, opp_rank=r))
        if sides:
            rows.append(max(sides, key=lambda d: d["p_model"]))
    return pd.DataFrame(rows)


def _flex_ev(ps: list[float]) -> float | None:
    _, flex = payouts.tables()
    n = len(ps)
    if n not in flex:
        return None
    ev = 0.0
    for combo in itertools.product([0, 1], repeat=n):
        pr = float(np.prod([p if c else 1 - p for c, p in zip(combo, ps)]))
        ev += pr * flex[n].get(sum(combo), 0)
    return ev


def _note(r) -> str:
    s = f"{'More' if r.side == 'over' else 'Less'} in {r.hits} of last {r.games} games"
    if r.nudge and r.opp_rank is not None and r.opp_rank == r.opp_rank:
        soft = r.opp_rank <= TIER
        s += f"; {teams.display(r.opp)} D allows the {'most' if soft else 'least'} of this (rank {int(r.opp_rank)})"
    return s


def best_lineups(est: pd.DataFrame, sizes=(2, 3, 4, 5, 6)) -> dict[str, list[dict]]:
    """Per size, the lineup of the highest estimates (one per player, at least two teams), priced like src/ev.py."""
    power, _ = payouts.tables()
    out = {}
    if est is None or est.empty:
        return {str(n): [] for n in sizes}
    pool = est.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates("player_id")
    for n in sizes:
        pick = pool.head(n)
        if len(pick) < n:
            out[str(n)] = []
            continue
        if pick["team"].nunique() < 2:                    # PrizePicks needs two teams: swap in the best other team
            other = pool[~pool["team"].isin(pick["team"])].head(1)
            if other.empty:
                out[str(n)] = []
                continue
            pick = pd.concat([pick.head(n - 1), other])
        ps = pick["p_model"].tolist()
        p_all = float(np.prod(ps))
        pw = p_all * power[n]
        fx = _flex_ev(ps)
        best = max(pw, fx or 0)
        out[str(n)] = [dict(
            n=n, recommended=bool(best > 1.0), basis="old model", play_as="Power" if pw >= (fx or 0) + 0.03 else "Flex",
            p_all=p_all, p_independent=p_all, power_ev=pw, flex_ev=fx, best_ev=best, min_power_mult=1 / p_all,
            legs=[dict(player=r.player, team=teams.display(r.team), stat=r.stat, side=r.side, line=r.line,
                       p_model=r.p_model, game_id=r.game_id,
                       note=_note(r))
                  for r in pick.itertuples()])]
    return out
