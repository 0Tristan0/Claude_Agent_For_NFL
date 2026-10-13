"""Game-specific matchup reasoning for each PrizePicks pick.

For one game, every factor compares this offense with this defense, using only stats available before kickoff
(team rates are recency weighted: this season counts most, last season a little). Each factor gives a score from
-1 to +1 for the pick's side, from where the matchup ranks among the 32 teams, and a short reason in plain words:

  pressure      the defense's pressure rate (PFR) vs the pressure this offense's line allows. Heavy pressure:
                fewer passing yards and touchdowns, fewer deep targets (receivers with an aDOT of 11+), more QB
                scrambles, more short checkdowns to backs and tight ends
  pass defense  EPA per dropback the defense allows: a soft pass defense helps passing and receiving overs
  run defense   EPA per rush the defense allows: a soft run defense helps rushing overs
  position      receiving yards per game the defense has allowed to this player's position (WR, TE or RB)
  game script   the spread: big underdogs throw more, big favorites run more late
  red zone      the defense's touchdown rate allowed per red-zone trip, for touchdown picks

The adjustment added to the pick's chance is the weighted sum of the scores, capped at 5 points either way.
These weights are rules of thumb, the same kind of judgment the original report used; they have not been tested
on past seasons yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import teams
from ..features.asof import visible

WEIGHTS = {"pressure": 0.02, "pass_def": 0.015, "run_def": 0.015, "position": 0.02, "script": 0.015, "red_zone": 0.015}
CAP = 0.05
DEEP_ADOT = 11.0
REASON_MIN = 0.35            # only factors this strong are written up as reasons

PASS_STATS = {"pass_yds", "pass_tds", "completions", "pass_rush_yds"}
REC_STATS = {"rec_yds", "receptions", "rush_rec_yds"}
RUSH_STATS = {"rush_yds", "rush_att"}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _pct_rank(series: pd.Series, value: float, higher_is_more: bool = True) -> float:
    """0..1, where 1 = the most favorable to More among the 32 teams."""
    s = series.dropna()
    if s.empty or value != value:
        return 0.5
    p = float((s < value).mean() + 0.5 * (s == value).mean())
    return p if higher_is_more else 1 - p


def _rank(series: pd.Series, value: float, descending: bool = True) -> int:
    s = series.dropna()
    return int((s > value).sum() + 1) if descending else int((s < value).sum() + 1)


def context(tables: dict, game_id: str, eng) -> dict:
    """Team and position-vs-defense numbers for both sides of one game, as of kickoff."""
    g = tables["games"].set_index("game_id").loc[game_id]
    ko, season = g["kickoff_utc"], int(g["season"])
    league = sorted(set(tables["games"].loc[tables["games"]["season"] == season, "home_team"]))
    tf = eng.team.features(league, ko, season).set_index("team")
    pg = visible(tables["player_games"], ko)
    pg = pg[(pg["season"] == season) & (pg["season_type"] == "REG")]
    allowed = {}
    for pos in ("WR", "TE", "RB"):
        x = pg[pg["position"] == pos].groupby(["opponent", "game_id"])["rec_yds"].sum().groupby("opponent").mean()
        allowed[pos] = x
    adot = (pg.groupby("player_id")["air_yards"].sum() / pg.groupby("player_id")["targets"].sum().replace(0, np.nan))
    return dict(game=g, tf=tf, allowed=allowed, adot=adot, spread_home=float(-g["spread_line"]))


def _factors(row, pos: str, ctx: dict) -> list[tuple[str, float, str]]:
    """(name, score for More, reason when it favors More, reason when it favors Less)."""
    g, tf = ctx["game"], ctx["tf"]
    team = row["team"]
    opp = g["away_team"] if team == g["home_team"] else g["home_team"]
    if team not in tf.index or opp not in tf.index:
        return []
    o, d = tf.loc[team], tf.loc[opp]
    T, O = teams.display(team), teams.display(opp)
    stat = row["stat"]
    out = []
    # pressure: defense's pressure generated and this line's pressure allowed, averaged as percentiles
    if "def_pressure_rate_pfr" in tf and "off_pressure_rate_pfr" in tf:
        p_def = _pct_rank(tf["def_pressure_rate_pfr"], d["def_pressure_rate_pfr"])
        p_off = _pct_rank(tf["off_pressure_rate_pfr"], o["off_pressure_rate_pfr"])
        heat = (p_def + p_off) / 2                                 # 1 = heavy pressure expected
        pr = _rank(tf["def_pressure_rate_pfr"], d["def_pressure_rate_pfr"])
        lr = _rank(tf["off_pressure_rate_pfr"], o["off_pressure_rate_pfr"])
        desc = (f"{O} pressure QBs on {d['def_pressure_rate_pfr']:.0%} of dropbacks ({_ordinal(pr)}); {T}'s line allows "
                f"{o['off_pressure_rate_pfr']:.0%} ({_ordinal(lr)} most)")
        adot = ctx["adot"].get(row["player_id"], np.nan)
        s = None
        if pos == "QB" and stat in PASS_STATS:
            s = -(2 * heat - 1)
            why = desc + (": a hard night to throw" if heat > 0.5 else ": a clean pocket")
        elif pos == "QB" and stat == "rush_yds":
            s = 2 * heat - 1
            why = desc + (": more scrambles" if heat > 0.5 else ": fewer reasons to run")
        elif stat in REC_STATS and pos == "WR" and adot == adot and adot >= DEEP_ADOT:
            s = -(2 * heat - 1)
            why = desc + (f": less time for deep throws (his targets travel {adot:.0f} yds)" if heat > 0.5
                          else f": time for deep throws (his targets travel {adot:.0f} yds)")
        elif stat in REC_STATS and pos in ("RB", "TE"):
            s = 2 * heat - 1
            why = desc + (": more quick checkdowns" if heat > 0.5 else ": fewer checkdowns")
        if s is not None:
            out.append(("pressure", s, why))
    # pass and run defense quality
    if stat in PASS_STATS | REC_STATS and pos != "RB" or (stat in REC_STATS and pos == "RB"):
        p = _pct_rank(tf["def_pass_epa"], d["def_pass_epa"])       # higher EPA allowed = softer
        r = _rank(tf["def_pass_epa"], d["def_pass_epa"])
        out.append(("pass_def", 2 * p - 1, f"{O}'s pass defense ranks {_ordinal(33 - r)} of 32 by EPA per dropback allowed"))
    if stat in RUSH_STATS and pos != "QB":
        p = _pct_rank(tf["def_rush_epa"], d["def_rush_epa"])
        r = _rank(tf["def_rush_epa"], d["def_rush_epa"])
        out.append(("run_def", 2 * p - 1, f"{O}'s run defense ranks {_ordinal(33 - r)} of 32 by EPA per rush allowed"))
    # position-specific receiving yards allowed
    if stat in REC_STATS and pos in ctx["allowed"]:
        a = ctx["allowed"][pos]
        if opp in a.index:
            p = _pct_rank(a, a[opp])
            r = _rank(a, a[opp])
            out.append(("position", 2 * p - 1, f"{O} allow {a[opp]:.0f} receiving yds a game to {pos}s ({_ordinal(r)} most)"))
    # game script from the spread (team spread < 0 = favorite)
    spread = ctx["spread_home"] if team == g["home_team"] else -ctx["spread_home"]
    size = float(np.clip(abs(spread) / 10, 0, 1))
    if abs(spread) >= 3:
        fav = spread < 0
        if pos == "QB" and stat in {"pass_att", "completions", "pass_yds"}:
            out.append(("script", -size if fav else size,
                        f"{T} {'favored' if fav else 'underdog'} by {abs(spread):g}: "
                        + ("may lean on the run late" if fav else "likely throwing to catch up")))
        elif pos == "RB" and stat in RUSH_STATS:
            out.append(("script", size if fav else -size,
                        f"{T} {'favored' if fav else 'underdog'} by {abs(spread):g}: "
                        + ("more carries to close it out" if fav else "may abandon the run if behind")))
        elif stat in REC_STATS and pos in ("WR", "TE"):
            out.append(("script", -size * 0.5 if fav else size * 0.5,
                        f"{T} {'favored' if fav else 'underdog'} by {abs(spread):g}: "
                        + ("fewer passes late" if fav else "more passes if trailing")))
    # red zone for touchdown picks
    if stat in {"anytime_td", "rush_rec_tds", "pass_tds"} and "def_rz_td_rate" in tf:
        p = _pct_rank(tf["def_rz_td_rate"], d["def_rz_td_rate"])
        out.append(("red_zone", 2 * p - 1, f"{O} allow touchdowns on {d['def_rz_td_rate']:.0%} of red-zone trips"))
    return out


def adjust(picks: pd.DataFrame, ctx: dict, positions: dict) -> pd.DataFrame:
    """Add the matchup adjustment and reasons to blended picks for one game (columns p_matchup, reasons)."""
    if picks is None or picks.empty:
        return picks
    picks = picks.copy()
    adj, reasons = [], []
    for _, r in picks.iterrows():
        sign = 1 if r["side"] == "over" else -1
        fs = _factors(r, positions.get(r["player_id"], ""), ctx)
        a = float(np.clip(sum(WEIGHTS[n] * s * sign for n, s, _ in fs), -CAP, CAP))
        adj.append(a)
        strong = sorted([(WEIGHTS[n] * s * sign, why) for n, s, why in fs if abs(s) >= REASON_MIN],
                        key=lambda t: -abs(t[0]))
        reasons.append([("For: " if w > 0 else "Against: ") + why for w, why in strong[:3]])
    picks["p_matchup"] = adj
    picks["reasons"] = reasons
    picks["p_model"] = np.clip(picks["p_model"] + picks["p_matchup"], 0.01, 0.99)
    return picks
