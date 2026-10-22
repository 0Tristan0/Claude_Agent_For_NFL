"""A small fake league in nflverse raw format, for offline tests.

8 teams x 3 seasons (2021-2023), 7 regular-season weeks plus one playoff week per season. Each week
has a Thursday night game, two Sunday 1:00 games, a Sunday 4:25 game (same-day slots) and, in some
weeks, a Monday game. Every code path the pipeline uses is exercised: neutral and garbage-time plays,
red zone and inside-10 plays, scrambles, kneels, spikes, sacks, interceptions, two-point tries,
no_play penalties, participation, FTN, PFR pressure and rushing tables, and one WR who takes snaps but
never sees a target (the zero-target spine case).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

TEAMS = ["BUF", "LA", "KC", "SF", "DAL", "PHI", "MIA", "DET"]
SEASONS = [2021, 2022, 2023]
STADIUM = {"BUF": "BUF00", "LA": "LAX01", "KC": "KAN00", "SF": "SFO01", "DAL": "DAL00", "PHI": "PHI00",
           "MIA": "MIA00", "DET": "DET00"}
SLOTS = [  # (day offset from Thursday, local ET time)
    (0, "20:15"), (3, "13:00"), (3, "13:00"), (3, "16:25"),
]
ROLES = [("QB", "QB1", 0.0, 0.0), ("RB", "RB1", 0.12, 0.65), ("RB", "RB2", 0.04, 0.30), ("WR", "WR1", 0.27, 0.0),
         ("WR", "WR2", 0.20, 0.0), ("WR", "WR3", 0.14, 0.0), ("TE", "TE1", 0.17, 0.0), ("TE", "TE2", 0.06, 0.0),
         ("WR", "WR4", 0.0, 0.0)]   # WR4 plays snaps but is never targeted


def pid(team: str, role: str, season: int) -> str:
    # One player changes team between seasons (KC WR1 -> BUF WR1 in 2023) to exercise history across teams.
    if team == "BUF" and role == "WR1" and season >= 2023:
        return "00-KCWR1"
    if team == "KC" and role == "WR1" and season >= 2023:
        return "00-KCWR1N"
    return f"00-{team}{role}"


def round_robin(teams):
    t = list(teams)
    out = []
    for r in range(len(t) - 1):
        pairs = [(t[i], t[-1 - i]) for i in range(len(t) // 2)]
        out.append(pairs)
        t = [t[0]] + [t[-1]] + t[1:-1]
    return out


def schedule() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        start = date(season, 9, 9)
        start += timedelta(days=(3 - start.weekday()) % 7)          # first Thursday on/after Sep 9
        rr = round_robin(TEAMS)
        for w, pairs in enumerate(rr, start=1):
            thu = start + timedelta(weeks=w - 1)
            for k, (a, b) in enumerate(pairs):
                home, away = (a, b) if (w + k + season) % 2 else (b, a)
                off, tm = SLOTS[k]
                day = thu + timedelta(days=off)
                if w % 3 == 0 and k == 3:
                    day, tm = thu + timedelta(days=4), "20:15"     # Monday night
                rows.append(dict(game_id=f"{season}_{w:02d}_{away}_{home}", season=season, game_type="REG", week=w,
                                 gameday=day.isoformat(), gametime=tm, away_team=away, home_team=home,
                                 location="Home", away_rest=7, home_rest=7, div_game=0,
                                 roof="dome" if home in ("LA", "DET", "DAL") else "outdoors", surface="grass",
                                 stadium_id=STADIUM[home], stadium=STADIUM[home], spread_line=float((k % 3) * 1.5),
                                 total_line=44.5 + k, home_moneyline=-150, away_moneyline=130, home_spread_odds=-110,
                                 away_spread_odds=-110, over_odds=-110, under_odds=-110,
                                 temp=np.nan, wind=np.nan, home_qb_name=f"{home} QB", away_qb_name=f"{away} QB"))
        wc = start + timedelta(weeks=8, days=2)                     # one playoff Saturday
        rows.append(dict(game_id=f"{season}_19_DAL_BUF", season=season, game_type="WC", week=19,
                         gameday=wc.isoformat(), gametime="16:30", away_team="DAL", home_team="BUF", location="Home",
                         away_rest=13, home_rest=13, div_game=0, roof="outdoors", surface="grass",
                         stadium_id="BUF00", stadium="BUF00", spread_line=3.0, total_line=47.5, home_moneyline=-150,
                         away_moneyline=130, home_spread_odds=-110, away_spread_odds=-110, over_odds=-110,
                         under_odds=-110, temp=np.nan, wind=np.nan, home_qb_name="x", away_qb_name="y"))
    return pd.DataFrame(rows)


def _players() -> pd.DataFrame:
    rows = {}
    for season in SEASONS:
        for t in TEAMS:
            for pos, role, _, _ in ROLES:
                g = pid(t, role, season)
                rows[g] = dict(gsis_id=g, pfr_id="pfr" + g[3:], display_name=f"{t} {role}" if g != "00-KCWR1"
                               else "Traded Receiver", position=pos)
    return pd.DataFrame(rows.values())


def _team_roster(team: str, season: int):
    return [(pid(team, role, season), pos, role, ts, cs) for pos, role, ts, cs in ROLES]


def _game_plays(rng, game, season):
    """Plays for one game: alternating drives, plus kneels, spikes, two-point tries and penalties."""
    home, away = game["home_team"], game["away_team"]
    plays, play_id = [], 1
    score = {home: 0, away: 0}
    secs = 3600
    drive = 0
    pos = away
    while secs > 0 and drive < 24:
        drive += 1
        off, de = pos, (home if pos == away else away)
        roster = _team_roster(off, season)
        yl = int(rng.integers(55, 80))
        n = int(rng.integers(3, 10))
        result = rng.choice(["Touchdown", "Field goal", "Punt", "Turnover"], p=[0.25, 0.2, 0.45, 0.10])
        score_pre = score[off]
        for j in range(n):
            if secs <= 0:
                break
            qtr = min(4, 1 + (3600 - secs) // 900)
            diff = score[off] - score[de]
            wp = float(np.clip(0.5 + diff / 40 + rng.normal(0, 0.05), 0.01, 0.99))
            last = j == n - 1
            is_pass = rng.random() < 0.58
            row = dict(game_id=game["game_id"], play_id=play_id, season=season, week=game["week"],
                       season_type="REG" if game["game_type"] == "REG" else "POST", posteam=off, defteam=de,
                       home_team=home, away_team=away, qtr=qtr, down=1 + j % 4, ydstogo=10, yardline_100=yl,
                       game_seconds_remaining=secs, half_seconds_remaining=secs % 1800 or 1800, fixed_drive=drive,
                       fixed_drive_result=result, wp=wp, score_differential=diff, posteam_score=score[off],
                       defteam_score=score[de], two_point_attempt=0.0, field_goal_attempt=0.0, punt_attempt=0.0,
                       extra_point_attempt=0.0, drive_inside20=float(yl <= 20), qb_hit=0.0, interception=0.0,
                       fumble_lost=0.0, sack=0.0, qb_scramble=0.0, qb_kneel=0.0, qb_spike=0.0, touchdown=0.0,
                       pass_touchdown=0.0, rush_touchdown=0.0, complete_pass=0.0, pass_attempt=0.0, rush_attempt=0.0,
                       qb_dropback=0.0, epa=float(rng.normal(0, 1)), success=float(rng.random() < 0.45),
                       xpass=float(rng.uniform(0.3, 0.8)))
            row["pass_oe"] = (float(is_pass) - row["xpass"]) * 100
            gained = int(np.clip(rng.gamma(2.0, 3.5) - 1, -5, 60))
            qb = roster[0][0]
            if is_pass:
                r = rng.random()
                if r < 0.06:      # sack
                    row.update(play_type="pass", **{"pass": 1.0}, qb_dropback=1.0, pass_attempt=1.0, sack=1.0,
                               passer_player_id=qb, yards_gained=-6, qb_hit=1.0)
                elif r < 0.11:    # scramble
                    row.update(play_type="run", **{"pass": 1.0}, qb_dropback=1.0, qb_scramble=1.0, rush_attempt=1.0,
                               rusher_player_id=qb, rushing_yards=gained, yards_gained=gained)
                else:
                    shares = np.array([x[3] for x in roster])
                    rec = roster[int(rng.choice(len(roster), p=shares / shares.sum()))][0]
                    air = float(rng.normal(8, 6))
                    comp = rng.random() < 0.65
                    row.update(play_type="pass", **{"pass": 1.0}, qb_dropback=1.0, pass_attempt=1.0,
                               passer_player_id=qb, receiver_player_id=rec, air_yards=air,
                               complete_pass=float(comp), yards_gained=gained if comp else 0,
                               passing_yards=gained if comp else np.nan, receiving_yards=gained if comp else np.nan,
                               yards_after_catch=max(0.0, gained - air) if comp else np.nan,
                               interception=float((not comp) and rng.random() < 0.05), qb_hit=float(rng.random() < 0.1))
            else:
                shares = np.array([x[4] for x in roster])
                ru = roster[int(rng.choice(len(roster), p=shares / shares.sum()))][0]
                row.update(play_type="run", rush=1.0, rush_attempt=1.0, rusher_player_id=ru, rushing_yards=gained,
                           yards_gained=gained, fumble_lost=float(rng.random() < 0.01))
            yl = int(np.clip(yl - max(gained, 0), 1, 99))
            if last and result == "Touchdown" and row["play_type"] in ("pass", "run") and row.get("sack", 0) == 0:
                if row.get("complete_pass", 0) == 1 or row["play_type"] == "run":
                    row["touchdown"] = 1.0
                    row["td_team"] = off
                    if row["play_type"] == "pass":
                        row["pass_touchdown"] = 1.0
                        row["td_player_id"] = row["receiver_player_id"]
                    else:
                        row["rush_touchdown"] = 1.0
                        row["td_player_id"] = row["rusher_player_id"]
                    row["yardline_100"] = min(row["yardline_100"], 9)
                    score[off] += 6
            row["posteam_score_post"] = score[off]
            plays.append(row)
            play_id += 1
            secs -= int(rng.integers(20, 45))
        if result == "Touchdown" and score[off] > score_pre:
            plays.append(dict(game_id=game["game_id"], play_id=play_id, season=season, week=game["week"],
                              season_type="REG", posteam=off, defteam=de, home_team=home, away_team=away, qtr=4,
                              play_type="pass", **{"pass": 1.0}, qb_dropback=1.0, pass_attempt=1.0,
                              two_point_attempt=1.0, passer_player_id=roster[0][0],
                              receiver_player_id=roster[3][0], complete_pass=1.0, fixed_drive=drive,
                              game_seconds_remaining=max(secs, 1), half_seconds_remaining=max(secs % 1800, 1),
                              yardline_100=2, wp=0.5))
            play_id += 1
            score[off] += 1
        elif result == "Field goal":
            score[off] += 3
        pos = de
    # garbage-time style kneel, a spike and a wiped-out penalty play at the end
    leader = home if score[home] >= score[away] else away
    qb = _team_roster(leader, season)[0][0]
    for kind in ("qb_kneel", "qb_spike", "no_play"):
        row = dict(game_id=game["game_id"], play_id=play_id, season=season, week=game["week"], season_type="REG",
                   posteam=leader, defteam=away if leader == home else home, home_team=home, away_team=away, qtr=4,
                   play_type=kind, game_seconds_remaining=20, half_seconds_remaining=20, fixed_drive=drive,
                   wp=0.98, yardline_100=50, score_differential=abs(score[home] - score[away]))
        if kind == "qb_kneel":
            row.update(qb_kneel=1.0, rush_attempt=1.0, rusher_player_id=qb, rushing_yards=-1, yards_gained=-1)
        elif kind == "qb_spike":
            row.update(qb_spike=1.0, pass_attempt=1.0, passer_player_id=qb)
        plays.append(row)
        play_id += 1
    return plays, score


def build_raw(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sched = schedule()
    all_plays, snaps, part, ftn, pfr_pass, pfr_rush, rosters = [], [], [], [], [], [], []
    for _, g in sched.iterrows():
        season = int(g["season"])
        plays, score = _game_plays(rng, g, season)
        sched.loc[sched["game_id"] == g["game_id"], ["home_score", "away_score"]] = [score[g["home_team"]],
                                                                                     score[g["away_team"]]]
        all_plays.extend(plays)
        dfp = pd.DataFrame(plays)
        for team, opp in ((g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])):
            n_off = int(((dfp["posteam"] == team) & dfp["play_type"].isin(["pass", "run"])).sum())
            on_field = {}
            for gsis, pos, role, ts, cs in _team_roster(team, season):
                share = {"QB": 1.0, "RB": 0.6 if role == "RB1" else 0.35, "WR": 0.85 if role != "WR4" else 0.2,
                         "TE": 0.7 if role == "TE1" else 0.3}[pos]
                k = int(round(n_off * share))
                on_field[gsis] = share
                snaps.append(dict(game_id=g["game_id"], pfr_game_id="x", season=season, game_type=g["game_type"],
                                  week=g["week"], player=f"{team} {role}", pfr_player_id="pfr" + gsis[3:],
                                  position=pos, team=team, opponent=opp, offense_snaps=float(k),
                                  offense_pct=k / max(n_off, 1), defense_snaps=0.0, defense_pct=0.0, st_snaps=1.0,
                                  st_pct=0.05))
                rosters.append(dict(season=season, week=g["week"], gsis_id=gsis, position=pos, team=team))
            mine = dfp[(dfp["posteam"] == team) & dfp["play_type"].isin(["pass", "run"])]
            for _, pl in mine.iterrows():
                ids = [k for k, sh in on_field.items() if rng.random() < sh]
                part.append(dict(nflverse_game_id=g["game_id"], play_id=pl["play_id"], offense_players=";".join(ids)))
                ftn.append(dict(nflverse_game_id=g["game_id"], nflverse_play_id=pl["play_id"], season=season,
                                week=g["week"], is_screen_pass=bool(rng.random() < 0.1),
                                is_play_action=bool(rng.random() < 0.2), is_motion=bool(rng.random() < 0.4)))
            qb = _team_roster(team, season)[0][0]
            db = int((mine["qb_dropback"] == 1).sum())
            press = int(rng.integers(5, 15))
            pfr_pass.append(dict(game_id=g["game_id"], season=season, week=g["week"], team=team, opponent=opp,
                                 pfr_player_id="pfr" + qb[3:], times_pressured=press,
                                 times_pressured_pct=press / max(db, 1)))
            rb = _team_roster(team, season)[1][0]
            car = int(((mine["rusher_player_id"] == rb)).sum())
            pfr_rush.append(dict(game_id=g["game_id"], season=season, week=g["week"], team=team, opponent=opp,
                                 pfr_player_id="pfr" + rb[3:], carries=car,
                                 rushing_yards_before_contact=car * 1.5, rushing_yards_after_contact=car * 2.5,
                                 rushing_broken_tackles=int(car > 10)))
    pbp = pd.DataFrame(all_plays)
    for c in ("receiver_player_id", "rusher_player_id", "passer_player_id", "td_player_id", "td_team"):
        if c not in pbp:
            pbp[c] = None
    sched["result"] = sched["home_score"] - sched["away_score"]
    sched["total"] = sched["home_score"] + sched["away_score"]
    sched["overtime"] = 0
    return {"schedules": sched, "players": _players(), "pbp": pbp, "snap_counts": pd.DataFrame(snaps),
            "rosters": pd.DataFrame(rosters), "participation": pd.DataFrame(part), "ftn": pd.DataFrame(ftn),
            "pfr_pass": pd.DataFrame(pfr_pass), "pfr_rush": pd.DataFrame(pfr_rush)}
