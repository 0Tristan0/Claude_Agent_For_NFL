"""Synthetic DEMO fixture, used only when real data cannot be downloaded (and in tests).

Everything generated here is fabricated. Team codes start with "DM" and player
names start with "DEMO", so the data can never be mistaken for real NFL
observations. The dashboard shows a persistent banner whenever demo data is
loaded.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DEMO_TEAMS = [f"DM{c}" for c in "ABCDEFGH"]


def demo_calendar(today: date | None = None) -> tuple[tuple[int, ...], int]:
    """Four demo seasons ending with the current one; weeks before today count as played.

    Anchoring the fixture to today's date keeps "upcoming" demo games in the future, so journal
    snapshots made in demo mode are genuinely pregame.
    """
    today = today or date.today()
    last = today.year if today.month >= 8 else today.year - 1
    start = date(last, 9, 10)
    done = (today - start).days // 7 + 1 if today >= start else 0
    return tuple(range(last - 3, last + 1)), int(np.clip(done, 0, 18))


def make_demo_raw(seasons=None, completed_weeks_last=None, n_weeks=17,
                  seed: int = 7) -> dict[str, pd.DataFrame]:
    """Synthetic raw tables. Defaults follow demo_calendar(); tests pass fixed values."""
    if seasons is None or completed_weeks_last is None:
        cal_seasons, cal_done = demo_calendar()
        seasons = seasons or cal_seasons
        completed_weeks_last = cal_done if completed_weeks_last is None else completed_weeks_last
    rng = np.random.default_rng(seed)
    # --- players -------------------------------------------------------------
    players = []
    pid = 0
    roster: dict[str, list[dict]] = {}
    for t in DEMO_TEAMS:
        roster[t] = []
        for pos, n in (("WR", 4), ("TE", 2), ("QB", 1)):
            for _ in range(n):
                pid += 1
                p = dict(
                    gsis_id=f"DEMO-{pid:04d}", pfr_id=f"DemoPf{pid:02d}",
                    display_name=f"DEMO {pos} {t[-1]}{pid:02d}", position=pos,
                    draft_round=float(rng.integers(1, 8)) if rng.random() < 0.8 else np.nan,
                    rookie_season=int(rng.choice(seasons)) if rng.random() < 0.3 else seasons[0] - 3,
                    birth_date="1999-01-01",
                    share=float(rng.dirichlet([1])[0]),
                )
                p["draft_pick"] = (p["draft_round"] - 1) * 32 + rng.integers(1, 33) if p["draft_round"] == p["draft_round"] else np.nan
                players.append(p)
                roster[t].append(p)
    for t in DEMO_TEAMS:
        rec = [p for p in roster[t] if p["position"] != "QB"]
        w = rng.dirichlet(np.ones(len(rec)) * 2.0) * 0.85
        for p, s in zip(rec, w):
            p["share"] = float(s)
            p["catch"] = float(np.clip(rng.normal(0.66 if p["position"] == "WR" else 0.72, 0.05), 0.4, 0.85))
            p["ypr"] = float(np.clip(rng.normal(12.5 if p["position"] == "WR" else 10.0, 2.0), 6, 19))
    players_df = pd.DataFrame(players).drop(columns=["share", "catch", "ypr"], errors="ignore")

    # --- schedules + stats ---------------------------------------------------
    games, stat_rows, snap_rows = [], [], []
    for season in seasons:
        start = date(season, 9, 10)
        for week in range(1, n_weeks + 2):
            post = week == n_weeks + 1
            order = rng.permutation(DEMO_TEAMS)
            pairs = [(order[i], order[i + 1]) for i in range(0, len(order), 2)]
            if post:
                pairs = pairs[:1]
            gday = start + timedelta(days=7 * (week - 1))
            for away, home in pairs:
                gid = f"{season}_{week:02d}_{away}_{home}"
                done = not (season == seasons[-1] and week > completed_weeks_last)
                scores = (int(rng.integers(10, 35)), int(rng.integers(10, 35))) if done else (np.nan, np.nan)
                games.append(dict(game_id=gid, season=season, game_type="WC" if post else "REG", week=week,
                                  gameday=gday.isoformat(), gametime="13:00", away_team=away, home_team=home,
                                  away_score=scores[0], home_score=scores[1],
                                  roof="dome" if home in ("DMA", "DMB") else "outdoors",
                                  away_rest=7.0, home_rest=7.0, location="Home", stadium=f"Demo Field {home}",
                                  temp=np.nan, wind=np.nan, away_qb_name=None, home_qb_name=None))
                if not done:
                    continue
                for team, opp in ((away, home), (home, away)):
                    dropbacks = int(rng.negative_binomial(20, 20 / (20 + 36)))
                    team_targets = int(round(dropbacks * 0.88))
                    plays = int(rng.integers(55, 72))
                    qb = [p for p in roster[team] if p["position"] == "QB"][0]
                    stat_rows.append(_stat(qb, season, week, gid, team, opp, 0, 0, 0, attempts=team_targets,
                                           sacks=dropbacks - team_targets))
                    snap_rows.append(_snap(qb, season, week, gid, team, opp, plays, 1.0))
                    for p in roster[team]:
                        if p["position"] == "QB" or rng.random() < 0.07:   # inactive
                            continue
                        snap_pct = float(np.clip(rng.beta(8, 3) * min(1.0, 0.4 + 2.5 * p["share"]), 0.02, 1.0))
                        lam = p["share"] * team_targets * rng.gamma(4.0, 0.25)
                        tg = int(rng.poisson(lam))
                        rc = int(rng.binomial(tg, p["catch"]))
                        yd = float(np.round(rng.gamma(1.4, p["ypr"] / 1.4, size=rc).sum())) if rc else 0.0
                        snap_rows.append(_snap(p, season, week, gid, team, opp, int(round(plays * snap_pct)), round(snap_pct, 2)))
                        if tg > 0:   # mimic nflverse: no stat row when nothing recorded
                            stat_rows.append(_stat(p, season, week, gid, team, opp, tg, rc, yd))
    sched = pd.DataFrame(games)
    stats = pd.DataFrame(stat_rows)
    snaps = pd.DataFrame(snap_rows)
    stats["season_type"] = np.where(stats["week"] > n_weeks, "POST", "REG")
    snaps["game_type"] = np.where(snaps["week"] > n_weeks, "WC", "REG")
    return {"schedules": sched, "players": players_df, "player_stats": stats, "snap_counts": snaps,
            "injuries": pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "report_status"]),
            "nextgen_receiving": pd.DataFrame(columns=["season", "week", "player_gsis_id", "avg_separation",
                                                       "avg_cushion", "avg_intended_air_yards"])}


def _stat(p, season, week, gid, team, opp, tg, rc, yd, attempts=0, sacks=0):
    air = tg * (11.0 if p["position"] == "WR" else 7.0)
    return dict(player_id=p["gsis_id"], player_display_name=p["display_name"], position=p["position"],
                season=season, week=week, game_id=gid, team=team, opponent_team=opp, targets=tg,
                receptions=rc, receiving_yards=yd, receiving_air_yards=air,
                receiving_yards_after_catch=round(yd * 0.4), receiving_tds=0,
                receiving_20=int(yd >= 20 and rc > 0), receiving_40=0, attempts=attempts, sacks_suffered=sacks)


def _snap(p, season, week, gid, team, opp, snaps, pct):
    return dict(game_id=gid, season=season, week=week, player=p["display_name"], pfr_player_id=p["pfr_id"],
                position=p["position"], team=team, opponent=opp, offense_snaps=snaps, offense_pct=pct)
