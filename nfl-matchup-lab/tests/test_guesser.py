"""Guesser method: the original Bills-Rams formula, same-game lineups (Entry A/B), the frozen $20 bet and grading."""
import json

import numpy as np
import pandas as pd
import pytest

from nfl_lab import guesser
from nfl_lab.report import picks_html


def test_estimate_reproduces_the_original_slip():
    # src/ev.py: Allen rush+rec TDs 4 of 4, nudge +0.04, market 52% -> 64.1%; Kincaid under 2 of 4, +0.08, 52% -> 55.0%
    assert guesser.estimate(4, 4, 0.04, 0.52) == pytest.approx(0.641, abs=5e-4)
    assert guesser.estimate(2, 4, 0.08, 0.52) == pytest.approx(0.550, abs=5e-4)
    assert guesser.tier(0.641) == "High" and guesser.tier(0.55) == "Medium" and guesser.tier(0.51) == "Coin flip"


def _picks(rows):
    base = dict(game_id="G", stat="rec_yds", line=40.5, side="under", hits=3, games=4, log=[1.0, 2.0, 3.0, 50.0],
                reasons=["For: something"])
    df = pd.DataFrame([dict(base, **r) for r in rows])
    df["tier"] = df["p_model"].map(guesser.tier)
    return df


def test_lineups_top_picks_one_per_player_both_teams():
    df = _picks([dict(player_id="a", player="A", team="KC", p_model=0.64),
                 dict(player_id="a", player="A", team="KC", p_model=0.63, stat="receptions", line=3.5),
                 dict(player_id="b", player="B", team="KC", p_model=0.62),
                 dict(player_id="c", player="C", team="KC", p_model=0.61),
                 dict(player_id="d", player="D", team="BUF", p_model=0.58),
                 dict(player_id="e", player="E", team="BUF", p_model=0.70, games=2),      # too few games
                 dict(player_id="f", player="F", team="BUF", p_model=0.52)])               # coin flip
    lu = guesser.lineups(df, "BUF at KC")
    A, B = lu["3"][0], lu["4"][0]
    assert (A["entry"], A["play_as"], B["entry"], B["play_as"]) == ("A", "Flex", "B", "Power")
    assert [x["player"] for x in A["legs"]] == ["A", "B", "D"]          # top 3 are all KC: third swapped for D
    assert [x["player"] for x in B["legs"]] == ["A", "B", "C", "D"]
    assert A["p_all"] == pytest.approx(0.64 * 0.62 * 0.58)
    assert B["power_ev"] == pytest.approx(0.64 * 0.62 * 0.61 * 0.58 * 10)
    assert "5" not in lu                                                 # E and F are kept out of lineups


def _game(ko):
    return pd.Series(dict(game_id="2026_06_BBB_AAA", season=2026, week=6, away_team="BUF", home_team="KC", kickoff_utc=ko))


def _lu():
    df = _picks([dict(player_id=k, player=k.upper(), team=t, p_model=p)
                 for k, t, p in (("a", "BUF", 0.62), ("b", "KC", 0.6), ("c", "BUF", 0.58), ("d", "KC", 0.56))])
    return guesser.lineups(df)


def test_bankroll_is_frozen_at_kickoff(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    path = guesser.write_bankroll(_game(ko), _lu(), ko - pd.Timedelta(hours=5))
    d = json.loads(path.read_text())
    assert [(x["id"], x["type"], x["stake"], len(x["legs"])) for x in d["entries"]] == [("A", "flex", 12.0, 3), ("B", "power", 8.0, 4)]
    assert d["entries"][0]["pays"] == {"3": 2.25, "2": 1.25} and d["entries"][1]["pays"] == {"4": 10}
    assert guesser.write_bankroll(_game(ko), {}, ko + pd.Timedelta(minutes=1)) is None and path.exists()   # after kickoff: kept
    guesser.write_bankroll(_game(ko), {}, ko - pd.Timedelta(hours=1))                                     # before: no entries, no bet
    assert not path.exists()


def test_grade_week(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    guesser.write_bankroll(_game(ko), _lu(), ko - pd.Timedelta(hours=5))
    stats = pd.DataFrame([dict(player_id=k, week=6, receiving_yards=v) for k, v in (("a", 10), ("b", 20), ("c", 90))])
    assert guesser.grade_week(2026, 6, stats) == []                      # d has no stats yet: wait
    stats = pd.concat([stats, pd.DataFrame([dict(player_id="d", week=6, receiving_yards=5)])])
    assert guesser.grade_week(2026, 6, stats) == ["2026_06_BBB_AAA"]
    d = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [v["hit"] for v in d["result"]["legs"].values()] == [True, True, False, True]


def test_lineup_view_lists_each_pick_once():
    out = picks_html.guesser_lineups(_lu(), prefix="")
    for name in ("A", "B", "C", "D"):
        assert out.count(f"<b>{name}</b>") == 1
    assert "Picks 1-3" in out and "Picks 1-4" in out and "Guesser $12" in out and "Guesser $8" in out
