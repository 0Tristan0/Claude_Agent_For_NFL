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


def _picks4():
    return _picks([dict(player_id=k, player=k.upper(), team=t, p_model=p)
                   for k, t, p in (("a", "BUF", 0.62), ("b", "KC", 0.6), ("c", "BUF", 0.58), ("d", "KC", 0.56))])


def test_plan_spends_twenty_in_whole_dollars_on_valid_entries():
    pl = guesser.plan(_picks4(), "BUF at KC")
    ents = pl["entries"]
    assert sum(x["stake"] for x in ents) == 20 and 1 <= len(ents) <= guesser.MAX_ENTRIES
    assert all(isinstance(x["stake"], int) and x["stake"] >= guesser.MIN_STAKE for x in ents)
    assert [x["entry"] for x in ents] == list("ABC"[:len(ents)])
    assert [x["stake"] for x in ents] == sorted((x["stake"] for x in ents), reverse=True)
    for x in ents:
        ids = [l["player_id"] for l in x["legs"]]
        assert len(ids) == x["n"] == len(set(ids)) and len({l["team"] for l in x["legs"]}) == 2
        assert x["play_as"] in ("Power", "Flex") and 0 < x["p_paid"] <= 1
    assert 0 <= pl["p_lose_all"] <= 1 - pl["p_profit"]
    assert pl["expected_back"] == pytest.approx(sum(x["stake"] * x["best_ev"] for x in ents))


def test_plan_needs_two_good_picks_from_both_teams():
    one_team = _picks([dict(player_id=k, player=k, team="KC", p_model=0.6) for k in "abc"])
    assert guesser.plan(one_team) == {}
    assert guesser.plan(_picks([dict(player_id="a", player="A", team="KC", p_model=0.6),
                                dict(player_id="b", player="B", team="BUF", p_model=0.51)])) == {}


def test_bankroll_is_frozen_at_kickoff(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    pl = guesser.plan(_picks4())
    path = guesser.write_bankroll(_game(ko), pl, ko - pd.Timedelta(hours=5))
    d = json.loads(path.read_text())
    assert sum(x["stake"] for x in d["entries"]) == 20 and d["bankroll"] == 20
    for x, L in zip(d["entries"], pl["entries"]):
        assert (x["id"], x["type"], x["stake"], len(x["legs"])) == (L["entry"], L["play_as"].lower(), L["stake"], L["n"])
        assert x["pays"] == ({"3": 2.25, "2": 1.25} if (x["type"], L["n"]) == ("flex", 3) else x["pays"])
    assert guesser.write_bankroll(_game(ko), {}, ko + pd.Timedelta(minutes=1)) is None and path.exists()   # after kickoff: kept
    guesser.write_bankroll(_game(ko), {}, ko - pd.Timedelta(hours=1))                                     # before: no play, no bet
    assert not path.exists()


def test_grade_week(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    guesser.write_bankroll(_game(ko), guesser.plan(_picks4()), ko - pd.Timedelta(hours=5))
    stats = pd.DataFrame([dict(player_id=k, week=6, receiving_yards=v) for k, v in (("a", 10), ("b", 20), ("c", 90))])
    d = json.loads(next(tmp_path.glob("*.json")).read_text())
    used = {l["player_id"] for en in d["entries"] for l in en["legs_detail"]}
    if "d" in used:
        assert guesser.grade_week(2026, 6, stats) == []                  # d has no stats yet: wait
    stats = pd.concat([stats, pd.DataFrame([dict(player_id="d", week=6, receiving_yards=5)])])
    assert guesser.grade_week(2026, 6, stats, {"2026_06_BBB_AAA": "AAA 24, BBB 20"}) == ["2026_06_BBB_AAA"]
    d = json.loads(next(tmp_path.glob("*.json")).read_text())
    want = {"a": True, "b": True, "c": False, "d": True}                 # all "under 40.5"
    for en in d["entries"]:
        for t, l in zip(en["legs"], en["legs_detail"]):
            assert d["result"]["legs"][t]["hit"] == want[l["player_id"]]
    assert d["result"]["final"] == "AAA 24, BBB 20" and d["result"]["provisional"] is False
    assert guesser.grade_week(2026, 6, stats) == []                      # official results are never regraded


def test_a_player_with_snaps_but_no_stat_row_is_graded_at_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    guesser.write_bankroll(_game(ko), guesser.plan(_picks4()), ko - pd.Timedelta(hours=5))
    stats = pd.DataFrame([dict(player_id=k, week=6, receiving_yards=10) for k in "abc"])   # "d" had no catches
    no_snaps = pd.DataFrame([dict(week=6, team="KC", player="D", offense_snaps=0, defense_snaps=0, st_snaps=0)])
    assert guesser.grade_week(2026, 6, stats, snaps=no_snaps) == []                    # would be a void: wait
    snaps = no_snaps.assign(offense_snaps=6)
    assert guesser.grade_week(2026, 6, stats, snaps=snaps) == ["2026_06_BBB_AAA"]
    d = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert d["result"]["legs"][guesser.leg_text(dict(player="D", stat="rec_yds", side="under", line=40.5))] == \
        dict(actual=0.0, hit=True)


def test_official_grade_keeps_a_provisional_final_score(tmp_path, monkeypatch):
    monkeypatch.setattr(guesser, "LIVE_DIR", tmp_path)
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    guesser.write_bankroll(_game(ko), guesser.plan(_picks4()), ko - pd.Timedelta(hours=5))
    path = next(tmp_path.glob("*.json"))
    d = json.loads(path.read_text())
    d["result"] = dict(final="AAA 24, BBB 20", provisional=True, legs={})
    path.write_text(json.dumps(d))
    stats = pd.DataFrame([dict(player_id=k, week=6, receiving_yards=10) for k in "abcd"])
    assert guesser.grade_week(2026, 6, stats) == ["2026_06_BBB_AAA"]
    r = json.loads(path.read_text())["result"]
    assert r["final"] == "AAA 24, BBB 20" and r["provisional"] is False


def test_the_play_shows_each_entry_with_its_stake_and_each_pick_once():
    pl = guesser.plan(_picks4())
    out = picks_html.guesser_play(pl, prefix="")
    assert "The play: $20" in out
    for L in pl["entries"]:
        assert f"<b>Entry {L['entry']}</b> · {L['n']}-pick {L['play_as']}" in out and f"${L['stake']}</span>" in out
    used = {l["player"] for L in pl["entries"] for l in L["legs"]}
    why = out.split("Why these picks")[1]
    for name in used:
        assert why.count(f"<b>{name}</b>") == 1
    assert [x["player"] for x in picks_html.play_summary(pl)] == sorted(used, key=lambda n: "abcd".index(n.lower()))


def test_news_nudges_move_the_side_they_favor_and_stay_capped():
    df = _picks([dict(player_id="a", player="Ann Bee", team="KC", p_model=0.60, side="over", stat="rec_yds"),
                 dict(player_id="b", player="Cal Dee", team="BUF", p_model=0.60, side="under", stat="rec_yds")])
    df["p_matchup"], df["reasons"] = [0.0, 0.07], [[], []]
    nudges = pd.DataFrame([dict(player="Ann Bee", stat="*", more_adj=0.04, reason="WR1 out: more targets"),
                           dict(player="Cal Dee", stat="rec_yds", more_adj=-0.06, reason="role cut")])
    out = guesser._apply_news(df, nudges)
    assert out.loc[0, "p_model"] == pytest.approx(0.64)                  # More pick, nudge toward More
    assert out.loc[1, "p_news"] == pytest.approx(guesser.NEWS_CAP - 0.07)  # Less pick, capped with its matchup
    assert out.loc[0, "reasons"] == ["For: News: WR1 out: more targets"]
    assert out.loc[1, "reasons"] == ["For: News: role cut"]


def test_book_prices_match_exact_lines_only():
    from nfl_lab.data import lines
    ko = pd.Timestamp("2026-10-18 17:00", tz="UTC")
    g = pd.Series(dict(game_id="G", home_team="KC", away_team="BUF", kickoff_utc=ko))
    props = pd.DataFrame([dict(player="Ann Bee", team="KC", stat="rec_yds", line=40.5, source="draftkings", game_id="",
                               captured_at=ko - pd.Timedelta(hours=3), fair_over=0.44),
                          dict(player="Ann Bee", team="KC", stat="rec_yds", line=40.5, source="fanduel", game_id="",
                               captured_at=ko - pd.Timedelta(hours=2), fair_over=0.46),
                          dict(player="Ann Bee", team="KC", stat="rec_yds", line=41.5, source="betmgm", game_id="",
                               captured_at=ko - pd.Timedelta(hours=2), fair_over=0.40),
                          dict(player="Ann Bee", team="KC", stat="rec_yds", line=40.5, source="caesars", game_id="",
                               captured_at=ko + pd.Timedelta(hours=1), fair_over=0.10)])       # after kickoff: ignored
    got = guesser._book_prices(props, g)
    assert got[(lines.norm_name("Ann Bee"), "rec_yds", 40.5)][0] == pytest.approx(0.45)
    assert got[(lines.norm_name("Ann Bee"), "rec_yds", 41.5)][0] == pytest.approx(0.40)
