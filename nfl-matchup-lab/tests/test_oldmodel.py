"""Old model: the original report's formula applied to PrizePicks lines, and its lineup pricing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_lab.picks import oldmodel

KO = pd.Timestamp("2026-10-18 17:00", tz="UTC")
COLS = ["rec_yds", "receptions", "rush_yds", "carries", "pass_yds", "pass_att", "completions", "pass_tds", "ints",
        "rush_tds", "rec_tds"]


def _tables(values):
    games = pd.DataFrame([dict(game_id="G", home_team="AAA", away_team="BBB", kickoff_utc=KO, season=2026)])
    rows = []
    for i, v in enumerate(values):                       # the player's past games (team AAA) against CCC
        t = KO - pd.Timedelta(days=7 * (len(values) - i))
        rows.append(dict(player_id="p", season=2026, season_type="REG", any_snaps=30, kickoff_utc=t, available_at=t + pd.Timedelta(hours=16),
                         opponent="CCC", game_id=f"g{i}", **{c: (float(v) if c == "rec_yds" else 0.0) for c in COLS}))
    t = KO - pd.Timedelta(days=3)                        # one game where BBB's defense allowed a lot
    rows.append(dict(player_id="q", season=2026, season_type="REG", any_snaps=30, kickoff_utc=t, available_at=t + pd.Timedelta(hours=16),
                     opponent="BBB", game_id="gx", **{c: (400.0 if c == "rec_yds" else 0.0) for c in COLS}))
    return {"games": games, "player_games": pd.DataFrame(rows)}


def _legs(line, p_market=np.nan):
    base = dict(game_id="G", player_id="p", player="P", team="AAA", stat="rec_yds", line=line, prizepicks=True, line_type="standard")
    return pd.DataFrame([dict(base, side="over", p_market=p_market), dict(base, side="under", p_market=np.nan)])


def test_estimate_uses_the_old_formula():
    tables = _tables([10, 50, 60, 70, 80])               # last 4: 50, 60, 70, 80
    est = oldmodel.estimates(_legs(55.5), tables).iloc[0]
    assert est["side"] == "over" and est["hits"] == 3 and est["games"] == 4
    nudge = est["nudge"]                                 # BBB allows the most receiving yards: soft defense
    assert nudge == pytest.approx(0.03)
    assert est["p_model"] == pytest.approx(((3 + 2.5) / (4 + 5) + nudge + 0.5) / 2)


def test_market_price_is_averaged_in_when_present():
    tables = _tables([60, 60, 60, 60])
    est = oldmodel.estimates(_legs(55.5, p_market=0.40), tables).iloc[0]
    assert est["p_model"] == pytest.approx(((4 + 2.5) / 9 + 0.03 + 0.40) / 2)


def test_lineup_pricing_matches_the_legacy_script():
    legacy = json.loads((Path(__file__).resolve().parents[1] / "src" / "ev.json").read_text())
    est = pd.DataFrame([dict(player_id=k, player=k, team=t, stat="rec_yds", line=1.5, side="over", p_model=v["est"], hits=0,
                             games=0, nudge=0.0, opp="X", opp_rank=None, game_id="G")
                        for (k, v), t in zip(legacy["props"].items(), ["BUF", "LA"] * 10)
                        if k in ("allen_td", "staff_py", "kyren_rec")])
    L = oldmodel.best_lineups(est, sizes=(3,))["3"][0]
    want = legacy["lineups"]["3"]
    assert L["p_all"] == pytest.approx(want["p_all"], abs=1e-4)
    assert L["power_ev"] == pytest.approx(want["power_ev"], abs=1e-3) and L["flex_ev"] == pytest.approx(want["flex_ev"], abs=1e-3)
    assert len({x["player"] for x in L["legs"]}) == 3 and len({x["team"] for x in L["legs"]}) == 2
