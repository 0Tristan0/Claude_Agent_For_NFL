import io
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nfl_lab import teams
from nfl_lab.data import context, ingest, lines, pbp, player_games, weather
from nfl_lab.data.sources import SOURCES


# ---- ingest ------------------------------------------------------------------------------------

class FakeResp:
    def __init__(self, status, content=b""):
        self.status_code, self.content, self.ok = status, content, status == 200

    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").HTTPError(str(self.status_code))

    def json(self):
        return json.loads(self.content)


def _parquet_bytes(n=3):
    buf = io.BytesIO()
    pq.write_table(pa.table({"a": list(range(n))}), buf)
    return buf.getvalue()


def test_fetch_content_addressed_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    body = _parquet_bytes()
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(200, body))
    r = ingest.fetch(SOURCES["pbp"], 2023, raw_dir=tmp_path, live_season=2026)
    assert r.status == "ok" and r.path.exists() and r.path.name.startswith("pbp_2023.")
    rows = ingest.read_manifest(tmp_path / "manifest.jsonl")
    assert rows[-1]["n_rows"] == 3 and len(rows[-1]["sha256"]) == 64
    again = ingest.fetch(SOURCES["pbp"], 2023, raw_dir=tmp_path, live_season=2026)
    assert again.status == "cached"
    assert ingest.load("pbp", seasons=[2023], raw_dir=tmp_path)["a"].tolist() == [0, 1, 2]


def test_failed_download_keeps_cache_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(200, _parquet_bytes()))
    first = ingest.fetch(SOURCES["pbp"], 2026, raw_dir=tmp_path, live_season=2026)

    def boom(*a, **k):
        raise ingest.requests.ConnectionError("down")
    monkeypatch.setattr(ingest.requests, "get", boom)
    r = ingest.fetch(SOURCES["pbp"], 2026, refresh=True, raw_dir=tmp_path, live_season=2026)
    assert r.status == "failed" and r.path == first.path and first.path.exists()
    assert ingest.read_manifest(tmp_path / "manifest.jsonl")[-1]["status"] == "failed"


def test_404_is_not_published_and_bad_parquet_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(404))
    assert ingest.fetch(SOURCES["participation"], 2026, raw_dir=tmp_path).status == "not_published"
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(200, b"not parquet"))
    r = ingest.fetch(SOURCES["pbp"], 2020, raw_dir=tmp_path)
    assert r.status == "failed" and "unreadable" in r.message


def test_refresh_never_overwrites_an_old_version(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(200, _parquet_bytes(3)))
    a = ingest.fetch(SOURCES["pbp"], 2024, raw_dir=tmp_path)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp(200, _parquet_bytes(4)))
    b = ingest.fetch(SOURCES["pbp"], 2024, refresh=True, raw_dir=tmp_path)
    assert a.path != b.path and a.path.exists() and b.path.exists()
    assert len(ingest.load("pbp", seasons=[2024], raw_dir=tmp_path)) == 4


# ---- teams ---------------------------------------------------------------------------------------

def test_team_aliases():
    assert teams.code("LAR") == "LA" and teams.code("lar") == "LA" and teams.display("LA") == "LAR"
    assert teams.code("JAC") == "JAX" and teams.code("OAK") == "LV" and teams.code("WSH") == "WAS"
    with pytest.raises(ValueError):
        teams.code("XYZ")


# ---- play flags ----------------------------------------------------------------------------------

def _play(**kw):
    base = dict(game_id="2024_01_A_B", play_id=1, season=2024, week=1, season_type="REG", posteam="BUF",
                defteam="KC", home_team="KC", away_team="BUF", qtr=1, down=1, ydstogo=10, yardline_100=50,
                game_seconds_remaining=3000, half_seconds_remaining=1200, fixed_drive=1, play_type="pass",
                pass_attempt=1.0, qb_dropback=1.0, wp=0.5, two_point_attempt=0.0, receiver_player_id="r",
                passer_player_id="q", yards_gained=5, score_differential=0)
    base.update(kw)
    return base


def test_play_flags():
    rows = [
        _play(play_id=1),                                                          # neutral target
        _play(play_id=2, wp=0.95),                                                 # not neutral
        _play(play_id=3, sack=1.0, receiver_player_id=None),                       # sack: dropback, no attempt
        _play(play_id=4, play_type="run", qb_scramble=1.0, rush_attempt=1.0, pass_attempt=0.0,
              receiver_player_id=None, rusher_player_id="q"),                      # scramble
        _play(play_id=5, play_type="qb_kneel", qb_kneel=1.0, rush_attempt=1.0, qb_dropback=0.0, pass_attempt=0.0,
              receiver_player_id=None, rusher_player_id="q"),
        _play(play_id=6, play_type="qb_spike", qb_spike=1.0, qb_dropback=0.0, receiver_player_id=None),
        _play(play_id=7, two_point_attempt=1.0),
        _play(play_id=8, play_type="no_play"),
        _play(play_id=9, yardline_100=8, yards_gained=25),
        _play(play_id=10, qtr=4, score_differential=-21),
        _play(play_id=11, play_type="run", rush_attempt=1.0, pass_attempt=0.0, qb_dropback=0.0,
              receiver_player_id=None, rusher_player_id="b", yards_gained=12),
    ]
    p = pbp.prepare(pd.DataFrame(rows)).set_index("play_id")
    assert p.loc[1, "neutral"] and p.loc[1, "target"] and p.loc[1, "pass_att"]
    assert not p.loc[2, "neutral"]
    assert p.loc[3, "dropback"] and p.loc[3, "is_sack"] and not p.loc[3, "pass_att"]
    assert p.loc[4, "scramble"] and p.loc[4, "dropback"] and p.loc[4, "rush_att"] and not p.loc[4, "designed_run"]
    assert p.loc[5, "kneel"] and p.loc[5, "rush_att"] and not p.loc[5, "neutral"]
    assert p.loc[6, "spike"] and p.loc[6, "pass_att"] and not p.loc[6, "target"] and not p.loc[6, "dropback"]
    assert not p.loc[7, "scrimmage"] and not p.loc[8, "scrimmage"]
    assert p.loc[9, "rz"] and p.loc[9, "i10"] and p.loc[9, "explosive"]
    assert p.loc[10, "garbage"] and not p.loc[1, "garbage"]
    assert p.loc[11, "designed_run"] and p.loc[11, "explosive"]


def test_pace_excludes_two_minute_drill_and_caps():
    rows = [_play(play_id=i, game_seconds_remaining=3000 - 30 * i, half_seconds_remaining=1200 - 30 * i)
            for i in range(1, 5)]
    rows[2]["game_seconds_remaining"] = rows[1]["game_seconds_remaining"] - 90      # long gap gets capped
    rows[3]["game_seconds_remaining"] = rows[2]["game_seconds_remaining"] - 30
    late = [_play(play_id=10 + i, fixed_drive=2, game_seconds_remaining=1900 - 20 * i, half_seconds_remaining=100 - 20 * i)
            for i in range(3)]
    p = pbp.prepare(pd.DataFrame(rows + late))
    pace = pbp.pace_seconds(p)
    got = pace.dropna().tolist()
    assert got == [30.0, 45.0, 30.0]                      # capped at 45, two-minute plays excluded


# ---- tables on the synthetic league --------------------------------------------------------------

def test_zero_target_player_kept_in_spine(tables):
    pg = tables["player_games"]
    wr4 = pg[pg["player_id"].str.endswith("WR4")]
    assert len(wr4) > 0 and (wr4["targets"] == 0).all() and (wr4["offense_snaps"] > 0).all()


def test_team_sums_equal_player_sums(tables):
    pg, tg = tables["player_games"], tables["team_games"]
    s = pg.groupby(["game_id", "team"])[["targets", "designed_carries"]].sum()
    t = tg.set_index(["game_id", "team"])[["off_targets", "off_designed_runs"]]
    j = s.join(t, how="inner")
    assert len(j) == len(tg)
    assert np.allclose(j["targets"], j["off_targets"]) and np.allclose(j["designed_carries"], j["off_designed_runs"])


def test_defense_mirrors_opponent_offense(tables):
    tg = tables["team_games"].set_index(["game_id", "team"])
    for (gid, team), r in tg.head(20).iterrows():
        o = tg.loc[(gid, r["opponent"])]
        assert r["def_pass_epa_sum"] == o["off_pass_epa_sum"] and r["def_plays"] == o["off_plays"]


def test_available_at_lags(tables):
    pg, tg = tables["player_games"], tables["team_games"]
    assert ((tg["available_at"] - tg["kickoff_utc"]) == pd.Timedelta(hours=16)).all()
    assert ((pg["available_at"] - pg["kickoff_utc"]) == pd.Timedelta(hours=28)).all()
    tp = tables["team_pressure"]
    assert ((tp["available_at"] - tp["kickoff_utc"]) == pd.Timedelta(hours=76)).all()


def test_no_blocked_columns_in_games(tables):
    from nfl_lab.data.games import BLOCKED_COLUMNS
    assert not (BLOCKED_COLUMNS & set(tables["games"].columns))
    assert {"home_score", "away_score"} <= set(tables["game_results"].columns)


def test_implied_points_from_spread_and_total():
    from nfl_lab.data.games import team_view
    g = pd.DataFrame([dict(game_id="2026_05_BUF_LA", season=2026, week=5, season_type="REG",
                           kickoff_utc=pd.Timestamp("2026-10-13 00:15", tz="UTC"), stadium_id="LAX01",
                           roof_pregame="dome", div_game=0, spread_line=3.5, total_line=54.5, home_team="LA",
                           away_team="BUF", home_rest=8, away_rest=8, location="Home")])
    tv = team_view(g).set_index("team")
    assert tv.loc["LA", "implied_points"] == 29.0 and tv.loc["BUF", "implied_points"] == 25.5
    assert tv.loc["LA", "team_spread"] == 3.5 and tv.loc["BUF", "team_spread"] == -3.5


def test_context_travel_and_short_week():
    tv = pd.DataFrame([
        dict(game_id="g1", season=2026, team="BUF", opponent="LA", kickoff_utc=pd.Timestamp("2026-10-13 00:15", tz="UTC"),
             is_home=0.0, rest_days=8, stadium_id="LAX01", roof_pregame="dome"),
        dict(game_id="g2", season=2026, team="LA", opponent="BUF", kickoff_utc=pd.Timestamp("2026-10-13 00:15", tz="UTC"),
             is_home=1.0, rest_days=4, stadium_id="LAX01", roof_pregame="dome")])
    games = pd.DataFrame([dict(season=2026, home_team="BUF", stadium_id="BUF01", location="Home"),
                          dict(season=2026, home_team="LA", stadium_id="LAX01", location="Home")])
    c = context.build(tv, games).set_index("team")
    assert c.loc["BUF", "tz_shift_hours"] == -3.0 and c.loc["LA", "tz_shift_hours"] == 0.0
    assert c.loc["LA", "short_week"] == 1.0 and c.loc["BUF", "short_week"] == 0.0
    assert c.loc["BUF", "dome"] == 1.0
    assert c.loc["BUF", "body_clock_kickoff_hour"] == pytest.approx(20.25)


def test_weather_disabled_or_dome_returns_none():
    k = pd.Timestamp("2026-10-13 00:15", tz="UTC")
    assert weather.kickoff_weather("BUF01", k, "outdoors") is None       # disabled by default
    assert weather.kickoff_weather("LAX01", k, "dome") is None


# ---- props.csv -----------------------------------------------------------------------------------

def test_props_parse_and_resolve(tmp_path):
    f = tmp_path / "props.csv"
    f.write_text("player,team,stat,line,source,over_odds,under_odds\n"
                 "Josh Allen,BUF,Pass Yards,249.5,fanduel,-115,-105\n"
                 "Puka Nacua,LAR,receiving yards,94.5,prizepicks,,\n"
                 "Kyren Williams,LA,Anytime TD,0.5,dk,-170,\n")
    p = lines.fair_probs(lines.load_props(f))
    assert p["stat"].tolist() == ["pass_yds", "rec_yds", "anytime_td"]
    assert p["team"].tolist() == ["BUF", "LA", "LA"]
    assert p.loc[0, "market_note"] == "two-way" and 0.5 < p.loc[0, "fair_over"] < 0.52
    assert p.loc[1, "market_note"] == "no price"
    assert p.loc[2, "market_note"] == "one-sided"                     # TD yes-only uses the bigger overround
    assert p.loc[2, "fair_over"] == pytest.approx((170 / 270) / 1.20)
    assert p["captured_at_assumed"].all()
    cands = pd.DataFrame({"player_id": ["a", "b", "c"], "player_name": ["Josh Allen", "Puka Nacua", "Kyren Williams"],
                          "team": ["BUF", "LA", "LA"]})
    r = lines.resolve_players(p, cands)
    assert r["player_id"].tolist() == ["a", "b", "c"]


def test_props_unknown_stat_raises(tmp_path):
    f = tmp_path / "props.csv"
    f.write_text("player,team,stat,line,source,over_odds,under_odds\nX,BUF,Punt Yards,40.5,x,-110,-110\n")
    with pytest.raises(ValueError, match="unknown stat"):
        lines.load_props(f)


def test_seeded_props_file_parses():
    p = lines.fair_probs(lines.load_props())
    assert len(p) >= 13 and (p["odds_kind"].isin(["legacy_blend", "none", ""])).all()
    assert (p.loc[p["odds_kind"] == "legacy_blend", "market_note"] == "legacy blended estimate (reference only)").all()
