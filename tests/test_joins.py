"""Player/game joins, participation spine, inactive players, franchise codes, trades, duplicate names."""
import numpy as np
import pandas as pd
import pytest

from nfl_agent.data import build
from nfl_agent.features.pregame import build_feature_table

from .conftest import tiny_raw


def test_player_games_unique_and_joined_to_games(demo_tables):
    pg, games = demo_tables["player_games"], demo_tables["games"]
    assert not pg.duplicated(["player_id", "game_id"]).any()
    assert pg["game_id"].isin(games["game_id"]).all()
    g = pg.merge(games[["game_id", "home_team", "away_team"]], on="game_id")
    home = g["team"] == g["home_team"]
    away = g["team"] == g["away_team"]
    assert (home | away).all(), "every player row belongs to one of the two teams in its game"
    assert ((home & (g["opponent"] == g["away_team"])) | (away & (g["opponent"] == g["home_team"]))).all()


def test_zero_target_games_recovered_from_snaps_not_fabricated():
    t = build.build_all(tiny_raw())["player_games"]
    p2 = t[t["player_id"] == "P2"].sort_values("week")
    # played weeks 1 and 3 (snap counts) with no stat row -> recorded 0 targets, flagged
    assert list(p2["week"]) == [1, 3]
    assert (p2["targets"] == 0).all() and p2["stats_zero_filled"].all()
    # inactive in week 2 -> no row at all (never a fabricated zero)
    assert not ((t["player_id"] == "P2") & (t["week"] == 2)).any()


def test_no_zero_fill_when_snap_counts_missing_for_game():
    raw = tiny_raw()
    raw["snap_counts"] = raw["snap_counts"][raw["snap_counts"]["week"] != 3]   # week 3 snaps not yet published
    t = build.build_all(raw)["player_games"]
    assert not ((t["player_id"] == "P2") & (t["week"] == 3)).any(), "unknown participation must stay unknown"
    p1w3 = t[(t["player_id"] == "P1") & (t["week"] == 3)]
    assert len(p1w3) == 1 and p1w3["snaps_missing"].iloc[0]        # stats row kept, snaps flagged missing
    assert np.isnan(p1w3["snap_share"].iloc[0])


def test_target_share_uses_team_targets_denominator():
    t = build.build_all(tiny_raw())["player_games"]
    r = t[(t["player_id"] == "P1") & (t["week"] == 2)].iloc[0]
    assert r["team_targets"] == 6        # only P1 has targets for AAA in week 2
    assert r["target_share"] == pytest.approx(1.0)


def test_relocated_franchise_codes_normalised():
    raw = tiny_raw()
    for k, cols in (("schedules", ["away_team"]), ("snap_counts", ["team"])):
        for c in cols:
            raw[k][c] = raw[k][c].replace({"AAA": "OAK"})
    raw["snap_counts"]["opponent"] = raw["snap_counts"]["opponent"]
    raw["player_stats"]["team"] = raw["player_stats"]["team"].replace({"AAA": "LV"})
    t = build.build_all(raw)
    pg = t["player_games"]
    assert set(pg["team"]) == {"LV"}
    assert pg["team_targets"].notna().all(), "stats (LV) and schedule/snaps (OAK) must join after normalisation"


def test_traded_player_team_change_flag(demo_tables):
    pg = demo_tables["player_games"].copy()
    tg = demo_tables["team_games"]
    pid = pg["player_id"].iloc[0]
    rows = pg.index[pg["player_id"] == pid]
    new_team_rows = rows[len(rows) // 2:]
    # simulate a trade: later games for another team (choose games where that team played)
    pg.loc[new_team_rows, "team"] = "DMZ"
    f = build_feature_table(pg, tg)
    f = f[f["player_id"] == pid].sort_values("kickoff_utc").reset_index(drop=True)
    first_new = f.index[f["team"] == "DMZ"][0]
    assert f.loc[first_new, "team_changed"] == 1
    assert f.loc[first_new, "games_with_team_prior"] == 0
    assert f.loc[first_new + 1, "games_with_team_prior"] == 1
    assert (f.loc[: first_new - 1, "team_changed"] == 0).all()


def test_duplicate_names_kept_separate_and_ambiguity_reported(demo_service):
    svc = demo_service
    f = svc.features
    a, b = f["player_id"].unique()[:2]
    name = f.loc[f["player_id"] == a, "player_name"].iloc[0]
    svc.features.loc[svc.features["player_id"] == b, "player_name"] = name
    try:
        assert f[f["player_name"] == name]["player_id"].nunique() == 2     # ids never merged by name
        with pytest.raises(ValueError, match="Ambiguous"):
            svc.resolve_player(name)
        assert svc.resolve_player(a) == a                                  # stable id always works
    finally:
        orig = svc.pg.drop_duplicates("player_id").set_index("player_id")["player_name"]
        svc.features["player_name"] = svc.features["player_id"].map(orig).fillna(svc.features["player_name"])


def test_postseason_games_labelled(demo_tables):
    pg = demo_tables["player_games"]
    post = pg[pg["game_type"] == "WC"]
    assert len(post) > 0 and (post["season_type"] == "POST").all()
    f = build_feature_table(pg, demo_tables["team_games"])
    assert (f.loc[f["season_type"] == "POST", "is_post"] == 1).all()
    assert (f.loc[f["season_type"] == "REG", "is_post"] == 0).all()


def test_kickoff_converted_from_eastern_to_utc():
    g = build.build_games(tiny_raw()["schedules"])
    k = g.loc[g["week"] == 1, "kickoff_utc"].iloc[0]
    assert str(k.tz) == "UTC" and k.hour == 17          # 13:00 EDT == 17:00 UTC
