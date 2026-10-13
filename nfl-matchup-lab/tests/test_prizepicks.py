"""PrizePicks board import: parsing the feed and replacing only the live board in props.csv."""
import pandas as pd

from nfl_lab.data import lines, prizepicks as PP

NOW = pd.Timestamp("2026-10-14 15:00", tz="UTC")


def proj(i, player_id, stat, line, odds_type="standard", status="pre_game"):
    return {"type": "projection", "id": str(i), "attributes": {"stat_type": stat, "line_score": line, "odds_type": odds_type,
            "status": status, "start_time": "2026-10-18T16:25:00-04:00"},
            "relationships": {"new_player": {"data": {"type": "new_player", "id": player_id}}}}


def player(pid, name, team, league="NFL"):
    return {"type": "new_player", "id": pid, "attributes": {"display_name": name, "team": team, "league": league}}


FEED = {"data": [proj(1, "a", "Pass Yards", 245.5), proj(2, "a", "Rush+Rec TDs", 0.5), proj(3, "b", "Receiving Yards", 62.5),
                 proj(4, "b", "Receiving Yards", 49.5, "goblin"), proj(5, "b", "Fantasy Score", 12.5),
                 proj(6, "c", "Receptions", 4.5), proj(7, "d", "Pass Yards", 230.5, status="in_progress"),
                 proj(8, "e", "Rush Yards", 60.5)],
        "included": [player("a", "Josh Allen", "BUF"), player("b", "Brock Bowers", "LV"),
                     player("c", "Bowers + Allen", "LV/BUF"), player("d", "Geno Smith", "LV"),
                     player("e", "Someone", "XYZ")]}


def test_parse_board():
    rows, skipped = PP.parse(FEED, NOW)
    got = {(r.player, r.stat, r.line, r.line_type) for r in rows.itertuples()}
    assert got == {("Josh Allen", "pass_yds", 245.5, "standard"), ("Josh Allen", "anytime_td", 0.5, "standard"),
                   ("Brock Bowers", "rec_yds", 62.5, "standard"), ("Brock Bowers", "rec_yds", 49.5, "goblin")}
    assert set(rows["source"]) == {"prizepicks"} and rows["game_id"].eq("").all()
    assert skipped["not pre-game"] == 1 and skipped["stat not modeled: Fantasy Score"] == 1
    assert skipped["not a single NFL player"] == 1 and skipped["unknown team: XYZ"] == 1


def test_write_replaces_only_the_live_board(tmp_path):
    path = tmp_path / "props.csv"
    pd.DataFrame([dict(player="Puka Nacua", team="LAR", stat="rec_yds", line=94.5, source="prizepicks", over_odds="",
                       under_odds="", captured_at="2026-10-12T17:56:32Z", game_id="2026_05_BUF_LA", odds_kind="",
                       line_type="standard", payout_multiplier="", notes=""),
                  dict(player="Old Line", team="BUF", stat="rec_yds", line=10.5, source="prizepicks", over_odds="",
                       under_odds="", captured_at="2026-10-13T12:00:00Z", game_id="", odds_kind="", line_type="standard",
                       payout_multiplier="", notes="")]).to_csv(path, index=False)
    rows, _ = PP.parse(FEED, NOW)
    PP.write(rows, path)
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert "Puka Nacua" in set(out["player"]) and "Old Line" not in set(out["player"])
    assert (out["player"] == "Brock Bowers").sum() == 2
    loaded = lines.load_props(path)                       # the regular loader accepts the imported rows
    assert set(loaded.loc[loaded["player"] == "Josh Allen", "stat"]) == {"pass_yds", "anytime_td"}
