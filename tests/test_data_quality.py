"""Validation, user CSV imports, demo labelling, router intents, ingestion failure handling."""
import numpy as np
import pandas as pd
import pytest

from nfl_agent import config
from nfl_agent.agent import router
from nfl_agent.data import demo, imports, ingest, validate
from nfl_agent.data.sources import SOURCES


def test_validation_flags_missing_columns_and_quirks():
    assert any("missing required" in i for i in validate.validate("player_stats", pd.DataFrame({"player_id": ["a"]})))
    df = pd.DataFrame({"player_id": ["a", "a"], "game_id": ["g", "g"], "targets": [1, 2], "receptions": [2, 1]})
    issues = validate.validate("player_stats", df)
    assert any("receptions > targets" in i for i in issues)
    assert any("duplicate" in i for i in issues)
    assert validate.validate("schedules", pd.DataFrame()) == ["schedules: table is empty or missing"]


CSV_OK = b"player_id,game_id,routes_run,route_participation,team_dropbacks\nP1,2030_01_AAA_BBB,30,0.8,38\nP1,2030_02_AAA_BBB,,0.75,40\n"


def test_import_valid_csv_keeps_blanks_missing(tmp_dirs):
    df, errs = imports.validate_routes_csv(CSV_OK, {"P1"}, {"2030_01_AAA_BBB", "2030_02_AAA_BBB"})
    assert errs == []
    assert np.isnan(df.loc[1, "routes_run"]), "blank must stay missing, never zero"
    meta = imports.save_import(df, CSV_OK, "routes.csv", "my charting notes")
    assert meta["used_in_model"] is False and "USER-SUPPLIED" in meta["provenance"]
    loaded, metas = imports.load_imports()
    assert (loaded["provenance"] == "user-supplied: my charting notes").all()


@pytest.mark.parametrize("raw,needle", [
    (b"player_id,game_id,routes_run\nP1,2030_01_AAA_BBB,-3\n", "negative"),
    (b"player_id,game_id,route_participation\nP1,2030_01_AAA_BBB,1.4\n", "between 0 and 1"),
    (b"player_id,game_id,routes_run\nP1,week1,3\n", "malformed game_id"),
    (b"player_id,game_id,routes_run\nPX,2030_01_AAA_BBB,3\n", "unknown player_id"),
    (b"player_id,routes_run\nP1,3\n", "Missing required"),
    (b"player_id,game_id,notes\nP1,2030_01_AAA_BBB,hi\n", "at least one"),
])
def test_import_rejects_bad_csv(raw, needle):
    df, errs = imports.validate_routes_csv(raw, {"P1"}, {"2030_01_AAA_BBB"})
    assert df is None and any(needle in e for e in errs)


def test_import_requires_source_label(tmp_dirs):
    df, _ = imports.validate_routes_csv(CSV_OK)
    with pytest.raises(ValueError):
        imports.save_import(df, CSV_OK, "x.csv", "  ")


def test_demo_data_is_unmistakably_labelled():
    raw = demo.make_demo_raw(seasons=(2021, 2022), completed_weeks_last=2)
    assert raw["schedules"]["home_team"].str.startswith("DM").all()
    assert raw["players"]["display_name"].str.startswith("DEMO").all()
    assert raw["player_stats"]["player_id"].str.startswith("DEMO").all()


def test_failed_download_keeps_cache_and_logs(tmp_dirs, monkeypatch):
    src = SOURCES["schedules"]
    path = ingest.cache_path(src, None)
    assert str(path).startswith(str(tmp_dirs)), "test must never touch the real cache"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": ["x"]}).to_parquet(path)

    def boom(*a, **k):
        raise RuntimeError("download failed after 3 attempts: ConnectionError")
    monkeypatch.setattr(ingest, "_get", boom)
    r = ingest.fetch(src, None, refresh=True)
    assert r.status == "failed" and r.path == path and path.exists()
    from nfl_agent import db
    with db.session() as con:
        row = con.execute("SELECT status, message FROM ingestion_log ORDER BY id DESC").fetchone()
    assert row["status"] == "failed" and "using cache" in row["message"]


def test_no_secrets_or_keys_required():
    import inspect
    from nfl_agent.data import ingest as ing
    src = inspect.getsource(ing)
    assert "Authorization" not in src and "API_KEY" not in src


@pytest.mark.parametrize("q,intent", [
    ("What is this receiver's projected receiving-yard distribution?", "distribution"),
    ("How uncertain is his projected target volume?", "targets"),
    ("What is the estimated probability of reaching 50, 75, 100, or 150 receiving yards?", "milestones"),
    ("How would reduced playing time change the forecast?", "scenario"),
    ("What evidence supports the forecast, and what could make it wrong?", "evidence"),
    ("How accurately has the model predicted comparable outcomes historically?", "accuracy"),
    ("Why can't his average yardage alone tell me the chance of a 150-yard game?", "average_vs_tail"),
])
def test_router_intents(q, intent):
    assert router.classify(q) == intent


def test_router_answers_use_computed_numbers(demo_service):
    svc = demo_service
    qrow = svc.features[svc.features["is_query"]].iloc[0]
    fc = svc.forecast(qrow["player_id"], qrow["game_id"], n_sims=2000, bootstrap=3)
    ans = router.answer("What is the probability of reaching 100 yards?", fc, {})
    from nfl_agent.explain.narrative import pct
    assert pct(fc.milestone_probabilities["100"]) in ans["markdown"]
    unknown = router.answer("tell me a joke", fc, {})
    assert unknown["intent"] == "unknown" and "I can answer" in unknown["markdown"]
