"""No-lookahead suite. These tests must fail if any feature uses same-game or future data.

1. Oracle: features from full raw data == features from raw data truncated to what existed at asof.
2. Poison: absurd values in every not-yet-available row change nothing; a positive control does.
3. Mutations: each planted leak must be caught by the oracle (otherwise the oracle is too weak).
4. Static tripwires: modelling code never reads outcomes, raw files or blocked columns.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import leakcheck
from conftest import game, kickoff
from nfl_lab import config
from nfl_lab.data import games as G, routes
from nfl_lab.features import asof as asof_mod, bundle, opponent

pytestmark = pytest.mark.leak

QUERIES = [(2022, 1, 3), (2022, 4, 0), (2023, 3, 3), (2023, 6, 1), (2023, 7, 3)]


@pytest.mark.parametrize("season,week,k", QUERIES)
def test_oracle_truncation_identical(raw, season, week, k):
    g = game(raw, season, week, k)
    leakcheck.assert_no_lookahead(raw, g["game_id"], first_season=2021)


def test_oracle_playoff_game(raw):
    leakcheck.assert_no_lookahead(raw, "2022_19_DAL_BUF", first_season=2021)


@pytest.mark.parametrize("season,week,k", [(2023, 5, 3), (2022, 6, 2)])
def test_poisoned_future_rows_change_nothing(raw, season, week, k):
    g = game(raw, season, week, k)
    ko = kickoff(raw, g["game_id"])
    a = leakcheck.features(raw, g["game_id"], first_season=2021)
    b = leakcheck.features(leakcheck.poison(raw, ko, season), g["game_id"], first_season=2021)
    leakcheck.assert_same(a, b)


def test_positive_control_past_rows_do_change(raw):
    g = game(raw, 2023, 5, 3)
    ko = kickoff(raw, g["game_id"])
    a = leakcheck.features(raw, g["game_id"], first_season=2021)
    p = {k: v.copy() for k, v in raw.items()}
    team = g["home_team"]
    pbp = p["pbp"]
    prior = pbp[(pbp["posteam"] == team) & (pbp["season"] == 2023) & (pbp["week"] == 4)]
    pbp.loc[prior.index, "epa"] = 9.0
    snaps = p["snap_counts"]
    m = (snaps["team"] == team) & (snaps["season"] == 2023) & (snaps["week"] == 4)
    snaps.loc[m, "offense_snaps"] = snaps.loc[m, "offense_snaps"] * 0 + 1
    b = leakcheck.features(p, g["game_id"], first_season=2021)
    with pytest.raises(AssertionError):
        leakcheck.assert_same(a, b)


# ---- planted leaks: each must make the oracle fail ---------------------------------------------

def _boundary_asof(raw, gid, hours):
    """A time exactly `hours` after the kickoff of the query home team's previous game (before the query)."""
    s = raw["schedules"]
    g = s[s["game_id"] == gid].iloc[0]
    ko = leakcheck.kickoffs(s)
    prev = s[((s["home_team"] == g["home_team"]) | (s["away_team"] == g["home_team"])) &
             s["game_id"].map(ko).lt(ko[gid])]
    last = prev["game_id"].map(ko).max()
    return last + pd.Timedelta(hours=hours)


def test_mutation_searchsorted_right_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 5, 3)["game_id"]
    t = _boundary_asof(raw, gid, 16)                 # a pbp row becomes available exactly now
    leakcheck.assert_no_lookahead(raw, gid, asof=t, first_season=2021)
    monkeypatch.setattr(asof_mod, "SIDE", "right")
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, asof=t, first_season=2021)


def test_mutation_zero_pbp_lag_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 5, 3)["game_id"]           # 4:25 game; 1:00 games that day must stay invisible
    real = config.lag_hours
    monkeypatch.setattr(config, "lag_hours", lambda k: 0.0 if k == "pbp" else real(k))
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, first_season=2021)


def test_mutation_score_in_allowlist_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 5, 3)["game_id"]
    monkeypatch.setattr(G, "PREGAME_COLUMNS", G.PREGAME_COLUMNS + ["home_score"])
    monkeypatch.setattr(bundle, "BLOCKED_COLUMNS", G.BLOCKED_COLUMNS - {"home_score"})
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, first_season=2021)


def test_mutation_same_season_participation_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 6, 3)["game_id"]
    monkeypatch.setattr(routes, "LOOKBACK", (0, 1))
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, first_season=2021)


def test_mutation_ridge_leq_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 5, 3)["game_id"]
    t = _boundary_asof(raw, gid, 16)
    monkeypatch.setattr(opponent, "visible", lambda df, a, time_col="available_at": df[df[time_col] <= a])
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, asof=t, first_season=2021)


def test_mutation_pfr_lag_removed_is_caught(raw, monkeypatch):
    gid = game(raw, 2023, 5, 3)["game_id"]
    t = _boundary_asof(raw, gid, 40)                 # 40 h after the previous game: PFR not public yet
    leakcheck.assert_no_lookahead(raw, gid, asof=t, first_season=2021)
    real = config.lag_hours
    monkeypatch.setattr(config, "lag_hours", lambda k: 16.0 if k in ("pfr", "ftn") else real(k))
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(raw, gid, asof=t, first_season=2021)


def test_same_day_early_game_invisible_to_league_fit(tables, raw):
    g = game(raw, 2023, 5, 3)
    early = game(raw, 2023, 5, 1)
    ko = kickoff(raw, g["game_id"])
    assert kickoff(raw, early["game_id"]) < ko                       # same Sunday, earlier slot
    tg = tables["team_games"]
    used = tg[tg["available_at"] < ko]
    assert early["game_id"] not in set(used["game_id"])


# ---- static tripwires ----------------------------------------------------------------------------

MODEL_DIRS = ["features", "project", "sim", "picks"]
BANNED_CALLS = {"read_parquet", "read_csv"}
BANNED_NAMES = {"game_results", "stats_player", "stats_official"}
BANNED_STRINGS = {"home_score", "away_score", "result", "total", "temp", "wind", "home_qb_name", "away_qb_name",
                  "home_qb_id", "away_qb_id", "overtime", "off_points", "def_points", "roof"}
# Fitting code may read past seasons' scores (it only ever sees seasons before the target season).
ALLOWED_STRINGS = {("project", "v0.py"): {"off_points", "def_points"}}


def _py_files():
    root = Path(__file__).resolve().parents[1] / "nfl_lab"
    for d in MODEL_DIRS:
        yield from (root / d).glob("*.py")


@pytest.mark.parametrize("path", list(_py_files()), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_static_tripwires(path):
    tree = ast.parse(path.read_text())
    allowed = ALLOWED_STRINGS.get((path.parent.name, path.name), set())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_CALLS:
            problems.append(f"line {node.lineno}: calls {node.attr} (read history only through the tables/AsofIndex)")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: uses {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                node.value in (BANNED_STRINGS | BANNED_NAMES) - allowed:
            problems.append(f"line {node.lineno}: string {node.value!r} names an outcome or blocked column")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            if "track" in mod.split(".") or any(n in ("track", "game_results") for n in names):
                problems.append(f"line {node.lineno}: imports grading code")
    assert not problems, "\n".join(problems)


# ---- AsofIndex against a brute-force oracle ---------------------------------------------------------

def _brute(df, ent, col, t, season, decay, disc):
    rows = df[(df["e"] == ent) & (df["available_at"] < t)].sort_values(["available_at", "game_id"], kind="mergesort")
    s, prev = 0.0, None
    for _, r in rows.iterrows():
        if prev is not None and r["season"] != prev:
            s *= disc ** (r["season"] - prev)
        s = s * decay + r[col]
        prev = r["season"]
    if prev is not None:
        s *= disc ** max(0, season - prev)
    return s


def test_asof_index_matches_brute_force_with_ties_and_mixed_timezones():
    rng = np.random.default_rng(3)
    n = 300
    base = pd.Timestamp("2022-09-01", tz="UTC")
    hours = rng.integers(0, 24 * 200, n)
    hours[::7] = hours[1::7][: len(hours[::7])]                       # exact ties
    t = base + pd.to_timedelta(hours, unit="h")
    df = pd.DataFrame({"e": rng.choice(list("abcd"), n), "x": rng.normal(size=n), "game_id": [f"g{i}" for i in range(n)],
                       "season": np.where(t < pd.Timestamp("2023-02-15", tz="UTC"), 2022, 2023),
                       "available_at": t.tz_convert("America/New_York")})     # non-UTC tz on purpose
    df = df.sample(frac=1.0, random_state=1)                                   # unsorted input
    idx = asof_mod.AsofIndex(df, "e", ["x"], half_life=3.0, season_discount=0.4)
    d = 0.5 ** (1 / 3.0)
    utc = df.assign(available_at=df["available_at"].dt.tz_convert("UTC"))
    for q in range(60):
        when = base + pd.Timedelta(hours=int(rng.integers(0, 24 * 200)))
        if q % 5 == 0:
            when = utc["available_at"].iloc[q]                                 # exactly at a row: excluded
        season = 2022 if when < pd.Timestamp("2023-02-15", tz="UTC") else 2023
        for e in "abcd":
            want = _brute(utc, e, "x", when, season, d, 0.4)
            got = idx.state(e, when, season)[0]
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
