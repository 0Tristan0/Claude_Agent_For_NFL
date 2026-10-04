"""Forecast service: turns a (player, game) selection into a complete, auditable forecast.

Model fitting policy (mirrors the evaluated walk-forward design): a forecast for a
game in season S uses model parameters fitted on seasons FIRST_TRAIN_SEASON..S-1
and features computed from every game played before kickoff. A historical game
is therefore forecast exactly as the backtest would have, never with a model
that saw that season's outcomes.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config, db
from ..data import pipeline
from ..features.pregame import FeatureParams, build_feature_table, player_features, position_priors
from ..models.baselines import RollingEmpiricalBaseline, WorkloadBaseline, lagged_outcomes
from ..models.simulate import stable_seed
from ..models.structured import StructuredModel

YT = config.RECEIVING_YARD_MILESTONES
RT = config.RECEPTION_MILESTONES
MIN_GAMES_FOR_CONFIDENCE = 4


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class Forecast:
    forecast_created_at: str
    player_id: str
    player_name: str
    position: str
    team: str
    opponent: str
    game_id: str
    season: int
    week: int
    season_type: str
    kickoff_utc: str
    game_completed: bool
    data_mode: str
    data_cutoff: str
    data_built_at: str
    model_version: str
    model_train_seasons: list
    scenario: dict
    conditional_on_participation: bool
    participation_probability: str
    yards: dict
    receptions: dict
    targets: dict
    milestone_probabilities: dict
    milestone_probability_ranges: dict
    reception_milestones: dict
    target_pmf: dict
    reception_pmf: dict
    sample_sizes: dict
    inputs: dict
    missing_inputs: list
    flags: list
    evidence_level: str
    baseline: dict
    model_params: dict
    yards_histogram: dict = field(default_factory=dict)
    contributions: list = field(default_factory=list)
    recent_games: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_json_default))

    def to_display_dict(self) -> dict:
        p = self.to_payload()
        p.pop("contributions", None)
        p.pop("recent_games", None)
        return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _histogram(y: np.ndarray, width: int = 5, top: int = 250) -> dict:
    """Share of simulated games per yardage bin (last bin open-ended)."""
    edges = np.arange(-10, top + width, width)
    counts, _ = np.histogram(np.clip(y, -10, top), bins=np.r_[edges, np.inf])
    return {"bin_start": edges.tolist(), "share": (counts / len(y)).tolist(), "width": width}


def _r(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd) if nd else int(round(float(x)))


class ForecastService:
    def __init__(self, tables: dict, meta: dict, params: FeatureParams = FeatureParams()):
        self.tables, self.meta, self.params = tables, meta, params
        self.games = tables["games"]
        self.pg = tables["player_games"]
        self.tg = tables["team_games"]
        self.data_mode = meta["data_mode"]
        self._models: dict[int, dict] = {}
        self._build_features()

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, prefer: str = "real") -> "ForecastService":
        tables, meta = pipeline.load_tables(prefer)
        return cls(tables, meta)

    def _candidate_rows(self) -> pd.DataFrame:
        """Query rows for upcoming games: WR/TE whose latest appearance was for one of the teams."""
        up = self.games[~self.games["completed"]]
        if up.empty:
            return pd.DataFrame()
        cur = int(up["season"].min())
        recent = self.pg[self.pg["season"] >= cur - 1].sort_values("kickoff_utc")
        last = recent.groupby("player_id").tail(1)
        # keep players who appeared in the current season, or last season if the current one barely started
        n_cur = (self.pg["season"] == cur).sum()
        if n_cur > 0:
            active = set(self.pg.loc[self.pg["season"] == cur, "player_id"])
            last = last[last["player_id"].isin(active)]
        rows = []
        tv = pd.concat([
            up.assign(team=up["home_team"], opponent=up["away_team"]),
            up.assign(team=up["away_team"], opponent=up["home_team"])])
        for _, g in tv.iterrows():
            for _, p in last[last["team"] == g["team"]].iterrows():
                rows.append(dict(player_id=p["player_id"], player_name=p["player_name"], position=p["position"],
                                 game_id=g["game_id"], season=g["season"], week=g["week"],
                                 season_type=g["season_type"], game_type=g["game_type"],
                                 kickoff_utc=g["kickoff_utc"], team=g["team"], opponent=g["opponent"],
                                 draft_pick_filled=p["draft_pick_filled"], draft_pick=p["draft_pick"],
                                 rookie_season=p["rookie_season"],
                                 is_rookie=float(g["season"] == p["rookie_season"]) if pd.notna(p["rookie_season"]) else np.nan))
        return pd.DataFrame(rows)

    def _build_features(self):
        warnings.simplefilter("ignore")
        q = self._candidate_rows()
        self.query_rows = q
        self.features = build_feature_table(self.pg, self.tg, self.params, query_rows=q if len(q) else None)
        self.features["played"] = self.features["targets"].notna()
        self.hist_yards = lagged_outcomes(self.features, "receiving_yards")
        self.hist_rec = lagged_outcomes(self.features, "receptions")
        self.priors = position_priors(self.features)

    # ------------------------------------------------------------ models
    def models_for_season(self, season: int) -> dict:
        if season in self._models:
            return self._models[season]
        key = f"{self.data_mode}_{self.meta['built_at']}_{config.MODEL_VERSION}_{season}".replace(":", "")
        path = (config.PROCESSED_DIR if self.data_mode == "real" else config.DEMO_DIR) / "models" / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                self._models[season] = pickle.load(fh)
            return self._models[season]
        f = self.features
        first = config.FIRST_TRAIN_SEASON if self.data_mode == "real" else int(f["season"].min())
        train = f[f["season"].between(first, season - 1) & f["played"]]
        if len(train) < 500:
            raise ValueError(f"Not enough training history before season {season}")
        bundle = {
            "structured": StructuredModel().fit(train),
            "b3_yards": WorkloadBaseline("receiving_yards").fit(train, YT),
            "b3_rec": WorkloadBaseline("receptions").fit(train, RT),
            "b2_yards": RollingEmpiricalBaseline().fit(train, "receiving_yards", YT),
            "train_seasons": sorted(int(s) for s in train["season"].unique()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        self._models[season] = bundle
        return bundle

    # ---------------------------------------------------------- selection
    def resolve_player(self, text: str) -> str:
        f = self.features
        if (f["player_id"] == text).any():
            return text
        m = f[f["player_name"].str.lower() == text.lower()]["player_id"].unique()
        if len(m) == 1:
            return m[0]
        if len(m) > 1:
            raise ValueError(f"Ambiguous name '{text}': {list(m)}. Use the GSIS id.")
        raise ValueError(f"No WR/TE named '{text}'")

    def game_options(self) -> pd.DataFrame:
        ids = set(self.features["game_id"])
        g = self.games[self.games["game_id"].isin(ids)].copy()
        return g.sort_values("kickoff_utc", ascending=False)

    def players_for_game(self, game_id: str) -> pd.DataFrame:
        f = self.features[self.features["game_id"] == game_id]
        return f[["player_id", "player_name", "position", "team", "opponent", "played", "is_query", "tpg_ew"]].sort_values(
            ["team", "tpg_ew"], ascending=[True, False])

    def row(self, player_id: str, game_id: str) -> pd.DataFrame:
        r = self.features[(self.features["player_id"] == player_id) & (self.features["game_id"] == game_id)]
        if r.empty:
            raise ValueError("This player has no forecastable row for that game (not on either roster "
                             "by recent appearances, or did not participate in a completed game).")
        return r

    # ---------------------------------------------------------- forecast
    def forecast(self, player_id: str, game_id: str, playing_time_multiplier: float = 1.0,
                 team_volume_multiplier: float = 1.0, n_sims: int = config.N_SIMS_APP,
                 bootstrap: int = 30) -> Forecast:
        row = self.row(player_id, game_id)
        r = row.iloc[0]
        season = int(r["season"])
        bundle = self.models_for_season(season)
        model: StructuredModel = bundle["structured"]
        mult = float(playing_time_multiplier) * float(team_volume_multiplier)
        seed = stable_seed(config.GLOBAL_SEED, player_id, game_id, round(mult, 4))
        draws, prm = model.simulate_draws(row, n_sims, seed, mult)
        T, C, Y = draws["targets"][0], draws["receptions"][0], draws["yards"][0]

        def dist(x):
            q = np.quantile(x, [0.025, 0.10, 0.25, 0.5, 0.75, 0.90, 0.975])
            return {"mean": _r(np.mean(x), 1), "median": _r(q[3]),
                    "interval_50": [_r(q[2]), _r(q[4])], "interval_80": [_r(q[1]), _r(q[5])],
                    "interval_95": [_r(q[0]), _r(q[6])]}

        probs = {str(t): float((Y >= t).mean()) for t in YT}
        rec_probs = {str(t): float((C >= t).mean()) for t in RT}
        tpmf = {str(k): float((T == k).mean()) for k in range(15)}
        tpmf["15+"] = float((T >= 15).mean())
        rpmf = {str(k): float((C == k).mean()) for k in range(12)}
        rpmf["12+"] = float((C >= 12).mean())
        ranges = self._probability_ranges(row, model, mult, bootstrap) if bootstrap else {}

        idx = np.flatnonzero((self.features["player_id"] == player_id).to_numpy() & (self.features["game_id"] == game_id).to_numpy())[0]
        b3 = bundle["b3_yards"].predict(row)
        b3r = bundle["b3_rec"].predict(row)
        b2 = bundle["b2_yards"].predict(row, self.hist_yards[[idx]])
        baseline = {
            "workload_baseline_version": config.BASELINE_VERSION,
            "workload_mean_yards": _r(b3["mean"][0], 1), "workload_median_yards": _r(b3["median"][0]),
            "workload_interval_80": [_r(b3["q100"][0]), _r(b3["q900"][0])],
            "workload_milestones": {str(t): float(b3[f"p_ge_{t}"][0]) for t in YT},
            "workload_mean_receptions": _r(b3r["mean"][0], 1),
            "empirical_milestones_last16": {str(t): float(b2[f"p_ge_{t}"][0]) for t in YT},
            "rolling_mean_last8": _r(b2["mean"][0], 1),
            "note": "Baselines are shown for comparison and are not adjusted by scenario multipliers.",
        }
        flags, missing = self._flags(r)
        n_prior = int(r["n_games_prior"])
        evidence = ("limited" if n_prior < MIN_GAMES_FOR_CONFIDENCE else
                    "moderate" if n_prior < 12 else "substantial")
        if r["team_changed"] == 1 or (r["games_with_team_prior"] < 3 and n_prior >= 3):
            evidence = "limited" if evidence == "limited" else "moderate"
        recent = self._recent_games(player_id, r["kickoff_utc"])
        contrib = model.target_contributions(row).head(8).to_dict("records")
        cutoff = self._data_cutoff(r["kickoff_utc"])
        game = self.games[self.games["game_id"] == game_id].iloc[0]
        context = {
            "listed_qb_for_team_in_schedule": (game["home_qb_name"] if r["team"] == game["home_team"] else game["away_qb_name"]),
            "listed_qb_note": "From the nflverse schedule. For completed games this is the actual starter (known only "
                              "afterwards); for upcoming games it is nflverse's projection with unknown timing. "
                              "Context only, not a model input.",
            "roof": game.get("roof"), "fixed_dome": None if pd.isna(game.get("fixed_dome")) else bool(game.get("fixed_dome")),
            "weather_note": "Weather is not used: the schedule's temp/wind fields are filled in after games "
                            "(observed, not forecast) and no archived pregame weather forecasts are ingested.",
            "injury_report": self._injury_context(r),
        }
        fc = Forecast(
            forecast_created_at=utcnow().isoformat(), player_id=player_id, player_name=r["player_name"],
            position=r["position"], team=r["team"], opponent=r["opponent"], game_id=game_id,
            season=season, week=int(r["week"]), season_type=r["season_type"],
            kickoff_utc=pd.Timestamp(r["kickoff_utc"]).isoformat(), game_completed=bool(game["completed"]),
            data_mode=self.data_mode, data_cutoff=cutoff, data_built_at=self.meta["built_at"],
            model_version=config.MODEL_VERSION, model_train_seasons=bundle["train_seasons"],
            scenario={"playing_time_multiplier": playing_time_multiplier,
                      "team_volume_multiplier": team_volume_multiplier,
                      "is_base": abs(mult - 1.0) < 1e-9,
                      "assumption": None if abs(mult - 1.0) < 1e-9 else
                      "HYPOTHETICAL: expected targets scaled proportionally by the multipliers. This is an "
                      "analyst assumption, not a validated causal effect."},
            conditional_on_participation=True,
            participation_probability="not modelled (forecast assumes the player is active and takes offensive snaps)",
            yards=dist(Y), receptions=dist(C), targets=dist(T),
            milestone_probabilities=probs, milestone_probability_ranges=ranges, reception_milestones=rec_probs,
            target_pmf=tpmf, reception_pmf=rpmf,
            sample_sizes={"prior_games_played": n_prior, "effective_weighted_games": _r(r["eff_games"], 1),
                          "games_with_current_team": int(r["games_with_team_prior"]),
                          "weighted_games_with_snap_data": _r(r["snap_history_games"], 1),
                          "model_training_rows": model.n_train, "monte_carlo_draws": n_sims},
            inputs={k: (None if pd.isna(r[k]) else float(r[k])) for k in
                    ("tpg_ew", "tshare_ew", "tshare_recent", "catch_rate_ew", "ypr_ew", "adot_ew",
                     "yac_per_rec_ew", "explosive_rate_ew", "snap_share_ew", "snap_share_recent",
                     "team_tpg_ew", "opp_def_tpg_rel", "opp_def_ypt_rel", "days_since_last_game")},
            missing_inputs=missing, flags=flags, evidence_level=evidence, baseline=baseline,
            model_params={"expected_targets": float(prm["mu_t"][0]), "catch_probability": float(prm["p_catch"][0]),
                          "yards_per_catch": float(prm["m_ypr"][0]), "nb_size": float(np.ravel(prm["nb_size"])[0]),
                          "beta_binomial_rho": float(prm["bb_rho"]), "per_catch_gamma_shape": float(prm["gamma_shape"])},
            yards_histogram=_histogram(Y), contributions=contrib, recent_games=recent, context=context)
        return fc

    # ---------------------------------------------------------- helpers
    def _probability_ranges(self, row: pd.DataFrame, model: StructuredModel, mult: float, B: int) -> dict:
        """Uncertainty about the probability itself from limited player history.

        Re-weights the player's prior games with Poisson(1) bootstrap weights, recomputes his
        features, and re-simulates. Reports the 10th-90th percentile of each milestone probability.
        Does not include model-parameter uncertainty (documented).
        """
        pid, gid = row.iloc[0]["player_id"], row.iloc[0]["game_id"]
        sub = self.features[self.features["player_id"] == pid].sort_values("kickoff_utc").reset_index(drop=True)
        base_cols = [c for c in row.columns]
        rng = np.random.default_rng(stable_seed("boot", pid, gid))
        ps = {t: [] for t in YT}
        pf_cols = None
        for b in range(B):
            w = rng.poisson(1.0, size=len(sub)).astype(float)
            pf = player_features(sub, self.params, self.priors, mult={pid: w})
            pf = pf[pf["game_id"] == gid]
            pf_cols = [c for c in pf.columns if c not in ("player_id", "game_id")]
            new = row.copy()
            for c in pf_cols:
                new[c] = pf[c].to_numpy()
            new["ypt_ew"] = new["catch_rate_ew"] * new["ypr_ew"]
            d, _ = model.simulate_draws(new, 4000, stable_seed("bootsim", pid, gid, b), mult)
            for t in YT:
                ps[t].append(float((d["yards"][0] >= t).mean()))
        return {str(t): [float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))] for t, v in ps.items()}

    def _flags(self, r) -> tuple[list[str], list[str]]:
        flags, missing = [], []
        n = int(r["n_games_prior"])
        if n == 0:
            flags.append("No prior NFL games in the data: the forecast is essentially the position-level prior "
                         "adjusted for draft position. Treat it as a weak estimate.")
        elif n < MIN_GAMES_FOR_CONFIDENCE:
            flags.append(f"Only {n} prior games: estimates are heavily shrunk toward position averages.")
        if r["team_changed"] == 1:
            flags.append("First game with a new team: history comes from a different offense.")
        elif r["games_with_team_prior"] < 3 and n >= 3:
            flags.append(f"Only {int(r['games_with_team_prior'])} prior games with the current team.")
        if pd.notna(r["days_since_last_game"]) and r["days_since_last_game"] > 21 and r["week"] > 1:
            flags.append(f"{int(r['days_since_last_game'])} days since his last game: possible return from injury "
                         "or a role change; playing time may differ from his history.")
        if r["snap_history_games"] < 0.5:
            missing.append("snap counts (no snap history for this player)")
        if pd.notna(r.get("snap_share_recent")) and pd.notna(r.get("snap_share_ew")) and \
                r["snap_share_recent"] < r["snap_share_ew"] - 0.15:
            flags.append("Recent snap share is well below his weighted history: role may be shrinking.")
        if pd.notna(r.get("tshare_recent")) and r["tshare_recent"] > r["tshare_ew"] * 1.4 and r["tshare_recent"] > 0.12:
            flags.append("Recent target share is well above his weighted history: role may be growing "
                         "(or a short-term spike).")
        missing.append("route participation / routes run (not available in-season; snap share is a different variable)")
        missing.append("probability of being active (injury reports lack publication timestamps)")
        missing.append("quarterback availability for this game (not a verified pregame input)")
        missing.append("pregame weather forecast (no archived forecasts ingested)")
        return flags, missing

    def _recent_games(self, player_id: str, before) -> list:
        h = self.pg[(self.pg["player_id"] == player_id) & (self.pg["kickoff_utc"] < before)].tail(8)
        return h[["game_id", "season", "week", "team", "opponent", "targets", "receptions", "receiving_yards",
                  "snap_share", "target_share"]].to_dict("records")

    def _data_cutoff(self, kickoff) -> str:
        """Kickoff of the latest game, before this one, whose box-score stats are in the ingested data."""
        with_stats = set(self.tg.loc[self.tg["has_stats"], "game_id"])
        done = self.games[self.games["game_id"].isin(with_stats) & (self.games["kickoff_utc"] < kickoff)]
        last = done["kickoff_utc"].max() if len(done) else None
        return pd.Timestamp(last).isoformat() if last is not None else "none"

    def _injury_context(self, r) -> dict | None:
        inj = self.tables.get("injuries")
        if inj is None or inj.empty or "gsis_id" not in inj:
            return None
        m = inj[(inj["gsis_id"] == r["player_id"]) & (inj["season"] == r["season"]) & (inj["week"] == r["week"])]
        if m.empty:
            return {"status": None, "note": "Not listed on this week's injury report in the ingested data "
                                            "(or the report is not yet published)."}
        x = m.iloc[-1]
        return {"status": x.get("report_status"), "practice": x.get("practice_status"),
                "injury": x.get("report_primary_injury"),
                "note": "Official injury report as ingested from nflverse. Publication time is not recorded, "
                        "so it is shown as context only and is not a model input."}

    # ---------------------------------------------------------- journal
    def save(self, fc: Forecast, path=None) -> str:
        return save_forecast(fc, path)

    def record_results(self, path=None) -> int:
        return record_results(self, path)


def save_forecast(fc: Forecast, path=None) -> str:
    payload = fc.to_payload()
    blob = json.dumps(payload, sort_keys=True)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    scen = "base" if fc.scenario["is_base"] else json.dumps(
        {k: fc.scenario[k] for k in ("playing_time_multiplier", "team_volume_multiplier")}, sort_keys=True)
    fid = str(uuid.uuid4())
    with db.session(path) as con:
        prev = con.execute("SELECT forecast_id, revision FROM forecasts WHERE player_id=? AND game_id=? AND "
                           "model_version=? AND scenario=? ORDER BY revision DESC LIMIT 1",
                           (fc.player_id, fc.game_id, fc.model_version, scen)).fetchone()
        rev = 0 if prev is None else prev["revision"] + 1
        sup = None if prev is None else prev["forecast_id"]
        is_pre = int(pd.Timestamp(fc.forecast_created_at) < pd.Timestamp(fc.kickoff_utc))
        con.execute("INSERT INTO forecasts (forecast_id, created_at, player_id, player_name, game_id, kickoff_utc, "
                    "is_pregame, model_version, data_cutoff, revision, supersedes, scenario, data_mode, payload_json, "
                    "payload_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fid, fc.forecast_created_at, fc.player_id, fc.player_name, fc.game_id, fc.kickoff_utc, is_pre,
                     fc.model_version, fc.data_cutoff, rev, sup, scen, fc.data_mode, blob, sha))
    return fid


def record_results(svc: ForecastService, path=None) -> int:
    """Attach actual outcomes for journal forecasts whose games are complete. Forecast rows are untouched."""
    n = 0
    with db.session(path) as con:
        rows = con.execute("SELECT DISTINCT f.player_id, f.game_id FROM forecasts f LEFT JOIN results r "
                           "ON f.player_id=r.player_id AND f.game_id=r.game_id WHERE r.player_id IS NULL").fetchall()
        done = set(svc.games.loc[svc.games["completed"], "game_id"])
        has_stats = set(svc.tg.loc[svc.tg["has_stats"], "game_id"])
        for row in rows:
            pid, gid = row["player_id"], row["game_id"]
            if gid not in done or gid not in has_stats:
                continue
            m = svc.pg[(svc.pg["player_id"] == pid) & (svc.pg["game_id"] == gid)]
            if m.empty:     # did not participate: recorded as such, never as zero yards
                con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                            (pid, gid, utcnow().isoformat(), 0, None, None, None, f"nflverse ({svc.data_mode})"))
            else:
                x = m.iloc[0]
                con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                            (pid, gid, utcnow().isoformat(), 1, int(x["targets"]), int(x["receptions"]),
                             float(x["receiving_yards"]), f"nflverse ({svc.data_mode})"))
            n += 1
    return n


def journal(path=None) -> pd.DataFrame:
    with db.session(path) as con:
        df = pd.read_sql_query(
            "SELECT f.forecast_id, f.created_at, f.player_name, f.player_id, f.game_id, f.kickoff_utc, f.is_pregame, "
            "f.model_version, f.data_cutoff, f.revision, f.supersedes, f.scenario, f.data_mode, f.payload_json, "
            "f.payload_sha256, r.played, r.targets AS actual_targets, r.receptions AS actual_receptions, "
            "r.receiving_yards AS actual_yards, r.recorded_at FROM forecasts f LEFT JOIN results r "
            "ON f.player_id=r.player_id AND f.game_id=r.game_id ORDER BY f.created_at DESC", con)
    return df
