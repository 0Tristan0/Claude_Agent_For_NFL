"""Walk-forward (expanding-window) evaluation.

For each evaluated season S, every model is fitted on seasons
[FIRST_TRAIN_SEASON, S-1] only (all preprocessing, imputation, dispersion
estimation and baseline priors included) and scored on season S. Features are
pregame by construction, so within season S each week's forecast uses only
games that had already been played. Whole games are never split: the unit of
splitting is the season.

Model parameters are refitted once per season (not weekly); this is
conservative for the model and mirrors a practical update cadence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..models.baselines import RollingEmpiricalBaseline, WorkloadBaseline, lagged_outcomes
from ..models.structured import ALL_GROUPS, DEFAULT_GROUPS, DEFAULT_LEARNER, StructuredModel
from . import metrics as M

YT = config.RECEIVING_YARD_MILESTONES
RT = config.RECEPTION_MILESTONES
OUTCOMES = {"yards": ("receiving_yards", YT), "rec": ("receptions", RT)}
INTERVALS = {"50": ("q250", "q750"), "80": ("q100", "q900"), "95": ("q025", "q975")}


@dataclass
class ModelSpec:
    name: str
    kind: str                    # structured | rolling | workload | workload_normal
    groups: tuple = DEFAULT_GROUPS
    learner: str = DEFAULT_LEARNER


DEFAULT_SPECS = [
    ModelSpec("structured", "structured"),
    ModelSpec("B1B2_rolling_empirical", "rolling"),
    ModelSpec("B3_workload", "workload"),
    ModelSpec("B3n_workload_normal", "workload_normal"),
]


def ablation_specs() -> list[ModelSpec]:
    """Feature-group ablation around the full model, plus the learner comparison."""
    specs = [ModelSpec("full_all_groups", "structured", ALL_GROUPS),
             ModelSpec("default_history_snaps", "structured", DEFAULT_GROUPS),
             ModelSpec("default_with_glm_targets", "structured", DEFAULT_GROUPS, "glm"),
             ModelSpec("history_only", "structured", ("player_history",))]
    for g in ALL_GROUPS:
        if g != "player_history":
            specs.append(ModelSpec(f"full_minus_{g}", "structured", tuple(x for x in ALL_GROUPS if x != g)))
    return specs


def _predict(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, hist: dict, seed: int,
             n_sims: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if spec.kind == "structured":
        m = StructuredModel(groups=spec.groups, target_learner=spec.learner).fit(train)
        actual = {"yards": test["receiving_yards"].to_numpy(float), "receptions": test["receptions"].to_numpy(float),
                  "targets": test["targets"].to_numpy(float)}
        r = m.simulate(test, n_sims, seed, actual=actual)
        for k, v in r.items():
            if k.startswith("yards_"):
                out[k] = v
            elif k.startswith("receptions_"):
                out["rec_" + k[len("receptions_"):]] = v
            elif k.startswith("targets_"):
                out["tgt_" + k[len("targets_"):]] = v
        out["mu_t"] = r["mu_t"]
        out["_params"] = m.params_
        return out
    for key, (col, thr) in OUTCOMES.items():
        if spec.kind == "rolling":
            b = RollingEmpiricalBaseline().fit(train, col, thr)
            r = b.predict(test, hist[col])
        else:
            b = WorkloadBaseline(outcome=col, normal=spec.kind == "workload_normal").fit(train, thr)
            r = b.predict(test)
            r["pit"] = b.pit(test, test[col].to_numpy(float), seed=seed)
        for k, v in r.items():
            out[f"{key}_{k}"] = v
    return out


def fold_masks(df: pd.DataFrame, test_season: int, first_train: int) -> tuple[pd.Series, pd.Series]:
    """Training rows: seasons [first_train, test_season - 1]; test rows: test_season. Played rows only."""
    tr = df["season"].between(first_train, test_season - 1) & df["played"]
    te = (df["season"] == test_season) & df["played"]
    return tr, te


def run(features: pd.DataFrame, test_seasons, specs=None, n_sims: int = config.N_SIMS_EVAL,
        first_train: int = config.FIRST_TRAIN_SEASON) -> pd.DataFrame:
    """Return a long table: one row per (model, test player-game) with predictions and outcomes."""
    specs = specs or DEFAULT_SPECS
    df = features.copy()
    if "played" not in df:
        df["played"] = df["targets"].notna()
    hist = {"receiving_yards": lagged_outcomes(df, "receiving_yards"),
            "receptions": lagged_outcomes(df, "receptions")}
    frames = []
    for S in test_seasons:
        tr_mask, te_mask = fold_masks(df, S, first_train)
        if te_mask.sum() == 0:
            continue
        train, test = df[tr_mask], df[te_mask]
        h_te = {k: v[te_mask.to_numpy()] for k, v in hist.items()}
        for spec in specs:
            pred = _predict(spec, train, test, h_te, seed=config.GLOBAL_SEED + S, n_sims=n_sims)
            pred.pop("_params", None)
            f = test[["player_id", "player_name", "position", "game_id", "season", "week", "season_type",
                      "team", "targets", "receptions", "receiving_yards", "n_games_prior", "eff_games",
                      "team_changed", "is_rookie"]].copy()
            for k, v in pred.items():
                f[k] = v
            f["model"] = spec.name
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def score(preds: pd.DataFrame, n_boot: int = 300) -> dict[str, pd.DataFrame]:
    """Summary metrics per model, with game-cluster bootstrap CIs for differences vs. B3."""
    rows, diffs = [], []
    ref_name = "B3_workload"
    by_model = {m: g.sort_values(["game_id", "player_id"]).reset_index(drop=True) for m, g in preds.groupby("model")}
    for model, g in by_model.items():
        for key, (col, thr) in OUTCOMES.items():
            y = g[col].to_numpy(float)
            if f"{key}_median" not in g:
                continue
            rows.append(dict(model=model, outcome=key, metric="MAE (median)", value=np.mean(np.abs(y - g[f"{key}_median"])), n=len(g)))
            qcols = {c[len(key) + 1:]: g[c].to_numpy(float) for c in g.columns if c.startswith(key + "_q")}
            rows.append(dict(model=model, outcome=key, metric="CRPS (approx.)", value=np.mean(M.crps_from_quantiles(qcols, y)), n=len(g)))
            for lvl, (lo, hi) in INTERVALS.items():
                hit = M.interval_hit(g[f"{key}_{lo}"], g[f"{key}_{hi}"], y)
                rows.append(dict(model=model, outcome=key, metric=f"naive coverage {lvl}%", value=hit.mean(), n=len(g)))
                if f"{key}_pit" in g and g[f"{key}_pit"].notna().all():
                    pit = g[f"{key}_pit"].to_numpy(float)
                    rows.append(dict(model=model, outcome=key, metric=f"coverage {lvl}%",
                                     value=float(np.mean(np.abs(pit - 0.5) <= int(lvl) / 200)), n=len(g)))
                rows.append(dict(model=model, outcome=key, metric=f"width {lvl}%", value=np.mean(g[f"{key}_{hi}"] - g[f"{key}_{lo}"]), n=len(g)))
            for t in thr:
                p = g[f"{key}_p_ge_{t}"].to_numpy(float)
                o = (y >= t).astype(float)
                rows.append(dict(model=model, outcome=key, metric=f"Brier >={t}", value=np.mean(M.brier(p, o)), n=len(g)))
                rows.append(dict(model=model, outcome=key, metric=f"LogLoss >={t}", value=np.mean(M.log_loss(p, o)), n=len(g)))
                rows.append(dict(model=model, outcome=key, metric=f"mean p >={t}", value=p.mean(), n=len(g)))
                rows.append(dict(model=model, outcome=key, metric=f"observed >={t}", value=o.mean(), n=len(g)))
    summary = pd.DataFrame(rows)
    if ref_name in by_model:
        ref = by_model[ref_name]
        for model, g in by_model.items():
            if model == ref_name or len(g) != len(ref):
                continue
            vals = {}
            y = g["receiving_yards"].to_numpy(float)
            vals["MAE yards"] = np.abs(y - g["yards_median"]) - np.abs(y - ref["yards_median"])
            for t in YT:
                o = (y >= t).astype(float)
                vals[f"LogLoss >={t}"] = M.log_loss(g[f"yards_p_ge_{t}"].to_numpy(), o) - M.log_loss(ref[f"yards_p_ge_{t}"].to_numpy(), o)
                vals[f"Brier >={t}"] = M.brier(g[f"yards_p_ge_{t}"].to_numpy(), o) - M.brier(ref[f"yards_p_ge_{t}"].to_numpy(), o)
            d = M.cluster_bootstrap(pd.DataFrame(vals), g["game_id"], n_boot=n_boot)
            d.insert(0, "model", model)
            diffs.append(d)
    diff = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    return {"summary": summary, "diff_vs_B3": diff}


def calibration(preds: pd.DataFrame, model: str, key: str = "yards") -> pd.DataFrame:
    col, thr = OUTCOMES[key]
    g = preds[preds["model"] == model]
    parts = []
    for t in thr:
        c = M.calibration_table(g[f"{key}_p_ge_{t}"].to_numpy(float), (g[col].to_numpy(float) >= t).astype(float))
        c.insert(0, "threshold", t)
        parts.append(c)
    return pd.concat(parts, ignore_index=True)
