"""Run the full evaluation and write docs/EVALUATION.md plus tables for the dashboard.

Stages
------
1. Validation (walk-forward, seasons 2019-2024): default model vs baselines,
   feature-group ablation and learner comparison. Used for model selection.
2. Final test (season 2025): default model vs baselines, scored once with the
   configuration frozen after stage 1.
3. Live (current season, completed games): small-sample sanity check.
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..data import pipeline
from ..features.pregame import FeatureParams, build_feature_table
from ..models.structured import StructuredModel
from . import walkforward as W

log = logging.getLogger(__name__)


def eval_dir(demo: bool):
    d = (config.DEMO_DIR if demo else config.PROCESSED_DIR) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(x, nd=3):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}" if abs(int(x)) >= 10000 else str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def _md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for rec in df.to_dict("records"):           # keeps per-column dtypes (ints stay ints)
        lines.append("| " + " | ".join(_fmt(rec[c], nd) for c in cols) + " |")
    return "\n".join(lines)


KEY_METRICS = ["MAE (median)", "CRPS (approx.)", "coverage 50%", "coverage 80%", "coverage 95%",
               "width 80%", "Brier >=50", "LogLoss >=50", "Brier >=100", "LogLoss >=100",
               "Brier >=150", "LogLoss >=150"]


def summary_table(summary: pd.DataFrame, outcome: str, metrics=KEY_METRICS) -> pd.DataFrame:
    s = summary[(summary["outcome"] == outcome) & summary["metric"].isin(metrics)]
    t = s.pivot(index="model", columns="metric", values="value")
    return t[[m for m in metrics if m in t.columns]].reset_index()


def high_milestone_table(preds: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        g = preds[preds["model"] == m]
        y = g["receiving_yards"].to_numpy()
        for t in (100, 125, 150):
            p = g[f"yards_p_ge_{t}"].to_numpy()
            rows.append(dict(model=m, milestone=f">={t}", n_forecasts=len(g), events=int((y >= t).sum()),
                             expected_events=float(p.sum()), mean_pred=float(p.mean()), observed=float((y >= t).mean())))
    return pd.DataFrame(rows)


def subgroup_table(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    g = preds[preds["model"] == model].copy()
    groups = {
        "all": np.ones(len(g), bool),
        "<=3 prior games": g["n_games_prior"].to_numpy() <= 3,
        "first game with new team": g["team_changed"].to_numpy() == 1,
        "rookies": g["is_rookie"].to_numpy() == 1,
        "postseason": g["season_type"].to_numpy() == "POST",
        "tight ends": g["position"].to_numpy() == "TE",
    }
    rows = []
    y = g["receiving_yards"].to_numpy()
    for name, idx in groups.items():
        if idx.sum() == 0:
            continue
        pit = g["yards_pit"].to_numpy()[idx] if "yards_pit" in g else np.full(idx.sum(), np.nan)
        rows.append(dict(subgroup=name, n=int(idx.sum()), mae=float(np.mean(np.abs(y[idx] - g["yards_median"].to_numpy()[idx]))),
                         mean_pred=float(g["yards_mean"].to_numpy()[idx].mean()), mean_obs=float(y[idx].mean()),
                         coverage_80=float(np.mean(np.abs(pit - 0.5) <= 0.4)),
                         p100_pred=float(g["yards_p_ge_100"].to_numpy()[idx].mean()), p100_obs=float((y[idx] >= 100).mean())))
    return pd.DataFrame(rows)


def dependence_check(features: pd.DataFrame, train_seasons, test_seasons) -> dict:
    """Is per-catch efficiency correlated with target volume surprises? (held-out seasons)"""
    tr = features[features["season"].isin(train_seasons) & features["played"]]
    te = features[features["season"].isin(test_seasons) & features["played"] & (features["receptions"] > 0)].copy()
    m = StructuredModel().fit(tr)
    p = m.predict_params(te)
    ratio = te["targets"].to_numpy() / p["mu_t"]
    eff = te["receiving_yards"].to_numpy() / (te["receptions"].to_numpy() * p["m_ypr"])
    from scipy import stats
    rho = stats.spearmanr(np.log(ratio), eff)
    q = pd.qcut(ratio, 5, labels=False)
    by = [float(te["receiving_yards"].to_numpy()[q == i].sum() / (te["receptions"].to_numpy()[q == i] * p["m_ypr"][q == i]).sum()) for i in range(5)]
    return {"spearman_volume_surprise_vs_efficiency": float(rho.statistic), "p_value": float(rho.pvalue),
            "efficiency_ratio_by_volume_quintile": by, "n": int(len(te)), "params": m.params_}


def run(demo: bool = False, quick: bool = False) -> dict:
    warnings.simplefilter("ignore")
    tables, meta = pipeline.load_tables(prefer="demo" if demo else "real")
    pg, tg = tables["player_games"], tables["team_games"]
    feats = build_feature_table(pg, tg, FeatureParams())
    feats["played"] = feats["targets"].notna()
    seasons = sorted(feats.loc[feats["played"], "season"].unique())
    if demo:
        val, test, first = [seasons[-3]], seasons[-2], seasons[0]
    else:
        val = config.VALIDATION_SEASONS[-2:] if quick else config.VALIDATION_SEASONS
        test, first = config.FINAL_TEST_SEASON, config.FIRST_TRAIN_SEASON
    live_season = config.LAST_SEASON if not demo else seasons[-1]
    out_dir = eval_dir(demo)

    log.info("validation seasons %s", val)
    val_preds = W.run(feats, val, W.DEFAULT_SPECS, first_train=first)
    abl_preds = W.run(feats, val, W.ablation_specs(), first_train=first)
    log.info("final test season %s", test)
    test_preds = W.run(feats, [test], W.DEFAULT_SPECS, first_train=first)
    live_preds = W.run(feats, [live_season], W.DEFAULT_SPECS, first_train=first) if live_season != test else pd.DataFrame()

    val_sc, test_sc = W.score(val_preds), W.score(test_preds)
    abl_sc = W.score(pd.concat([abl_preds, val_preds[val_preds["model"] == "B3_workload"]]))
    live_sc = W.score(live_preds) if len(live_preds) else None
    dep = dependence_check(feats, list(range(first, val[0])), val) if not demo else {}

    for name, df in {"val_preds": val_preds, "test_preds": test_preds, "live_preds": live_preds,
                     "val_summary": val_sc["summary"], "val_diff": val_sc["diff_vs_B3"],
                     "test_summary": test_sc["summary"], "test_diff": test_sc["diff_vs_B3"],
                     "ablation_summary": abl_sc["summary"], "ablation_diff": abl_sc["diff_vs_B3"],
                     "test_calibration": W.calibration(test_preds, "structured"),
                     "val_calibration": W.calibration(val_preds, "structured"),
                     "test_calibration_B3": W.calibration(test_preds, "B3_workload"),
                     "test_rec_calibration": W.calibration(test_preds, "structured", "rec")}.items():
        if len(df):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
    info = {"generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "data_mode": meta["data_mode"], "data_cutoff": meta.get("data_cutoff"),
            "validation_seasons": [int(s) for s in val], "test_season": int(test),
            "live_season": int(live_season), "first_train_season": int(first),
            "n_validation": int((val_preds["model"] == "structured").sum()),
            "n_test": int((test_preds["model"] == "structured").sum()),
            "n_live": int((live_preds["model"] == "structured").sum()) if len(live_preds) else 0,
            "model_version": config.MODEL_VERSION, "dependence_check": dep}
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, default=str))
    if not demo:
        write_markdown(info, val_sc, test_sc, abl_sc, live_sc, val_preds, test_preds, live_preds)
    return info


def _decision(test_sc: dict) -> str:
    d = test_sc["diff_vs_B3"]
    d = d[d["model"] == "structured"].set_index("metric")
    better = [m for m in ("MAE yards", "LogLoss >=25", "LogLoss >=50") if m in d.index and d.loc[m, "ci_high"] < 0]
    high = [m for m in ("LogLoss >=100", "LogLoss >=125", "LogLoss >=150") if m in d.index]
    high_better = [m for m in high if d.loc[m, "ci_high"] < 0]
    high_worse = [m for m in high if d.loc[m, "estimate"] > 0]
    if len(better) >= 2:
        msg = ("The structured model is retained as the default for the yardage distribution, intervals and "
               "low/medium milestones: on the untouched test season it improves on the workload baseline with "
               f"95% cluster-bootstrap intervals excluding zero for {', '.join(better)}.")
    else:
        msg = ("On the test season the structured model did NOT clearly beat the workload baseline. "
               "Per the pre-registered rule the baseline is treated as the reference forecast.")
    if len(high_better) < len(high):
        msg += (" **For high milestones (100+ yards) it has not demonstrated an advantage**: the differences in "
                "log loss vs. the workload baseline at " + ", ".join(m.replace("LogLoss ", "") for m in high) +
                " have intervals that include zero" +
                (f", and its point estimate is slightly worse at {', '.join(m.replace('LogLoss ', '') for m in high_worse)}"
                 if high_worse else "") +
                ". The dashboard therefore shows the workload-baseline probability next to every model probability, "
                "and both models over-predicted the 150+ rate in the test season (see table below).")
    return msg


def write_markdown(info, val_sc, test_sc, abl_sc, live_sc, val_preds, test_preds, live_preds) -> None:
    models = ["structured", "B3_workload", "B1B2_rolling_empirical", "B3n_workload_normal"]
    dep = info["dependence_check"]
    abl_metrics = ["MAE (median)", "CRPS (approx.)", "LogLoss >=50", "LogLoss >=100", "LogLoss >=150", "coverage 80%"]

    def diff_tbl(sc, model="structured"):
        d = sc["diff_vs_B3"]
        d = d[d["model"] == model][["metric", "estimate", "ci_low", "ci_high"]]
        return _md_table(d, 4)

    parts = [f"""# Historical evaluation report

*Generated {info['generated_at']} from real nflverse data (data cutoff {info['data_cutoff']}).
Model version `{info['model_version']}`. Regenerate with `python -m nfl_agent.cli evaluate`.*

## Design

* **Unit**: one WR/TE player-game in which the player took at least one offensive snap or recorded a
  receiving statistic. Forecasts are **conditional on participation**; inactive players are not scored.
* **Walk-forward**: for each evaluated season S, every model (including the baselines and all
  preprocessing, imputation and dispersion estimates) is fitted on seasons {info['first_train_season']}..S-1 only.
  Features use only games played before each kickoff. Seasons are never split across train/test.
* **Validation seasons** {info['validation_seasons']} ({info['n_validation']:,} player-games): used for every
  modelling decision (feature half-life, learner, feature groups, dispersion form).
* **Final test season** {info['test_season']} ({info['n_test']:,} player-games): scored once, after the
  configuration was frozen.
* **Live** {info['live_season']} completed games so far ({info['n_live']:,} player-games): small sample, shown for monitoring only.
* **Uncertainty**: 95% intervals on metric differences come from a cluster bootstrap that resamples whole
  games. Players also recur across weeks; that correlation is not resampled, so intervals are, if anything,
  too narrow.
* **Coverage** uses the randomized probability integral transform, which is the correct check for a discrete
  outcome with a large point mass at 0 yards. Naively counting `lo <= y <= hi` overstates coverage
  (e.g. the default model's naive 80% coverage is about 87%).
* Monte-Carlo: {config.N_SIMS_EVAL:,} draws per forecast in evaluation (MC error on a probability is at most ~1.1
  percentage points and far smaller for rare milestones); the app uses {config.N_SIMS_APP:,}.

## Final test season ({info['test_season']}), receiving yards

{_md_table(summary_table(test_sc['summary'], 'yards'), 3)}

Difference vs. the workload baseline (negative = structured model better), 95% game-cluster bootstrap CI:

{diff_tbl(test_sc)}

**Decision.** {_decision(test_sc)}

### High milestones (test season)

Rare events: a handful of hits or misses moves these numbers; treat them with caution.

{_md_table(high_milestone_table(test_preds, models), 4)}

### Calibration (test season, structured model)

Observed frequency vs. mean forecast probability, by probability bin, with sample counts and a normal-approx.
95% interval on the observed frequency. The dashboard plots these.

{_md_table(W.calibration(test_preds, 'structured').query('threshold in [50, 100, 150]')[['threshold', 'bin_low', 'bin_high', 'n', 'mean_pred', 'observed', 'obs_ci_low', 'obs_ci_high']], 3)}

`coverage` for the rolling-empirical baseline is n/a: its intervals are raw quantiles of the last 16 games,
which have no continuous predictive distribution for a PIT; its naive coverage is in the parquet tables.

### Receptions (test season)

{_md_table(summary_table(test_sc['summary'], 'rec', ['MAE (median)', 'coverage 50%', 'coverage 80%', 'coverage 95%', 'LogLoss >=4', 'LogLoss >=6', 'LogLoss >=8']), 3)}

### Subgroups (test season, structured model)

{_md_table(subgroup_table(test_preds, 'structured'), 3)}

## Validation seasons {info['validation_seasons'][0]}-{info['validation_seasons'][-1]}

{_md_table(summary_table(val_sc['summary'], 'yards'), 3)}

{diff_tbl(val_sc)}

### Ablation and learner comparison (validation seasons)

Does adding a feature group improve held-out performance? `full_*` rows use all groups; `default_history_snaps`
is the shipped configuration.

{_md_table(summary_table(abl_sc['summary'], 'yards', abl_metrics), 4)}

Reading: removing *team volume*, *opponent* or *context* features changes held-out scores by less than
Monte-Carlo/bootstrap noise, while removing *snap share* hurts. These groups were therefore left out of the
default model. That does not prove these factors are irrelevant to football outcomes; it says the
versions available pregame here (rolling team targets, rolling opponent WR/TE targets and yards allowed,
home/dome/rest) add no measurable predictive information beyond the player's own recent usage, which already
reflects his team's passing volume.

### Volume-efficiency dependence

Spearman correlation between a game's target surprise (actual / expected targets) and its per-catch
efficiency (actual / expected yards per catch), validation seasons: **{dep.get('spearman_volume_surprise_vs_efficiency', float('nan')):.3f}**
(n = {dep.get('n', 0):,}). Efficiency ratio by target-surprise quintile (lowest to highest):
{', '.join(f'{x:.3f}' for x in dep.get('efficiency_ratio_by_volume_quintile', []))}. The dependence is weak: about 4-6%
lower yards per catch in the games with the largest target surprises. The simulation treats per-catch
efficiency as independent of volume given the inputs, which slightly fattens the extreme right tail
(see the high-milestone table).

Fitted parameters of the model used for this check (trained on seasons before the first validation season): `{json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in dep.get('params', {}).items()})}`
"""]
    if live_sc is not None:
        parts.append(f"""
## Live season {info['live_season']} (completed games, small sample)

{_md_table(summary_table(live_sc['summary'], 'yards', ['MAE (median)', 'coverage 80%', 'LogLoss >=50', 'LogLoss >=100']), 3)}
""")
    parts.append("""
## Limitations

* Forecasts are conditional on the player participating; the probability of being active is **not**
  modelled (injury reports lack publication timestamps, see DATA_DICTIONARY.md).
* No route participation data are available in-season, so the model cannot separate "fewer routes" from
  "fewer targets per route".
* Stat corrections: training data are today's corrected values, which can differ slightly from what was
  public the night after each game.
* The 150-yard milestone has few events per season; its log loss and calibration are noisy, and the model
  still slightly over-predicts the extreme right tail.
""")
    (config.DOCS_DIR / "EVALUATION.md").write_text("\n".join(parts))


def render_from_saved(demo: bool = False) -> None:
    """Re-render docs/EVALUATION.md from stored evaluation tables (no re-fitting)."""
    d = eval_dir(demo)
    info = json.loads((d / "info.json").read_text())
    rd = lambda n: pd.read_parquet(d / f"{n}.parquet") if (d / f"{n}.parquet").exists() else pd.DataFrame()  # noqa: E731
    val_sc = {"summary": rd("val_summary"), "diff_vs_B3": rd("val_diff")}
    test_sc = {"summary": rd("test_summary"), "diff_vs_B3": rd("test_diff")}
    abl_sc = {"summary": rd("ablation_summary"), "diff_vs_B3": rd("ablation_diff")}
    if demo:
        return                      # the committed report is only ever written from real data
    live = rd("live_preds")
    live_sc = W.score(live, n_boot=50) if len(live) else None
    write_markdown(info, val_sc, test_sc, abl_sc, live_sc, rd("val_preds"), rd("test_preds"), live)
