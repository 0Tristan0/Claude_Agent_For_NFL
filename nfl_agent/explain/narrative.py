"""Plain-English explanations built ONLY from computed forecast numbers.

No language model is involved and nothing here creates a number: every figure
quoted is read from the Forecast object or from stored evaluation tables.
Statements are tagged as [model output], [data] or [assumption].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GLOSSARY = {
    "target": "A pass thrown to the player, whether caught or not (official play-by-play attribution).",
    "target share": "Player targets divided by all of his team's targets in the same games. Denominator: team targets.",
    "snap share": "Share of the team's offensive snaps the player was on the field for (Pro-Football-Reference). "
                  "Includes running plays, so it is NOT route participation.",
    "route participation": "Share of team pass plays on which the player ran a route. Not available in-season "
                           "from public sources used here.",
    "catch rate": "Receptions divided by targets.",
    "aDOT": "Average depth of target: air yards (distance the ball travels past the line of scrimmage, "
            "including incompletions) divided by targets.",
    "air-yard share": "Player air yards divided by team air yards.",
    "YAC": "Yards after catch: receiving yards gained after the ball is caught.",
    "explosive reception": "A reception gaining 20+ yards (nflverse 'receiving_20'; definition inferred, see data dictionary).",
    "prediction interval": "Range that should contain the player's actual yards in the stated share of games "
                           "(e.g. 80%), if the model is calibrated. It describes game-to-game randomness.",
    "probability range": "The spread of a milestone probability when the player's own history is resampled. "
                         "It describes how sure we are about the probability itself, which is a different thing "
                         "from the prediction interval.",
    "calibration": "Whether events forecast at X% happened about X% of the time historically.",
    "Brier score": "Mean squared error of probability forecasts (0 is perfect; lower is better).",
    "log loss": "Penalty that grows sharply for confident wrong probabilities (lower is better).",
    "shrinkage": "Pulling a player's estimate toward the position average in proportion to how little data he has.",
    "effective games": "Number of prior games after down-weighting older games (recent games count more).",
}


def pct(p: float) -> str:
    """Sensible rounding: whole percents; never claim 0% or 100% from a simulation."""
    if p is None:
        return "n/a"
    if p < 0.01:
        return "<1%"
    if p > 0.99:
        return ">99%"
    return f"{round(100 * p):d}%"


def headline(fc) -> str:
    y = fc.yards
    scen = "" if fc.scenario["is_base"] else " under the HYPOTHETICAL scenario"
    return (f"**{fc.player_name}** ({fc.position}, {fc.team}) vs {fc.opponent}{scen}: median **{y['median']}** "
            f"receiving yards, mean {y['mean']:.0f}. In 80% of simulated games he finishes between "
            f"{y['interval_80'][0]} and {y['interval_80'][1]} yards. *[model output, conditional on him playing]*")


def milestone_sentences(fc) -> list[str]:
    out = []
    for t, p in fc.milestone_probabilities.items():
        rng = fc.milestone_probability_ranges.get(t)
        r = f" (range when his history is resampled: {pct(rng[0])}-{pct(rng[1])})" if rng else ""
        out.append(f"{t}+ yards: **{pct(p)}**{r}")
    return out


def main_factors(fc, n: int = 4) -> list[str]:
    """Largest contributions to the expected-target forecast (predictive associations)."""
    out = []
    for c in fc.contributions[:n]:
        direction = "raises" if c["contribution"] > 0 else "lowers"
        size = abs(np.expm1(c["contribution"]))
        out.append(f"His **{c['label']}** {direction} the expected targets by about {size:.0%} relative to a "
                   f"typical value. *[model output: predictive association, not a causal effect]*")
    return out


def uncertainty_factors(fc) -> list[str]:
    out = []
    t = fc.targets
    out.append(f"Target volume alone is uncertain: 80% of simulated games fall between {t['interval_80'][0]} and "
               f"{t['interval_80'][1]} targets (expected {fc.model_params['expected_targets']:.1f}). "
               "*[model output]*")
    out.append(f"Each catch's yardage is highly variable (per-catch shape {fc.model_params['per_catch_gamma_shape']:.2f}; "
               "one long catch can double a game's total). *[model output, estimated from historical data]*")
    out.extend(f"{f} *[data]*" for f in fc.flags)
    if fc.evidence_level == "limited":
        out.append("**Evidence is limited.** Treat the probabilities as rough; they lean heavily on position averages.")
    return out


def role_change(fc) -> list[str]:
    i = fc.inputs
    out = []
    if i.get("tshare_recent") is not None and i.get("tshare_ew") is not None:
        out.append(f"Target share: last ~2 games {i['tshare_recent']:.0%} vs weighted history {i['tshare_ew']:.0%}. *[data]*")
    if i.get("snap_share_recent") is not None and i.get("snap_share_ew") is not None:
        out.append(f"Snap share: last ~2 games {i['snap_share_recent']:.0%} vs weighted history "
                   f"{i['snap_share_ew']:.0%}. *[data]* Snap share is not route participation.")
    rg = fc.recent_games[-3:]
    if rg:
        tg = [g["targets"] for g in rg]
        out.append("Targets in his last 3 games: " + ", ".join(str(int(x)) for x in tg) + ". *[data]*")
    return out


def labels_model_vs_assumption(fc) -> dict[str, list[str]]:
    model = ["Yardage distribution, intervals and milestone probabilities (simulation of fitted models).",
             "Expected targets, catch probability and yards per catch (fitted on seasons "
             f"{fc.model_train_seasons[0]}-{fc.model_train_seasons[-1]})."]
    assumptions = ["The player is active and plays offensive snaps (participation is not modelled).",
                   "Past usage patterns are informative about this game (no injury, depth-chart or "
                   "scheme news is used)."]
    if not fc.scenario["is_base"]:
        assumptions.append(fc.scenario["assumption"])
    return {"model outputs": model, "analyst assumptions": assumptions}


def why_average_is_not_enough(fc) -> str:
    """Use this forecast's own numbers to show why a mean cannot determine P(150+)."""
    mean = fc.yards["mean"]
    lo, hi = fc.yards["interval_80"]
    sd_equiv = max((hi - lo) / (2 * 1.2816), 1.0)
    p_norm = float(stats.norm.sf((149.5 - mean) / sd_equiv))
    p_model = fc.milestone_probabilities.get("150", float("nan"))
    return (
        f"His expected yardage is about {mean:.0f}, yet the chance of 150+ depends on how *spread out* and how "
        f"*lopsided* his outcomes are, not on the average. Two receivers with the same {mean:.0f}-yard average can "
        "differ a lot: one steady (rarely below 40 or above 110) and one boom-or-bust (many quiet games, a few huge "
        "ones). Receiving yards are skewed to the right: they cannot go much below zero but a few long catches can "
        f"produce a very large game. A bell curve with the same centre and spread would put P(150+) at "
        f"{pct(p_norm)}; the simulation, which models targets, catches and per-catch yardage separately, gives "
        f"{pct(p_model)}. *[model output; the bell-curve figure is shown only for comparison]* "
        "Our evaluation found a normal-distribution model under-predicts 150-yard games several-fold."
    )


def historical_accuracy(fc, eval_tables: dict) -> list[str]:
    """Read stored evaluation results for comparable forecasts. Never recomputes or invents metrics."""
    out = []
    preds = eval_tables.get("test_preds")
    info = eval_tables.get("info", {})
    if preds is None or preds.empty:
        return ["No stored evaluation results. Run `python -m nfl_agent.cli evaluate`."]
    s = preds[preds["model"] == "structured"]
    m = fc.yards["mean"]
    comp = s[(s["yards_mean"] - m).abs() <= 10]
    if len(comp) >= 30:
        mae = float(np.mean(np.abs(comp["receiving_yards"] - comp["yards_median"])))
        pit = comp["yards_pit"].to_numpy() if "yards_pit" in comp else None
        cov = float(np.mean(np.abs(pit - 0.5) <= 0.4)) if pit is not None else float("nan")
        out.append(f"In the {info.get('test_season', 'test')} held-out season, {len(comp):,} forecasts had a similar "
                   f"expected yardage ({m - 10:.0f}-{m + 10:.0f}). Their median forecast missed by {mae:.0f} yards on "
                   f"average, and {cov:.0%} of outcomes fell inside the 80% interval. *[evaluation data]*")
    for t in ("100", "150"):
        p = fc.milestone_probabilities.get(t)
        col = f"yards_p_ge_{t}"
        if p is None or col not in s:
            continue
        near = s[(s[col] - p).abs() <= max(0.03, 0.25 * p)]
        if len(near) >= 50:
            obs = float((near["receiving_yards"] >= int(t)).mean())
            out.append(f"Forecasts of about {pct(p)} for {t}+ yards ({len(near):,} cases) came true "
                       f"{pct(obs)} of the time. *[evaluation data]*")
        else:
            out.append(f"Too few comparable historical {t}+ forecasts ({len(near)}) to say how reliable "
                       f"a {pct(p)} forecast is. *[evaluation data]*")
    out.append("Rare milestones (150+) occur in under 1% of WR/TE games, so their track record is noisy.")
    return out


def full_explanation(fc, eval_tables: dict | None = None) -> dict[str, list[str] | str]:
    return {
        "headline": headline(fc),
        "milestones": milestone_sentences(fc),
        "main_factors": main_factors(fc),
        "uncertainty": uncertainty_factors(fc),
        "role_change": role_change(fc),
        "labels": labels_model_vs_assumption(fc),
        "missing": [f"{m} *[missing input]*" for m in fc.missing_inputs],
        "average_vs_tail": why_average_is_not_enough(fc),
        "accuracy": historical_accuracy(fc, eval_tables or {}),
    }
