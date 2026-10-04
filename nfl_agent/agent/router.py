"""Deterministic question router: maps a plain-English question to computed answers.

This is the "research agent" interface. It uses keyword rules, not a language
model, so it works without any API key and can never invent a number: every
answer is assembled from the Forecast object and stored evaluation tables.
"""
from __future__ import annotations

import re

from ..explain import narrative as N

INTENTS = [
    ("scenario", r"playing time|snap|reduced|fewer|less playing|limited role|pitch count|volume change|pass(ing)? volume"),
    ("targets", r"target"),
    ("accuracy", r"accura|historic|track record|how good|reliab|calibrat|past (forecast|predict)"),
    ("average_vs_tail", r"average|mean alone|why can'?t|150[- ]yard|big game"),
    ("evidence", r"evidence|support|wrong|why|risk|uncertain|explain|factor"),
    ("milestones", r"probab|chance|odds|reach|hit|\b\d{2,3}\b|milestone|over|at least"),
    ("receptions", r"reception|catches|catch"),
    ("distribution", r"distribution|projection|projected|forecast|expect|range|interval"),
]

SUPPORTED = [
    "What is this receiver's projected receiving-yard distribution?",
    "How uncertain is his projected target volume?",
    "What is the estimated probability of reaching 50, 75, 100, or 150 receiving yards?",
    "How would reduced playing time change the forecast?",
    "What evidence supports the forecast, and what could make it wrong?",
    "How accurately has the model predicted comparable outcomes historically?",
    "Why can't his average yardage alone tell me the chance of a 150-yard game?",
]


def classify(question: str) -> str:
    q = question.lower()
    for name, pat in INTENTS:
        if re.search(pat, q):
            return name
    return "unknown"


def answer(question: str, fc, eval_tables: dict | None = None, scenario_fc=None) -> dict:
    """Return {'intent', 'markdown'}; the UI renders it. Numbers come only from fc / eval_tables."""
    intent = classify(question)
    ex = N.full_explanation(fc, eval_tables)
    if intent == "distribution":
        y = fc.yards
        md = [ex["headline"],
              f"- 50% interval: {y['interval_50'][0]}-{y['interval_50'][1]} yards",
              f"- 80% interval: {y['interval_80'][0]}-{y['interval_80'][1]} yards",
              f"- 95% interval: {y['interval_95'][0]}-{y['interval_95'][1]} yards",
              "These describe game-to-game randomness *if the model is right*; see 'probability range' for "
              "uncertainty about the estimates themselves."]
    elif intent == "targets":
        t = fc.targets
        md = [f"Expected targets: **{fc.model_params['expected_targets']:.1f}** (median {t['median']}).",
              f"- 50% of simulated games: {t['interval_50'][0]}-{t['interval_50'][1]} targets",
              f"- 80%: {t['interval_80'][0]}-{t['interval_80'][1]}; 95%: {t['interval_95'][0]}-{t['interval_95'][1]}",
              "Target counts are more variable than a simple Poisson count would imply (negative-binomial "
              "dispersion estimated from history). *[model output]*"] + ex["role_change"]
    elif intent == "receptions":
        r = fc.receptions
        md = [f"Median receptions **{r['median']}** (mean {r['mean']:.1f}); 80% interval "
              f"{r['interval_80'][0]}-{r['interval_80'][1]}. Catch probability per target "
              f"{fc.model_params['catch_probability']:.0%}. *[model output]*"] + \
             [f"{k}+ receptions: **{N.pct(v)}**" for k, v in fc.reception_milestones.items()]
    elif intent == "milestones":
        md = ["Estimated probability of reaching each milestone, conditional on playing *[model output]*:"] + \
             [f"- {s}" for s in ex["milestones"]] + \
             ["Workload baseline for comparison: " + ", ".join(
                 f"{k}+: {N.pct(v)}" for k, v in fc.baseline["workload_milestones"].items())]
    elif intent == "scenario":
        if scenario_fc is None:
            md = ["Use the **Hypothetical scenario** sliders (playing time / team pass volume) on the Player "
                  "forecast page, then ask again. Scenarios scale expected targets proportionally; that is an "
                  "**analyst assumption**, not a validated causal effect."]
        else:
            s = scenario_fc.scenario
            md = [f"HYPOTHETICAL: playing time x{s['playing_time_multiplier']:.2f}, team pass volume "
                  f"x{s['team_volume_multiplier']:.2f}.",
                  f"- Median yards: {fc.yards['median']} -> **{scenario_fc.yards['median']}**",
                  f"- 80% interval: {fc.yards['interval_80']} -> {scenario_fc.yards['interval_80']}"] + \
                 [f"- {t}+ yards: {N.pct(fc.milestone_probabilities[t])} -> **{N.pct(p)}**"
                  for t, p in scenario_fc.milestone_probabilities.items()] + \
                 [f"*{s['assumption']}*"]
    elif intent == "evidence":
        md = ["**Main factors (predictive associations):**"] + [f"- {x}" for x in ex["main_factors"]] + \
             ["**What could make it wrong:**"] + [f"- {x}" for x in ex["uncertainty"]] + \
             [f"- Missing: {x}" for x in ex["missing"]]
    elif intent == "accuracy":
        md = ex["accuracy"]
    elif intent == "average_vs_tail":
        md = [ex["average_vs_tail"]]
    else:
        md = ["I can answer these questions about the selected player and game:"] + [f"- {q}" for q in SUPPORTED]
    return {"intent": intent, "markdown": "\n\n".join(md)}
