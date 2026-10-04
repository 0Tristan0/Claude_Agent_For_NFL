# Model methodology

Version `structured-gbm-v1.0`. Results are in [EVALUATION.md](EVALUATION.md); data definitions are in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Scope

* **Outcome**: receiving yards and receptions for WR and TE in one game.
* **Population**: player-games in which the player took at least one offensive snap or recorded a receiving
  stat. **Every forecast is conditional on the player participating.** The probability of being active is not
  modelled, because injury reports carry no publication timestamp (see the data dictionary).
* **Purpose**: education and forecast-accuracy research. No sportsbook data, wager advice or stake logic.

## Baselines (always computed, always shown)

| Name | Point forecast | Distribution / milestones |
|---|---|---|
| B1 rolling | Mean of last 8 games played | Empirical quantiles of last 16 games (position quantiles if < 4 games) |
| B2 empirical | — | Share of last 16 games reaching the milestone, shrunk toward the position rate with 6 pseudo-games (Beta-binomial posterior mean) |
| B3 workload | `tpg_ew × ypt_ew` (shrunk targets/game × yards/target), rescaled to be unbiased on training data | Mean × empirical ratio (outcome / mean) from training rows in the same mean quintile |
| B3n normal | Same mean | Normal errors with quintile-specific SD (only to show what a normal assumption does to tails) |

## Structured model

Interpretable chain, fitted on the training seasons only:

1. **Target volume.** Expected targets μ from a gradient-boosted Poisson regression
   (`HistGradientBoostingRegressor(loss="poisson")`, 250 trees, 15 leaves, ≥200 rows per leaf) on player-history
   and snap features. Game-to-game variation: negative binomial with size `r(μ) = 1 / exp(c0 + c1·log μ)`. The
   dispersion was made mean-dependent after diagnostics on validation seasons showed that a single size
   under-dispersed low-volume players (observed/predicted variance 1.65) and over-dispersed high-volume players
   (0.75), which distorted tail probabilities. `c0, c1` are fitted by weighted regression of per-decile
   method-of-moments estimates.
2. **Catches given targets.** Logistic regression for per-target catch probability p (inputs: shrunk catch
   rate, aDOT, TE flag, experience). Beta-binomial ρ estimated by moments captures game-level catch-rate
   variation, so targets in a game are not treated as independent coin flips (estimated ρ ≈ 0.01: small).
3. **Yards given catches.** Gamma regression (log link) for mean yards per catch m (inputs: shrunk YPR, aDOT,
   YAC/reception, explosive-reception rate, TE flag, experience). Each catch's yards ~ Gamma(shape a, mean m),
   with a game-level multiplicative shock s ~ Gamma(mean 1, variance τ²) shared by all catches in a game.
   `a` and `τ²` come from regressing `(Y − k·m)²/m²` on k and k² (k = catches): the moment identity
   `E = (1+τ²)k/a + τ²k²`. Estimates: a ≈ 1.35–1.40 (per-catch yardage is highly variable) and τ² ≈ 0–0.003.
4. **Simulation.** 20,000 draws per forecast in the app (2,000 in evaluation). Receptions ≤ targets by
   construction; yards are rounded to whole yards. Seeds are derived from (player, game, scenario), so the same
   request returns identical numbers.

**Why not a normal distribution?** Receiving yards have a point mass near zero and a long right tail. In the
2025 test season, the normal-error variant expected 6 games of 150+ yards when 20 occurred; the structured model
expected 30 and the workload baseline 31 (see the evaluation report for direction and size of these errors).

**Volume–efficiency dependence.** We checked whether games with more targets than expected have different
yards per catch. In validation seasons, the Spearman correlation between target surprise and per-catch efficiency
was 0.014; yards per catch were about 4% lower in the top quintile of target surprise. The simulation does not
model this weak negative dependence, which slightly fattens the extreme right tail. This is a documented
limitation, not a hidden assumption.

## Small samples, rookies, team and quarterback changes

* **Shrinkage**: every rate is a weighted ratio of sums plus pseudo-observations at the position prior from
  *earlier* seasons (see the data dictionary for pseudo-counts). A player with no NFL games gets the prior.
* **Rookies**: the model also receives `rookie_flag`, draft position (undrafted = 260) and effective sample size,
  so it learns from history how far typical rookies sit from the position prior. Forecasts with fewer than 4 prior
  games are labelled **limited evidence**.
* **Team changes**: the first game with a new team is flagged; tenure with the team is an input. History from
  the old team is kept (it remains informative) but the forecast carries a warning.
* **Quarterback changes**: QB availability is **not** an input, because the schedule's QB field is retrospective
  for completed games and of unknown timing for upcoming ones. The forecast page lists it as context and as
  missing information. Users can explore a different-volume scenario; that is labelled hypothetical.

## Feature selection (validation seasons only)

Candidate groups: player history; snap share; team passing volume; opponent defense (WR/TE targets, yards per
target and catch rate allowed); context (home, fixed dome, rest, postseason). An ablation on 2019–2024 showed
team-volume, opponent and context groups did not improve held-out MAE, CRPS or log loss beyond noise, while snap
share did. The default therefore uses **player history + snaps**. The GBM target learner beat the Poisson GLM
(MAE 16.36 vs 16.48; log loss ≥50 0.353 vs 0.357). The player half-life (10 vs 6 games) was chosen on the same
seasons. All of these decisions were frozen before the 2025 test season was scored.

No fixed weights were assigned to any factor, and there are no hand rules ("a missing teammate adds 3 targets").
Relationships are learned from history; scenario sliders are labelled assumptions.

## Uncertainty reported for each forecast

* **Prediction intervals (50/80/95%)**: game-to-game randomness *if the model is right*.
* **Probability range**: the 10th–90th percentile of each milestone probability when the player's own prior games
  are reweighted with Poisson(1) bootstrap weights (30 resamples) and the forecast recomputed. This describes
  uncertainty about the probability itself due to his limited history. It excludes model-parameter uncertainty.
* **Evidence level**: limited (< 4 prior games, or first game with a new team), moderate (< 12 or < 3 games with
  the team), substantial.
* **Historical accuracy**: the forecast page quotes stored test-season results for comparable forecasts (similar
  expected yards; similar milestone probability), with counts, and says when too few comparable cases exist.

## Explanations

Feature contributions are model-agnostic: for each input, the expected-target forecast is recomputed with that
input set to its training median; the change in log expected targets is that input's contribution. These are
**predictive associations within the model**, not causal effects, and correlated inputs share credit unevenly.
All explanation text is generated deterministically from computed numbers (`nfl_agent/explain/narrative.py`);
no language model is used and no API key is needed. The question box routes questions to these computed
answers with keyword rules (`nfl_agent/agent/router.py`).

## Scenario analysis

Playing-time and team-pass-volume sliders multiply expected targets. That proportionality is an **analyst
assumption** (it is not a validated causal effect), and the scenario output is labelled HYPOTHETICAL. Baseline
numbers are not scenario-adjusted.

## Model fitting policy in the app

A forecast for a game in season S uses parameters fitted on seasons 2014..S−1 (the evaluated design) and
features from every game played before kickoff. A completed game is therefore shown exactly as the backtest
would have forecast it.

## Default model decision

The structured model is the default because, on the untouched 2025 test season, it beat the workload baseline on
MAE and on log loss for 25+ and 50+ yards with cluster-bootstrap intervals excluding zero. **For 100+ yard
milestones it showed no demonstrated advantage** (intervals include zero, point estimate slightly worse at 125+
and 150+), so the dashboard shows the workload-baseline probability next to every model probability.
