# Historical evaluation report

*Generated 2026-10-04T18:02:27+00:00 from real nflverse data (data cutoff 2026-10-04 13:30:00+00:00).
Model version `structured-gbm-v1.0`. Regenerate with `python -m nfl_agent.cli evaluate`.*

## Design

* **Unit**: one WR/TE player-game in which the player took at least one offensive snap or recorded a
  receiving statistic. Forecasts are **conditional on participation**; inactive players are not scored.
* **Walk-forward**: for each evaluated season S, every model (including the baselines and all
  preprocessing, imputation and dispersion estimates) is fitted on seasons 2014..S-1 only.
  Features use only games played before each kickoff. Seasons are never split across train/test.
* **Validation seasons** [2019, 2020, 2021, 2022, 2023, 2024] (26,802 player-games): used for every
  modelling decision (feature half-life, learner, feature groups, dispersion form).
* **Final test season** 2025 (4,649 player-games): scored once, after the
  configuration was frozen.
* **Live** 2026 completed games so far (823 player-games): small sample, shown for monitoring only.
* **Uncertainty**: 95% intervals on metric differences come from a cluster bootstrap that resamples whole
  games. Players also recur across weeks; that correlation is not resampled, so intervals are, if anything,
  too narrow.
* **Coverage** uses the randomized probability integral transform, which is the correct check for a discrete
  outcome with a large point mass at 0 yards. Naively counting `lo <= y <= hi` overstates coverage
  (e.g. the default model's naive 80% coverage is about 87%).
* Monte-Carlo: 2,000 draws per forecast in evaluation (MC error on a probability is at most ~1.1
  percentage points and far smaller for rare milestones); the app uses 20,000.

## Final test season (2025), receiving yards

| model | MAE (median) | CRPS (approx.) | coverage 50% | coverage 80% | coverage 95% | width 80% | Brier >=50 | LogLoss >=50 | Brier >=100 | LogLoss >=100 | Brier >=150 | LogLoss >=150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1B2_rolling_empirical | 16.275 | 9.357 | n/a | n/a | n/a | 45.787 | 0.108 | 0.353 | 0.030 | 0.120 | 0.004 | 0.023 |
| B3_workload | 15.851 | 8.686 | 0.499 | 0.810 | 0.954 | 52.833 | 0.105 | 0.332 | 0.029 | 0.110 | 0.004 | 0.020 |
| B3n_workload_normal | 17.805 | 9.588 | 0.458 | 0.904 | 0.958 | 60.204 | 0.105 | 0.341 | 0.029 | 0.117 | 0.004 | 0.022 |
| structured | 15.176 | 8.345 | 0.507 | 0.809 | 0.952 | 51.719 | 0.103 | 0.325 | 0.029 | 0.110 | 0.004 | 0.021 |

Difference vs. the workload baseline (negative = structured model better), 95% game-cluster bootstrap CI:

| metric | estimate | ci_low | ci_high |
|---|---|---|---|
| MAE yards | -0.6750 | -0.8438 | -0.5132 |
| LogLoss >=25 | -0.0195 | -0.0255 | -0.0127 |
| Brier >=25 | -0.0078 | -0.0102 | -0.0053 |
| LogLoss >=50 | -0.0072 | -0.0111 | -0.0029 |
| Brier >=50 | -0.0014 | -0.0025 | -0.0002 |
| LogLoss >=75 | -0.0023 | -0.0053 | 0.0005 |
| Brier >=75 | 0.0001 | -0.0005 | 0.0007 |
| LogLoss >=100 | -0.0004 | -0.0027 | 0.0017 |
| Brier >=100 | 0.0002 | -0.0002 | 0.0006 |
| LogLoss >=125 | 0.0007 | -0.0009 | 0.0024 |
| Brier >=125 | 0.0001 | -0.0002 | 0.0004 |
| LogLoss >=150 | 0.0014 | -0.0000 | 0.0031 |
| Brier >=150 | 0.0001 | -0.0001 | 0.0003 |

**Decision.** The structured model is retained as the default for the yardage distribution, intervals and low/medium milestones: on the untouched test season it improves on the workload baseline with 95% cluster-bootstrap intervals excluding zero for MAE yards, LogLoss >=25, LogLoss >=50. **For high milestones (100+ yards) it has not demonstrated an advantage**: the differences in log loss vs. the workload baseline at >=100, >=125, >=150 have intervals that include zero, and its point estimate is slightly worse at >=125, >=150. The dashboard therefore shows the workload-baseline probability next to every model probability, and both models over-predicted the 150+ rate in the test season (see table below).

### High milestones (test season)

Rare events: a handful of hits or misses moves these numbers; treat them with caution.

| model | milestone | n_forecasts | events | expected_events | mean_pred | observed |
|---|---|---|---|---|---|---|
| structured | >=100 | 4649 | 156 | 175.0595 | 0.0377 | 0.0336 |
| structured | >=125 | 4649 | 63 | 74.1780 | 0.0160 | 0.0136 |
| structured | >=150 | 4649 | 20 | 30.2650 | 0.0065 | 0.0043 |
| B3_workload | >=100 | 4649 | 156 | 169.1199 | 0.0364 | 0.0336 |
| B3_workload | >=125 | 4649 | 63 | 72.4116 | 0.0156 | 0.0136 |
| B3_workload | >=150 | 4649 | 20 | 30.9493 | 0.0067 | 0.0043 |
| B1B2_rolling_empirical | >=100 | 4649 | 156 | 192.8022 | 0.0415 | 0.0336 |
| B1B2_rolling_empirical | >=125 | 4649 | 63 | 73.3635 | 0.0158 | 0.0136 |
| B1B2_rolling_empirical | >=150 | 4649 | 20 | 28.8459 | 0.0062 | 0.0043 |
| B3n_workload_normal | >=100 | 4649 | 156 | 108.0896 | 0.0233 | 0.0336 |
| B3n_workload_normal | >=125 | 4649 | 63 | 29.8769 | 0.0064 | 0.0136 |
| B3n_workload_normal | >=150 | 4649 | 20 | 6.3176 | 0.0014 | 0.0043 |

### Calibration (test season, structured model)

Observed frequency vs. mean forecast probability, by probability bin, with sample counts and a normal-approx.
95% interval on the observed frequency. The dashboard plots these.

| threshold | bin_low | bin_high | n | mean_pred | observed | obs_ci_low | obs_ci_high |
|---|---|---|---|---|---|---|---|
| 50 | 0.000 | 0.020 | 1031 | 0.010 | 0.003 | 0.000 | 0.006 |
| 50 | 0.020 | 0.050 | 789 | 0.032 | 0.030 | 0.018 | 0.042 |
| 50 | 0.050 | 0.100 | 548 | 0.073 | 0.069 | 0.048 | 0.091 |
| 50 | 0.100 | 0.200 | 634 | 0.143 | 0.115 | 0.090 | 0.140 |
| 50 | 0.200 | 0.300 | 441 | 0.248 | 0.234 | 0.194 | 0.273 |
| 50 | 0.300 | 0.400 | 436 | 0.351 | 0.303 | 0.260 | 0.346 |
| 50 | 0.400 | 0.500 | 288 | 0.450 | 0.431 | 0.373 | 0.488 |
| 50 | 0.500 | 0.600 | 320 | 0.545 | 0.512 | 0.458 | 0.567 |
| 50 | 0.600 | 0.700 | 124 | 0.642 | 0.694 | 0.612 | 0.775 |
| 50 | 0.700 | 0.800 | 38 | 0.719 | 0.711 | 0.566 | 0.855 |
| 100 | 0.000 | 0.020 | 2922 | 0.004 | 0.003 | 0.001 | 0.005 |
| 100 | 0.020 | 0.050 | 592 | 0.033 | 0.027 | 0.014 | 0.040 |
| 100 | 0.050 | 0.100 | 480 | 0.071 | 0.050 | 0.031 | 0.069 |
| 100 | 0.100 | 0.200 | 508 | 0.145 | 0.130 | 0.101 | 0.159 |
| 100 | 0.200 | 0.300 | 138 | 0.241 | 0.290 | 0.214 | 0.366 |
| 100 | 0.300 | 0.400 | 9 | 0.314 | 0.222 | 0.000 | 0.494 |
| 150 | 0.000 | 0.020 | 4112 | 0.002 | 0.001 | 0.000 | 0.002 |
| 150 | 0.020 | 0.050 | 417 | 0.032 | 0.010 | 0.000 | 0.019 |
| 150 | 0.050 | 0.100 | 117 | 0.065 | 0.103 | 0.048 | 0.158 |
| 150 | 0.100 | 0.200 | 3 | 0.107 | 0.000 | 0.000 | 0.000 |

`coverage` for the rolling-empirical baseline is n/a: its intervals are raw quantiles of the last 16 games,
which have no continuous predictive distribution for a PIT; its naive coverage is in the parquet tables.

### Receptions (test season)

| model | MAE (median) | coverage 50% | coverage 80% | coverage 95% | LogLoss >=4 | LogLoss >=6 | LogLoss >=8 |
|---|---|---|---|---|---|---|---|
| B1B2_rolling_empirical | 1.219 | n/a | n/a | n/a | 0.387 | 0.229 | 0.111 |
| B3_workload | 1.177 | 0.508 | 0.816 | 0.955 | 0.364 | 0.212 | 0.099 |
| B3n_workload_normal | 1.296 | 0.485 | 0.870 | 0.966 | 0.363 | 0.215 | 0.104 |
| structured | 1.095 | 0.498 | 0.809 | 0.952 | 0.349 | 0.206 | 0.099 |

### Subgroups (test season, structured model)

| subgroup | n | mae | mean_pred | mean_obs | coverage_80 | p100_pred | p100_obs |
|---|---|---|---|---|---|---|---|
| all | 4649 | 15.176 | 24.777 | 23.636 | 0.809 | 0.038 | 0.034 |
| <=3 prior games | 283 | 8.505 | 11.915 | 11.110 | 0.799 | 0.008 | 0.011 |
| first game with new team | 122 | 8.848 | 13.629 | 10.992 | 0.820 | 0.012 | 0.000 |
| rookies | 739 | 12.851 | 19.467 | 18.242 | 0.815 | 0.021 | 0.015 |
| postseason | 208 | 16.505 | 27.084 | 24.827 | 0.827 | 0.041 | 0.043 |
| tight ends | 1743 | 11.406 | 17.722 | 17.377 | 0.813 | 0.015 | 0.011 |

## Validation seasons 2019-2024

| model | MAE (median) | CRPS (approx.) | coverage 50% | coverage 80% | coverage 95% | width 80% | Brier >=50 | LogLoss >=50 | Brier >=100 | LogLoss >=100 | Brier >=150 | LogLoss >=150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1B2_rolling_empirical | 17.740 | 10.007 | n/a | n/a | n/a | 49.164 | 0.120 | 0.383 | 0.038 | 0.145 | 0.007 | 0.038 |
| B3_workload | 17.189 | 9.367 | 0.499 | 0.801 | 0.950 | 55.799 | 0.115 | 0.362 | 0.037 | 0.134 | 0.007 | 0.033 |
| B3n_workload_normal | 19.161 | 10.296 | 0.447 | 0.892 | 0.953 | 62.500 | 0.115 | 0.373 | 0.037 | 0.143 | 0.007 | 0.038 |
| structured | 16.361 | 8.983 | 0.501 | 0.798 | 0.947 | 53.957 | 0.113 | 0.353 | 0.037 | 0.133 | 0.007 | 0.032 |

| metric | estimate | ci_low | ci_high |
|---|---|---|---|
| MAE yards | -0.8286 | -0.9045 | -0.7527 |
| LogLoss >=25 | -0.0241 | -0.0265 | -0.0215 |
| Brier >=25 | -0.0092 | -0.0101 | -0.0080 |
| LogLoss >=50 | -0.0090 | -0.0107 | -0.0074 |
| Brier >=50 | -0.0019 | -0.0024 | -0.0014 |
| LogLoss >=75 | -0.0026 | -0.0040 | -0.0014 |
| Brier >=75 | -0.0001 | -0.0004 | 0.0002 |
| LogLoss >=100 | -0.0009 | -0.0021 | 0.0001 |
| Brier >=100 | 0.0000 | -0.0002 | 0.0002 |
| LogLoss >=125 | -0.0004 | -0.0012 | 0.0005 |
| Brier >=125 | 0.0000 | -0.0001 | 0.0001 |
| LogLoss >=150 | -0.0005 | -0.0012 | -0.0001 |
| Brier >=150 | -0.0000 | -0.0001 | 0.0000 |

### Ablation and learner comparison (validation seasons)

Does adding a feature group improve held-out performance? `full_*` rows use all groups; `default_history_snaps`
is the shipped configuration.

| model | MAE (median) | CRPS (approx.) | LogLoss >=50 | LogLoss >=100 | LogLoss >=150 | coverage 80% |
|---|---|---|---|---|---|---|
| B3_workload | 17.1892 | 9.3670 | 0.3624 | 0.1337 | 0.0328 | 0.8012 |
| default_history_snaps | 16.3606 | 8.9832 | 0.3534 | 0.1327 | 0.0323 | 0.7980 |
| default_with_glm_targets | 16.4829 | 9.0599 | 0.3565 | 0.1336 | 0.0332 | 0.8025 |
| full_all_groups | 16.3522 | 8.9884 | 0.3539 | 0.1326 | 0.0322 | 0.7996 |
| full_minus_context | 16.3595 | 8.9846 | 0.3536 | 0.1327 | 0.0322 | 0.7980 |
| full_minus_opponent | 16.3595 | 8.9880 | 0.3540 | 0.1327 | 0.0323 | 0.7974 |
| full_minus_snaps | 16.4177 | 9.0156 | 0.3545 | 0.1326 | 0.0323 | 0.7992 |
| full_minus_team_volume | 16.3420 | 8.9790 | 0.3534 | 0.1326 | 0.0322 | 0.8007 |
| history_only | 16.4397 | 9.0274 | 0.3547 | 0.1329 | 0.0326 | 0.7979 |

Reading: removing *team volume*, *opponent* or *context* features changes held-out scores by less than
Monte-Carlo/bootstrap noise, while removing *snap share* hurts. These groups were therefore left out of the
default model. That does not prove these factors are irrelevant to football outcomes; it says the
versions available pregame here (rolling team targets, rolling opponent WR/TE targets and yards allowed,
home/dome/rest) add no measurable predictive information beyond the player's own recent usage, which already
reflects his team's passing volume.

### Volume-efficiency dependence

Spearman correlation between a game's target surprise (actual / expected targets) and its per-catch
efficiency (actual / expected yards per catch), validation seasons: **0.014**
(n = 17,659). Efficiency ratio by target-surprise quintile (lowest to highest):
1.033, 1.017, 1.011, 0.988, 0.958. The dependence is weak: about 4-6%
lower yards per catch in the games with the largest target surprises. The simulation treats per-catch
efficiency as independent of volume given the inputs, which slightly fattens the extreme right tail
(see the high-milestone table).

Fitted parameters of the model used for this check (trained on seasons before the first validation season): `{"nb_size_const": 9.7174, "disp_intercept": -0.2374, "disp_slope": -1.215, "bb_rho": 0.0097, "gamma_shape": 1.4001, "tau2": 0.0028, "n_train": 20318}`


## Live season 2026 (completed games, small sample)

| model | MAE (median) | coverage 80% | LogLoss >=50 | LogLoss >=100 |
|---|---|---|---|---|
| B1B2_rolling_empirical | 17.220 | n/a | 0.377 | 0.122 |
| B3_workload | 16.440 | 0.796 | 0.362 | 0.103 |
| B3n_workload_normal | 18.933 | 0.900 | 0.373 | 0.111 |
| structured | 15.694 | 0.787 | 0.355 | 0.102 |


## Limitations

* Forecasts are conditional on the player participating; the probability of being active is **not**
  modelled (injury reports lack publication timestamps, see DATA_DICTIONARY.md).
* No route participation data are available in-season, so the model cannot separate "fewer routes" from
  "fewer targets per route".
* Stat corrections: training data are today's corrected values, which can differ slightly from what was
  public the night after each game.
* The 150-yard milestone has few events per season; its log loss and calibration are noisy, and the model
  still slightly over-predicts the extreme right tail.
