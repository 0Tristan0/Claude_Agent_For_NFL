"""Structured probabilistic model: targets -> catches | targets -> yards | catches.

Three generalized linear models (fitted only on the training split), plus
dispersion parameters estimated from training residuals by the method of moments:

1. Target volume:   Poisson GLM (log link) for the mean, negative-binomial size
   r for overdispersion. An optional gradient-boosted alternative is provided
   for comparison.
2. Catch rate:      logistic GLM per target, beta-binomial rho for game-to-game
   variation in catch rate (targets within a game are not independent).
3. Yards per catch: gamma GLM (log link) for the mean per-catch yardage; the
   per-catch shape `a` and the game-level shock variance tau2 come from
   regressing squared residuals on k and k^2 (k = catches):
       E[(Y - k m)^2 / m^2] = (1 + tau2) k / a + tau2 k^2

Coefficients are predictive associations learned from historical data, not
causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from . import simulate as sim

EPS = 1e-6

FEATURE_GROUPS = {
    "player_history": ["log_tpg_ew", "log_tshare_ew", "log_tshare_recent", "log1p_eff_games",
                       "rookie_flag", "log_draft_pick", "is_te", "team_changed",
                       "log1p_games_with_team", "log_days_since_last"],
    "snaps": ["snap_share_ew", "snap_share_recent"],
    "team_volume": ["log_team_tpg_ew"],
    "opponent": ["log_opp_def_tpg_rel"],
    "context": ["ctx_is_home", "ctx_fixed_dome", "ctx_rest_days_c", "is_post"],
}
CATCH_FEATURES = {
    "player_history": ["logit_catch_rate_ew", "log_adot_ew", "is_te", "log1p_eff_games"],
    "opponent": ["log_opp_def_catch_rel"],
    "context": ["ctx_fixed_dome", "is_post"],
}
YPR_FEATURES = {
    "player_history": ["log_ypr_ew", "log_adot_ew", "log_yac_per_rec_ew", "explosive_rate_ew",
                       "is_te", "log1p_eff_games"],
    "opponent": ["log_opp_def_ypt_rel"],
    "context": ["ctx_fixed_dome", "is_post"],
}
ALL_GROUPS = tuple(FEATURE_GROUPS)
# Frozen default, selected on validation seasons 2019-2024 (see docs/EVALUATION.md):
# adding team volume, opponent and context groups did not improve held-out scores,
# so the simpler model is the default.
DEFAULT_GROUPS = ("player_history", "snaps")
DEFAULT_LEARNER = "gbm"

FEATURE_LABELS = {
    "log_tpg_ew": "recent targets per game", "log_tshare_ew": "target share (weighted history)",
    "log_tshare_recent": "target share (last ~2 games)", "log1p_eff_games": "amount of recent history",
    "rookie_flag": "rookie season", "log_draft_pick": "draft position (later pick = higher)",
    "is_te": "tight end", "team_changed": "first game with a new team",
    "log1p_games_with_team": "games with current team", "log_days_since_last": "days since last game played",
    "snap_share_ew": "offensive snap share (weighted history)", "snap_share_recent": "snap share (last ~2 games)",
    "log_team_tpg_ew": "team pass targets per game", "log_opp_def_tpg_rel": "opponent WR/TE targets allowed",
    "ctx_is_home": "home game", "ctx_fixed_dome": "fixed dome stadium", "ctx_rest_days_c": "days of rest",
    "is_post": "postseason game", "logit_catch_rate_ew": "catch rate (shrunk)",
    "log_adot_ew": "average depth of target", "log_opp_def_catch_rel": "opponent catch rate allowed",
    "log_ypr_ew": "yards per reception (shrunk)", "log_yac_per_rec_ew": "yards after catch per reception",
    "explosive_rate_ew": "share of receptions gaining 20+ yards", "log_opp_def_ypt_rel": "opponent yards per target allowed",
}


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Transform pregame features into model inputs. Missing stays missing (imputed in-pipeline)."""
    X = pd.DataFrame(index=df.index)
    X["log_tpg_ew"] = np.log(df["tpg_ew"].clip(lower=0.05))
    X["log_tshare_ew"] = np.log(df["tshare_ew"].clip(lower=0.005))
    X["log_tshare_recent"] = np.log(df["tshare_recent"].clip(lower=0.005))
    X["log1p_eff_games"] = np.log1p(df["eff_games"])
    X["rookie_flag"] = df["rookie_flag"]
    X["log_draft_pick"] = df["log_draft_pick"]
    X["is_te"] = df["is_te"]
    X["team_changed"] = df["team_changed"]
    X["log1p_games_with_team"] = np.log1p(df["games_with_team_prior"])
    X["log_days_since_last"] = np.log(df["days_since_last_game"].clip(3, 400))
    X["snap_share_ew"] = df["snap_share_ew"]
    X["snap_share_recent"] = df["snap_share_recent"]
    X["log_team_tpg_ew"] = np.log(df["team_tpg_ew"])
    X["log_opp_def_tpg_rel"] = np.log(df["opp_def_tpg_rel"])
    X["ctx_is_home"] = df["ctx_is_home"]
    X["ctx_fixed_dome"] = df["ctx_fixed_dome"]
    X["ctx_rest_days_c"] = df["ctx_rest_days"].clip(4, 14)
    X["is_post"] = df["is_post"]
    cr = df["catch_rate_ew"].clip(0.05, 0.95)
    X["logit_catch_rate_ew"] = np.log(cr / (1 - cr))
    X["log_adot_ew"] = np.log(df["adot_ew"].clip(lower=0.5))
    X["log_opp_def_catch_rel"] = np.log(df["opp_def_catch_rel"])
    X["log_ypr_ew"] = np.log(df["ypr_ew"].clip(lower=1))
    X["log_yac_per_rec_ew"] = np.log(df["yac_per_rec_ew"].clip(lower=0.5))
    X["explosive_rate_ew"] = df["explosive_rate_ew"]
    X["log_opp_def_ypt_rel"] = np.log(df["opp_def_ypt_rel"])
    return X


def _cols(groups_map: dict, groups) -> list[str]:
    cols: list[str] = []
    for g in groups:
        for c in groups_map.get(g, []):
            if c not in cols:
                cols.append(c)
    return cols


def fit_dispersion_curve(mu: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Negative-binomial dispersion that varies with the mean: 1/r(mu) = exp(c0 + c1 log mu).

    A single constant size under-disperses low-volume players and over-disperses
    high-volume ones (checked on held-out seasons; see docs/METHODOLOGY.md), which
    distorts tail probabilities. Per-bin method-of-moments estimates of 1/r are
    regressed on log(mean bin mu), weighted by bin size.
    """
    edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(mu, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        idx = b == i
        if idx.sum() < 50:
            continue
        m_ = mu[idx]
        inv_r = (np.mean((y[idx] - m_) ** 2) - np.mean(m_)) / np.mean(m_ ** 2)
        if inv_r > 1e-4:
            xs.append(np.log(np.mean(m_)))
            ys.append(np.log(inv_r))
            ws.append(idx.sum())
    if len(xs) < 2:
        return (float(np.log(0.1)), 0.0)
    c1, c0 = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    return (float(c0), float(c1))


def nb_size_for(mu: np.ndarray, coef: tuple[float, float]) -> np.ndarray:
    inv_r = np.exp(coef[0] + coef[1] * np.log(np.clip(mu, 0.05, None)))
    return 1.0 / np.clip(inv_r, 0.002, 3.0)


def _glm(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


@dataclass
class StructuredModel:
    groups: tuple = DEFAULT_GROUPS
    target_learner: str = DEFAULT_LEARNER   # "gbm" (default) or "glm"
    version: str = config.MODEL_VERSION
    params_: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame) -> "StructuredModel":
        tr = train[train["played"]].copy()
        X = design(tr)
        self.t_cols = _cols(FEATURE_GROUPS, self.groups)
        self.c_cols = _cols(CATCH_FEATURES, self.groups)
        self.y_cols = _cols(YPR_FEATURES, self.groups)

        # 1. targets
        yT = tr["targets"].to_numpy(float)
        self.t_medians = X[self.t_cols].median().to_dict()
        if self.target_learner == "gbm":
            self.t_model = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.05,
                                                         max_leaf_nodes=15, min_samples_leaf=200,
                                                         random_state=0)
            self.t_model.fit(X[self.t_cols], yT)
        else:
            self.t_model = _glm(PoissonRegressor(alpha=1e-4, max_iter=1000)).fit(X[self.t_cols], yT)
        mu = self.t_model.predict(X[self.t_cols])
        self.disp_coef = fit_dispersion_curve(mu, yT)
        inv_r = np.sum((yT - mu) ** 2 - mu) / np.sum(mu ** 2)
        self.nb_size = float(1.0 / max(inv_r, 1e-3))   # constant-size summary, for reporting

        # 2. catches | targets (expanded to caught / not-caught rows with weights)
        ct = tr[tr["targets"] > 0]
        Xc = design(ct)[self.c_cols]
        rec = ct["receptions"].to_numpy(float)
        tg = ct["targets"].to_numpy(float)
        Xcc = pd.concat([Xc, Xc], ignore_index=True)
        yc = np.r_[np.ones(len(Xc)), np.zeros(len(Xc))]
        wc = np.r_[rec, tg - rec]
        keep = wc > 0
        self.c_model = _glm(LogisticRegression(C=100.0, max_iter=2000))
        self.c_model.fit(Xcc[keep], yc[keep], logisticregression__sample_weight=wc[keep])
        p = self.c_model.predict_proba(Xc)[:, 1]
        m2 = tg >= 2
        num = np.sum((rec - tg * p) ** 2 - tg * p * (1 - p))
        den = np.sum((tg * (tg - 1) * p * (1 - p))[m2])
        self.bb_rho = float(np.clip(num / den, 0.0, 0.5))

        # 3. yards per catch
        yr = tr[tr["receptions"] > 0]
        Xy = design(yr)[self.y_cols]
        k = yr["receptions"].to_numpy(float)
        Y = yr["receiving_yards"].to_numpy(float)
        ypr = np.clip(Y / k, 0.5, None)
        self.y_model = _glm(GammaRegressor(alpha=1e-4, max_iter=1000))
        self.y_model.fit(Xy, ypr, gammaregressor__sample_weight=k)
        m = self.y_model.predict(Xy)
        z = (Y - k * m) ** 2 / m ** 2
        A = np.column_stack([k, k ** 2])
        w = 1.0 / k ** 2
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)
        tau2 = float(max(coef[1], 0.0))
        c1 = float(max(coef[0], 1e-3))
        self.tau2 = tau2
        self.gamma_shape = float(np.clip((1 + tau2) / c1, 0.2, 20.0))
        self.n_train = int(len(tr))
        self.train_seasons = sorted(int(s) for s in tr["season"].unique())
        self.params_ = dict(nb_size_const=self.nb_size, disp_intercept=self.disp_coef[0],
                            disp_slope=self.disp_coef[1], bb_rho=self.bb_rho, gamma_shape=self.gamma_shape,
                            tau2=self.tau2, n_train=self.n_train)
        return self

    # ------------------------------------------------------------ predict
    def predict_params(self, df: pd.DataFrame, target_multiplier: np.ndarray | float = 1.0) -> dict:
        X = design(df)
        mu = self.t_model.predict(X[self.t_cols]) * target_multiplier
        p = self.c_model.predict_proba(X[self.c_cols])[:, 1]
        m = self.y_model.predict(X[self.y_cols])
        return dict(mu_t=mu, p_catch=p, m_ypr=m, nb_size=nb_size_for(mu, self.disp_coef), bb_rho=self.bb_rho,
                    gamma_shape=self.gamma_shape, tau2=self.tau2)

    def simulate(self, df: pd.DataFrame, n_sims: int, seed: int, target_multiplier=1.0,
                 chunk: int = 400, actual: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Simulate rows in chunks; returns summaries (not raw draws) for every row.

        If ``actual`` is given ({"yards"|"receptions"|"targets": observed values}), the
        randomized PIT of each observation is returned too (for interval calibration of
        discrete outcomes).
        """
        params = self.predict_params(df, target_multiplier)
        n = len(df)
        out: dict[str, list] = {}
        pit_rng = np.random.default_rng([seed, 999])
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            sub = {k: (v[sl] if isinstance(v, np.ndarray) and np.ndim(v) == 1 else v) for k, v in params.items()}
            rng = np.random.default_rng([seed, start])
            draws = sim.simulate(sub, n_sims, rng)
            for name, thr in (("yards", config.RECEIVING_YARD_MILESTONES),
                              ("receptions", config.RECEPTION_MILESTONES),
                              ("targets", ())):
                s = sim.summarize(draws[name], thr)
                for key, val in s.items():
                    out.setdefault(f"{name}_{key}", []).append(val)
                if actual is not None and name in actual:
                    yv = np.asarray(actual[name], float)[sl][:, None]
                    below = (draws[name] < yv).mean(axis=1)
                    at = (draws[name] == yv).mean(axis=1)
                    out.setdefault(f"{name}_pit", []).append(below + pit_rng.random(len(below)) * at)
        res = {k: np.concatenate(v) for k, v in out.items()}
        res.update({"mu_t": params["mu_t"], "p_catch": params["p_catch"], "m_ypr": params["m_ypr"]})
        return res

    def simulate_draws(self, df_row: pd.DataFrame, n_sims: int, seed: int, target_multiplier=1.0):
        params = self.predict_params(df_row, target_multiplier)
        return sim.simulate(params, n_sims, np.random.default_rng(seed)), params

    # -------------------------------------------------------- explanation
    def target_contributions(self, df_row: pd.DataFrame) -> pd.DataFrame:
        """How much each input moves this forecast's expected targets (model-agnostic).

        For each feature, the input is replaced by its training median and the expected
        target count recomputed; contribution = log(mu) - log(mu with that input typical).
        Positive = this player's value of the input raises his forecast relative to a typical
        value. These are *predictive associations inside the model*, not causal effects, and
        correlated inputs share credit unevenly.
        """
        X = design(df_row)[self.t_cols]
        base = float(self.t_model.predict(X)[0])
        rows = []
        for c in self.t_cols:
            Xr = X.copy()
            Xr[c] = self.t_medians[c]
            alt = float(self.t_model.predict(Xr)[0])
            rows.append(dict(feature=c, label=FEATURE_LABELS.get(c, c), value=X.iloc[0][c],
                             typical=self.t_medians[c], contribution=np.log(max(base, 1e-6)) - np.log(max(alt, 1e-6))))
        return pd.DataFrame(rows).sort_values("contribution", key=np.abs, ascending=False, ignore_index=True)
