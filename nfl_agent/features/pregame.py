"""Leakage-safe pregame features.

Every feature for a player-game row is computed from that player's (or team's)
*earlier* games only. The recursion below reads the accumulated state BEFORE
adding the current game, which is the "shift by one" that excludes the current
game. Rows without a realised outcome (upcoming games) contribute nothing to
later rows.

Weighting
---------
Exponentially-weighted sums over prior games with a per-game decay
(half-life in games) and an extra multiplicative discount when a season
boundary is crossed (off-season changes make older seasons less relevant).

Shrinkage (partial pooling)
---------------------------
Rates are shrunk toward a position-level prior by adding k pseudo-observations
at the prior value (empirical-Bayes style). Priors for season S are computed
from seasons strictly before S, so they never use future information. With no
history at all, a feature equals its prior; the model also receives the
effective sample size, rookie flag and draft position, so it can learn how
much (little) such players' priors should be trusted.

Definitions and denominators are documented in docs/DATA_DICTIONARY.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureParams:
    player_half_life: float = 10.0      # games (chosen on 2019-2024 validation: 10 vs 6)
    short_half_life: float = 1.5        # games: "recent role"
    team_half_life: float = 8.0
    season_discount: float = 0.6        # applied to the weight state at each season boundary
    k_target_share: float = 3.0         # pseudo-games
    k_tpg: float = 3.0                  # pseudo-games
    k_catch: float = 25.0               # pseudo-targets
    k_ypr: float = 15.0                 # pseudo-receptions
    k_adot: float = 20.0                # pseudo-targets
    k_explosive: float = 20.0           # pseudo-receptions
    k_snap: float = 2.0                 # pseudo-games
    k_def: float = 4.0                  # pseudo-games for defensive rates


# Per-game quantities accumulated for players: (name, value column, weight-mask column or None)
PLAYER_SUMS = [
    ("g", None),
    ("tg", "targets"),
    ("rec", "receptions"),
    ("yds", "receiving_yards"),
    ("air", "receiving_air_yards"),
    ("yac", "receiving_yards_after_catch"),
    ("r20", "receiving_20"),
    ("team_tg", "team_targets"),
    ("snap", "snap_share"),
    ("g_snap", None),
]


def _ew_state(values: np.ndarray, valid: np.ndarray, season: np.ndarray, decay: float,
              season_discount: float, mult: np.ndarray | None = None) -> np.ndarray:
    """Return pregame EW sums for each row (state before the row is added).

    values: (n, k) per-row quantities; valid: (n,) whether row has an outcome;
    mult: optional per-row weight multipliers (bootstrap).
    """
    n, k = values.shape
    out = np.zeros((n, k))
    state = np.zeros(k)
    cur_season = None
    for i in range(n):
        if cur_season is not None and season[i] != cur_season:
            state = state * season_discount      # applied once per season boundary
        cur_season = season[i]
        out[i] = state                           # pregame: before adding row i
        if valid[i]:
            w = 1.0 if mult is None else mult[i]
            state = state * decay + w * values[i]
    return out


def _decay(half_life: float) -> float:
    return 0.5 ** (1.0 / half_life)


def _first_finite(*vals):
    for v in vals:
        if v is not None and np.isfinite(v):
            return float(v)
    return np.nan


def position_priors(pg: pd.DataFrame) -> pd.DataFrame:
    """Per (season, position) priors computed from seasons strictly before."""
    played = pg[pg["played"]]
    agg = played.groupby(["season", "position"]).agg(
        tg=("targets", "sum"), rec=("receptions", "sum"), yds=("receiving_yards", "sum"),
        air=("receiving_air_yards", "sum"), yac=("receiving_yards_after_catch", "sum"),
        r20=("receiving_20", "sum"), team_tg=("team_targets", "sum"), g=("targets", "size"),
        snap=("snap_share", "mean"),
    ).reset_index()
    rows = []
    seasons = sorted(pg["season"].dropna().unique())
    for pos in agg["position"].unique():
        a = agg[agg["position"] == pos].set_index("season")
        for s in seasons:
            prior = a[a.index < s]
            if prior.empty:     # first season in the data: warm-up only, use own season
                prior = a[a.index == s]
            t = prior.sum()
            rows.append(dict(season=s, position=pos,
                             p_ts=t.tg / t.team_tg, p_tpg=t.tg / t.g, p_catch=t.rec / t.tg,
                             p_ypr=t.yds / t.rec, p_adot=t.air / t.tg, p_yac=t.yac / t.rec,
                             p_r20=t.r20 / t.rec,
                             # 2012 has no snap data: fall back to all available seasons, then 0.5
                             p_snap=_first_finite(prior["snap"].mean(), a["snap"].mean(), 0.5)))
    return pd.DataFrame(rows)


def league_def_priors(tg: pd.DataFrame) -> pd.DataFrame:
    done = tg[tg["has_stats"]]
    agg = done.groupby("season").agg(t=("def_wrte_targets_allowed", "sum"),
                                     r=("def_wrte_receptions_allowed", "sum"),
                                     y=("def_wrte_yards_allowed", "sum"),
                                     n=("def_wrte_targets_allowed", "size"),
                                     tt=("team_targets", "sum"),
                                     db=("team_dropbacks", "sum"))
    rows = []
    for s in sorted(tg["season"].unique()):
        prior = agg[agg.index < s]
        if prior.empty:
            prior = agg[agg.index == s]
        t = prior.sum()
        rows.append(dict(season=s, l_def_tpg=t.t / t.n, l_def_ypt=t.y / t.t, l_def_catch=t.r / t.t,
                         l_team_tpg=t.tt / t.n, l_team_dbpg=t.db / t.n))
    return pd.DataFrame(rows)


def team_features(team_games: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """Pregame team-offence volume and opponent-defence features, per (game_id, team)."""
    tg = team_games.sort_values(["team", "kickoff_utc"]).reset_index(drop=True)
    d = _decay(params.team_half_life)
    off_cols = ["team_targets", "team_dropbacks", "team_offense_plays"]
    def_cols = ["def_wrte_targets_allowed", "def_wrte_receptions_allowed", "def_wrte_yards_allowed"]
    out_parts = []
    for team, grp in tg.groupby("team", sort=False):
        valid = grp["has_stats"].to_numpy()
        vals = grp[off_cols + def_cols].fillna(0).to_numpy(dtype=float)
        vals = np.column_stack([np.ones(len(grp)), vals, grp["team_offense_plays"].notna().to_numpy(float)])
        st = _ew_state(vals, valid, grp["season"].to_numpy(), d, params.season_discount)
        f = pd.DataFrame(st, columns=["w"] + off_cols + def_cols + ["w_plays"], index=grp.index)
        f["game_id"] = grp["game_id"].to_numpy()
        f["team"] = team
        f["season"] = grp["season"].to_numpy()
        out_parts.append(f)
    st = pd.concat(out_parts)
    lp = league_def_priors(tg)
    st = st.merge(lp, on="season", how="left")
    k = params.k_def
    kr = 2.0 * params.k_def        # per-target defensive rates are noisier: more pseudo-games
    res = pd.DataFrame({"game_id": st["game_id"], "team": st["team"]})
    res["team_tpg_ew"] = (st["team_targets"] + k * st["l_team_tpg"]) / (st["w"] + k)
    res["team_dbpg_ew"] = (st["team_dropbacks"] + k * st["l_team_dbpg"]) / (st["w"] + k)
    res["team_plays_ew"] = np.where(st["w_plays"] > 0, st["team_offense_plays"] / st["w_plays"].clip(lower=1e-9), np.nan)
    res["def_tpg_allowed_rel"] = ((st["def_wrte_targets_allowed"] + k * st["l_def_tpg"]) / (st["w"] + k)) / st["l_def_tpg"]
    res["def_ypt_allowed_rel"] = ((st["def_wrte_yards_allowed"] + kr * st["l_def_tpg"] * st["l_def_ypt"]) /
                                  (st["def_wrte_targets_allowed"] + kr * st["l_def_tpg"])) / st["l_def_ypt"]
    res["def_catch_allowed_rel"] = ((st["def_wrte_receptions_allowed"] + kr * st["l_def_tpg"] * st["l_def_catch"]) /
                                    (st["def_wrte_targets_allowed"] + kr * st["l_def_tpg"])) / st["l_def_catch"]
    res["team_games_ew"] = st["w"]
    return res


def player_features(pg: pd.DataFrame, params: FeatureParams, priors: pd.DataFrame,
                    mult: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Pregame player features for every row of ``pg`` (rows sorted by kickoff per player).

    ``pg`` must have a boolean ``played`` column: True when the outcome is
    realised and the row may serve as history for later rows.
    """
    pg = pg.sort_values(["player_id", "kickoff_utc"]).reset_index(drop=True)
    dl, ds = _decay(params.player_half_life), _decay(params.short_half_life)
    cols = ["targets", "receptions", "receiving_yards", "receiving_air_yards",
            "receiving_yards_after_catch", "receiving_20", "team_targets"]
    parts = []
    for pid, grp in pg.groupby("player_id", sort=False):
        valid = grp["played"].to_numpy()
        snap = grp["snap_share"].to_numpy(dtype=float)
        has_snap = ~np.isnan(snap)
        vals = np.column_stack([np.ones(len(grp)), grp[cols].fillna(0).to_numpy(float),
                                np.where(has_snap, snap, 0.0), has_snap.astype(float)])
        # team targets only count where known, otherwise the share denominator is biased
        tt_known = grp["team_targets"].notna().to_numpy(float)
        vals = np.column_stack([vals, grp["targets"].fillna(0).to_numpy(float) * tt_known])
        seasons = grp["season"].to_numpy()
        m = None if mult is None else mult.get(pid)
        long = _ew_state(vals, valid, seasons, dl, params.season_discount, m)
        short = _ew_state(vals[:, [0, 1, 7, 8, 9, 10]], valid, seasons, ds, params.season_discount, m)
        f = pd.DataFrame(long, columns=["g", "tg", "rec", "yds", "air", "yac", "r20", "team_tg",
                                        "snap", "g_snap", "tg_tt"])
        sf = pd.DataFrame(short, columns=["s_g", "s_tg", "s_team_tg", "s_snap", "s_g_snap", "s_tg_tt"])
        f = pd.concat([f, sf], axis=1)
        f.index = grp.index
        # career and team-tenure counts (unweighted, prior games only)
        played_cum = np.cumsum(valid) - valid
        f["n_games_prior"] = played_cum
        teams = grp["team"].to_numpy()
        kick = grp["kickoff_utc"].to_numpy()
        last_team, last_kick, tenure = None, None, 0
        tenure_arr, changed_arr, days_arr = np.zeros(len(grp)), np.zeros(len(grp)), np.full(len(grp), np.nan)
        for i in range(len(grp)):
            changed_arr[i] = float(last_team is not None and teams[i] != last_team)
            tenure_arr[i] = 0 if changed_arr[i] else tenure
            if last_kick is not None:
                days_arr[i] = (kick[i] - last_kick) / np.timedelta64(1, "D")
            if valid[i]:
                tenure = (0 if changed_arr[i] else tenure) + 1
                last_team, last_kick = teams[i], kick[i]
        f["games_with_team_prior"] = tenure_arr
        f["team_changed"] = changed_arr
        f["days_since_last_game"] = days_arr
        parts.append(f)
    st = pd.concat(parts).sort_index()
    base = pg[["player_id", "game_id", "season", "position"]].join(st)
    base = base.merge(priors, on=["season", "position"], how="left")
    p = params
    out = pd.DataFrame({"player_id": base["player_id"], "game_id": base["game_id"]})
    out["eff_games"] = base["g"]
    out["n_games_prior"] = base["n_games_prior"]
    out["tpg_ew"] = (base["tg"] + p.k_tpg * base["p_tpg"]) / (base["g"] + p.k_tpg)
    avg_tt = np.where(base["g"] > 0, base["team_tg"] / base["g"].clip(lower=1e-9), 33.0)
    avg_tt = np.where(avg_tt > 0, avg_tt, 33.0)
    out["tshare_ew"] = (base["tg_tt"] + p.k_target_share * base["p_ts"] * avg_tt) / (base["team_tg"] + p.k_target_share * avg_tt)
    s_avg_tt = np.where(base["s_g"] > 0, base["s_team_tg"] / base["s_g"].clip(lower=1e-9), avg_tt)
    s_avg_tt = np.where(s_avg_tt > 0, s_avg_tt, 33.0)
    out["tshare_recent"] = (base["s_tg_tt"] + 1.0 * out["tshare_ew"] * s_avg_tt) / (base["s_team_tg"] + 1.0 * s_avg_tt)
    out["catch_rate_ew"] = (base["rec"] + p.k_catch * base["p_catch"]) / (base["tg"] + p.k_catch)
    out["ypr_ew"] = (base["yds"] + p.k_ypr * base["p_ypr"]) / (base["rec"] + p.k_ypr)
    out["adot_ew"] = (base["air"] + p.k_adot * base["p_adot"]) / (base["tg"] + p.k_adot)
    out["yac_per_rec_ew"] = (base["yac"] + p.k_ypr * base["p_yac"]) / (base["rec"] + p.k_ypr)
    out["explosive_rate_ew"] = (base["r20"] + p.k_explosive * base["p_r20"]) / (base["rec"] + p.k_explosive)
    out["ypt_ew"] = out["catch_rate_ew"] * out["ypr_ew"]
    out["snap_share_ew"] = (base["snap"] + p.k_snap * base["p_snap"]) / (base["g_snap"] + p.k_snap)
    out["snap_share_recent"] = (base["s_snap"] + 1.0 * out["snap_share_ew"]) / (base["s_g_snap"] + 1.0)
    out["snap_history_games"] = base["g_snap"]
    out["games_with_team_prior"] = base["games_with_team_prior"]
    out["team_changed"] = base["team_changed"]
    out["days_since_last_game"] = base["days_since_last_game"]
    return out


def add_played_flag(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.copy()
    pg["played"] = pg["targets"].notna()
    return pg


def build_feature_table(player_games: pd.DataFrame, team_games: pd.DataFrame,
                        params: FeatureParams = FeatureParams(),
                        query_rows: pd.DataFrame | None = None,
                        mult: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Return player_games (+ optional query rows for upcoming games) joined to pregame features.

    query_rows: rows for games not yet played, columns player_id, player_name,
    position, game_id, season, week, season_type, kickoff_utc, team, opponent
    (outcome columns absent/NaN). They never contribute history.
    """
    pg = add_played_flag(player_games)
    if query_rows is not None and len(query_rows):
        q = query_rows.copy()
        q["played"] = False
        q["is_query"] = True
        pg["is_query"] = False
        # never forecast a game twice: drop realised rows for the same player-game
        pg = pg[~pg.set_index(["player_id", "game_id"]).index.isin(q.set_index(["player_id", "game_id"]).index)]
        pg = pd.concat([pg, q], ignore_index=True)
    else:
        pg["is_query"] = False
    priors = position_priors(pg)
    pf = player_features(pg, params, priors, mult)
    tf = team_features(team_games, params)
    df = pg.merge(pf, on=["player_id", "game_id"], how="left")
    df = df.merge(tf, on=["game_id", "team"], how="left")
    opp = tf[["game_id", "team", "def_tpg_allowed_rel", "def_ypt_allowed_rel", "def_catch_allowed_rel"]].rename(
        columns={"team": "opponent", "def_tpg_allowed_rel": "opp_def_tpg_rel",
                 "def_ypt_allowed_rel": "opp_def_ypt_rel", "def_catch_allowed_rel": "opp_def_catch_rel"})
    df = df.merge(opp, on=["game_id", "opponent"], how="left")
    ctx = team_games[["game_id", "team", "is_home", "rest_days", "fixed_dome"]].rename(
        columns={"is_home": "ctx_is_home", "rest_days": "ctx_rest_days", "fixed_dome": "ctx_fixed_dome"})
    df = df.merge(ctx, on=["game_id", "team"], how="left")
    df["is_te"] = (df["position"] == "TE").astype(float)
    df["is_post"] = (df["season_type"] == "POST").astype(float)
    df["log_draft_pick"] = np.log(df.get("draft_pick_filled", pd.Series(260.0, index=df.index)).fillna(260.0))
    df["rookie_flag"] = df.get("is_rookie", pd.Series(np.nan, index=df.index))
    return df.sort_values(["kickoff_utc", "player_id"], ignore_index=True)
