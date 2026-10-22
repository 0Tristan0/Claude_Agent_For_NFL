"""v0 projections: opportunity x efficiency x a market-anchored game environment.

A deliberately simple first model so the system is usable now; Phases 2-3 replace pieces of it.

Environment   team points from the market (spread/total); plays and pass rate from fitted regressions on
              team history plus the spread (game script); sack, scramble and interception rates shrunk to
              league averages.
Opportunity   target, carry, red-zone target and inside-10 carry shares: recency-weighted (half-life 4
              games, last season at half weight) and shrunk toward a prior built from the player's snap
              share (position share-per-snap), then normalized per team with an "other" bucket.
Efficiency    catch rate, yards per catch, yards per carry: shrunk hard toward position averages.

Fits use only seasons before the game's season and never later than 2024: 2025 stays a holdout until
the final test. Every fitted number is stored in Fit so the report can show where it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..features import bundle
from ..features.asof import AsofIndex, visible

FIT_LAST_SEASON = 2024
FIT_YEARS = 3
POSITIONS = ("QB", "RB", "WR", "TE")

# Pseudo-counts for shrinkage (opportunity light, efficiency heavy)
K_TARGET_SHARE = 20.0      # pseudo team targets (~2/3 of a game): opportunity is the stable signal
K_CARRY_SHARE = 40.0       # pseudo team designed carries (~1.5 games)
K_RZ = 10.0
K_I10 = 8.0
K_CATCH = 40.0             # pseudo targets
K_YPC_REC = 30.0           # pseudo catches
K_YPC_RUSH = 60.0          # pseudo carries
K_SCRAMBLE = 30.0          # pseudo non-sack dropbacks (rates read with long memory, see qb_long)
NO_HISTORY_SNAP = 0.15     # assumed snap share for a player with no games yet (depth role)
GADGET_DB_PG = 5.0         # a non-starting QB with fewer dropbacks per game than this keeps his designed carries
K_INT = 400.0              # pseudo attempts
K_SACK = 150.0             # pseudo dropbacks
OTHER_FLOOR = 0.015        # share of targets/carries left to players not listed (trick plays, call-ups)


def fit_seasons(season: int) -> list[int]:
    last = min(season - 1, FIT_LAST_SEASON)
    return list(range(max(config.FIRST_SEASON, last - FIT_YEARS + 1), last + 1))


@dataclass
class Fit:
    season: int
    seasons: list
    pos: pd.DataFrame                        # per-position priors
    league: dict                             # league rates
    plays_coef: np.ndarray = None
    plays_sd: float = 0.0
    pr_coef: np.ndarray = None
    pr_sd: float = 0.0
    script_beta: float = 0.0                 # change in pass rate per point of (actual - expected) margin
    points_sd: float = 10.0
    points_corr: float = 0.0
    carry_emp: dict = field(default_factory=dict)     # position -> real designed-carry yards (bootstrap source)
    notes: dict = field(default_factory=dict)


def _pos_priors(pg: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos, g in pg.groupby("position"):
        snap = g["offense_snaps"].sum() / g["team_offense_snaps"].sum()
        rows.append(dict(
            position=pos,
            tgt_per_snap=(g["targets"].sum() / g["team_targets"].sum()) / snap,
            car_per_snap=(g["designed_carries"].sum() / g["team_designed_carries"].sum()) / snap,
            rz_ratio=(g["rz_targets"].sum() / g["team_rz_targets"].sum()) / max(g["targets"].sum() / g["team_targets"].sum(), 1e-9),
            i10_ratio=(g["i10_carries"].sum() / g["team_i10_carries"].sum()) /
            max(g["designed_carries"].sum() / g["team_designed_carries"].sum(), 1e-9),
            catch_rate=g["receptions"].sum() / max(g["targets"].sum(), 1),
            yds_per_catch=g["rec_yds"].sum() / max(g["receptions"].sum(), 1),
            yds_per_carry=g["designed_rush_yds"].sum() / max(g["designed_carries"].sum(), 1),
        ))
    pri = pd.DataFrame(rows).set_index("position")
    # target/carry share as a function of snap share (convex: starters get more per snap than depth players)
    edges = np.array([0, .2, .4, .6, .8, 1.01])
    for pos_, g in pg.groupby("position"):
        b = np.digitize(g["offense_snaps"] / g["team_offense_snaps"], edges[1:-1])
        ts = (g["targets"] / g["team_targets"]).groupby(b).mean()
        cs = (g["designed_carries"] / g["team_designed_carries"].replace(0, np.nan)).groupby(b).mean()
        sn = (g["offense_snaps"] / g["team_offense_snaps"]).groupby(b).mean()
        pri.loc[pos_, "curve_snap"] = ",".join(f"{v:.4f}" for v in sn.reindex(range(5)).interpolate().bfill().ffill())
        pri.loc[pos_, "curve_tgt"] = ",".join(f"{v:.4f}" for v in ts.reindex(range(5)).interpolate().bfill().ffill())
        pri.loc[pos_, "curve_car"] = ",".join(f"{v:.4f}" for v in cs.reindex(range(5)).fillna(0))
    # per-event spreads from play-by-play (gamma shape for catch yards, sd for carry yards)
    pos = pg.set_index(["game_id", "player_id"])["position"]
    catches = plays[plays["target"] & (plays["complete_pass"] == 1)]
    cpos = pos.reindex(list(zip(catches["game_id"], catches["receiver_player_id"]))).to_numpy()
    runs = plays[plays["designed_run"] & plays["rusher_player_id"].notna()]
    rpos = pos.reindex(list(zip(runs["game_id"], runs["rusher_player_id"]))).to_numpy()
    for p in pri.index:
        y = catches.loc[cpos == p, "receiving_yards"].astype(float)
        pri.loc[p, "catch_yds_shape"] = float(np.clip(y.mean() ** 2 / y.var(), 0.5, 3.0)) if len(y) > 50 else 1.0
        r = runs.loc[rpos == p, "rushing_yards"].astype(float)
        pri.loc[p, "carry_yds_sd"] = float(r.std()) if len(r) > 50 else 6.0
    return pri


def _team_rows(tables, seasons) -> pd.DataFrame:
    """Historical team-games with their own pregame features (as of each kickoff) for the volume fits."""
    tg = tables["team_games"]
    tg = tg[tg["season"].isin(seasons) & (tg["season_type"] == "REG")].copy()
    cols = ["off_plays", "off_dropbacks", "off_designed_runs", "def_plays", "def_dropbacks", "def_designed_runs"]
    idx = AsofIndex(tables["team_games"], "team", cols, half_life=8.0, season_discount=0.5)
    tv = tables["team_view"].set_index(["game_id", "team"])
    feats = []
    for _, r in tg.iterrows():
        a = idx.state(r["team"], r["kickoff_utc"], r["season"])
        b = idx.state(r["opponent"], r["kickoff_utc"], r["season"])
        v = tv.loc[(r["game_id"], r["team"])]
        feats.append(dict(
            plays_pg=a[0] / a[-1] if a[-1] > 0 else np.nan,
            pass_rate=a[1] / (a[1] + a[2]) if a[1] + a[2] > 0 else np.nan,
            opp_def_plays_pg=b[3] / b[-1] if b[-1] > 0 else np.nan,
            opp_def_pass_rate=b[4] / (b[4] + b[5]) if b[4] + b[5] > 0 else np.nan,
            total_line=v["total_line"], team_spread=v["team_spread"], implied=v["implied_points"]))
    f = pd.DataFrame(feats, index=tg.index)
    out = pd.concat([tg, f], axis=1)
    out["pass_rate_game"] = out["off_dropbacks"] / (out["off_dropbacks"] + out["off_designed_runs"])
    return out.dropna(subset=["plays_pg", "pass_rate", "opp_def_plays_pg", "opp_def_pass_rate", "total_line"])


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid.std(ddof=X.shape[1])), resid


def fit(tables: dict, season: int) -> Fit:
    seasons = fit_seasons(season)
    pg = tables["player_games"]
    pg = pg[pg["season"].isin(seasons) & (pg["season_type"] == "REG") & (pg["offense_snaps"] > 0)]
    plays = tables["plays"]
    plays = plays[plays["season"].isin(seasons) & (plays["season_type"] == "REG")]
    pos = _pos_priors(pg, plays)

    tg = tables["team_games"]
    t = tg[tg["season"].isin(seasons) & (tg["season_type"] == "REG")]
    sc = plays[plays["scramble"]]
    league = dict(
        sack_rate=t["off_sacks"].sum() / t["off_dropbacks"].sum(),
        scramble_rate=t["off_scrambles"].sum() / (t["off_dropbacks"].sum() - t["off_sacks"].sum()),
        int_rate=plays.loc[plays["pass_att"], "interception"].fillna(0).mean(),
        targets_per_att=t["off_targets"].sum() / t["off_pass_att"].sum(),
        td_drive_rate=t["off_td_drives"].sum() / t["off_drives"].sum(),
        drives_pg=float(t["off_drives"].mean()),
        pass_td_share=float(pg["rec_tds"].sum() / (pg["rec_tds"].sum() + pg["rush_tds"].sum())),
        scramble_yds_mean=float(sc["rushing_yards"].mean()), scramble_yds_sd=float(sc["rushing_yards"].std()),
        # per non-sack dropback, matching how the sim applies it
        scramble_rate_nonsack=t["off_scrambles"].sum() / (t["off_dropbacks"].sum() - t["off_sacks"].sum()),
        other_catch_rate=0.62, other_yds_per_catch=9.5, other_yds_per_carry=4.0,
    )
    # share of points scored on offensive TDs (rest: FGs, defense/special-teams scores, safeties)
    league["td_point_share"] = float((t["off_td_drives"] * 6.95).sum() / t["off_points"].sum())
    league["fg_rate_non_td"] = float(((t["off_points"] - t["off_td_drives"] * 6.95).clip(lower=0) / 3).sum() /
                                     (t["off_drives"] - t["off_td_drives"]).sum())

    rows = _team_rows(tables, seasons)
    X = np.column_stack([np.ones(len(rows)), rows["plays_pg"], rows["opp_def_plays_pg"], rows["total_line"]])
    plays_coef, plays_sd, _ = _ols(X, rows["off_plays"].to_numpy(float))
    Xp = np.column_stack([np.ones(len(rows)), rows["pass_rate"], rows["opp_def_pass_rate"], rows["team_spread"]])
    pr_coef, pr_sd, pr_res = _ols(Xp, rows["pass_rate_game"].to_numpy(float))
    margin_dev = (rows["off_points"] - rows["def_points"]) - rows["team_spread"]
    script_beta = float(np.polyfit(margin_dev.to_numpy(float), pr_res, 1)[0])
    pr_sd_after = float(np.std(pr_res - script_beta * margin_dev.to_numpy(float)))
    res = rows["off_points"] - rows["implied"]
    paired = rows.assign(res=res).pivot_table(index="game_id", columns=rows.groupby("game_id").cumcount(), values="res")
    points_corr = float(paired.dropna().corr().iloc[0, 1]) if paired.shape[1] == 2 else 0.0
    pid_pos = pg.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])["position"]
    runs = plays[plays["designed_run"] & plays["rusher_player_id"].notna()]
    rpos = pid_pos.reindex(list(zip(runs["game_id"], runs["rusher_player_id"]))).to_numpy()
    carry_emp = {p: runs.loc[rpos == p, "rushing_yards"].fillna(0).to_numpy(float) for p in ("QB", "RB", "WR", "TE")}
    carry_emp = {p: v for p, v in carry_emp.items() if len(v) >= 100}
    return Fit(season=season, seasons=seasons, pos=pos, league=league, plays_coef=plays_coef, plays_sd=plays_sd, carry_emp=carry_emp,
               pr_coef=pr_coef, pr_sd=pr_sd_after, script_beta=script_beta, points_sd=float(res.std()),
               points_corr=points_corr, notes=dict(n_team_games=len(rows)))


# ---- one game -----------------------------------------------------------------------------------

def _curve(row, key, snap):
    """Interpolate a position's share-vs-snap-share curve (stored as comma-separated bin means)."""
    if not isinstance(row.get(key), str):
        return 0.0
    xs = np.r_[0.0, [float(v) for v in row["curve_snap"].split(",")]]
    ys = np.r_[0.0, [float(v) for v in row[key].split(",")]]
    return float(np.interp(snap, xs, ys))


QB_LONG_COLS = ["scrambles", "dropbacks", "sacks_taken", "scramble_yds", "ints", "pass_att"]


def _qb_long(eng, tables):
    """Long-memory as-of sums for QB rates (no game decay, 0.8 per season back), built once per engine."""
    if getattr(eng, "qb_long", None) is None:
        pg = tables["player_games"]
        eng.qb_long = AsofIndex(pg[pg["position"] == "QB"], "player_id", QB_LONG_COLS, half_life=None,
                                season_discount=0.8)
    return eng.qb_long


def _pick_starter(tables, team, asof, season, P, is_qb, weighted_db) -> int:
    """Starting QB: an explicit override if given, else the active QB with the most dropbacks in the
    team's most recent game (ties and week 1: most recency-weighted dropbacks)."""
    if not is_qb.any():
        return -1
    if "force_starter" in P.columns and P["force_starter"].fillna(False).astype(bool).any():
        return int(np.flatnonzero(P["force_starter"].fillna(False).astype(bool).to_numpy())[0])
    pg = visible(tables["player_games"], asof)
    pg = pg[(pg["team"] == team) & (pg["position"] == "QB")]
    last = pg[pg["season"] == season]
    score = np.where(is_qb, weighted_db / 1000.0, -1.0)
    if len(last):
        g = last[last["kickoff_utc"] == last["kickoff_utc"].max()].set_index("player_id")["dropbacks"]
        recent = P["player_id"].map(g).fillna(0).to_numpy(float)
        score = np.where(is_qb, recent + weighted_db / 1000.0, -1.0)
    return int(np.argmax(score))


def _shrink(num, den, k, prior):
    return (num + k * prior) / (den + k)


def active_players(tables: dict, team: str, asof: pd.Timestamp, season: int, out_ids=()) -> pd.DataFrame:
    """Players who took a snap for `team` in one of its last 2 games before asof (this season; last season
    in week 1), minus anyone listed as out. Returns player_id, player_name, position."""
    pg = visible(tables["player_games"], asof)
    pg = pg[(pg["team"] == team) & (pg["any_snaps"] > 0)]
    cur = pg[pg["season"] == season]
    use = cur if len(cur) else pg[pg["season"] == season - 1]
    recent = sorted(use["game_id"].unique(), key=lambda g: use.loc[use["game_id"] == g, "kickoff_utc"].iloc[0])[-2:]
    r = use[use["game_id"].isin(recent) & use["position"].isin(POSITIONS)]
    r = r.sort_values("kickoff_utc").groupby("player_id").tail(1)
    r = r[~r["player_id"].isin(set(out_ids))]
    return r[["player_id", "player_name", "position"]].reset_index(drop=True)


@dataclass
class TeamProj:
    team: str
    opponent: str
    is_home: float
    implied_points: float
    team_spread: float
    plays: float
    pass_rate: float
    sack_rate: float
    players: pd.DataFrame     # one row per player with shares, efficiency and effective sample sizes


def project_game(tables: dict, game_id: str, fit_: Fit, asof: pd.Timestamp | None = None, out_ids=(),
                 eng=None, active_override: dict | None = None) -> dict:
    f = bundle.game_features(tables, game_id, asof, eng)
    g = f["game"].iloc[0]
    asof = f["asof"]["asof"].iloc[0]
    season = int(g["season"])
    eng = eng or bundle.engines(tables)
    pri, lg = fit_.pos, fit_.league
    tview = f["team_view"].set_index("team")
    tfeat = f["team"].set_index("team")
    teams_out = {}
    for team in (g["home_team"], g["away_team"]):
        opp = g["away_team"] if team == g["home_team"] else g["home_team"]
        v, tf, of = tview.loc[team], tfeat.loc[team], tfeat.loc[opp]
        # volume from the fitted regressions (team and opponent tendencies + market)
        ts, os_ = eng.team.idx.state(team, asof, season), eng.team.idx.state(opp, asof, season)
        cols = eng.team.idx.values
        tsum = dict(zip(cols + ["eff_games"], ts))
        osum = dict(zip(cols + ["eff_games"], os_))
        plays_pg = tsum["off_plays"] / tsum["eff_games"] if tsum["eff_games"] else np.nan
        opp_plays = osum["def_plays"] / osum["eff_games"] if osum["eff_games"] else np.nan
        pr = tsum["off_dropbacks"] / max(tsum["off_dropbacks"] + tsum["off_designed_runs"], 1)
        opp_pr = osum["def_dropbacks"] / max(osum["def_dropbacks"] + osum["def_designed_runs"], 1)
        plays = float(fit_.plays_coef @ [1, plays_pg, opp_plays, v["total_line"]])
        pass_rate = float(np.clip(fit_.pr_coef @ [1, pr, opp_pr, v["team_spread"]], 0.35, 0.80))
        sack_rate = _shrink(tsum["off_sacks"], tsum["off_dropbacks"], K_SACK, lg["sack_rate"])

        act = active_override[team] if active_override and team in active_override else \
            active_players(tables, team, asof, season, out_ids)
        pf = eng.player.features(act["player_id"].tolist(), asof, season)
        P = act.merge(pf, on="player_id", how="left")
        pp = pri.reindex(P["position"])
        snap = P["snap_share"].fillna(NO_HISTORY_SNAP).clip(0, 1).to_numpy()
        tgt_prior = np.array([_curve(row, "curve_tgt", sv) for (_, row), sv in zip(pp.iterrows(), snap)]).clip(0, 0.6)
        car_prior = np.array([_curve(row, "curve_car", sv) for (_, row), sv in zip(pp.iterrows(), snap)]).clip(0, 0.9)
        S = lambda c: P[f"sum_{c}"].fillna(0).to_numpy(float)
        tshare = _shrink(S("targets"), S("team_targets"), K_TARGET_SHARE, tgt_prior)
        cshare = _shrink(S("designed_carries"), S("team_designed_carries"), K_CARRY_SHARE, car_prior)
        tshare[P["position"].to_numpy() == "QB"] = 0.0
        # normalize per team so the active players share all the volume except OTHER_FLOOR. Shares are
        # measured over the games each player played, so when a teammate is out his targets are
        # redistributed to the actives in proportion to their shares (vacated volume never disappears).
        for arr in (tshare, cshare):
            tot = arr.sum()
            if tot > 0:
                arr *= (1 - OTHER_FLOOR) / tot
        rz = _shrink(S("rz_targets"), S("team_rz_targets"), K_RZ, tshare * pp["rz_ratio"].to_numpy())
        i10 = _shrink(S("i10_carries"), S("team_i10_carries"), K_I10, cshare * pp["i10_ratio"].to_numpy())
        cr = _shrink(S("receptions"), S("targets"), K_CATCH, pp["catch_rate"].to_numpy())
        ypc = _shrink(S("rec_yds"), S("receptions"), K_YPC_REC, pp["yds_per_catch"].to_numpy())
        ypr = _shrink(S("designed_rush_yds"), S("designed_carries"), K_YPC_RUSH, pp["yds_per_carry"].to_numpy())
        is_qb = P["position"].to_numpy() == "QB"
        starter = _pick_starter(tables, team, asof, season, P, is_qb, S("dropbacks"))
        P = P.assign(
            target_share=tshare, carry_share=cshare, rz_share=rz.clip(0, 0.9), i10_share=i10.clip(0, 0.95),
            catch_rate=cr.clip(0.2, 0.95), yds_per_catch=ypc, yds_per_carry=ypr,
            n_targets=S("targets"), n_receptions=S("receptions"), n_carries=S("designed_carries"),
            n_team_targets=S("team_targets"), n_team_carries=S("team_designed_carries"),
            catch_yds_shape=pp["catch_yds_shape"].to_numpy(), carry_yds_sd=pp["carry_yds_sd"].to_numpy(),
            starter_qb=False, scramble_rate=np.nan, yds_per_scramble=np.nan, int_rate=np.nan)
        if starter >= 0:
            P.loc[starter, "starter_qb"] = True
            q = _qb_long(eng, tables).state(P.loc[starter, "player_id"], asof, season)
            sc, db, sk, sy, it, pa = q[:6]
            P.loc[starter, "scramble_rate"] = _shrink(sc, db - sk, K_SCRAMBLE, lg["scramble_rate_nonsack"])
            P.loc[starter, "yds_per_scramble"] = _shrink(sy, sc, 15.0, lg["scramble_yds_mean"])
            P.loc[starter, "int_rate"] = _shrink(it, pa, K_INT, lg["int_rate"])
            # backup QBs keep no designed carries unless they have a gadget role (few dropbacks per game)
            eff = np.maximum(P["eff_games"].fillna(0).to_numpy(float), 1e-9)
            gadget = S("dropbacks") / eff < GADGET_DB_PG
            backups = is_qb & (np.arange(len(P)) != starter) & ~gadget
            P.loc[backups, "carry_share"] = 0.0
        teams_out[team] = TeamProj(team=team, opponent=opp, is_home=float(v["is_home"]),
                                   implied_points=float(v["implied_points"]), team_spread=float(v["team_spread"]),
                                   plays=plays, pass_rate=pass_rate, sack_rate=float(sack_rate), players=P)
    return {"game": g, "asof": asof, "teams": teams_out, "features": f, "fit": fit_}
