"""v0 Monte Carlo game simulator (team-level, correlated). Seeded and vectorized over sims.

Per sim, for each team:
  1. Scoring: a shared game shock and a team shock move each team's TD rate per drive; TDs and field
     goals are drawn per drive, so points average the market-implied total and spread.
  2. Volume: plays around the projection (pace shock shared by both teams); pass rate shifts with how the
     game went versus the spread (teams that fall behind throw more); dropbacks, sacks, scrambles,
     pass attempts and targets follow.
  3. Allocation: each player's target and carry share is drawn from its posterior (parameter
     uncertainty), then targets and carries are dealt out multinomially, with an "other" bucket.
  4. Outcomes: catches ~ Binomial(targets, catch rate), yards from right-skewed gamma draws per catch or
     carry, scaled by the team shock (so yards, catches and points move together).
  5. TDs: pass TDs go to receivers who caught a ball in that sim (weighted by red-zone and overall target
     share), rush TDs to ball carriers (inside-10 and overall carry share). QB pass yards are the sum of
     receiver yards, so stacks are internally consistent. Winning QBs take two kneels (-1 yd each).

Every prop and every parlay is read from these same draws, so joint probabilities come from joint outcomes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

TEAM_SHOCK_ON_CATCH = 0.12      # logit shift of catch rate per unit team shock
TEAM_SHOCK_ON_YARDS = 0.07      # log shift of yards per catch/carry per unit team shock
GAME_SHOCK_WEIGHT = 0.35        # correlation of the two teams' scoring shocks
KNEELS_IF_WIN = 2
CARRY_SHIFT = 3.0               # carry yards = Gamma - 3 so losses are possible
OTHER_FLOOR = 0.015             # same as project.v0.OTHER_FLOOR
KAPPA_TARGETS = 40.0            # game-to-game role noise (Dirichlet concentration) on top of parameter uncertainty
KAPPA_CARRIES = 30.0
STATS = ("targets", "receptions", "rec_yds", "rec_tds", "carries", "rush_att", "rush_yds", "rush_tds",
         "pass_att", "completions", "pass_yds", "pass_tds", "ints")


def seed_for(game_id: str, base: int = 20261012) -> np.random.SeedSequence:
    h = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
    return np.random.SeedSequence([h, base])


@lru_cache(maxsize=64)
def _scoring_sigma(points_sd: float, drives: int, tau: float) -> float:
    """Scoring-shock size so a 22.5-point team's simulated points SD matches the fitted SD."""
    rng = np.random.default_rng(0)
    best, best_err = 0.0, 1e9
    implied = 22.5
    p0 = tau * implied / (6.95 * drives)
    pf = (1 - tau) * implied / (3 * drives * (1 - p0))
    for s in np.linspace(0.0, 0.8, 41):
        e = rng.standard_normal(20000)
        p = np.clip(p0 * np.exp(s * e - s * s / 2), 0.005, 0.9)
        td = rng.binomial(drives, p)
        fg = rng.binomial(drives - td, np.clip(pf, 0, 0.9))
        sd = (6.95 * td + 3 * fg).std()
        if abs(sd - points_sd) < best_err:
            best, best_err = float(s), abs(sd - points_sd)
    return best


def _beta_draw(rng, mean, conc, n):
    mean = np.clip(np.asarray(mean, float), 1e-4, 1 - 1e-4)
    conc = np.clip(np.asarray(conc, float), 2.0, 2000.0)
    return rng.beta((mean * conc)[:, None], ((1 - mean) * conc)[:, None], size=(len(mean), n))


def _allocate(rng, total, shares, floor):
    """Deal `total` (n,) events to players by share rows (k, n) plus an 'other' bucket."""
    k, n = shares.shape
    s = shares.copy()
    tot = s.sum(0)
    ok = tot > 0
    s[:, ok] *= (1 - floor) / tot[ok]                 # per-sim draws renormalized like the projection
    other = 1 - s.sum(0)
    pv = np.vstack([s, other[None, :]]).T          # (n, k+1)
    pv = pv / pv.sum(1, keepdims=True)
    return rng.multinomial(total.astype(np.int64), pv).T     # (k+1, n)


def _gamma_sum(rng, count, mean, shape):
    """Sum of `count` iid gamma(shape, mean/shape) draws (0 where count == 0)."""
    count = np.asarray(count)
    out = rng.gamma(np.maximum(count * shape, 1e-9), np.maximum(mean, 1e-6) / shape)
    return np.where(count > 0, out, 0.0)


@dataclass
class SimResult:
    n: int
    game_id: str
    players: dict = field(default_factory=dict)     # player_id -> {stat: array}
    meta: dict = field(default_factory=dict)        # player_id -> {name, team, position}
    teams: dict = field(default_factory=dict)       # team -> {points, plays, dropbacks, ...}

    def stat(self, player_id: str, stat: str) -> np.ndarray:
        p = self.players[player_id]
        if stat == "rush_rec_yds":
            return p["rush_yds"] + p["rec_yds"]
        if stat == "anytime_td":
            return (p["rec_tds"] + p["rush_tds"] >= 1).astype(float)
        if stat == "rush_att":
            return p["rush_att"]
        return p[stat]


def simulate(proj: dict, n_sims: int = 20000, seed: np.random.SeedSequence | None = None) -> SimResult:
    g, fit = proj["game"], proj["fit"]
    lg = fit.league
    rng = np.random.default_rng(seed or seed_for(str(g["game_id"])))
    n = n_sims
    drives = int(round(lg["drives_pg"]))
    tau = lg["td_point_share"]
    sigma = _scoring_sigma(round(fit.points_sd, 2), drives, round(tau, 3))
    z_game = rng.standard_normal(n)
    z_pace = rng.standard_normal(n)
    teams = list(proj["teams"])
    out = SimResult(n=n, game_id=str(g["game_id"]))

    # 1. scoring for both teams
    e, pts, tds = {}, {}, {}
    for t in teams:
        tp = proj["teams"][t]
        e[t] = GAME_SHOCK_WEIGHT * z_game + np.sqrt(1 - GAME_SHOCK_WEIGHT ** 2) * rng.standard_normal(n)
        p0 = tau * tp.implied_points / (6.95 * drives)
        p_td = np.clip(p0 * np.exp(sigma * e[t] - sigma ** 2 / 2), 0.005, 0.9)
        p_fg = np.clip((1 - tau) * tp.implied_points / (3 * drives * (1 - p0)), 0.0, 0.9)
        td = rng.binomial(drives, p_td)
        fg = rng.binomial(drives - td, p_fg)
        tds[t], pts[t] = td, 6.95 * td + 3 * fg
    for t in teams:
        tp = proj["teams"][t]
        o = tp.opponent
        margin_dev = (pts[t] - pts[o]) - tp.team_spread
        P = tp.players.reset_index(drop=True)
        k = len(P)
        shock_mult = np.exp(TEAM_SHOCK_ON_YARDS * e[t] - TEAM_SHOCK_ON_YARDS ** 2 / 2)

        # 2. volume
        plays = np.maximum(30, np.round(tp.plays + fit.plays_sd * (0.5 * z_pace + np.sqrt(0.75) * rng.standard_normal(n))))
        pass_rate = np.clip(tp.pass_rate + fit.script_beta * margin_dev + fit.pr_sd * rng.standard_normal(n), 0.25, 0.9)
        dropbacks = np.round(plays * pass_rate).astype(np.int64)
        designed = (plays - dropbacks).astype(np.int64)
        sacks = rng.binomial(dropbacks, tp.sack_rate)
        starter = np.flatnonzero(P["starter_qb"].to_numpy())
        scr_rate = float(P.loc[starter[0], "scramble_rate"]) if len(starter) else lg["scramble_rate"]
        scrambles = rng.binomial(dropbacks - sacks, scr_rate)
        pass_att = dropbacks - sacks - scrambles
        targets_team = rng.binomial(pass_att, lg["targets_per_att"])

        # 3. allocation with posterior draws of shares
        # concentration = parameter uncertainty (sample size) combined with per-game role noise
        t_conc = 1 / (1 / (P["n_team_targets"].to_numpy() + 20.0) + 1 / KAPPA_TARGETS)
        c_conc = 1 / (1 / (P["n_team_carries"].to_numpy() + 40.0) + 1 / KAPPA_CARRIES)
        ts = _beta_draw(rng, P["target_share"].to_numpy(), t_conc, n) * (P["target_share"].to_numpy() > 0)[:, None]
        cs = _beta_draw(rng, P["carry_share"].to_numpy(), c_conc, n) * (P["carry_share"].to_numpy() > 0)[:, None]
        T = _allocate(rng, targets_team, ts, OTHER_FLOOR)          # (k+1, n)
        C = _allocate(rng, designed, cs, OTHER_FLOOR)

        # 4. outcomes
        cr = _beta_draw(rng, P["catch_rate"].to_numpy(), P["n_targets"].to_numpy() + 40.0, n)
        cr = 1 / (1 + np.exp(-(np.log(cr / (1 - cr)) + TEAM_SHOCK_ON_CATCH * e[t])))
        rec = rng.binomial(T[:k], cr)
        rec_o = rng.binomial(T[k], lg["other_catch_rate"])
        ypc_unc = np.exp(rng.standard_normal((k, n)) * (0.85 / np.sqrt(P["n_receptions"].to_numpy() + 30.0))[:, None])
        ypc = P["yds_per_catch"].to_numpy()[:, None] * ypc_unc * shock_mult
        shape = P["catch_yds_shape"].to_numpy()[:, None]
        rec_yds = _gamma_sum(rng, rec, ypc, shape)
        rec_yds_o = _gamma_sum(rng, rec_o, lg["other_yds_per_catch"] * shock_mult, 1.2)
        m = P["yds_per_carry"].to_numpy()[:, None] * shock_mult + CARRY_SHIFT
        sd = P["carry_yds_sd"].to_numpy()[:, None]
        rush_yds = _gamma_sum(rng, C[:k], m, m ** 2 / sd ** 2) - CARRY_SHIFT * C[:k]
        rush_att = C[:k].astype(float)
        scr_yds = np.zeros(n)
        if len(starter):
            i = starter[0]
            ys = float(P.loc[i, "yds_per_scramble"])
            scr_yds = _gamma_sum(rng, scrambles, ys * shock_mult, (ys / lg["scramble_yds_sd"]) ** 2)
            kneel = np.where(pts[t] > pts[o], KNEELS_IF_WIN, 0)
            rush_att[i] = rush_att[i] + scrambles + kneel
            rush_yds[i] = rush_yds[i] + scr_yds - kneel

        # 5. touchdowns
        pass_share = float(np.clip(lg["pass_td_share"] * tp.pass_rate / 0.6, 0.35, 0.8))
        pass_td = rng.binomial(tds[t], pass_share)
        rush_td = tds[t] - pass_td
        other_t = 1 - P["target_share"].sum()
        w = (0.65 * P["rz_share"].to_numpy() + 0.35 * P["target_share"].to_numpy())[:, None] * (rec > 0)
        wo = (0.65 * other_t + 0.35 * other_t) * (rec_o > 0)
        W = np.vstack([w, wo[None, :]]).T
        W[W.sum(1) == 0, -1] = 1.0
        rec_td = rng.multinomial(pass_td, W / W.sum(1, keepdims=True)).T
        other_c = 1 - P["carry_share"].sum()
        wr = (0.7 * P["i10_share"].to_numpy() + 0.3 * P["carry_share"].to_numpy())[:, None] * (C[:k] > 0)
        if len(starter):
            wr[starter[0]] = wr[starter[0]] + 0.05 * (scrambles > 0)
        wro = (0.7 * other_c + 0.3 * other_c) * (C[k] > 0)
        WR = np.vstack([wr, wro[None, :]]).T
        WR[WR.sum(1) == 0, -1] = 1.0
        rush_tds = rng.multinomial(rush_td, WR / WR.sum(1, keepdims=True)).T

        pass_yds = rec_yds.sum(0) + rec_yds_o
        completions = rec.sum(0) + rec_o
        ints = rng.binomial(pass_att, float(P.loc[starter[0], "int_rate"]) if len(starter) else lg["int_rate"])
        for j, r in P.iterrows():
            d = dict(targets=T[j].astype(float), receptions=rec[j].astype(float), rec_yds=np.round(rec_yds[j]),
                     rec_tds=rec_td[j].astype(float), carries=C[j].astype(float), rush_att=rush_att[j],
                     rush_yds=np.round(rush_yds[j]), rush_tds=rush_tds[j].astype(float),
                     pass_att=np.zeros(n), completions=np.zeros(n), pass_yds=np.zeros(n), pass_tds=np.zeros(n),
                     ints=np.zeros(n))
            if r["starter_qb"]:
                d.update(pass_att=pass_att.astype(float), completions=completions.astype(float),
                         pass_yds=np.round(pass_yds), pass_tds=pass_td.astype(float), ints=ints.astype(float))
            out.players[r["player_id"]] = d
            out.meta[r["player_id"]] = dict(name=r["player_name"], team=t, position=r["position"],
                                            starter_qb=bool(r["starter_qb"]))
        out.teams[t] = dict(points=pts[t], plays=plays, dropbacks=dropbacks.astype(float), pass_att=pass_att.astype(float),
                            designed_runs=designed.astype(float), tds=tds[t].astype(float), pass_yds=pass_yds)
    return out


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def summary(sim: SimResult, stats=("targets", "receptions", "rec_yds", "carries", "rush_yds", "rush_rec_yds",
                                   "pass_att", "completions", "pass_yds", "pass_tds", "ints", "anytime_td")):
    import pandas as pd
    rows = []
    for pid, meta in sim.meta.items():
        for s in stats:
            x = sim.stat(pid, s)
            if s.startswith("pass") or s in ("completions", "ints"):
                if not meta["starter_qb"]:
                    continue
            if not x.any():
                continue
            q = np.quantile(x, QUANTILES)
            rows.append(dict(player_id=pid, player=meta["name"], team=meta["team"], position=meta["position"], stat=s,
                             mean=float(x.mean()), q10=q[0], q25=q[1], median=q[2], q75=q[3], q90=q[4]))
    return pd.DataFrame(rows)


def p_over(sim: SimResult, player_id: str, stat: str, line: float) -> dict:
    x = sim.stat(player_id, stat)
    return dict(over=float((x > line).mean()), under=float((x < line).mean()), push=float((x == line).mean()))
