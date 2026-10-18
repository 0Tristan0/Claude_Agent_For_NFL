"""v0 Monte Carlo game simulator (team-level, correlated). Seeded and vectorized over sims.

Per sim, for each team:
  1. Scoring: TDs and field goals are drawn per drive, so points average the market-implied total and
     spread. Each team's efficiency shock is tied to its simulated points, so yards, catches and TDs
     rise and fall with scoring (same-game stacks are correlated like real games).
  2. Volume: plays around the projection (pace shock shared by both teams); pass rate shifts with how the
     game went versus the spread (teams that fall behind throw more); dropbacks, sacks, scrambles,
     pass attempts and targets follow.
  3. Allocation: each player's target and carry share is drawn from its posterior (parameter
     uncertainty), then targets and carries are dealt out multinomially, with an "other" bucket.
  4. Outcomes: catches ~ Binomial(targets, catch rate), yards from right-skewed gamma draws per catch or
     carry, scaled by the team shock (so yards, catches and points move together).
  5. TDs: allocated per catch and per carry, weighted by red-zone/inside-10 shares and by position
     conversion rates measured in play-by-play (QB runs near the goal line score far more often than RB
     runs; RB red-zone catches less often than WR/TE); scrambles are TD chances too. QB pass yards are the
     sum of receiver yards. Kneels and a mid-game starter exit are random, at their real rates.

Every prop and every parlay is read from these same draws, so joint probabilities come from joint outcomes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

TEAM_SHOCK_ON_CATCH = 0.25      # logit shift of catch rate per unit efficiency shock
TEAM_SHOCK_ON_YARDS = 0.16      # log shift of yards per catch/carry per unit efficiency shock
CARRY_SHIFT = 3.0               # carry yards = Gamma - 3 so losses are possible
OTHER_FLOOR = 0.015             # same as project.v0.OTHER_FLOOR
KAPPA_TARGETS = 40.0            # game-to-game role noise (Dirichlet concentration) on top of parameter uncertainty
KAPPA_CARRIES = 30.0
STATS = ("targets", "receptions", "rec_yds", "rec_tds", "carries", "rush_att", "rush_yds", "rush_tds",
         "pass_att", "completions", "pass_yds", "pass_tds", "ints")


def seed_for(game_id: str, base: int = 20261012) -> np.random.SeedSequence:
    h = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16)
    return np.random.SeedSequence([h, base])


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
        if stat == "pass_rush_yds":
            return p["pass_yds"] + p["rush_yds"]
        if stat == "anytime_td":
            return (p["rec_tds"] + p["rush_tds"] >= 1).astype(float)
        if stat == "rush_rec_tds":
            return p["rec_tds"] + p["rush_tds"]
        if stat == "rush_att":
            return p["rush_att"]
        return p[stat]


# Rushing-TD conversion relative to an RB carry (2022-24 play-by-play: QB inside-10 designed runs score at
# about 1.6x the RB rate, outside the 10 about 1.9x), and red-zone catch conversion relative to WR/TE.
RUSH_TD_CONV_I10 = {"QB": 1.58, "RB": 1.0, "WR": 1.3, "TE": 1.3}
RUSH_TD_CONV_OUT = {"QB": 1.9, "RB": 1.0, "WR": 2.5, "TE": 2.5}
REC_TD_RZ_CONV = {"RB": 0.55, "WR": 1.0, "TE": 1.0, "QB": 1.0}
SCRAMBLE_TD_WEIGHT = 1.05        # a scramble is a TD chance about like an average carry
KNEEL_RATE = {"win": 1.32, "lose": 0.22, "tie": 0.75}    # kneels per game (2022-24), about -1.1 yd each
STARTER_EXIT_P = 0.06            # starter leaves mid-game in about 6% of games (injury, benching, blowout)


def _per_event_weights(events, base):
    """TD weight per sim proportional to how many chances (catches/carries) a player had in that sim,
    normalized so each player's expected weight equals `base` (his intended share of TDs)."""
    exp = np.maximum(events.mean(1, keepdims=True), 1e-9)
    return events * (base[:, None] / exp)


def _carry_yards(rng, counts, ypc, mult, emp, ypc_pos):
    """Sum of per-carry yards: bootstrap the position's real carry distribution (keeps breakaways and
    losses), rescaled so the player's mean matches ypc * mult. counts: (n,) carries per sim."""
    n = counts.shape[0]
    total = int(counts.sum())
    if total == 0:
        return np.zeros(n)
    draws = emp[rng.integers(0, len(emp), total)] + CARRY_SHIFT
    sums = np.zeros(n)
    idx = np.repeat(np.arange(n), counts.astype(np.int64))
    np.add.at(sums, idx, draws)
    scale = (ypc * mult + CARRY_SHIFT) / (ypc_pos + CARRY_SHIFT)
    return scale * sums - CARRY_SHIFT * counts


def simulate(proj: dict, n_sims: int = 20000, seed: np.random.SeedSequence | None = None) -> SimResult:
    g, fit = proj["game"], proj["fit"]
    lg = fit.league
    rng = np.random.default_rng(seed or seed_for(str(g["game_id"])))
    n = n_sims
    drives = int(round(lg["drives_pg"]))
    tau = lg["td_point_share"]
    z_pace = rng.standard_normal(n)
    z_poss = rng.standard_normal(n)        # possession trade-off: one team's extra plays are the other's fewer
    teams = list(proj["teams"])
    out = SimResult(n=n, game_id=str(g["game_id"]))

    # 1. scoring: TDs and field goals per drive, averaging the market-implied points
    e, pts, tds = {}, {}, {}
    for t in teams:
        tp = proj["teams"][t]
        p0 = tau * tp.implied_points / (6.95 * drives)
        p_fg = np.clip((1 - tau) * tp.implied_points / (3 * drives * (1 - p0)), 0.0, 0.9)
        td = rng.binomial(drives, np.clip(p0, 0.005, 0.9), size=n)
        fg = rng.binomial(drives - td, p_fg)
        tds[t], pts[t] = td, 6.95 * td + 3 * fg
        # efficiency shock tied to how the scoring went: good scoring sims are good yardage sims
        zp = (pts[t] - pts[t].mean()) / max(pts[t].std(), 1e-9)
        e[t] = 0.9 * zp + np.sqrt(0.19) * rng.standard_normal(n)
    for t in teams:
        tp = proj["teams"][t]
        o = tp.opponent
        margin_dev = (pts[t] - pts[o]) - tp.team_spread
        P = tp.players.reset_index(drop=True)
        k = len(P)
        pos = P["position"].to_numpy()
        shock_mult = np.exp(TEAM_SHOCK_ON_YARDS * e[t] - TEAM_SHOCK_ON_YARDS ** 2 / 2)
        side = 1.0 if tp.is_home >= 0.5 else -1.0
        zp_t = (pts[t] - pts[t].mean()) / max(pts[t].std(), 1e-9)

        # 2. volume: pace shared, possessions traded off between the teams, more plays when scoring
        noise = 0.3 * z_pace + 0.7 * side * z_poss + 0.2 * zp_t + 0.62 * rng.standard_normal(n)
        plays = np.maximum(30, np.round(tp.plays + fit.plays_sd * noise))
        pass_rate = np.clip(tp.pass_rate + fit.script_beta * margin_dev + fit.pr_sd * rng.standard_normal(n), 0.25, 0.9)
        dropbacks = np.round(plays * pass_rate).astype(np.int64)
        designed = (plays - dropbacks).astype(np.int64)
        sacks = rng.binomial(dropbacks, tp.sack_rate)
        starter = np.flatnonzero(P["starter_qb"].to_numpy())
        scr_rate = float(P.loc[starter[0], "scramble_rate"]) if len(starter) else lg["scramble_rate"]
        scrambles = rng.binomial(dropbacks - sacks, scr_rate)
        pass_att = dropbacks - sacks - scrambles
        targets_team = rng.binomial(pass_att, lg["targets_per_att"])

        # 3. allocation with posterior draws of shares (parameter uncertainty + per-game role noise)
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
        rush_yds = np.zeros((k, n))
        for j in range(k):
            if C[j].any():
                emp = fit.carry_emp.get(pos[j], fit.carry_emp["RB"])
                rush_yds[j] = _carry_yards(rng, C[j], float(P.loc[j, "yds_per_carry"]), shock_mult, emp,
                                           float(fit.pos.loc[pos[j], "yds_per_carry"]) if pos[j] in fit.pos.index else 4.3)
        rush_att = C[:k].astype(float)
        frac = np.ones(n)
        if len(starter):
            i = starter[0]
            ys = float(P.loc[i, "yds_per_scramble"])
            scr_yds = _gamma_sum(rng, scrambles, ys * shock_mult, (ys / lg["scramble_yds_sd"]) ** 2)
            lam = np.where(pts[t] > pts[o], KNEEL_RATE["win"], np.where(pts[t] < pts[o], KNEEL_RATE["lose"], KNEEL_RATE["tie"]))
            kneel = rng.poisson(lam)
            rush_att[i] = rush_att[i] + scrambles + kneel
            rush_yds[i] = rush_yds[i] + scr_yds - 1.1 * kneel
            # the starter leaves mid-game in a few sims: he keeps only the share of pass volume before he left
            frac = np.where(rng.random(n) < STARTER_EXIT_P, rng.random(n), 1.0)

        # 5. touchdowns: per catch / per carry, with position conversion rates
        pass_share = float(np.clip(lg["pass_td_share"] * tp.pass_rate / 0.6, 0.35, 0.8))
        pass_td = rng.binomial(tds[t], pass_share)
        rush_td = tds[t] - pass_td
        rz_conv = np.array([REC_TD_RZ_CONV.get(p, 1.0) for p in pos])
        base_rec = 0.65 * P["rz_share"].to_numpy() * rz_conv + 0.35 * P["target_share"].to_numpy()
        other_t = max(1 - P["target_share"].sum(), 1e-6)
        W = _per_event_weights(np.vstack([rec, rec_o[None, :]]).astype(float), np.append(base_rec, other_t)).T
        W[W.sum(1) == 0, -1] = 1.0
        rec_td = rng.multinomial(pass_td, W / W.sum(1, keepdims=True)).T
        conv_i10 = np.array([RUSH_TD_CONV_I10.get(p, 1.0) for p in pos])
        conv_out = np.array([RUSH_TD_CONV_OUT.get(p, 1.0) for p in pos])
        base_rush = 0.7 * P["i10_share"].to_numpy() * conv_i10 + 0.3 * P["carry_share"].to_numpy() * conv_out
        other_c = max(1 - P["carry_share"].sum(), 1e-6)
        WR = _per_event_weights(np.vstack([C[:k], C[k][None, :]]).astype(float), np.append(base_rush, other_c))
        if len(starter):
            per_carry = np.append(base_rush, other_c).sum() / max(C.mean(1).sum(), 1e-9)
            WR[starter[0]] = WR[starter[0]] + SCRAMBLE_TD_WEIGHT * per_carry * scrambles
        WR = WR.T
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
                d.update(pass_att=np.round(pass_att * frac), completions=np.round(completions * frac),
                         pass_yds=np.round(pass_yds * frac), pass_tds=rng.binomial(pass_td, frac).astype(float),
                         ints=rng.binomial(ints, frac).astype(float))
            out.players[r["player_id"]] = d
            out.meta[r["player_id"]] = dict(name=r["player_name"], team=t, position=r["position"],
                                            starter_qb=bool(r["starter_qb"]))
        out.teams[t] = dict(points=pts[t], plays=plays, dropbacks=dropbacks.astype(float), pass_att=pass_att.astype(float),
                            designed_runs=designed.astype(float), tds=tds[t].astype(float), pass_yds=pass_yds,
                            completions=completions.astype(float), pass_tds=pass_td.astype(float),
                            rush_tds=rush_td.astype(float))
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
