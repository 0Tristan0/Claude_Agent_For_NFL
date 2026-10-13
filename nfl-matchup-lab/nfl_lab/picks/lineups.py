"""Leg qualification and PrizePicks lineup building from joint sim outcomes.

Only PrizePicks lines (source "prizepicks" in data/props.csv) go into lineups. A PrizePicks leg qualifies
when the model's probability beats the PrizePicks break-even for the entry size by the margin in
config/picks.toml (default 3 points) and also beats the market reference by that margin:
  - "model + market": a sportsbook price was pasted on the same row; the reference is its no-vig chance.
  - "model only": no sportsbook price; the PrizePicks standard line is treated as a coin flip (50%), which
    is what PrizePicks aims for when it sets a line. Weaker evidence, and labeled as such.
Rows from other sources (legacy blended estimates, sportsbook-only rows) are shown for reference only.

Never qualifying: goblin/demon lines (their payouts differ and aren't modeled yet), integer lines on
count stats (a push would reduce the entry; not modeled), and lines the model is very far from.

When no lineup clears the bar, best_lineups still returns the best one it can build per size from the
PrizePicks legs the model leans toward, flagged recommended=False, so the page can show it with its
expected return instead of an empty box.

Lineups (3-6 legs, one leg per player, at least two teams) are ranked by expected value from the joint hit
matrix of the sims, reweighted ("raked") so every leg hits at a probability shrunk 25% toward
max(market, break-even) (tempers the optimizer's curse). The reported numbers re-rake each finalist to the
calibrated leg probabilities, which keeps the sims' correlation structure for both Power and Flex.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .. import config
from ..features import market
from . import calibration, payouts

FAR_FROM_LINE = 0.72       # a model probability this extreme at a real line usually means news the model lacks

QUALIFYING_KINDS = {"american", "exchange", "prob"}
LINEUP_SOURCES = {"prizepicks"}
SIZES = (2, 3, 4, 5, 6)


def leg_table(props: pd.DataFrame, sims: dict, resolve) -> pd.DataFrame:
    """One row per (prop line, side) with model probability, market probability and break-even checks.

    props: lines.fair_probs(load_props()) rows for these games; sims: game_id -> SimResult;
    resolve: function(props_rows_for_team_game) -> player_id column."""
    cfg = config.load("picks")
    margin = cfg["margin"]
    be = payouts.breakevens()
    cal = calibration.load()
    count_stats = set(config.load("stats")["count_stats"]["keys"])
    rows = []
    for _, r in props.iterrows():
        sim = sims.get(r["game_id"])
        if sim is None or pd.isna(r.get("player_id")) or r["player_id"] not in sim.players:
            continue
        x = sim.stat(r["player_id"], r["stat"])
        raw_over = float(np.clip((x > r["line"]).mean(), 0.01, 0.99))
        raw_under = float(np.clip((x < r["line"]).mean(), 0.01, 0.99))
        p_over = float(calibration.apply(raw_over, cal, r["stat"]))
        p_under = float(calibration.apply(raw_under, cal, r["stat"]))
        kind = str(r.get("odds_kind") or "")
        has_mkt = kind in QUALIFYING_KINDS and not np.isnan(r.get("fair_over", np.nan))
        # exchange/probability quotes already are probabilities: no sportsbook overround to remove
        orr = 0.0 if kind in ("exchange", "prob") else cfg["one_sided_overround"].get(
            r["stat"], cfg["one_sided_overround"]["default"])
        integer_push = r["stat"] in count_stats and float(r["line"]).is_integer()
        is_pp = str(r.get("source", "")).strip().lower() in LINEUP_SOURCES
        for side, p, raw, fair in (("over", p_over, raw_over, r.get("fair_over")),
                                   ("under", p_under, raw_under, r.get("fair_under"))):
            if side == "under" and r["stat"] == "anytime_td":
                continue
            edge, basis = None, "no price"
            if has_mkt:
                po, pu = r["p_over_implied"], r["p_under_implied"]
                edge = market.min_edge(p, po if side == "over" else pu, pu if side == "over" else po,
                                       cfg["devig_methods"], orr)
                basis = "model + market"
            elif is_pp:
                edge, basis = p - 0.5, "model only"          # PrizePicks standard line taken as a coin flip
            rows.append(dict(game_id=r["game_id"], player_id=r["player_id"], player=r["player"], team=r["team"],
                             stat=r["stat"], line=r["line"], side=side, p_model=p, p_sim=raw,
                             far_from_line=bool(max(p, 1 - p) >= FAR_FROM_LINE and r.get("source", "") != "standin"),
                             p_market=float(fair) if has_mkt and fair == fair else np.nan,
                             market_edge=edge, source=r["source"], odds_kind=kind or "none",
                             line_type=r.get("line_type", "standard") or "standard", integer_push=integer_push,
                             prizepicks=is_pp, basis=basis))
    legs = pd.DataFrame(rows)
    if legs.empty:
        return legs
    legs["lineup_ok"] = (legs["prizepicks"] & legs["line_type"].eq("standard") & ~legs["far_from_line"]
                         & ~legs["integer_push"])
    for n in SIZES:
        b = min(be[f"power{n}"], be.get(f"flex{n}", 1))      # the lower of the Power and Flex break-evens
        legs[f"be{n}"] = b
        legs[f"qual{n}"] = (legs["lineup_ok"] & (legs["p_model"] >= b + margin)
                            & legs["market_edge"].fillna(-1).ge(margin))
    legs["qualified_any"] = legs[[f"qual{n}" for n in SIZES]].any(axis=1)
    return legs


def hit_matrix(legs: pd.DataFrame, sims: dict) -> np.ndarray:
    rows = []
    for _, l in legs.iterrows():
        x = sims[l["game_id"]].stat(l["player_id"], l["stat"])
        rows.append(x > l["line"] if l["side"] == "over" else x < l["line"])
    return np.array(rows)


def best_lineups(legs: pd.DataFrame, sims: dict, sizes=SIZES, top: int = 3) -> dict[int, list[dict]]:
    """Per size: up to `top` recommended lineups (qualified legs, EV minus 2 Monte Carlo SE above 1). If none,
    the single best lineup from PrizePicks legs the model leans toward (p >= 50%), with recommended=False."""
    out = {}
    for n in sizes:
        res = _search(legs[legs[f"qual{n}"]], n, sims, top, require_ev=True) if f"qual{n}" in legs else []
        if not res and "lineup_ok" in legs:
            lean = legs[legs["lineup_ok"] & (legs["p_model"] >= 0.5)]
            res = _search(lean, n, sims, 1, require_ev=False)
        out[n] = res
    return out


def _search(pool: pd.DataFrame, n: int, sims: dict, top: int, require_ev: bool) -> list[dict]:
    """Exhaustive search over the pool (cut to the legs with the biggest edges when too large)."""
    from math import comb
    lc = config.load("picks")["lineups"]
    min_teams = int(config.load("payouts")["rules"]["min_teams"])
    if len(pool) < n:
        return []
    pool = pool.assign(_edge=pool["p_model"] - pool[f"be{n}"]).sort_values("_edge", ascending=False, kind="mergesort")
    K = len(pool)
    while K > n and comb(K, n) > lc["max_exhaustive"]:
        K -= 1
    if K < len(pool):
        print(f"note: {n}-pick search uses the {K} legs with the biggest edges ({len(pool) - K} left out)")
    pool = pool.head(K).reset_index(drop=True)
    H = hit_matrix(pool, sims)
    s_ = lc["shrink_to_reference"]
    ref = np.maximum(pool["p_market"].fillna(0).to_numpy(), pool[f"be{n}"].to_numpy())
    shrunk = (1 - s_) * pool["p_model"].to_numpy() + s_ * ref
    w_pool = payouts.rake(H, shrunk)
    corr = np.corrcoef(H.astype(float)) if len(pool) > 1 else np.ones((1, 1))
    players, teams_ = pool["player_id"].to_numpy(), pool["team"].to_numpy()
    cands = []
    for c in combinations(range(len(pool)), n):
        c = list(c)
        if len(set(players[c])) < n or len(set(teams_[c])) < min_teams:
            continue
        sub = corr[np.ix_(c, c)]
        if (sub[np.triu_indices(n, 1)] < lc["negative_corr_block"]).any():
            continue
        ev = payouts.entry_ev(H[c], weights=w_pool)
        cands.append((max(ev["power_ev"] or 0, ev["flex_ev"] or 0), c))
    cands.sort(key=lambda t: -t[0])
    res = []
    for _, c in cands:
        if len(res) >= top:
            break
        w = payouts.rake(H[c], pool["p_model"].to_numpy()[c])
        if payouts.rake_error(H[c], w, pool["p_model"].to_numpy()[c]) > 1e-3:
            continue
        ev = payouts.entry_ev(H[c], weights=w)
        best_ev = max(ev["power_ev"] or 0, ev["flex_ev"] or 0)
        clears = best_ev - lc["min_ev_se"] * ev["power_ev_se"] > 1.0
        if require_ev and not clears:
            continue
        res.append(dict(legs=pool.iloc[c].drop(columns="_edge").to_dict("records"), **ev,
                        play_as="Power" if (ev["power_ev"] or 0) >= (ev["flex_ev"] or 0) + 0.03 else "Flex",
                        correlation_lift=ev["p_all"] / ev["p_independent"] if ev["p_independent"] else np.nan,
                        best_ev=best_ev, recommended=bool(require_ev and clears),
                        basis="model + market" if all(l["basis"] == "model + market" for l in pool.iloc[c].to_dict("records"))
                        else "model only"))
    return res
