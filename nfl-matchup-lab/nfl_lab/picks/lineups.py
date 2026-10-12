"""Leg qualification and lineup building from joint sim outcomes.

A leg qualifies only when the model's probability beats BOTH the PrizePicks break-even for the entry size
and the devigged market probability by the margin in config/picks.toml (default 3 points). Legs without
a real two-way or one-way market price (PrizePicks-only lines, legacy blended estimates) are shown with
their probabilities but never qualify: paste a sportsbook price for the same line to test them.

Lineups (3-6 legs, one leg per player, at least two teams) are ranked by expected value computed from the
joint hit matrix of the sims. Leg probabilities used for ranking are shrunk 25% toward max(market,
break-even) to temper the optimizer's curse; reported numbers are the unshrunk sim values.
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


def leg_table(props: pd.DataFrame, sims: dict, resolve) -> pd.DataFrame:
    """One row per (prop line, side) with model probability, market probability and break-even checks.

    props: lines.fair_probs(load_props()) rows for these games; sims: game_id -> SimResult;
    resolve: function(props_rows_for_team_game) -> player_id column."""
    cfg = config.load("picks")
    margin = cfg["margin"]
    be = payouts.breakevens()
    cal = calibration.load()
    rows = []
    for _, r in props.iterrows():
        sim = sims.get(r["game_id"])
        if sim is None or pd.isna(r.get("player_id")) or r["player_id"] not in sim.players:
            continue
        x = sim.stat(r["player_id"], r["stat"])
        raw_over, raw_under = float((x > r["line"]).mean()), float((x < r["line"]).mean())
        p_over = float(calibration.apply(raw_over, cal))
        p_under = float(calibration.apply(raw_under, cal))
        kind = str(r.get("odds_kind") or "")
        has_mkt = kind in QUALIFYING_KINDS and not np.isnan(r.get("fair_over", np.nan))
        orr = cfg["one_sided_overround"].get(r["stat"], cfg["one_sided_overround"]["default"])
        for side, p, raw, fair in (("over", p_over, raw_over, r.get("fair_over")),
                                   ("under", p_under, raw_under, r.get("fair_under"))):
            if side == "under" and r["stat"] == "anytime_td":
                continue
            edge = None
            if has_mkt:
                po, pu = r["p_over_implied"], r["p_under_implied"]
                edge = market.min_edge(p, po if side == "over" else pu, pu if side == "over" else po,
                                       cfg["devig_methods"], orr)
            rows.append(dict(game_id=r["game_id"], player_id=r["player_id"], player=r["player"], team=r["team"],
                             stat=r["stat"], line=r["line"], side=side, p_model=p, p_sim=raw,
                             far_from_line=bool(max(p, 1 - p) >= FAR_FROM_LINE and r.get("source", "") != "standin"),
                             p_market=float(fair) if has_mkt and fair == fair else np.nan,
                             market_edge=edge, source=r["source"], odds_kind=kind or "none",
                             line_type=r.get("line_type", "standard")))
    legs = pd.DataFrame(rows)
    if legs.empty:
        return legs
    for n in (3, 4, 5, 6):
        b = min(be[f"power{n}"], be.get(f"flex{n}", 1))      # the lower of the Power and Flex break-evens
        legs[f"be{n}"] = b
        legs[f"qual{n}"] = ((legs["p_model"] >= b + margin) & legs["market_edge"].fillna(-1).ge(margin)
                            & ~legs["far_from_line"])
    legs["qualified_any"] = legs[[f"qual{n}" for n in (3, 4, 5, 6)]].any(axis=1)
    return legs


def hit_matrix(legs: pd.DataFrame, sims: dict) -> np.ndarray:
    rows = []
    for _, l in legs.iterrows():
        x = sims[l["game_id"]].stat(l["player_id"], l["stat"])
        rows.append(x > l["line"] if l["side"] == "over" else x < l["line"])
    return np.array(rows)


def best_lineups(legs: pd.DataFrame, sims: dict, sizes=(3, 4, 5, 6), top: int = 3) -> dict[int, list[dict]]:
    """Exhaustive search (with a beam above the configured size) for the highest-EV qualified lineups."""
    cfg = config.load("picks")
    lc = cfg["lineups"]
    be = payouts.breakevens()
    out = {}
    for n in sizes:
        pool = legs[legs[f"qual{n}"]].reset_index(drop=True)
        if len(pool) < n:
            out[n] = []
            continue
        H = hit_matrix(pool, sims)
        cal_ratio = (pool["p_model"] / pool["p_sim"].clip(lower=1e-6)).to_numpy()
        ref = np.maximum(pool["p_market"].fillna(0).to_numpy(), be[f"power{n}"])
        p_leg = H.mean(1)
        shrink = lc["shrink_to_reference"]
        adj = (1 - shrink) + shrink * (ref / np.maximum(p_leg, 1e-9))     # scale applied to joint prob for ranking
        corr = np.corrcoef(H.astype(float)) if len(pool) > 1 else np.ones((1, 1))
        players, teams_ = pool["player_id"].to_numpy(), pool["team"].to_numpy()
        cands = []
        combos = combinations(range(len(pool)), n)
        for c in combos:
            if len(set(players[list(c)])) < n or len(set(teams_[list(c)])) < 2:
                continue
            sub = corr[np.ix_(c, c)]
            if (sub[np.triu_indices(n, 1)] < lc["negative_corr_block"]).any():
                continue
            ev = payouts.entry_ev(H[list(c)], scale=float(np.prod(cal_ratio[list(c)])))
            rank = max(ev["power_ev"] or 0, ev["flex_ev"] or 0) * float(np.prod(adj[list(c)]))
            cands.append((rank, c, ev))
            if len(cands) > lc["max_exhaustive"]:
                break
        cands.sort(key=lambda t: -t[0])
        res = []
        for rank, c, _ in cands[:top]:
            # final numbers: sims reweighted so every leg hits at its calibrated chance (keeps correlations)
            c = list(c)
            w = payouts.rake(H[c], pool["p_model"].to_numpy()[c])
            ev = payouts.entry_ev(H[c], weights=w)
            if max(ev["power_ev"] or 0, ev["flex_ev"] or 0) - 2 * ev["power_ev_se"] <= 1.0:
                continue
            res.append(dict(legs=pool.iloc[list(c)].to_dict("records"), **ev,
                            play_as="Power" if (ev["power_ev"] or 0) >= (ev["flex_ev"] or 0) + 0.03 else "Flex",
                            correlation_lift=ev["p_all"] / ev["p_independent"] if ev["p_independent"] else np.nan))
        out[n] = res
    return out
