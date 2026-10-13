"""Best parlays where the new model (simulation) and the old model (recent hit rate, src/ev.py method) agree.

A pick is eligible when both models take the same side of a standard PrizePicks line and the new model's own
filters pass (not too far from the line, no whole-number push). Its chance is the average of the two models.
Lineups (one pick per player, at least two teams) are searched over the agreed picks and priced from the
simulated games reweighted to those averaged chances, so picks that move together are counted together.
"""
from __future__ import annotations

import pandas as pd

from . import lineups

KEYS = ["game_id", "player_id", "stat", "line"]


def agreed_legs(legs: pd.DataFrame, old_est: pd.DataFrame) -> pd.DataFrame:
    if legs is None or legs.empty or old_est is None or old_est.empty:
        return pd.DataFrame()
    new = (legs[legs["lineup_ok"]].sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates(KEYS))
    old = old_est[KEYS + ["side", "p_model", "hits", "games"]].rename(columns={"side": "old_side", "p_model": "p_old"})
    m = new.merge(old, on=KEYS, how="inner")
    m = m[m["side"] == m["old_side"]].copy()
    m["p_new"] = m["p_model"]
    m["p_model"] = (m["p_new"] + m["p_old"]) / 2
    return m[m["p_model"] >= 0.5].reset_index(drop=True)


def best(legs: pd.DataFrame, old_est: pd.DataFrame, sims: dict, sizes=(3, 4, 5, 6)) -> dict:
    pool = agreed_legs(legs, old_est)
    out = {"lineups": {}, "seventh": None, "n_agreed": int(len(pool))}
    if pool.empty:
        return out
    for n in sizes:
        res = lineups._search(pool, n, sims, top=1, require_ev=False)
        for L in res:
            L["recommended"] = L["clears"]
            L["basis"] = "both models agree"
            for leg in L["legs"]:
                leg["note"] = f"new model {leg['p_new']:.0%}, old model {leg['p_old']:.0%} (over in {leg['hits']} of last {leg['games']})" \
                    if leg["side"] == "over" else \
                    f"new model {leg['p_new']:.0%}, old model {leg['p_old']:.0%} (under in {leg['hits']} of last {leg['games']})"
        out["lineups"][n] = res
    six = out["lineups"].get(max(sizes)) or []
    used = {l["player_id"] for L in six for l in L["legs"]}
    rest = pool[~pool["player_id"].isin(used)].sort_values("p_model", ascending=False)
    if len(rest):
        r = rest.iloc[0]
        out["seventh"] = dict(player=r["player"], team=r["team"], stat=r["stat"], side=r["side"], line=float(r["line"]),
                              p_model=float(r["p_model"]), p_new=float(r["p_new"]), p_old=float(r["p_old"]))
    return out
