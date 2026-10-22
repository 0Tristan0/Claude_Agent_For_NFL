"""One model for PrizePicks picks: game simulation + recent form + this game's matchups, lineups from one game.

For each standard PrizePicks line and side:
  sim       the simulation model's chance (market-line calibration; pulled halfway to 50% without a sportsbook
            price), from picks/lineups.leg_table
  form      the original report's formula without its opponent nudge: last-4-game hit rate at the line pulled
            toward 50%, averaged with the market (picks/oldmodel.py); opponents are handled by the matchup layer
  chance    W_SIM x sim + (1 - W_SIM) x form + matchup adjustment (picks/matchups.py, capped at 5 points)
Each line keeps the side with the higher chance. Lineups use picks from ONE game only (one pick per player, both
teams), are priced from that game's simulations reweighted to the chances (picks that move together count
together), and per size the lineup with the highest expected return per $1 (Power or Flex) is kept.
"""
from __future__ import annotations

import pandas as pd

from . import lineups

W_SIM = 0.5
KEYS = ["game_id", "player_id", "stat", "line"]


def blend(legs: pd.DataFrame, old_est: pd.DataFrame) -> pd.DataFrame:
    """Both sides of every usable PrizePicks line, with sim, form and blended chance."""
    if legs is None or legs.empty:
        return pd.DataFrame()
    new = legs[legs["lineup_ok"]].copy()
    if new.empty:
        return new
    new["p_sim"] = new["p_model"]
    if old_est is not None and len(old_est):
        o = old_est[KEYS + ["side", "p_model", "nudge", "hits", "games"]].rename(
            columns={"side": "form_side", "p_model": "form_p", "nudge": "form_nudge"})
        new = new.merge(o, on=KEYS, how="left")
        clean = new["form_p"] - new["form_nudge"].fillna(0) / 2          # drop the old opponent nudge
        new["p_form"] = clean.where(new["side"] == new["form_side"], 1 - clean)
        new["p_form"] = new["p_form"].fillna(new["p_sim"])                # no form estimate: sim only
    else:
        new["p_form"], new["hits"], new["games"], new["form_side"] = new["p_sim"], 0, 0, new["side"]
    new["p_model"] = W_SIM * new["p_sim"] + (1 - W_SIM) * new["p_form"]
    new["p_matchup"], new["reasons"] = 0.0, [[] for _ in range(len(new))]
    return new.reset_index(drop=True)


def pick_sides(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values("p_model", ascending=False, kind="mergesort").drop_duplicates(KEYS).reset_index(drop=True)


def _note(leg: dict) -> str:
    hits, games = leg.get("hits"), leg.get("games")
    form = ""
    if games == games and games and hits == hits:
        h = int(hits) if leg["side"] == leg.get("form_side") else int(games) - int(hits)
        form = f" ({'More' if leg['side'] == 'over' else 'Less'} in {h} of last {int(games)})"
    m = round((leg.get("p_matchup", 0) or 0) * 100)
    mt = f"matchup {m:+d} pts" if m else "matchup even"
    return f"sim {leg['p_sim']:.0%}, recent form {leg['p_form']:.0%}{form}, {mt}"


def lineups_for_game(picks: pd.DataFrame, sims: dict, sizes=(3, 4, 5, 6)) -> dict:
    """Top lineup per size from one game's picks (chance 50% or better)."""
    pool = picks[picks["p_model"] >= 0.5] if len(picks) else picks
    out = {}
    if pool is None or pool.empty:
        return out
    for n in sizes:
        res = lineups._search(pool, n, sims, top=1, require_ev=False)
        for L in res:
            L["recommended"] = L["clears"]
            L["basis"] = "sim + form + matchups"
            for leg in L["legs"]:
                leg["note"] = _note(leg)
        out[n] = res
    return out


def best_across_games(per_game: dict, labels: dict) -> dict:
    """For each size, the best single-game lineup of the week (lineups never mix games)."""
    out = {}
    for gid, best in per_game.items():
        for n, lst in best.items():
            for L in lst:
                L = dict(L, game=labels.get(gid, gid))
                if n not in out or L["best_ev"] > out[n][0]["best_ev"]:
                    out[n] = [L]
    return out
