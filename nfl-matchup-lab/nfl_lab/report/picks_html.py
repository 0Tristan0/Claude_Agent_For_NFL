"""PrizePicks lines and lineups as HTML, shared by the game pages and the site's Lineups tab.

Lineups are passed around as plain dicts (to_json) so src/site.py can render them from saved files.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from .. import teams
from . import theme

e = html.escape

PP_STAT = {"rec_yds": "Receiving Yards", "rush_yds": "Rush Yards", "pass_yds": "Pass Yards", "receptions": "Receptions",
           "anytime_td": "Rush+Rec TDs", "rush_rec_tds": "Rush+Rec TDs", "rush_rec_yds": "Rush+Rec Yds",
           "pass_tds": "Pass TDs", "pass_att": "Pass Attempts", "completions": "Pass Completions",
           "rush_att": "Rush Attempts", "ints": "INT", "pass_rush_yds": "Pass+Rush Yds"}
SIDE = {"over": "More", "under": "Less"}

HOW_TO_ADD = ("Add PrizePicks lines to <code>data/props.csv</code> with source <code>prizepicks</code> (player, team, stat, "
              "line), or send me screenshots of the PrizePicks board and I'll enter them. A sportsbook price for the same "
              "line on the same row makes the check stronger.")


def stat_name(s: str) -> str:
    return PP_STAT.get(s, s.replace("_", " ").title())


def _pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def lines_table(legs: pd.DataFrame | None, blended: pd.DataFrame | None = None, used: set | None = None,
                prefix: str = "../") -> str:
    """PrizePicks lines for one game: the pick and its blended chance (sim + recent form) per standard line."""
    if legs is None or legs.empty or not legs["prizepicks"].any():
        other = 0 if legs is None or legs.empty else legs[["player", "stat", "line"]].drop_duplicates().shape[0]
        extra = f" ({other} reference lines from other sources are on file but don't go into lineups.)" if other else ""
        return f"<p class='sub'>No PrizePicks lines for this game yet.{extra} {HOW_TO_ADD}</p>"
    used = used or set()
    allpp = legs[legs["prizepicks"]]
    std = allpp[allpp["line_type"].eq("standard")]
    n_alt = allpp.loc[~allpp["line_type"].eq("standard"), ["player_id", "stat", "line"]].drop_duplicates().shape[0]
    rows = []
    b = blended if blended is not None else pd.DataFrame()
    for _, l in (b.sort_values("p_model", ascending=False) if len(b) else b).iterrows():
        top = (l["player_id"], l["stat"], l["line"]) in used
        rows.append(f"<tr><th scope='row'>{theme.logo(l['team'], 18, prefix)} {e(l['player'])}</th>"
                    f"<td class='l'>{e(stat_name(l['stat']))}</td><td class='num'>{l['line']:g}</td>"
                    f"<td class='l'>{SIDE[l['side']]}</td><td class='num'>{_pct(l['p_model'])}"
                    f"<span class='rng'>sim {_pct(l['p_sim'])} &middot; form {_pct(l['p_form'])} &middot; matchup "
                    f"{round((l.get('p_matchup', 0) or 0) * 100):+d}</span></td>"
                    f"<td class='l {'q-yes' if top else 'q-no'}'>{'In a top lineup' if top else ''}</td></tr>")
    left = std[~std["lineup_ok"]].sort_values("p_model_alone", ascending=False).drop_duplicates(["player_id", "stat", "line"])
    for _, l in left.iterrows():
        why = ("model far from line, check news" if l["far_from_line"] else
               "whole-number line can push" if l["integer_push"] else "left out")
        rows.append(f"<tr><th scope='row'>{theme.logo(l['team'], 18, prefix)} {e(l['player'])}</th>"
                    f"<td class='l'>{e(stat_name(l['stat']))}</td><td class='num'>{l['line']:g}</td>"
                    f"<td class='l'>{SIDE[l['side']]}</td><td class='num muted'>sim {_pct(l['p_model_alone'])}</td>"
                    f"<td class='l q-no'>Left out: {why}</td></tr>")
    n_other = legs.loc[~legs["prizepicks"], ["player", "stat", "line"]].drop_duplicates().shape[0]
    note = (f"<p class='small'>{n_other} reference lines from other sources are on file for this game; they are not "
            "PrizePicks lines, so they stay out.</p>") if n_other else ""
    if n_alt:
        note += (f"<p class='small'>{n_alt} goblin and demon lines are on the board too. They pay differently from standard "
                 "picks and those payouts aren't modeled yet, so they are left out.</p>")
    if not rows:
        return "<p class='sub'>No standard PrizePicks lines for this game that the model simulates.</p>" + note
    return ("<div class='tbl-wrap'><table class='pt'><thead><tr><th scope='col'>Player</th><th scope='col' class='l'>Stat</th>"
            "<th scope='col'>Line</th><th scope='col' class='l'>Pick</th><th scope='col'>Chance</th>"
            "<th scope='col' class='l'></th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + note)


def retro_note(retro: dict) -> str:
    """Explains a before-kickoff test that has been graded against the real result."""
    return (f"<div class='retro'><p><b>Graded test.</b> These lineups were built with only what was known before kickoff "
            f"(stats through the previous game and the lines below), then checked against the final stats: "
            f"{e(retro.get('final', ''))}. Lines here are the ones on file before kickoff, mostly sportsbook lines; "
            f"PrizePicks had pulled its board by the time it was saved.</p></div>")


def to_json(best: dict) -> dict:
    """best_lineups output -> plain dicts (only what the pages need)."""
    out = {}
    for n, lst in best.items():
        out[str(n)] = [dict(
            n=int(n), recommended=bool(L.get("recommended", True)), basis=L.get("basis", ""), play_as=L["play_as"],
            p_all=float(L["p_all"]), p_independent=float(L["p_independent"]),
            power_ev=None if L["power_ev"] is None else float(L["power_ev"]),
            flex_ev=None if L["flex_ev"] is None else float(L["flex_ev"]),
            best_ev=float(L.get("best_ev", max(L["power_ev"] or 0, L["flex_ev"] or 0))),
            min_power_mult=float(L["min_power_mult"]),
            legs=[dict(player=x["player"], team=teams.display(x["team"]), stat=x["stat"], side=x["side"],
                       line=float(x["line"]), p_model=float(x["p_model"]), game_id=x["game_id"],
                       player_id=x.get("player_id"),
                       **({"note": x["note"]} if x.get("note") else {}),
                       **({"reasons": list(x["reasons"])} if isinstance(x.get("reasons"), (list, tuple)) and len(x["reasons"]) else {}),
                       **({"result": x["result"]} if x.get("result") else {})) for x in L["legs"]],
            **({"outcome": L["outcome"]} if L.get("outcome") else {}),
            **({"game": L["game"]} if L.get("game") else {}))
            for L in lst]
    return out


def lineup_cards(best_json: dict, prefix: str = "../", no_lines_msg: str = "") -> str:
    cards = []
    for n in sorted(best_json, key=int):
        for L in best_json[n][:1]:
            legs = "".join(
                f"<li>{theme.logo(x['team'], 18, prefix)}<span><b>{e(x['player'])}</b> {SIDE[x['side']]} {x['line']:g} "
                f"{e(stat_name(x['stat']))}" + (f"<span class='vs-lbl'>{e(x['note'])}</span>" if x.get("note") else "")
                + "".join(f"<span class='vs-lbl why'>{e(w)}</span>" for w in x.get("reasons", []))
                + (f"<span class='vs-lbl res {'hit' if x['result']['hit'] else 'miss'}'>{'Hit' if x['result']['hit'] else 'Missed'}: "
                   f"{e(str(x['result']['actual']))}</span>" if x.get("result") else "")
                + f"</span><span class='pp'>{x['p_model']:.0%}</span></li>" for x in L["legs"])
            old = L.get("basis") == "old model"
            hit_rows = (f"<div><dt>All hit (legs multiplied)</dt><dd>{L['p_all']:.1%}</dd></div>" if old else
                        f"<div><dt>All hit (simulated)</dt><dd>{L['p_all']:.1%}</dd></div>"
                        f"<div><dt>If legs were independent</dt><dd>{L['p_independent']:.1%}</dd></div>")
            tag = ("<span class='tag ok'>Recommended</span>" if L["recommended"] else
                   "<span class='tag'>Not recommended</span>")
            if L.get("outcome"):
                o = L["outcome"]
                tag = (f"<span class='tag ok'>Won ${o['returned']:.2f} per $1</span>" if o["returned"] > 0 else
                       "<span class='tag'>Lost</span>")
            if L.get("basis") == "old model":
                tag = ("<span class='tag ok'>Above break-even</span>" if L["recommended"] else
                       "<span class='tag'>Below break-even</span>")
            flex = (f"<div><dt>Flex return per $1</dt><dd>${L['flex_ev']:.2f}</dd></div>" if L["flex_ev"] is not None else "")
            who = "old model" if old else "model"
            why = ("" if L["recommended"] else
                   f"<p class='small'>Best the {who} can build from these lines, but it expects ${L['best_ev']:.2f} back "
                   "per $1, which is below break-even.</p>")
            skip = (f"<div><dt>Skip if the app pays under</dt><dd>{L['min_power_mult']:.1f}x</dd></div>"
                    if L["recommended"] else "")
            game = f"<span class='small'>{e(L['game'])}</span>" if L.get("game") else ""
            cards.append(f"""<article class="lu{'' if L['recommended'] else ' dim'}"><header><div><h3>{n}-pick {L['play_as']}</h3>{game}</div>{tag}</header>
<ol class="lul">{legs}</ol><dl class="lustats">
{hit_rows}
<div><dt>Power return per $1</dt><dd>${(L['power_ev'] or 0):.2f}</dd></div>{flex}{skip}
</dl>{why}<p class='small'>Basis: {e(L['basis'])}</p></article>""")
    if not cards:
        return f"<p class='sub'>{no_lines_msg or 'No PrizePicks lineup can be built yet.'} {HOW_TO_ADD}</p>"
    return "<div class='lus'>" + "".join(cards) + "</div>"
