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
           "rush_att": "Rush Attempts", "ints": "INT Thrown"}
SIDE = {"over": "More", "under": "Less"}

HOW_TO_ADD = ("Add PrizePicks lines to <code>data/props.csv</code> with source <code>prizepicks</code> (player, team, stat, "
              "line), or send me screenshots of the PrizePicks board and I'll enter them. A sportsbook price for the same "
              "line on the same row makes the check stronger.")


def stat_name(s: str) -> str:
    return PP_STAT.get(s, s.replace("_", " ").title())


def _pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def lines_table(legs: pd.DataFrame | None, prefix: str = "../") -> str:
    """PrizePicks lines for one game with the model's chance on each side. Other rows are counted, not shown."""
    if legs is None or legs.empty or not legs["prizepicks"].any():
        other = 0 if legs is None or legs.empty else legs[["player", "stat", "line"]].drop_duplicates().shape[0]
        extra = f" ({other} reference lines from other sources are on file but don't go into lineups.)" if other else ""
        return f"<p class='sub'>No PrizePicks lines for this game yet.{extra} {HOW_TO_ADD}</p>"
    pp = legs[legs["prizepicks"]].sort_values("p_model", ascending=False)
    rows = []
    for _, l in pp.iterrows():
        if l["qualified_any"]:
            q, cls = ("Yes, model + market" if l["basis"] == "model + market" else "Yes, model only"), "q-yes"
        elif l["far_from_line"]:
            q, cls = "No: model far from line, check news", "q-no"
        elif l["line_type"] != "standard":
            q, cls = f"No: {l['line_type']} line", "q-no"
        elif l["integer_push"]:
            q, cls = "No: whole-number line can push", "q-no"
        else:
            q, cls = "No: below break-even + 3 pts", "q-no"
        rows.append(f"<tr><th scope='row'>{theme.logo(l['team'], 18, prefix)} {e(l['player'])}</th>"
                    f"<td class='l'>{e(stat_name(l['stat']))}</td><td class='num'>{l['line']:g}</td>"
                    f"<td class='l'>{SIDE[l['side']]}</td><td class='num'>{_pct(l['p_model'])}</td>"
                    f"<td class='num'>{_pct(l['p_market']) or '<span class=muted>none</span>'}</td>"
                    f"<td class='l {cls}'>{q}</td></tr>")
    n_other = legs.loc[~legs["prizepicks"], ["player", "stat", "line"]].drop_duplicates().shape[0]
    note = (f"<p class='small'>{n_other} reference lines from other sources (old blended estimates, sportsbook-only rows) "
            "are on file for this game; they are not PrizePicks lines, so they stay out of lineups.</p>") if n_other else ""
    return ("<div class='tbl-wrap'><table class='pt'><thead><tr><th scope='col'>Player</th><th scope='col' class='l'>Stat</th>"
            "<th scope='col'>Line</th><th scope='col' class='l'>Pick</th><th scope='col'>Model</th>"
            "<th scope='col'>Sportsbook (no vig)</th><th scope='col' class='l'>Qualifies</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>" + note)


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
                       line=float(x["line"]), p_model=float(x["p_model"]), game_id=x["game_id"]) for x in L["legs"]])
            for L in lst]
    return out


def lineup_cards(best_json: dict, prefix: str = "../", no_lines_msg: str = "") -> str:
    cards = []
    for n in sorted(best_json, key=int):
        for L in best_json[n][:1]:
            legs = "".join(
                f"<li>{theme.logo(x['team'], 18, prefix)}<span><b>{e(x['player'])}</b> {SIDE[x['side']]} {x['line']:g} "
                f"{e(stat_name(x['stat']))}</span><span class='pp'>{x['p_model']:.0%}</span></li>" for x in L["legs"])
            tag = ("<span class='tag ok'>Recommended</span>" if L["recommended"] else
                   "<span class='tag'>Not recommended</span>")
            flex = (f"<div><dt>Flex return per $1</dt><dd>${L['flex_ev']:.2f}</dd></div>" if L["flex_ev"] is not None else "")
            why = ("" if L["recommended"] else
                   f"<p class='small'>Best the model can build from these lines, but it expects ${L['best_ev']:.2f} back "
                   "per $1, which is below break-even.</p>")
            skip = (f"<div><dt>Skip if the app pays under</dt><dd>{L['min_power_mult']:.1f}x</dd></div>"
                    if L["recommended"] else "")
            cards.append(f"""<article class="lu{'' if L['recommended'] else ' dim'}"><header><h3>{n}-pick {L['play_as']}</h3>{tag}</header>
<ol class="lul">{legs}</ol><dl class="lustats">
<div><dt>All hit (simulated)</dt><dd>{L['p_all']:.1%}</dd></div>
<div><dt>If legs were independent</dt><dd>{L['p_independent']:.1%}</dd></div>
<div><dt>Power return per $1</dt><dd>${(L['power_ev'] or 0):.2f}</dd></div>{flex}{skip}
</dl>{why}<p class='small'>Basis: {e(L['basis'])}</p></article>""")
    if not cards:
        return f"<p class='sub'>{no_lines_msg or 'No PrizePicks lineup can be built yet.'} {HOW_TO_ADD}</p>"
    return "<div class='lus'>" + "".join(cards) + "</div>"
