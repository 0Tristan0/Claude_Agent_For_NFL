"""Game page for the Guesser method (nfl_lab/guesser.py): props board, lineups, unit vs unit. No simulation."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from . import picks_html, theme

e = html.escape
SIDE = {"over": "More", "under": "Less"}

CSS = """.board2 td.why{white-space:normal;min-width:300px;max-width:520px;font-size:.84rem;color:var(--muted)}
.board2 td.why span{display:block}
.tier{font-size:.78rem;font-weight:600}
.tier.High{color:var(--pos)} .tier.Med-High{color:var(--t2)} .tier.Medium{color:var(--t3)}
.tier.Lean,.tier.Coin{color:var(--muted)}
.ut td,.ut th{white-space:nowrap}
.adv{font-size:.78rem;color:var(--muted)}
.adv.o{color:var(--pos)} .adv.d{color:var(--neg)}"""


def _pill(rank: int) -> str:
    t = "t1" if rank <= 8 else "t2" if rank <= 16 else "t3" if rank <= 24 else "t4"
    return f"<span class='pill {t}'>{rank}</span>"


def _fmt(stat: str, v: float) -> str:
    return f"{v:.0%}" if "rate" in stat.lower() else f"{v:+.3f}" if "EPA" in stat else f"{v:.2f}"


def _board(picks: list[dict]) -> str:
    if not picks:
        return ("<p class='sub'>No standard PrizePicks lines for this game yet. " + picks_html.HOW_TO_ADD + "</p>")
    rows = []
    for p in sorted(picks, key=lambda d: -d["p_model"]):
        log = (", ".join("TD" if v else "-" for v in p["log"]) if p["stat"] == "anytime_td"
               else ", ".join(f"{v:g}" for v in p["log"]))
        small = " <span class='muted'>(small sample)</span>" if p["games"] < 3 else ""
        why = "".join(f"<span>{e(w)}</span>" for w in p.get("reasons", [])) or "<span>No strong matchup signal.</span>"
        rows.append(f"<tr><th scope='row'>{theme.logo(p['team'], 18, '../')} {e(p['player'])}<span class='vs-lbl'>{e(p['position'])}</span></th>"
                    f"<td class='l'>{e(picks_html.stat_name(p['stat']))}</td><td class='num'>{p['line']:g}</td>"
                    f"<td class='l'><b>{SIDE[p['side']]}</b></td>"
                    f"<td class='l'><span class='tier {p['tier'].split()[0]}'>{e(p['tier'])}</span><span class='vs-lbl'>{p['p_model']:.0%}</span></td>"
                    f"<td class='l'>{SIDE[p['side']]} in {p['hits']} of {p['games']}{small}<span class='vs-lbl'>{e(log)}</span></td>"
                    f"<td class='why'>{why}</td></tr>")
    return ("<div class='tbl-wrap'><table class='pt board2'><thead><tr><th scope='col'>Player</th><th scope='col' class='l'>Prop</th>"
            "<th scope='col'>Line</th><th scope='col' class='l'>Lean</th><th scope='col' class='l'>Confidence</th>"
            "<th scope='col' class='l'>Last 4</th><th scope='col' class='l'>Why</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def _units(units: list[dict]) -> str:
    if not units:
        return "<p class='sub'>Not enough games yet this season to rank the units.</p>"
    out = []
    for off in dict.fromkeys(u["offense"] for u in units):
        rows = []
        for u in [u for u in units if u["offense"] == off]:
            edge = ("<span class='adv o'>Offense</span>" if u["off_rank"] + 6 <= u["def_rank"] else
                    "<span class='adv d'>Defense</span>" if u["def_rank"] + 6 <= u["off_rank"] else "<span class='adv'>Even</span>")
            rows.append(f"<tr><th scope='row'>{e(u['stat'])}</th><td class='l'>{_pill(u['off_rank'])}{_fmt(u['stat'], u['off_value'])}</td>"
                        f"<td class='l'>{_pill(u['def_rank'])}{_fmt(u['stat'], u['def_value'])}</td><td class='l'>{edge}</td></tr>")
        d = units[[u["offense"] for u in units].index(off)]["defense"]
        out.append(f"<div class='tbl-wrap'><table class='pt ut'><caption>{theme.logo(off, 22, '../')} {e(off)} offense vs "
                   f"{theme.logo(d, 22, '../')} {e(d)} defense</caption><thead><tr><th scope='col'>Stat</th>"
                   f"<th scope='col' class='l'>{e(off)} offense</th><th scope='col' class='l'>{e(d)} defense</th>"
                   f"<th scope='col' class='l'>Edge</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    return ("<div class='legend'><span><span class='pill t1'>1-8</span>Top quarter</span><span><span class='pill t2'>9-16</span>"
            "Above avg</span><span><span class='pill t3'>17-24</span>Below avg</span><span><span class='pill t4'>25-32</span>"
            "Bottom quarter</span><span>Ranks out of 32; 1 = best for that unit. Pressure: the line's rank is pressure allowed, "
            "the defense's is pressure created.</span></div>" + "".join(out))


def render(rec: dict, out_path: Path) -> Path:
    a, h = rec["away"], rec["home"]
    kick = pd.Timestamp(rec["kickoff_utc"]).tz_convert("America/New_York")
    spread, total = rec.get("home_spread"), rec.get("total")
    if spread is not None and total is not None:
        imp_h, imp_a = (total - spread) / 2, (total + spread) / 2
        fav = h if spread < 0 else a
        line = f"{fav} {-abs(spread):+g} · O/U {total:g} · implied {a} {imp_a:.1f}, {h} {imp_h:.1f}"
    else:
        line = "No market line yet"
    lu = rec.get("lineups") or {}
    body = f"""<main class="wrap">
{theme.nav("home", "../")}<header class="head"><a class="back" href="../index.html">&larr; All games</a>
<div class="meta"><span>Week {rec['week']}</span><span>{kick:%a %b %-d, %-I:%M %p} ET</span></div>
<div class="match"><div class="ms away">{theme.logo(a, 64, '../')}<div class="who"><span class="code">{a}</span></div></div>
<span class="vsat">at</span><div class="ms home">{theme.logo(h, 64, '../')}<div class="who"><span class="code">{h}</span></div></div></div>
<dl class="facts"><div class="wide"><dt>Line</dt><dd>{e(line)}</dd></div>
<div class="wide"><dt>Injuries</dt><dd>{e(rec.get('injury_note') or 'not checked')}</dd></div></dl></header>

<section class="pane" id="lineups" data-tab="Lineups" aria-labelledby="h-lu"><div><h2 id="h-lu">Lineups</h2>
<p class="sub">Built the way the Bills-Rams $20 entries were: the top picks by estimate from this game only, one per player,
both teams (if the top 3 are all one team, the other team's best pick takes the 3rd spot). Entry A is the top 3 as a Flex
($12 in the Guesser), Entry B the top 4 as a Power ($8). The 5- and 6-pick add the next picks. A lineup's chance multiplies its picks' estimates.</p></div>
{picks_html.guesser_lineups(lu, no_lines_msg="Not enough standard PrizePicks lines for this game yet.")}</section>

<section class="pane" id="props" data-tab="Props board" aria-labelledby="h-pb"><div><h2 id="h-pb">Props board</h2>
<p class="sub">Every standard PrizePicks line for this game, rated with the original formula: (recent + matchup + market) / 2.
Recent is his last 4 games at the line pulled toward 50%; matchup is this game's matchups (up to 8 points before halving);
market is 50% when there is no sportsbook price. High is 60% or more, Med-High 57.5%, Medium 55%, Lean 52.5%.</p></div>{_board(rec.get('picks') or [])}</section>

<section class="pane" id="units" data-tab="Unit vs unit" aria-labelledby="h-uu"><div><h2 id="h-uu">Unit vs unit</h2>
<p class="sub">Each offense against the defense it faces tonight, recency weighted (this season counts most).</p></div>
{_units(rec.get('units') or [])}</section>

<footer class="notes"><p>Method: nfl_lab/guesser.py. Data: nflverse play-by-play, snap counts, rosters and injuries
(CC-BY 4.0); pressure rates from PFR via nflverse; lines from the PrizePicks board saved before kickoff. Only stats from before
kickoff are used. Estimates with real uncertainty, not betting advice.</p></footer>
</main>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(theme.document(f"{a} at {h}, Week {rec['week']}", body, CSS + picks_html.GUESSER_CSS))
    theme.copy_logos(out_path.parent.parent)
    return out_path
