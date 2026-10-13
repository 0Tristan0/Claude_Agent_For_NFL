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
.board2 td.pk2{white-space:nowrap}
.tier{font-size:.78rem;font-weight:600}
.tier.High{color:var(--pos)} .tier.Med-High{color:var(--t2)} .tier.Medium{color:var(--t3)}
.tier.Lean,.tier.Coin{color:var(--muted)}
.ut td,.ut th{white-space:nowrap}
.adv{font-size:.78rem;color:var(--muted)}
.adv.o{color:var(--pos)} .adv.d{color:var(--neg)}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.how summary{cursor:pointer;color:var(--muted);font-size:.85rem}
@media (max-width:560px){.match{grid-template-columns:1fr auto 1fr}.ms.home{flex-direction:row-reverse;text-align:right}
  .vsat{display:block}.match .lg{--s:40px!important;width:40px;height:40px}.ms{gap:10px}}
.how[open] summary{margin-bottom:6px}
.board2 tr.inplay th,.board2 tr.inplay td{background:var(--surface)}
.board2 tr.inplay th{box-shadow:inset 3px 0 0 var(--text)}
.inp{color:var(--text);font-weight:600}"""


def _pill(rank: int) -> str:
    t = "t1" if rank <= 8 else "t2" if rank <= 16 else "t3" if rank <= 24 else "t4"
    return f"<span class='pill {t}'>{rank}</span>"


def _fmt(stat: str, v: float) -> str:
    return f"{v:.0%}" if "rate" in stat.lower() else f"{v:+.3f}" if "EPA" in stat else f"{v:.2f}"


def _board(picks: list[dict], pl: dict | None = None) -> str:
    play = {picks_html._key(x) for x in picks_html.play_summary(pl or {})}
    if not picks:
        return ("<p class='sub'>No standard PrizePicks lines for this game yet. " + picks_html.HOW_TO_ADD + "</p>")
    rows = []
    for p in sorted(picks, key=lambda d: -d["p_model"]):
        log = (", ".join("TD" if v else "-" for v in p["log"]) if p["stat"] == "anytime_td"
               else ", ".join(f"{v:g}" for v in p["log"]))
        small = " <span class='muted'>(small sample)</span>" if p["games"] < 3 else ""
        why = "".join(f"<span>{e(w)}</span>" for w in p.get("reasons", [])) or "<span>No strong matchup signal.</span>"
        inplay = picks_html._key(p) in play
        rows.append(f"<tr{' class=inplay' if inplay else ''}><th scope='row'>{theme.logo(p['team'], 18, '../')} {e(p['player'])}"
                    f"<span class='vs-lbl'>{e(p['position'])}" + (" · <span class='inp'>&#9733; In the play</span>" if inplay else "")
                    + f"</span></th>"
                    f"<td class='l pk2'>{picks_html.side_chip(p['side'])} <b>{p['line']:g}</b> {e(picks_html.stat_name(p['stat']))}</td>"
                    f"<td class='l'><b>{p['p_model']:.0%}</b><span class='vs-lbl tier {p['tier'].split()[0]}'>{e(p['tier'])}</span></td>"
                    f"<td class='l'>{SIDE[p['side']]} in {p['hits']} of {p['games']}{small}<span class='vs-lbl'>{e(log)}</span></td>"
                    f"<td class='why'>{why}</td></tr>")
    return ("<div class='tbl-wrap'><table class='pt board2'><thead><tr><th scope='col'>Player</th><th scope='col' class='l'>Pick</th>"
            "<th scope='col' class='l'>Chance</th>"
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
    pl = rec.get("plan") or {}
    n_lean = len({p["player_id"] for p in rec.get("picks") or [] if p["p_model"] >= 0.525 and p["games"] >= 3})
    no_play = ("No standard PrizePicks lines for this game yet." if not rec.get("picks") else
               f"{len(rec['picks'])} line{'s' if len(rec['picks']) != 1 else ''} on file, but only {n_lean} player"
               f"{'s' if n_lean != 1 else ''} rated Lean (52.5%) or better from both teams, so no play yet. A fresh "
               "PrizePicks board closer to kickoff may add more.")
    body = f"""<main class="wrap">
{theme.nav("home", "../")}<header class="head"><a class="back" href="../index.html">&larr; All games</a>
<div class="meta"><span>Week {rec['week']}</span><span>{kick:%a %b %-d, %-I:%M %p} ET</span></div>
<div class="match"><div class="ms away">{theme.logo(a, 64, '../')}<div class="who"><span class="code">{a}</span></div></div>
<span class="vsat">at</span><div class="ms home">{theme.logo(h, 64, '../')}<div class="who"><span class="code">{h}</span></div></div></div>
<dl class="facts"><div class="wide"><dt>Line</dt><dd>{e(line)}</dd></div>
<div class="wide"><dt>Injuries</dt><dd>{e(rec.get('injury_note') or 'not checked')}</dd></div></dl></header>

<section class="pane" id="play" data-tab="The play" aria-labelledby="h-lu"><h2 id="h-lu" class="sr-only">The play</h2>
{picks_html.guesser_play(pl, lu, no_lines_msg=no_play)}
<details class="how"><summary>How the play is made</summary><p class="sub">Each standard PrizePicks line gets an estimate the way
the original Bills-Rams report did it: (recent + matchup + market) / 2, where recent is his last 4 games at the line pulled toward
50%, matchup is this game's matchups, and market is 50% when the board has no price. The best line per player (3+ games, 52.5% or
better) is a pick; entries are the top picks from this game only, both teams. The Guesser then splits $20 across up to 3 entries
(2 to 6 picks, Power or Flex) to grow a pretend bankroll fastest, which puts most of it on the entry most likely to pay. Chances
multiply the picks' estimates as if independent.</p></details></section>

<section class="pane" id="props" data-tab="Props board" aria-labelledby="h-pb"><div><h2 id="h-pb">Props board</h2>
<p class="sub">Every standard PrizePicks line for this game, rated with the original formula: (recent + matchup + market) / 2.
Recent is his last 4 games at the line pulled toward 50%; matchup is this game's matchups (up to 8 points before halving);
market is 50% when there is no sportsbook price. High is 60% or more, Med-High 57.5%, Medium 55%, Lean 52.5%. Rows marked &#9733; are in the play.</p></div>{_board(rec.get('picks') or [], pl)}</section>

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
