"""HTML page for one game from the v0 model, in the site theme (report/theme.py)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, teams
from ..picks import payouts
from . import theme

e = html.escape


def _pct(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def _team_table(sim, team: str) -> str:
    rows = []
    pids = [p for p, m in sim.meta.items() if m["team"] == team]
    def mean(pid, s):
        return float(sim.stat(pid, s).mean())
    pids.sort(key=lambda p: -(mean(p, "targets") + mean(p, "carries") + 30 * sim.meta[p]["starter_qb"]))
    for pid in pids:
        m = sim.meta[pid]
        if mean(pid, "targets") + mean(pid, "carries") < 0.6 and not m["starter_qb"]:
            continue
        def cell(s):
            x = sim.stat(pid, s)
            q = np.quantile(x, [0.1, 0.5, 0.9]) if x.any() else np.zeros(3)
            if q[2] == 0:
                return "<td class='num'></td>"                 # 90% of sims at zero: leave blank
            return f"<td class='num'>{q[1]:.0f}<span class='rng'> ({q[0]:.0f} to {q[2]:.0f})</span></td>"
        pass_cell = cell("pass_yds") if m["starter_qb"] else "<td></td>"
        rows.append(f"<tr><th scope='row'>{e(m['name'])}<span class='vs-lbl'>{m['position']}</span></th>"
                    f"<td class='num'>{mean(pid, 'targets'):.1f}</td><td class='num'>{mean(pid, 'receptions'):.1f}</td>"
                    f"{cell('rec_yds')}<td class='num'>{mean(pid, 'rush_att'):.1f}</td>{cell('rush_yds')}{pass_cell}"
                    f"<td class='num'>{_pct(float(sim.stat(pid, 'anytime_td').mean()))}</td></tr>")
    return ("<div class='tbl-wrap'><table class='pt'><caption>" + theme.swatch(teams.display(team)) + e(teams.name(team))
            + "</caption><thead><tr>"
            "<th scope='col'>Player</th><th scope='col'>Targets</th><th scope='col'>Catches</th>"
            "<th scope='col'>Rec yds (10th to 90th)</th><th scope='col'>Rushes</th><th scope='col'>Rush yds</th>"
            "<th scope='col'>Pass yds</th><th scope='col'>Any TD</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def _props_table(legs: pd.DataFrame) -> str:
    if legs is None or legs.empty:
        return ("<p class='sub'>No lines for this game in data/props.csv yet. Paste PrizePicks lines plus a sportsbook "
                "price for the same line to get qualified picks.</p>")
    rows = []
    for _, l in legs.sort_values("p_model", ascending=False).iterrows():
        q = "Yes" if l["qualified_any"] else ("Check news: model far from line" if l.get("far_from_line") else
                                             ("No market price" if pd.isna(l["p_market"]) else "No"))
        rows.append(f"<tr><th scope='row'>{e(l['player'])}<span class='vs-lbl'>{e(teams.display(l['team']))} · "
                    f"{e(l['source'])}</span></th><td class='l'>{e(l['stat'].replace('_', ' '))}</td><td class='num'>{l['line']:g}</td>"
                    f"<td class='l'>{l['side']}</td><td class='num'>{_pct(l['p_model'])}</td><td class='num'>{_pct(l['p_market'])}</td>"
                    f"<td class='l {'q-yes' if l['qualified_any'] else 'q-no'}'>{q}</td></tr>")
    return ("<div class='tbl-wrap'><table class='pt'><thead><tr><th scope='col'>Player</th><th scope='col' class='l'>Stat</th>"
            "<th scope='col'>Line</th><th scope='col' class='l'>Side</th><th scope='col'>Model</th><th scope='col'>Market (no vig)</th>"
            "<th scope='col' class='l'>Qualifies</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _lineups(best: dict) -> str:
    cards = []
    for n, lst in sorted(best.items()):
        for L in lst[:1]:
            legs = "".join(f"<li>{e(x['player'])} {e(x['stat'].replace('_', ' '))} {x['side']} {x['line']:g} "
                           f"<span class='pp'>{x['p_model']:.0%}</span></li>" for x in L["legs"])
            cards.append(f"""<article class="lu"><header><h3>{n}-pick</h3><span class="rec">Play it as <b>{L['play_as']}</b></span></header>
<ol>{legs}</ol><dl class="lustats">
<div><dt>All hit (sim)</dt><dd>{L['p_all']:.1%}</dd></div>
<div><dt>If legs were independent</dt><dd>{L['p_independent']:.1%}</dd></div>
<div><dt>Return per $1 (Power)</dt><dd>${L['power_ev']:.2f}</dd></div>
<div><dt>Return per $1 (Flex)</dt><dd>${(L['flex_ev'] or 0):.2f}</dd></div>
<div><dt>Skip if app shows under</dt><dd>{L['min_power_mult']:.1f}x</dd></div>
</dl></article>""")
    if not cards:
        return "<p class='sub'>No lineup qualifies yet. A leg needs the model to beat both the payout break-even and the no-vig market price by 3 points.</p>"
    return "<div class='lus'>" + "".join(cards) + "</div>"


def _trust(check: dict | None) -> str:
    if not check:
        return ""
    o, meta = check["overall"], check.get("meta", {})
    season = meta.get("season", 2024)
    fs = meta.get("fit_seasons") or []
    cal = check["calibration"]
    rows = "".join(f"<tr><td class='num'>{r.mean_p:.0%}</td><td class='num'>{r.hit_rate:.0%}</td>"
                   f"<td class='num'>{max(r.ci_lo, 0):.0%} to {min(r.ci_hi, 1):.0%}</td><td class='num'>{int(r.n):,}</td></tr>"
                   for r in cal.itertuples())
    span = f"{min(fs)} to {max(fs)}" if fs else "earlier seasons"
    return f"""<section aria-labelledby="h-trust"><div><h2 id="h-trust">How much to trust this</h2>
<p class="sub">Checked on every {season} regular-season game from week 3 ({o['games']} games, {o['n']:,} graded props), with the
model fitted on {span} only. Lines in that check are stand-ins (each player's recent median), not PrizePicks lines, so real hit
rates will be lower. {season} was also used to choose some v0 settings, so this is somewhat optimistic; the untouched test is the
2025 season. Brier score {o['brier_model']:.3f} vs {o['brier_old']:.3f} for a simple baseline (his last-8-game hit rate pulled toward
50%, no market; lower is better). When the model said X%, here is how often it happened:</p></div>
<div class="tbl-wrap"><table class="pt"><thead><tr><th scope="col">Model said</th><th scope="col">Happened</th>
<th scope="col">95% range</th><th scope="col">Props</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""


def render(proj: dict, sim, legs: pd.DataFrame | None, best: dict | None, check: dict | None, out_path: Path) -> Path:
    g = proj["game"]
    home, away = g["home_team"], g["away_team"]
    h_pts, a_pts = sim.teams[home]["points"], sim.teams[away]["points"]
    win = float((h_pts > a_pts).mean() + 0.5 * (h_pts == a_pts).mean())
    tph, tpa = proj["teams"][home], proj["teams"][away]
    be = payouts.breakevens()
    title = f"{teams.display(away)} at {teams.display(home)}, Week {int(g['week'])}"
    played = pd.Timestamp(g["kickoff_utc"]) <= pd.Timestamp.now(tz="UTC")
    line_note = "closing line" if played else "latest market line"
    lines_sentence = ("Market lines shown are closing lines." if played else
                      "Market lines are the latest in the nflverse schedule when this page was built.")
    a_d, h_d = teams.display(away), teams.display(home)
    kick = pd.Timestamp(g['kickoff_utc']).tz_convert('America/New_York')
    body = f"""<main class="wrap">
{theme.nav("games", "../")}<div class="board"><span class="eyebrow">Week {int(g['week'])} · {kick:%a %b %d, %-I:%M %p} ET<span class="v0tag">Model v0</span></span>
<div class="teams"><div class="team away">{theme.pennant(a_d)}<span class="nm">{e(teams.name(away))}</span>
<span class="rec">Implied {tpa.implied_points:.1f} pts · sim win {1 - win:.0%}</span></div><span class="at">at</span>
<div class="team home">{theme.pennant(h_d)}<span class="nm">{e(teams.name(home))}</span>
<span class="rec">Implied {tph.implied_points:.1f} pts · sim win {win:.0%}</span></div></div>
<div class="facts"><dl><div><dt>Spread / total</dt><dd>{e(h_d)} {-g['spread_line']:+g} / {g['total_line']:g} ({line_note})</dd></div>
<div><dt>Sim median score</dt><dd>{a_d} {np.median(a_pts):.0f}, {h_d} {np.median(h_pts):.0f}</dd></div>
<div><dt>Plays / pass rate</dt><dd>{a_d} {tpa.plays:.0f} / {tpa.pass_rate:.0%} · {h_d} {tph.plays:.0f} / {tph.pass_rate:.0%}</dd></div>
<div><dt>Sims</dt><dd>{sim.n:,}</dd></div>
<div><dt>Injuries</dt><dd>{e(str(proj.get('injury_note', 'not checked')))}</dd></div>
<div><dt>Run at</dt><dd>{pd.Timestamp(proj.get('run_at', pd.Timestamp.now(tz='UTC'))).tz_convert('America/New_York'):%b %d %-I:%M %p} ET</dd></div></dl></div></div>

<section aria-labelledby="h-proj"><div><h2 id="h-proj">Projections</h2>
<p class="sub">Medians with the 10th to 90th percentile range from {sim.n:,} simulated games. Targets and carries come from each
player's recent share of team volume (shrunk toward what his snap share predicts), team volume from the market total and
spread, and efficiency shrunk toward position averages.</p></div>
{_team_table(sim, away)}{_team_table(sim, home)}</section>

<section aria-labelledby="h-props"><div><h2 id="h-props">Your lines</h2>
<p class="sub">From data/props.csv. Model chances are calibrated on 2023 (raw sim chances were too confident at the
extremes). When the model is very far from a real line (72% or more on one side), the line probably reflects news the
model doesn't have, so those legs never qualify. Break-even per leg: 3-pick {min(be['power3'], be['flex3']):.1%}, 4-pick {min(be['power4'], be['flex4']):.1%},
5-pick {min(be['power5'], be['flex5']):.1%}, 6-pick {min(be['power6'], be['flex6']):.1%}. A leg qualifies only if the model beats that and the
no-vig market price by 3 points.</p></div>{_props_table(legs)}</section>

<section aria-labelledby="h-lu"><div><h2 id="h-lu">Lineups</h2>
<p class="sub">Hit chance comes from the simulated games, so legs that move together (a QB and his receiver) are counted together.
Same-game lineups can get lower multipliers in the app: skip a lineup if the app shows less than the listed minimum.</p></div>{_lineups(best or {})}</section>
{_trust(check)}
<footer class="notes"><p>Model v0: a simplified first version (team-level simulation, fixed shrinkage strengths). Data: nflverse
play-by-play, snap counts, rosters, injuries and schedules (CC-BY 4.0); stadium time zones from the greerreNFL/Stadiums dataset;
PrizePicks payouts from its help center. {lines_sentence} Actives: players with a snap in the team's last 2 games, minus Out/Doubtful; add
returning players in data/overrides.csv. These are estimates with real uncertainty, not betting advice.</p></footer>
</main>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(theme.document(title, body))
    # small summary next to the page, used by the site index (ticket for each game)
    out_path.with_suffix(".json").write_text(json.dumps(dict(
        game_id=g["game_id"], season=int(g["season"]), week=int(g["week"]), away=a_d, home=h_d,
        kickoff_et=f"{kick:%a %b %-d, %-I:%M %p} ET", kickoff_utc=str(pd.Timestamp(g["kickoff_utc"])),
        home_spread=float(-g["spread_line"]), total=float(g["total_line"]),
        away_implied=round(float(tpa.implied_points), 1), home_implied=round(float(tph.implied_points), 1),
        home_win=round(win, 3), median_score=[float(np.median(a_pts)), float(np.median(h_pts))]), indent=1))
    return out_path
