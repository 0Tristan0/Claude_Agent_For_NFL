"""HTML page for one game from the v0 model, in the site theme (report/theme.py)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, teams
from ..picks import payouts
from . import picks_html, theme

e = html.escape

PAGE_CSS = """.th2{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.th2 .muted{font-weight:400;font-size:.85rem;margin-left:4px}"""


def _pct(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def _team_table(sim, team: str, jerseys: dict, implied: float) -> str:
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
            return f"<td class='num'>{q[1]:.0f}<span class='rng'>{q[0]:.0f}&ndash;{q[2]:.0f}</span></td>"
        pass_cell = cell("pass_yds") if m["starter_qb"] else "<td></td>"
        jn = jerseys.get(pid)
        jn = "" if jn is None or (isinstance(jn, float) and np.isnan(jn)) else f"{int(jn)}"
        rows.append(f"<tr><th scope='row'><span class='jn'>{jn}</span>{e(m['name'])}<span class='pos'>{m['position']}</span></th>"
                    f"<td class='num'>{mean(pid, 'targets'):.1f}</td><td class='num'>{mean(pid, 'receptions'):.1f}</td>"
                    f"{cell('rec_yds')}<td class='num'>{mean(pid, 'rush_att'):.1f}</td>{cell('rush_yds')}{pass_cell}"
                    f"<td class='num'>{_pct(float(sim.stat(pid, 'anytime_td').mean()))}</td></tr>")
    d = teams.display(team)
    return (f"<section class='pane' id='{d.lower()}' data-tab='{d} players' data-logo='../logos/{teams.code(team)}.png' "
            f"aria-labelledby='h-{d.lower()}'><div><h2 class='th2' id='h-{d.lower()}'>{theme.logo(team, 28, '../')}{e(teams.name(team))}"
            f"<span class='muted'>{implied:.1f} implied pts</span></h2><p class='sub'>Medians from {sim.n:,} simulated games; the small "
            "numbers are the 10th to 90th percentile range. Any TD is the share of simulations where he scored.</p></div>"
            "<div class='tbl-wrap'><table class='pt'><thead><tr>"
            "<th scope='col'><span class='jn'>#</span>Player</th><th scope='col'>Targets</th><th scope='col'>Catches</th>"
            "<th scope='col'>Rec yds</th><th scope='col'>Rushes</th><th scope='col'>Rush yds</th>"
            "<th scope='col'>Pass yds</th><th scope='col'>Any TD</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></section>")


def _trust(check: dict | None) -> str:
    if not check:
        return ""
    o, meta = check["overall"], check.get("meta", {})
    season = meta.get("season", 2024)
    fs = meta.get("fit_seasons") or []
    cal = check["calibration"]
    rows = "".join(f"<tr><td class='num l'>{r.mean_p:.0%}</td><td class='num'>{r.hit_rate:.0%}</td>"
                   f"<td class='num'>{max(r.ci_lo, 0):.0%} to {min(r.ci_hi, 1):.0%}</td><td class='num'>{int(r.n):,}</td></tr>"
                   for r in cal.itertuples())
    span = f"{min(fs)} to {max(fs)}" if fs else "earlier seasons"
    return f"""<section class="pane" id="model" data-tab="Model check" aria-labelledby="h-trust"><div><h2 id="h-trust">How much to trust this</h2>
<p class="sub">Checked on every {season} regular-season game from week 3 ({o['games']} games, {o['n']:,} graded props), with the
model fitted on {span} only. Lines in that check are stand-ins (each player's recent median), not PrizePicks lines, so real hit
rates will be lower. {season} was also used to choose some v0 settings, so this is somewhat optimistic; the untouched test is the
2025 season. Brier score {o['brier_model']:.3f} vs {o['brier_old']:.3f} for a simple baseline (his last-8-game hit rate pulled toward
50%, no market; lower is better). When the model said X%, here is how often it happened:</p></div>
<div class="tbl-wrap"><table class="pt"><thead><tr><th scope="col">Model said</th><th scope="col">Happened</th>
<th scope="col">95% range</th><th scope="col">Props</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="sub">How the projections are built: targets and carries come from each player's recent share of team volume (shrunk
toward what his snap share predicts), team volume from the market total and spread, and efficiency shrunk toward position
averages. Each game is then simulated {{n_sims}} times.</p></section>"""


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
    run_at = pd.Timestamp(proj.get('run_at', pd.Timestamp.now(tz='UTC'))).tz_convert('America/New_York')
    jerseys = proj.get("jerseys", {})
    body = f"""<main class="wrap">
{theme.nav("home", "../")}<header class="head"><a class="back" href="../index.html">&larr; All games</a>
<div class="meta"><span>Week {int(g['week'])}</span><span>{kick:%a %b %-d, %-I:%M %p} ET</span><span class="v0tag">Model v0</span></div>
<div class="match"><div class="ms away">{theme.logo(away, 64, '../')}<div class="who"><span class="code">{a_d}</span>
<span class="nm">{e(teams.name(away))}</span></div></div><span class="vsat">at</span>
<div class="ms home">{theme.logo(home, 64, '../')}<div class="who"><span class="code">{h_d}</span>
<span class="nm">{e(teams.name(home))}</span></div></div></div>
<div class="wp"><div class="wpbar" role="img" aria-label="Simulated win chance: {a_d} {1 - win:.0%}, {h_d} {win:.0%}">
<span class="a" style="width:{(1 - win) * 100:.1f}%"></span><span class="h" style="width:{win * 100:.1f}%"></span></div>
<div class="lbl"><span>{a_d} {1 - win:.0%} &middot; {tpa.implied_points:.1f} pts implied</span><span>{tph.implied_points:.1f} pts implied &middot; {h_d} {win:.0%}</span></div></div>
<dl class="facts"><div><dt>Spread</dt><dd>{e(h_d)} {-g['spread_line']:+g}</dd></div><div><dt>Total</dt><dd>{g['total_line']:g}</dd></div>
<div><dt>Sim median score</dt><dd>{a_d} {np.median(a_pts):.0f}, {h_d} {np.median(h_pts):.0f}</dd></div>
<div><dt>Plays / pass rate</dt><dd>{tpa.plays:.0f} / {tpa.pass_rate:.0%} &middot; {tph.plays:.0f} / {tph.pass_rate:.0%}</dd></div>
<div><dt>Simulations</dt><dd>{sim.n:,}</dd></div><div><dt>Run</dt><dd>{run_at:%b %-d, %-I:%M %p} ET</dd></div>
<div class="wide"><dt>Injuries</dt><dd>{e(str(proj.get('injury_note', 'not checked')))}</dd></div>
<div class="wide"><dt>Line source</dt><dd>Spread and total: {line_note}</dd></div></dl></header>

{_team_table(sim, away, jerseys, tpa.implied_points)}
{_team_table(sim, home, jerseys, tph.implied_points)}

<section class="pane" id="lines" data-tab="Lines" aria-labelledby="h-props"><div><h2 id="h-props">PrizePicks lines</h2>
<p class="sub">The model's chance of More and Less at each PrizePicks line (calibrated on 2023). A pick qualifies when it clears
the PrizePicks break-even by 3 points: 2-pick {be['power2']:.1%}, 3-pick {min(be['power3'], be['flex3']):.1%}, 4-pick {min(be['power4'], be['flex4']):.1%},
5-pick {min(be['power5'], be['flex5']):.1%}, 6-pick {min(be['power6'], be['flex6']):.1%} per leg. With a sportsbook price on the row it must also beat that
price by 3 points. Without one ("model only"), the PrizePicks line is the market: the model's lean is pulled halfway
back toward 50%, because v0 has not yet been graded against real PrizePicks lines. When the model alone is 72% or more on
one side, the line probably reflects news the model doesn't have, so it never qualifies.</p></div>{picks_html.lines_table(legs)}</section>

<section class="pane" id="lineups" data-tab="Lineups" aria-labelledby="h-lu"><div><h2 id="h-lu">PrizePicks lineups</h2>
<p class="sub">Built only from this game's PrizePicks lines, one pick per player, both teams. Hit chances come from the simulated
games, so picks that move together (a QB and his receiver) are counted together. Same-game entries can pay less in the app:
skip a lineup if the app shows less than the listed multiplier. The Lineups tab at the top mixes games across the week.</p></div>
{picks_html.lineup_cards(picks_html.to_json(best or {}), no_lines_msg=("No PrizePicks lineup for this game yet." if legs is not None and len(legs) and legs["prizepicks"].any() else "No PrizePicks lines for this game yet."))}</section>
{_trust(check).replace("{n_sims}", f"{sim.n:,}")}
<footer class="notes"><p>Model v0: a simplified first version (team-level simulation, fixed shrinkage strengths). Data: nflverse
play-by-play, snap counts, rosters, injuries and schedules (CC-BY 4.0); stadium time zones from the greerreNFL/Stadiums dataset;
PrizePicks payouts from its help center. {lines_sentence} Actives: players with a snap in the team's last 2 games, minus Out/Doubtful; add
returning players in data/overrides.csv. These are estimates with real uncertainty, not betting advice.</p></footer>
</main>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(theme.document(title, body, PAGE_CSS))
    theme.copy_logos(out_path.parent.parent)
    # small summary next to the page, used by the site index (ticket for each game)
    out_path.with_suffix(".json").write_text(json.dumps(dict(
        game_id=g["game_id"], season=int(g["season"]), week=int(g["week"]), away=a_d, home=h_d,
        kickoff_et=f"{kick:%a %b %-d, %-I:%M %p} ET", kickoff_utc=str(pd.Timestamp(g["kickoff_utc"])),
        home_spread=float(-g["spread_line"]), total=float(g["total_line"]),
        away_implied=round(float(tpa.implied_points), 1), home_implied=round(float(tph.implied_points), 1),
        home_win=round(win, 3), median_score=[float(np.median(a_pts)), float(np.median(h_pts))]), indent=1))
    return out_path
