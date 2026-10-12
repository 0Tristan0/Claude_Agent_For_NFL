"""Build the website: site/index.html (game list + track record) and one page per game report.

index.html is a page fragment (the Artifact publisher wraps it in <html>/<head>/<body>).
Game pages are published as extra files, so they get a full document here.
Run after the report scripts:  python3 src/site.py
"""
import html, json, os, re, shutil
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
G = json.load(open(os.path.join(ROOT, "games.json")))["games"]

# head/style shared with the report template so both pages look like one site
TPL = open(os.path.join(ROOT, "src", "template.html")).read()
HEAD = TPL[:TPL.index("<main")]
HEAD = HEAD.replace("<title>Bills at Rams MNF Matchup</title>", "<title>NFL Matchup Lab</title>")
e = html.escape


def fmt_date(s):
    d = date.fromisoformat(s)
    return d.strftime("%a %b ") + str(d.day)


def game_page(g):
    src = open(os.path.join(ROOT, g["report"])).read()
    i = src.index("<main")
    back = '<nav class="back"><a href="../index.html">All games</a></nav>'
    body = re.sub(r'(<main class="wrap">)', r"\1\n  " + back, src[i:], count=1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            + src[:i] +
            '<style>body{margin:0}.back a{font-family:var(--display);font-weight:600;letter-spacing:.08em;'
            'text-transform:uppercase;color:var(--muted)}.back a::before{content:"\\2039  "}</style>\n'
            '</head>\n<body>\n' + body + '</body>\n</html>\n')


def card(g):
    match = f'<span class="tchip">{e(g["away"])}</span> at <span class="tchip">{e(g["home"])}</span>'
    meta = f'Week {g["week"]} · {e(g["label"])} · {fmt_date(g["date"])}'
    if g["status"] == "pending":
        tag = '<span class="tag t3">Picks locked</span>'
        body = (f'<p>{e(g["picks"])}</p>'
                f'<p class="fine">Frozen {e(g["frozen_at"])} in commit {e(g["frozen_commit"])}, before kickoff.</p>')
    else:
        tag = '<span class="tag t2">Graded</span>'
        body = f'<p>Final: {e(g["final"])}. Props {g["props"]["hit"]} of {g["props"]["total"]}.</p>'
        if not g.get("blind", True):
            body += '<p class="fine">Not a blind test: stats loaded for it already included this game.</p>'
    link = (f'<a class="open" href="games/{g["slug"]}.html">Open report</a>' if g.get("report")
            else '<span class="fine">No full report page for this one</span>')
    return (f'<article class="game"><header><span class="eyebrow">{meta}</span>{tag}</header>'
            f'<h3>{match}</h3><span class="nm">{e(g["away_name"])} at {e(g["home_name"])}</span>'
            f'{body}{link}</article>')


def record_rows():
    rows, ph, pt, units = [], 0, 0, 0.0
    for g in G:
        name = f'{e(g["away"])} at {e(g["home"])}<span class="vs-lbl">{fmt_date(g["date"])}</span>'
        if g["status"] != "graded":
            rows.append(f'<tr><th scope="row">{name}</th><td colspan="3" class="muted">Waiting on the game</td>'
                        f'<td><span class="tag t3">Clean test</span></td></tr>')
            continue
        gl, p = g["game_lines"], g["props"]
        ph, pt, units = ph + p["hit"], pt + p["total"], units + p["units"]
        blind = '<span class="tag t3">Clean test</span>' if g.get("blind", True) else '<span class="tag t4">Not blind</span>'
        rows.append(f'<tr><th scope="row">{name}</th><td class="num">{gl["hit"]} of {gl["total"]}'
                    f'<span class="vs-lbl">{e(gl["note"])}</span></td><td class="num">{p["hit"]} of {p["total"]}</td>'
                    f'<td class="num">{p["units"]:+.1f}u</td><td>{blind}</td></tr>')
    total = (f'<tr class="tot"><th scope="row">All graded</th><td></td><td class="num">{ph} of {pt}</td>'
             f'<td class="num">{units:+.1f}u</td><td></td></tr>') if pt else ""
    return "".join(rows) + total


STYLE = """<style>
.hero{display:grid;gap:8px}
.hero h1{font-size:clamp(2.6rem,8vw,4rem);font-weight:700;text-transform:uppercase;letter-spacing:.01em}
.hero h1 span{color:var(--lar-strong)}
.games{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px}
.game{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;display:grid;gap:8px;align-content:start;min-width:0}
.game header{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.game h3{font-size:2.2rem;font-weight:700}
.game .tchip{color:var(--fg)}
.game .nm{color:var(--muted);font-weight:600}
.game p{margin:0}
.fine{color:var(--muted);font-size:.85rem}
.tag{display:inline-block;font-size:.75rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:2px 9px;border-radius:999px;white-space:nowrap}
.tag.t2{color:var(--t1);background:var(--t1bg)} .tag.t3{color:var(--t3);background:var(--t3bg)} .tag.t4{color:var(--t4);background:var(--t4bg)}
.open{justify-self:start;margin-top:4px;font-family:var(--display);font-weight:700;font-size:1.05rem;letter-spacing:.05em;text-transform:uppercase;color:var(--surface);background:var(--fg);padding:6px 14px;border-radius:6px;text-decoration:none}
.open:hover{background:var(--lar)}
.rt{min-width:620px}
.rt .num{font-weight:700;white-space:nowrap}
.rt .muted{color:var(--muted)}
.rt tr.tot th,.rt tr.tot td{border-top:2px solid var(--fg)}
</style>
"""


def index_page():
    upcoming = [g for g in G if g["status"] == "pending"]
    graded = [g for g in G if g["status"] == "graded"]
    sec = lambda h, gs: (f'<section aria-labelledby="h-{h[0]}"><h2 id="h-{h[0]}">{h}</h2>'
                         f'<div class="games">{"".join(card(g) for g in gs)}</div></section>') if gs else ""
    return (HEAD + STYLE + '<main class="wrap">\n'
            '<header class="hero"><span class="eyebrow">2026 season · Game reports and pick tracking</span>'
            '<h1>NFL Matchup <span>Lab</span></h1>'
            '<p class="sub">One page per game: unit ranks, coverage, 1-on-1 matchups, a props board and PrizePicks '
            'lineups. Picks are locked in git before kickoff and graded after the game.</p></header>\n'
            + sec("Upcoming", upcoming) + sec("Graded", graded) +
            '<section aria-labelledby="h-rec"><div><h2 id="h-rec">Track record</h2>'
            '<p class="sub">Units assume 1 unit per prop at the approximate price listed in each results file. '
            'A handful of picks says almost nothing yet about whether the method works.</p></div>'
            '<div class="tbl-wrap"><table class="pt rt"><thead><tr><th scope="col">Game</th>'
            '<th scope="col">Game lines</th><th scope="col">Props hit</th><th scope="col">Units</th>'
            '<th scope="col">Test</th></tr></thead><tbody>' + record_rows() + '</tbody></table></div></section>\n'
            '<footer class="notes"><p>Built from <code>nfl-matchup-lab/games.json</code> by <code>src/site.py</code>. '
            'Each report lists its own sources at the bottom. These are estimates from small samples, '
            'not betting advice.</p></footer>\n</main>\n')


if __name__ == "__main__":
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "games"))
    for g in G:
        if g.get("report"):
            open(os.path.join(SITE, "games", g["slug"] + ".html"), "w").write(game_page(g))
    open(os.path.join(SITE, "index.html"), "w").write(index_page())
    print("wrote", SITE)
