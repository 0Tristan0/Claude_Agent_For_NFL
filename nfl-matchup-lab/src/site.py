"""Build the website: site/index.html (games), site/guesser.html (paper bankroll) and one page per game.

index.html is a page fragment (the Artifact publisher wraps it in <html>/<head>/<body>).
Game pages and the Guesser page are published as extra files, so they get a full document here.
Every page shares the look in nfl_lab/report/theme.py.
Run after the report scripts:  python3 src/site.py
"""
import glob, html, json, os, re, shutil, subprocess, sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfl_lab.report import theme  # noqa: E402

SITE = os.path.join(ROOT, "site")
G = json.load(open(os.path.join(ROOT, "games.json")))["games"]
V0_DIR = os.path.join(ROOT, "output", "v0")
LIVE_DIR = os.path.join(ROOT, "backtests", "live")
e = html.escape


def fmt_date(s):
    d = date.fromisoformat(s)
    return d.strftime("%a %b ") + str(d.day)


def money(x):
    return f"${x:,.2f}"


# ---- older Bills-Rams report ----------------------------------------------------------------------

def game_page(g):
    """The locked legacy report in the site theme. Only its look changes; its content is the frozen record."""
    src = open(os.path.join(ROOT, g["report"])).read()
    body = src[src.index("<main"):]
    body = re.sub(r'(<main class="wrap">)', lambda m: m.group(1) + "\n" + theme.nav("games", "../"), body, count=1)
    body = re.sub(r'<span class="ab">([A-Z]{2,3})</span>', lambda m: theme.pennant(m.group(1)), body)
    return theme.document(f'{g["away"]} at {g["home"]}, Week {g["week"]} report', body)


def locked_card(g):
    meta = f'Week {g["week"]} · {e(g["label"])} · {fmt_date(g["date"])}'
    if g["status"] == "pending":
        tag = '<span class="tag t3">Picks locked</span>'
        body = (f'<p>{e(g["picks"])}</p>'
                f'<p class="fine">Frozen {e(g["frozen_at"])} in commit {e(g["frozen_commit"])}, before kickoff.</p>')
    else:
        tag = '<span class="tag t1">Graded</span>'
        body = f'<p>Final: {e(g["final"])}. Props {g["props"]["hit"]} of {g["props"]["total"]}.</p>'
        if not g.get("blind", True):
            body += '<p class="fine">Not a blind test: stats loaded for it already included this game.</p>'
    link = (f'<a class="open" href="games/{g["slug"]}.html">Open report</a>' if g.get("report")
            else '<span class="fine">No full report page for this one</span>')
    return (f'<article class="game"><header><span class="eyebrow">{meta}</span>{tag}</header>'
            f'<h3>{theme.swatch(g["away"])}{e(g["away"])} <i>at</i> {theme.swatch(g["home"])}{e(g["home"])}</h3>'
            f'<span class="nm">{e(g["away_name"])} at {e(g["home_name"])}</span>{body}{link}</article>')


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


# ---- model v0 pages ---------------------------------------------------------------------------------

def v0_pages():
    """Model v0 game pages (output/v0/<season>-w<week>-<away>-<home>.html), newest week first."""
    if not os.path.isdir(V0_DIR):
        return []
    out = []
    for name in sorted(os.listdir(V0_DIR)):
        m = re.match(r"(\d{4})-w(\d{2})-([a-z]+)-([a-z]+)\.html$", name)
        if m:
            meta_path = os.path.join(V0_DIR, name[:-5] + ".json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            out.append(dict(file=name, season=int(m[1]), week=int(m[2]), away=m[3].upper(), home=m[4].upper(), meta=meta))
    return sorted(out, key=lambda d: (-d["season"], -d["week"], d["meta"].get("kickoff_utc", ""), d["file"]))


def ticket(p):
    m = p["meta"]
    a, h = p["away"], p["home"]
    if m:
        day, _, rest = m["kickoff_et"].partition(" ")
        clock = rest.split(", ")[-1].replace(" ET", "")
        fav = h if m["home_spread"] < 0 else a
        spread = f'{fav} {-abs(m["home_spread"]):+g}' if m["home_spread"] else "Pick'em"
        hw = m["home_win"]
        lead = f'{h} {hw:.0%}' if hw >= 0.5 else f'{a} {1 - hw:.0%}'
        lines = (f'<span class="meta">{e(spread)} · O/U {m["total"]:g}</span>'
                 f'<span class="meta">Sim win {e(lead)} · median {a} {m["median_score"][0]:.0f}, {h} {m["median_score"][1]:.0f}</span>')
    else:
        day, clock, lines = "", "", ""
    return (f'<a class="ticket" href="games/v0-{p["file"]}"><span class="stub"><span class="wk">Wk {p["week"]}</span>'
            f'<span class="day">{e(day)}</span><span class="time">{e(clock)}</span></span>'
            f'<span class="main"><span class="match">{theme.swatch(a)}<b style="{theme.team_vars(a)}">{e(a)}</b> <i>at</i> '
            f'{theme.swatch(h)}<b style="{theme.team_vars(h)}">{e(h)}</b></span>{lines}</span></a>')


def v0_section():
    pages = v0_pages()
    if not pages:
        return ""
    weeks = {}
    for p in pages:
        weeks.setdefault((p["season"], p["week"]), []).append(p)
    blocks = []
    for (season, week), ps in weeks.items():
        blocks.append(f'<div class="week"><span class="eyebrow">{season} · Week {week} · {len(ps)} '
                      f'{"game" if len(ps) == 1 else "games"}</span><div class="tickets">{"".join(ticket(p) for p in ps)}</div></div>')
    return ('<section aria-labelledby="h-v0"><div><h2 id="h-v0">Model projections</h2>'
            '<p class="sub">Model v0 simulates each game 20,000 times: player projections with ranges, model chances for '
            'the lines in data/props.csv, and lineups built from the simulated games. Each page shows how well calibrated '
            'it was on the 2024 season.</p></div>' + "".join(blocks) + '</section>\n')


INDEX_CSS = """
.hero{display:grid;gap:10px;padding-block:8px 4px}
.hero h1{font-size:clamp(2.8rem,9vw,5.2rem);font-weight:600;letter-spacing:-.02em;line-height:1}
.hero h1 em{font-style:italic;font-weight:500;color:var(--brass)}
.week{display:grid;gap:12px}
.tickets{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),1fr));gap:16px}
.ticket{position:relative;display:grid;grid-template-columns:78px 1fr;background:var(--panel);border:1px solid var(--rule);
  border-radius:4px;color:var(--ivory);text-decoration:none;transition:border-color .15s,transform .15s;min-width:0}
.ticket:hover,.ticket:focus-visible{border-color:var(--brass-dim);color:var(--ivory);transform:translateY(-1px)}
.ticket::before,.ticket::after{content:"";position:absolute;left:71px;width:14px;height:14px;border-radius:50%;background:var(--ground)}
.ticket::before{top:-8px} .ticket::after{bottom:-8px}
.stub{display:grid;align-content:center;justify-items:center;gap:1px;padding:12px 6px;border-right:1.5px dashed var(--rule-hi);text-align:center}
.stub .wk{font-size:.6rem;letter-spacing:.17em;text-transform:uppercase;color:var(--brass);font-weight:600}
.stub .day{font-family:var(--serif);font-size:1.1rem;font-weight:600}
.stub .time{font-size:.76rem;color:var(--sage);font-variant-numeric:tabular-nums}
.main{display:grid;gap:3px;padding:12px 16px;min-width:0}
.match{font-family:var(--varsity);font-size:1.3rem;letter-spacing:.02em;line-height:1.25}
.match b{font-weight:400;color:var(--ti)}
.match i,.game h3 i{font-family:var(--serif);font-style:italic;font-size:.85em;color:var(--brass);margin-inline:2px}
.sw{display:inline-block;width:16px;height:9px;margin-right:6px;background:var(--tc);clip-path:polygon(0 0,100% 50%,0 100%);vertical-align:2px}
.meta{font-size:.82rem;color:var(--sage);font-variant-numeric:tabular-nums}
.games{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px}
.game{background:var(--panel);border:1px solid var(--rule);border-top:2px solid var(--brass-dim);border-radius:3px;padding:18px;display:grid;gap:10px;align-content:start;min-width:0}
.game header{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.game h3{font-family:var(--varsity);font-weight:400;font-size:2rem;letter-spacing:.02em}
.game .nm{color:var(--sage);font-family:var(--serif);font-size:1.05rem}
.fine{color:var(--sage);font-size:.84rem}
.tag{display:inline-block;font-size:.64rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;padding:2px 8px;border-radius:2px;white-space:nowrap}
.tag.t1{color:var(--t1);background:var(--t1bg)} .tag.t3{color:var(--t3);background:var(--t3bg)} .tag.t4{color:var(--t4);background:var(--t4bg)}
.open{justify-self:start;margin-top:4px;font-size:.74rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--ground);
  background:var(--brass);padding:8px 14px;border-radius:2px;text-decoration:none}
.open:hover{background:var(--ivory);color:var(--ground)}
.rt{min-width:620px}
.rt .num{font-weight:700;white-space:nowrap}
.rt .muted{color:var(--sage)}
.rt tr.tot th,.rt tr.tot td{border-top:3px double var(--brass-dim)}
"""


def index_page():
    upcoming = [g for g in G if g["status"] == "pending"]
    graded = [g for g in G if g["status"] == "graded"]
    sec = lambda key, h, sub, gs: (f'<section aria-labelledby="h-{key}"><div><h2 id="h-{key}">{h}</h2>'
                                   f'<p class="sub">{sub}</p></div>'
                                   f'<div class="games">{"".join(locked_card(g) for g in gs)}</div></section>') if gs else ""
    pages = v0_pages()
    week = f'Week {pages[0]["week"]}' if pages else "2026 season"
    return (theme.head("NFL Matchup Lab") + f"<style>{INDEX_CSS}</style>\n" + '<main class="wrap">\n' + theme.nav("games") +
            f'<header class="hero"><span class="eyebrow">2026 season · {week}</span>'
            '<h1>NFL Matchup <em>Lab</em></h1>'
            '<p class="sub">Game-by-game projections from a simulation model, the picks locked in before kickoff, and how '
            'they graded afterward. Estimates with real uncertainty, never sure things.</p></header>\n'
            + v0_section()
            + sec("up", "Locked picks", "Hand-built reports whose picks were committed to git before kickoff.", upcoming)
            + sec("gr", "Graded", "Games that have been played and scored against the locked picks.", graded) +
            '<section aria-labelledby="h-rec"><div><h2 id="h-rec">Track record</h2>'
            '<p class="sub">Units assume 1 unit per prop at the approximate price listed in each results file. '
            'A handful of picks says almost nothing yet about whether the method works.</p></div>'
            '<div class="tbl-wrap"><table class="rt"><thead><tr><th scope="col">Game</th>'
            '<th scope="col">Game lines</th><th scope="col">Props hit</th><th scope="col">Units</th>'
            '<th scope="col">Test</th></tr></thead><tbody>' + record_rows() + '</tbody></table></div></section>\n'
            '<footer class="notes"><p>Built from <code>nfl-matchup-lab/games.json</code> and the model pages by '
            '<code>src/site.py</code>. Each report lists its own sources at the bottom. These are estimates from small '
            'samples, not betting advice.</p></footer>\n</main>\n')


# ---- the Guesser: paper bankroll on the locked picks ------------------------------------------------

def bankrolls():
    out = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "*-bankroll.json"))):
        b = json.load(open(path))
        b["_path"] = os.path.relpath(path, ROOT)
        out.append(b)
    return out


def grade_entry(entry, result):
    """Return (status, amount returned or None). Status: pending, won, lost."""
    over = ((result or {}).get("entries") or {}).get(entry["id"], {})
    if "returned" in over:
        r = float(over["returned"])
        return ("won" if r > 0 else "lost"), r
    legs = (result or {}).get("legs") or {}
    if not all(l in legs for l in entry["legs"]):
        return "pending", None
    k = sum(bool(legs[l]["hit"]) for l in entry["legs"])
    mult = float(entry["pays"].get(str(k), 0)) if entry["type"] == "flex" else (
        float(entry["pays"][str(len(entry["legs"]))]) if k == len(entry["legs"]) else 0.0)
    r = entry["stake"] * mult
    return ("won" if r > 0 else "lost"), r


def leg_mark(leg, result):
    g = ((result or {}).get("legs") or {}).get(leg)
    if g is None:
        return '<span class="mk pend" title="Not graded yet">&#9675;</span>'
    actual = f' <span class="act">({e(str(g.get("actual")))})</span>' if g.get("actual") is not None else ""
    return ('<span class="mk hit" title="Hit">&#10003;</span>' if g["hit"] else
            '<span class="mk miss" title="Missed">&#10005;</span>'), actual


def kind_label(k):
    """'flex3' -> '3-pick Flex'."""
    m = re.match(r"([a-z]+)(\d+)$", k)
    return f"{m[2]}-pick {m[1].title()}" if m else k


def slip(b):
    res = b.get("result")
    away, home = b["game"].split("_")[2], b["game"].split("_")[3]
    away_d, home_d = ("LAR" if away == "LA" else away), ("LAR" if home == "LA" else home)
    entries_html, staked, back, pending = [], 0.0, 0.0, False
    for en in b["entries"]:
        status, r = grade_entry(en, res)
        staked += en["stake"]
        pending |= status == "pending"
        back += r or 0.0
        legs = []
        for leg in en["legs"]:
            mk = leg_mark(leg, res)
            mark, actual = (mk, "") if isinstance(mk, str) else mk
            legs.append(f"<li>{mark}<span>{e(leg)}{actual}</span></li>")
        n = len(en["legs"])
        pays = (" · ".join(f'{k} of {n} pays {money(en["stake"] * float(v))}' for k, v in
                           sorted(en["pays"].items(), key=lambda kv: -int(kv[0]))))
        outcome = {"pending": '<span class="tag t3">Pending</span>',
                   "won": f'<span class="tag t1">Returned {money(r or 0)}</span>',
                   "lost": '<span class="tag t4">Lost</span>'}[status]
        entries_html.append(f'<article class="entry"><header><h3>Entry {e(en["id"])}<span class="kind">{n}-pick '
                            f'{en["type"].title()}</span></h3><span class="stake num">{money(en["stake"])}</span></header>'
                            f'<ul class="legs">{"".join(legs)}</ul><footer><span class="fine">{pays}</span>{outcome}</footer></article>')
    profit = None if pending else back - staked
    ev = b.get("expected_return_per_dollar", {})
    ev_rows = "".join(
        f'<tr><th scope="row">Entry {e(k.split("_")[0])}<span class="vs-lbl">{e(kind_label(k.split("_")[1]))}</span></th>'
        f'<td class="num">{v["legacy_estimates"]:.2f}</td><td class="num">{v["market_no_vig"]:.2f}</td>'
        f'<td class="num">{v["model_v0"]:.2f}</td></tr>' for k, v in ev.items() if isinstance(v, dict))
    head_tag = ('<span class="tag t3">Waiting on the game</span>' if pending else
                (f'<span class="tag t1">Up {money(profit)}</span>' if profit > 0 else
                 f'<span class="tag t4">Down {money(-profit)}</span>' if profit < 0 else '<span class="tag t3">Even</span>'))
    final = f' · Final {e(res["final"])}' if res and res.get("final") else ""
    return dict(staked=staked, back=back, pending=pending, html=f"""
<section class="slip" aria-labelledby="h-{e(b['game'])}">
<header class="slip-head"><div><span class="eyebrow">Week {e(b['game'].split('_')[1].lstrip('0'))} · kickoff {e(b['kickoff_utc'])}{final}</span>
<h2 id="h-{e(b['game'])}">{theme.swatch(away_d)}{e(away_d)} <i>at</i> {theme.swatch(home_d)}{e(home_d)}</h2></div>{head_tag}</header>
<p class="sub">Bankroll {money(b['bankroll'])}, all of it in play. Frozen {e(b['frozen_at'])}, before kickoff. Picks: {e(b['picks_source'])}.</p>
<div class="entries">{''.join(entries_html)}</div>
<div class="tbl-wrap"><table class="pt evt"><caption>What each view expected back per $1, before kickoff</caption><thead><tr>
<th scope="col">Entry</th><th scope="col">Locked estimates</th><th scope="col">No-vig market</th><th scope="col">Model v0</th></tr></thead>
<tbody>{ev_rows}</tbody></table></div>
<p class="fine">Above $1.00 means the view expected a profit. The locked estimates mixed a 4-game sample with market prices; the model
and the market both expected a loss. Legs are treated as independent here. Record: <code>{e(b['_path'])}</code>.</p>
</section>""")


GUESSER_CSS = """
.hero{display:grid;gap:10px;padding-block:8px 4px}
.hero h1{font-size:clamp(2.8rem,9vw,5rem);font-weight:600;letter-spacing:-.02em;line-height:1}
.hero h1 em{font-style:italic;font-weight:500;color:var(--brass)}
.ledger{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));border:1px solid var(--rule-hi);border-radius:3px;
  background:var(--panel);box-shadow:inset 0 0 0 5px var(--panel),inset 0 0 0 6px color-mix(in srgb,var(--brass) 35%,transparent);padding:10px}
.ledger div{display:grid;gap:4px;padding:14px 16px}
.ledger dt{font-size:.64rem;letter-spacing:.16em;text-transform:uppercase;color:var(--sage);font-weight:600}
.ledger dd{margin:0;font-family:var(--serif);font-size:clamp(1.7rem,4.5vw,2.4rem);font-weight:600;line-height:1.05;font-variant-numeric:lining-nums tabular-nums}
.ledger dd.total{text-decoration:underline double var(--brass-dim);text-underline-offset:7px;text-decoration-thickness:1px}
.ledger dd.up{color:var(--pine)} .ledger dd.down{color:var(--claret)} .ledger dd.wait{color:var(--sage)}
.slip{display:grid;gap:18px}
.slip h2::before{display:none}
.ledger-wrap{display:grid;gap:10px}
.slip-head{display:flex;justify-content:space-between;align-items:end;gap:12px;flex-wrap:wrap}
.slip h2 i{font-style:italic;font-weight:500;color:var(--brass);font-size:.8em}
.slip .sw{display:inline-block;width:20px;height:11px;margin-right:8px;background:var(--tc);clip-path:polygon(0 0,100% 50%,0 100%);vertical-align:4px}
.entries{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px}
.entry{background:var(--panel);border:1px solid var(--rule);border-radius:3px;display:grid;align-content:start;min-width:0}
.entry header{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:16px 18px 12px;border-bottom:1.5px dashed var(--rule-hi)}
.entry h3{font-size:1.35rem;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.entry .kind{font-family:var(--sans);font-size:.66rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--brass)}
.entry .stake{font-family:var(--serif);font-size:1.6rem;font-weight:600}
.legs{list-style:none;margin:0;padding:12px 18px;display:grid;gap:8px}
.legs li{display:grid;grid-template-columns:22px 1fr;gap:8px;align-items:baseline}
.mk{font-weight:700;text-align:center}
.mk.pend{color:var(--sage)} .mk.hit{color:var(--pine)} .mk.miss{color:var(--claret)}
.act{color:var(--sage);font-size:.85rem}
.entry footer{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 18px 16px;border-top:1px solid var(--rule)}
.fine{color:var(--sage);font-size:.84rem;max-width:78ch}
.tag{display:inline-block;font-size:.64rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;padding:3px 8px;border-radius:2px;white-space:nowrap}
.tag.t1{color:var(--t1);background:var(--t1bg)} .tag.t3{color:var(--t3);background:var(--t3bg)} .tag.t4{color:var(--t4);background:var(--t4bg)}
.evt{min-width:520px}
.empty{color:var(--sage)}
"""


def guesser_page():
    bs = bankrolls()
    slips = [slip(b) for b in bs]
    start = sum(b["bankroll"] for b in bs)
    staked = sum(s["staked"] for s in slips)
    back = sum(s["back"] for s in slips if not s["pending"])
    done_staked = sum(s["staked"] for s in slips if not s["pending"])
    any_pending = any(s["pending"] for s in slips)
    profit = back - done_staked
    if not slips:
        now, now_cls, prof, prof_cls = money(0), "wait", "None yet", "wait"
    else:
        bal = start - staked + back + (sum(s["staked"] for s in slips if s["pending"]) if any_pending else 0)
        now = money(bal) + (" *" if any_pending else "")
        now_cls = "total"
        prof = (f"+{money(profit)}" if profit > 0 else f"-{money(-profit)}" if profit < 0 else money(0)) if done_staked else "Pending"
        prof_cls = "up" if profit > 0 and done_staked else "down" if profit < 0 else "wait"
    pend_note = ('<p class="fine">* Counts stakes on games not graded yet at face value until they are graded.</p>'
                 if any_pending else "")
    body = ('<main class="wrap">\n' + theme.nav("guesser") +
            '<header class="hero"><span class="eyebrow">Paper bankroll · pretend money</span><h1>The <em>Guesser</em></h1>'
            '<p class="sub">Before a game, a pretend $20 goes on the picks that were locked in before kickoff, split into '
            'PrizePicks-style entries. After the game each leg gets graded and this page keeps the running score.</p></header>\n'
            f'<div class="ledger-wrap"><dl class="ledger"><div><dt>Started with</dt><dd>{money(start)}</dd></div>'
            f'<div><dt>Put in play</dt><dd>{money(staked)}</dd></div>'
            f'<div><dt>Paid back (graded)</dt><dd class="{"wait" if not done_staked else ""}">'
            f'{money(back) if done_staked else "Pending"}</dd></div>'
            f'<div><dt>Profit</dt><dd class="{prof_cls}">{prof}</dd></div>'
            f'<div><dt>Bankroll now</dt><dd class="{now_cls}">{now}</dd></div></dl>{pend_note}</div>\n'
            + ("".join(s["html"] for s in slips) or '<p class="empty">No games yet.</p>') +
            '\n<footer class="notes"><p>Payouts use the standard PrizePicks tables in config/payouts.toml (Power pays only '
            'when every leg hits; Flex pays on most legs). The app can offer lower multipliers on same-game entries; '
            'those would be noted on the slip. Pretend money, small samples, and not betting advice.</p></footer>\n</main>\n')
    return theme.document("The Guesser", body, GUESSER_CSS)


if __name__ == "__main__":
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "games"))
    for g in G:
        if g.get("report"):
            open(os.path.join(SITE, "games", g["slug"] + ".html"), "w").write(game_page(g))
    for p in v0_pages():
        doc = open(os.path.join(V0_DIR, p["file"])).read()
        open(os.path.join(SITE, "games", "v0-" + p["file"]), "w").write(theme.refresh(doc))
    open(os.path.join(SITE, "index.html"), "w").write(index_page())
    open(os.path.join(SITE, "guesser.html"), "w").write(guesser_page())
    print("wrote", SITE)
