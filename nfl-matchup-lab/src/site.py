"""Build the website: site/index.html (games), site/guesser.html (paper bankroll) and one page per game.

index.html is a page fragment (the Artifact publisher wraps it in <html>/<head>/<body>).
Game pages and the Guesser page are published as extra files, so they get a full document here.
Every page shares the look in nfl_lab/report/theme.py; team logos are copied to site/logos.
Run after the report scripts:  python3 src/site.py
"""
import glob, html, json, os, re, shutil, sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nfl_lab.report import theme  # noqa: E402

SITE = os.path.join(ROOT, "site")
G = json.load(open(os.path.join(ROOT, "games.json")))["games"]
V0_DIR = os.path.join(ROOT, "output", "v0")
LIVE_DIR = os.path.join(ROOT, "backtests", "live")
ET = ZoneInfo("America/New_York")
e = html.escape


def fmt_date(s):
    d = date.fromisoformat(s)
    return d.strftime("%a %b ") + str(d.day)


def money(x):
    return f"${x:,.2f}"


def nickname(code):
    import csv
    with open(os.path.join(ROOT, "config", "teams.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if code in (r["team"], r["display"]):
                return r["name"].split()[-1]
    return code


# ---- older Bills-Rams report ----------------------------------------------------------------------

def game_page(g):
    """The locked legacy report in the site theme. Only its look changes; its content is the frozen record."""
    src = open(os.path.join(ROOT, g["report"])).read()
    body = src[src.index("<main"):]
    body = re.sub(r'(<main class="wrap">)', lambda m: m.group(1) + "\n" + theme.nav("home", "../"), body, count=1)
    body = body.replace('<div class="board" aria-label="Game info">',
                        '<div class="board" aria-label="Game info"><a class="back" href="../index.html">&larr; All games</a>', 1)
    body = re.sub(r'<span class="ab">([A-Z]{2,3})</span>',
                  lambda m: f'<span class="tm">{theme.logo(m.group(1), 52, "../")}<span class="ab">{m.group(1)}</span></span>', body)
    body = re.sub(r'<section aria-labelledby="(h-[\w-]+)"( id="[^"]+")?>', lambda m: legacy_pane(m), body)
    return theme.document(f'{g["away"]} at {g["home"]}, Week {g["week"]} report', body)


LEGACY_TABS = {"h-big": ("summary", "Summary"), "h-unit": ("units", "Unit vs unit"), "h-cov": ("coverage", "Coverage"),
               "h-play": ("players", "Players"), "h-1v1": ("matchups", "1-on-1"), "h-props": ("props", "Props"),
               "h-pay": ("payouts", "Payouts"), "h-lu": ("lineups", "Lineups")}


def legacy_pane(m):
    key = m.group(1)
    pid, label = LEGACY_TABS.get(key, (key[2:], key[2:].title()))
    return f'<section class="pane" id="{pid}" data-tab="{label}" aria-labelledby="{key}">'


def matchup(away, home, size=22, prefix=""):
    return (f'<span class="mu2">{theme.logo(away, size, prefix)}<b>{e(away)}</b><span class="at2">at</span>'
            f'{theme.logo(home, size, prefix)}<b>{e(home)}</b></span>')


def locked_row(g):
    if g["status"] == "pending":
        tag = '<span class="tag">Picks locked</span>'
        body = (f'<p>{e(g["picks"])}</p>'
                f'<p class="fine">Frozen {e(g["frozen_at"])} in commit {e(g["frozen_commit"])}, before kickoff.</p>')
    else:
        tag = '<span class="tag">Graded</span>'
        body = f'<p>Final: {e(g["final"])}. Props {g["props"]["hit"]} of {g["props"]["total"]}.</p>'
        if not g.get("blind", True):
            body += '<p class="fine">Not a blind test: stats loaded for it already included this game.</p>'
    link = (f'<a class="more" href="games/{g["slug"]}.html">Open report &rarr;</a>' if g.get("report")
            else '<span class="fine">No full report page for this one</span>')
    return (f'<article class="lock"><header>{matchup(g["away"], g["home"])}{tag}</header>'
            f'<span class="fine">Week {g["week"]} · {e(g["label"])} · {fmt_date(g["date"])}</span>{body}{link}</article>')


def record_rows():
    rows, ph, pt, units = [], 0, 0, 0.0
    for g in G:
        name = f'{e(g["away"])} at {e(g["home"])}<span class="vs-lbl">{fmt_date(g["date"])}</span>'
        if g["status"] != "graded":
            rows.append(f'<tr><th scope="row">{name}</th><td colspan="3" class="muted">Waiting on the game</td>'
                        f'<td class="muted">Clean test</td></tr>')
            continue
        gl, p = g["game_lines"], g["props"]
        ph, pt, units = ph + p["hit"], pt + p["total"], units + p["units"]
        blind = '<span class="muted">Clean test</span>' if g.get("blind", True) else '<span class="neg">Not blind</span>'
        rows.append(f'<tr><th scope="row">{name}</th><td class="num">{gl["hit"]} of {gl["total"]}'
                    f'<span class="vs-lbl">{e(gl["note"])}</span></td><td class="num">{p["hit"]} of {p["total"]}</td>'
                    f'<td class="num">{p["units"]:+.1f}u</td><td>{blind}</td></tr>')
    total = (f'<tr class="tot"><th scope="row">All graded</th><td></td><td class="num">{ph} of {pt}</td>'
             f'<td class="num">{units:+.1f}u</td><td></td></tr>') if pt else ""
    return "".join(rows) + total


# ---- model v0 pages ---------------------------------------------------------------------------------

def v0_pages():
    """Model v0 game pages (output/v0/<season>-w<week>-<away>-<home>.html), newest week first, then by kickoff."""
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


def kickoff(p):
    k = p["meta"].get("kickoff_utc")
    return datetime.fromisoformat(k).astimezone(ET) if k else None


def status(p, now):
    """Status at build time; the page's script recomputes it when it is viewed."""
    k = kickoff(p)
    if k is None or now < k:
        return "up"
    return "live" if now < k + timedelta(hours=4) else "final"


STATUS_TXT = {"up": "", "live": "Live", "final": "Final"}


def kick_attr(p):
    k = p["meta"].get("kickoff_utc")
    return f' data-kick="{e(datetime.fromisoformat(k).astimezone(timezone.utc).isoformat())}"' if k else ""


def line_text(m, a, h):
    if not m:
        return ""
    fav = h if m["home_spread"] < 0 else a
    spread = f'{fav} {-abs(m["home_spread"]):+g}' if m["home_spread"] else "Pick'em"
    return f'{e(spread)}<span class="muted"> · </span>O/U {m["total"]:g}'


def game_row(p, now):
    m, a, h = p["meta"], p["away"], p["home"]
    k = kickoff(p)
    st = status(p, now)
    sides = ([(a, m["away_implied"], 1 - m["home_win"]), (h, m["home_implied"], m["home_win"])] if m
             else [(a, None, None), (h, None, None)])
    trs = "".join(
        f'<span class="tr{" fav" if w is not None and w >= 0.5 else ""}">{theme.logo(t, 24)}<b>{e(t)}</b>'
        f'<span class="tn">{e(nickname(t))}</span><span class="ip">{"" if ip is None else f"{ip:.1f}"}</span>'
        f'<span class="wn">{"" if w is None else f"{w:.0%}"}</span></span>' for t, ip, w in sides)
    return (f'<a class="gm" href="games/v0-{p["file"]}"{kick_attr(p)} data-status="{st}"><span class="time">'
            f'{k.strftime("%-I:%M %p") if k else ""}<span class="st">{STATUS_TXT[st]}</span></span>'
            f'<span class="trs">{trs}</span><span class="ln">{line_text(m, a, h)}</span><span class="go" aria-hidden="true">&rsaquo;</span></a>')


def chip(p, now):
    m, a, h = p["meta"], p["away"], p["home"]
    k = kickoff(p)
    st = status(p, now)
    when = k.strftime("%a %-I:%M %p") if k else ""
    return (f'<a class="chip" href="games/v0-{p["file"]}"{kick_attr(p)} data-status="{st}">'
            f'<span class="when">{e(when)}<span class="st">{STATUS_TXT[st]}</span></span>'
            f'<span class="pair">{theme.logo(a, 22)}<b>{e(a)}</b></span><span class="pair">{theme.logo(h, 22)}<b>{e(h)}</b></span>'
            f'<span class="ln">{line_text(m, a, h)}</span></a>')


def game_list(pages, now, newest_first):
    ordered = sorted(pages, key=lambda p: p["meta"].get("kickoff_utc", ""), reverse=newest_first)
    days = {}
    for p in ordered:
        k = kickoff(p)
        label = f'{k.strftime("%A, %B %-d")} · Week {p["week"]}' if k else f'Week {p["week"]}'
        days.setdefault(label, []).append(p)
    return "".join(
        f'<div class="day"><div class="dayh"><span>{e(d)}</span><span class="cols"><span>Pts</span><span>Win</span>'
        f'<span class="lnh">Line</span><span></span></span></div><div class="list">{"".join(game_row(p, now) for p in dps)}</div></div>'
        for d, dps in days.items())


def games_sections(now):
    pages = v0_pages()
    if not pages:
        return ""
    nxt = sorted([p for p in pages if status(p, now) != "final"], key=lambda p: p["meta"].get("kickoff_utc", ""))
    up = (f'<section aria-labelledby="h-next"><div class="secth"><h2 id="h-next">Up next</h2>'
          f'<span class="muted">Tap a game to open it</span></div><div class="next">{"".join(chip(p, now) for p in nxt)}</div></section>\n'
          if nxt else "")
    return (up + '<section aria-labelledby="h-all"><div class="secth"><h2 id="h-all">All games</h2>'
            '<div class="sort" role="group" aria-label="Order" hidden><button type="button" data-o="desc" aria-pressed="true">Latest first</button>'
            '<button type="button" data-o="asc" aria-pressed="false">Soonest first</button></div></div>'
            f'<div id="list-desc">{game_list(pages, now, True)}</div>'
            f'<div id="list-asc" hidden>{game_list(pages, now, False)}</div></section>\n')


INDEX_JS = """
(function(){
  var now=Date.now(), TXT={up:'',live:'Live',final:'Final'};
  [].forEach.call(document.querySelectorAll('[data-kick]'),function(el){
    var k=Date.parse(el.getAttribute('data-kick')); if(isNaN(k)) return;
    var s=now<k?'up':(now<k+4*3600e3?'live':'final');
    el.setAttribute('data-status',s);
    var b=el.querySelector('.st'); if(b) b.textContent=TXT[s];
    if(el.classList.contains('chip')) el.hidden=s==='final';
  });
  var next=document.querySelector('.next');
  if(next && !next.querySelector('.chip:not([hidden])')) next.closest('section').hidden=true;
  var tg=document.querySelector('.sort'); if(!tg) return;
  var lists={desc:document.getElementById('list-desc'),asc:document.getElementById('list-asc')};
  function set(o){
    lists.desc.hidden=o!=='desc'; lists.asc.hidden=o!=='asc';
    [].forEach.call(tg.querySelectorAll('button'),function(b){b.setAttribute('aria-pressed',b.getAttribute('data-o')===o?'true':'false');});
    try{localStorage.setItem('lab-sort',o);}catch(e){}
  }
  var saved='desc'; try{saved=localStorage.getItem('lab-sort')||'desc';}catch(e){}
  [].forEach.call(tg.querySelectorAll('button'),function(b){b.addEventListener('click',function(){set(b.getAttribute('data-o'));});});
  tg.hidden=false; set(saved==='asc'?'asc':'desc');
})();
"""


INDEX_CSS = """
.hero{display:grid;gap:8px}
.secth{display:flex;justify-content:space-between;align-items:center;gap:8px 16px;flex-wrap:wrap}
.secth .muted{font-size:.85rem}
.next{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:8px}
@media (max-width:640px){.next{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;padding-bottom:4px;scrollbar-width:none}
  .next::-webkit-scrollbar{display:none}.chip{flex:0 0 148px;scroll-snap-align:start}}
.chip{display:grid;gap:6px;padding:12px;background:var(--surface);border:1px solid var(--line);border-radius:8px;color:var(--text);text-decoration:none}
.chip:hover,.chip:focus-visible{border-color:var(--line-2);background:var(--raise)}
.chip .when{display:flex;justify-content:space-between;gap:6px;font-size:.76rem;color:var(--muted)}
.chip .pair{display:flex;align-items:center;gap:8px}
.chip b{font-weight:700;font-stretch:85%}
.chip .ln{font-size:.76rem;color:var(--muted)}
.st{font-weight:650;color:var(--text)}
[data-status="live"] .st{color:var(--pos)}
[data-status="final"] .st{color:var(--muted)}
.gm .st{display:block;font-size:.72rem;margin-top:2px}
.sort{display:inline-flex;border:1px solid var(--line-2);border-radius:999px;padding:2px}
.sort button{appearance:none;border:0;background:none;color:var(--muted);font:inherit;font-size:.8rem;padding:4px 12px;border-radius:999px;cursor:pointer}
.sort button[aria-pressed="true"]{background:var(--raise);color:var(--text)}
.day{display:grid;margin-top:8px}
.dayh{display:flex;justify-content:space-between;align-items:end;gap:12px;font-size:.75rem;color:var(--muted);padding-bottom:8px;border-bottom:1px solid var(--line-2)}
.cols{display:grid;grid-template-columns:44px 44px 150px 16px;gap:0 10px;text-align:right;flex:none}
.list{display:grid}
.gm{display:grid;grid-template-columns:72px 1fr 150px 16px;align-items:center;gap:0 10px;padding:12px 0;border-bottom:1px solid var(--line);
  color:var(--text);text-decoration:none}
.gm:hover{background:var(--surface)}
.gm[data-status="final"]{opacity:.62}
.gm .time{color:var(--muted);font-size:.82rem}
.trs{display:grid;gap:6px;min-width:0}
.tr{display:grid;grid-template-columns:24px 40px 1fr 44px 44px;align-items:center;gap:0 10px;color:var(--muted)}
.tr b{font-weight:700;font-stretch:85%;color:var(--text)}
.tr.fav .wn{color:var(--text);font-weight:600}
.tr .tn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.88rem}
.tr .ip,.tr .wn{text-align:right;font-size:.88rem}
.gm .ln{text-align:right;font-size:.85rem}
.gm .go{color:var(--faint);font-size:1.2rem;text-align:right}
@media (max-width:640px){
  .cols{grid-template-columns:44px 44px}.cols span:nth-child(n+3){display:none}
  .gm{grid-template-columns:56px 1fr;row-gap:6px}.gm .ln{grid-column:2;text-align:left;color:var(--muted);font-size:.8rem}.gm .go{display:none}}
.mu2{display:inline-flex;align-items:center;gap:8px;font-weight:700;font-stretch:85%;font-size:1.05rem}
.at2{color:var(--faint);font-weight:400;font-size:.85rem;margin-inline:2px}
.locks{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px}
.lock{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;display:grid;gap:8px;align-content:start;min-width:0}
.lock header{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.tag{font-size:.72rem;color:var(--muted);border:1px solid var(--line-2);border-radius:999px;padding:1px 9px;white-space:nowrap}
.fine{color:var(--muted);font-size:.82rem}
.more{justify-self:start;font-size:.88rem;text-decoration:none;color:var(--text)}
.more:hover{text-decoration:underline}
.rt{min-width:620px}
.rt .num{font-weight:600;white-space:nowrap}
.rt tr.tot th,.rt tr.tot td{border-top:1px solid var(--line-2)}
.neg{color:var(--neg)}
"""


def index_page():
    now = datetime.now(timezone.utc)
    locks = [g for g in G if g["status"] == "pending"] + [g for g in G if g["status"] == "graded"]
    pages = v0_pages()
    open_weeks = [p["week"] for p in pages if status(p, now) != "final"] or [p["week"] for p in pages]
    week = f"Week {max(open_weeks)}" if pages else "Games"
    return (theme.head("NFL Matchup Lab") + f"<style>{INDEX_CSS}</style>\n" + '<main class="wrap">\n' + theme.nav("home") +
            f'<header class="hero"><span class="eyebrow">2026 season</span><h1>{week}</h1>'
            '<p class="sub">Each game simulated 20,000 times from market lines and player usage. Pts is the market-implied '
            'score, Win is the share of simulations each team won. Open a game for player projections.</p></header>\n'
            + games_sections(now)
            + ('<section aria-labelledby="h-lock"><div><h2 id="h-lock">Locked picks</h2><p class="sub">Hand-built reports '
               'whose picks were committed to git before kickoff, then graded.</p></div>'
               f'<div class="locks">{"".join(locked_row(g) for g in locks)}</div></section>\n' if locks else "")
            + '<section aria-labelledby="h-rec"><div><h2 id="h-rec">Track record</h2>'
            '<p class="sub">Units assume 1 unit per prop at the approximate price listed in each results file. '
            'A handful of picks says almost nothing yet about whether the method works.</p></div>'
            '<div class="tbl-wrap"><table class="rt"><thead><tr><th scope="col">Game</th>'
            '<th scope="col">Game lines</th><th scope="col">Props hit</th><th scope="col">Units</th>'
            '<th scope="col">Test</th></tr></thead><tbody>' + record_rows() + '</tbody></table></div></section>\n'
            '<footer class="notes"><p>Data: nflverse (CC-BY 4.0), team logos from nflverse. Built by <code>src/site.py</code>. '
            'Estimates with real uncertainty, not betting advice.</p></footer>\n</main>\n'
            f'<script>{INDEX_JS}</script>\n')


# ---- the Guesser: paper bankroll on the locked picks ------------------------------------------------

def bankrolls():
    out = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "*-bankroll.json"))):
        b = json.load(open(path))
        b["_path"] = os.path.relpath(path, ROOT)
        out.append(b)
    return out


def grade_entry(entry, result):
    """(status, amount returned or None). Status: pending, won, lost."""
    over = ((result or {}).get("entries") or {}).get(entry["id"], {})
    if "returned" in over:
        r = float(over["returned"])
        return ("won" if r > 0 else "lost"), r
    legs = (result or {}).get("legs") or {}
    if not all(l in legs for l in entry["legs"]):
        return "pending", None
    k = sum(bool(legs[l]["hit"]) for l in entry["legs"])
    n = len(entry["legs"])
    mult = float(entry["pays"].get(str(k), 0)) if entry["type"] == "flex" else (float(entry["pays"][str(n)]) if k == n else 0.0)
    r = entry["stake"] * mult
    return ("won" if r > 0 else "lost"), r


def leg_html(leg, result):
    g = ((result or {}).get("legs") or {}).get(leg)
    if g is None:
        mark, actual = '<span class="mk" title="Not graded yet">&#9675;</span>', ""
    else:
        mark = '<span class="mk hit" title="Hit">&#10003;</span>' if g["hit"] else '<span class="mk miss" title="Missed">&#10005;</span>'
        actual = f'<span class="act">{e(str(g.get("actual")))}</span>' if g.get("actual") is not None else ""
    return f"<li>{mark}<span>{e(leg)}</span>{actual}</li>"


def kind_label(k):
    """'flex3' -> '3-pick Flex'."""
    m = re.match(r"([a-z]+)(\d+)$", k)
    return f"{m[2]}-pick {m[1].title()}" if m else k


def slip(b):
    res = b.get("result")
    _, wk, away, home = b["game"].split("_")
    entries_html, staked, back, pending = [], 0.0, 0.0, False
    for en in b["entries"]:
        status, r = grade_entry(en, res)
        staked += en["stake"]
        pending |= status == "pending"
        back += r or 0.0
        n = len(en["legs"])
        pays = " · ".join(f'{k} of {n}: {money(en["stake"] * float(v))}'
                          for k, v in sorted(en["pays"].items(), key=lambda kv: -int(kv[0])))
        outcome = {"pending": '<span class="tag">Pending</span>',
                   "won": f'<span class="tag pos">Returned {money(r or 0)}</span>',
                   "lost": '<span class="tag neg">Lost</span>'}[status]
        entries_html.append(f'<article class="entry"><header><div><h3>Entry {e(en["id"])}</h3><span class="kind">{n}-pick '
                            f'{en["type"].title()}</span></div><span class="stake">{money(en["stake"])}</span></header>'
                            f'<ul class="legs">{"".join(leg_html(l, res) for l in en["legs"])}</ul>'
                            f'<footer><span class="fine">Pays {pays}</span>{outcome}</footer></article>')
    profit = None if pending else back - staked
    ev = b.get("expected_return_per_dollar", {})
    ev_rows = "".join(
        f'<tr><th scope="row">Entry {e(k.split("_")[0])}<span class="vs-lbl">{e(kind_label(k.split("_")[1]))}</span></th>'
        f'<td class="num">${v["legacy_estimates"]:.2f}</td><td class="num">${v["market_no_vig"]:.2f}</td>'
        f'<td class="num">${v["model_v0"]:.2f}</td></tr>' for k, v in ev.items() if isinstance(v, dict))
    tag = ('<span class="tag">Waiting on the game</span>' if pending else
           f'<span class="tag pos">Up {money(profit)}</span>' if profit > 0 else
           f'<span class="tag neg">Down {money(-profit)}</span>' if profit < 0 else '<span class="tag">Even</span>')
    final = f' · Final {e(res["final"])}' if res and res.get("final") else ""
    if res and res.get("provisional"):
        final += " · graded from web recaps, waiting on official stats"
    a_d, h_d = ("LAR" if away == "LA" else away), ("LAR" if home == "LA" else home)
    return dict(staked=staked, back=back, pending=pending, html=f"""
<section class="slip" aria-labelledby="h-{e(b['game'])}">
<header class="slip-head"><div class="sh"><h2 id="h-{e(b['game'])}">{matchup(a_d, h_d, 32)}</h2>
<span class="fine">Week {int(wk)} · kickoff {e(b['kickoff_utc'])}{final} · frozen {e(b['frozen_at'])}</span></div>{tag}</header>
<div class="entries">{''.join(entries_html)}</div>
<div class="tbl-wrap"><table class="pt evt"><thead><tr><th scope="col">Expected back per $1</th><th scope="col">Locked estimates</th>
<th scope="col">No-vig market</th><th scope="col">Model v0</th></tr></thead><tbody>{ev_rows}</tbody></table></div>
<p class="fine">Above $1.00 means that view expected a profit. The locked estimates mixed a 4-game sample with market prices;
the market and the model both expected a loss. Picks: {e(b['picks_source'])}. Record: <code>{e(b['_path'])}</code>.</p>
</section>""")


GUESSER_CSS = """
.hero{display:grid;gap:8px}
.ledger{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr));gap:0;margin:0;border-top:1px solid var(--line-2);border-bottom:1px solid var(--line)}
.ledger div{display:grid;gap:4px;padding:16px 16px 16px 0}
.ledger dt{font-size:.75rem;color:var(--muted)}
.ledger dd{margin:0;font-size:clamp(1.5rem,4vw,2rem);font-weight:750;font-stretch:88%;line-height:1.1}
.ledger dd.wait{color:var(--faint)} .ledger dd.pos{color:var(--pos)} .ledger dd.neg{color:var(--neg)}
.ledger-wrap{display:grid;gap:10px}
.slip{display:grid;gap:18px}
.slip-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.sh{display:grid;gap:6px}
.mu2{display:inline-flex;align-items:center;gap:10px;font-weight:750;font-stretch:85%;font-size:1.4rem}
.at2{color:var(--faint);font-weight:400;font-size:.9rem;margin-inline:2px}
.entries{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:12px}
.entry{background:var(--surface);border:1px solid var(--line);border-radius:8px;display:grid;align-content:start;min-width:0}
.entry header{display:flex;justify-content:space-between;align-items:start;gap:10px;padding:16px 16px 12px}
.entry h3{font-size:1rem}
.entry .kind{font-size:.78rem;color:var(--muted)}
.entry .stake{font-size:1.4rem;font-weight:750;font-stretch:88%}
.legs{list-style:none;margin:0;padding:4px 16px 12px;display:grid;gap:8px}
.legs li{display:grid;grid-template-columns:18px 1fr auto;gap:10px;align-items:baseline}
.mk{color:var(--faint);text-align:center}
.mk.hit{color:var(--pos)} .mk.miss{color:var(--neg)}
.act{color:var(--muted);font-size:.85rem}
.entry footer{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px 14px;border-top:1px solid var(--line)}
.fine{color:var(--muted);font-size:.82rem;max-width:80ch}
.tag{font-size:.72rem;color:var(--muted);border:1px solid var(--line-2);border-radius:999px;padding:1px 9px;white-space:nowrap}
.tag.pos{color:var(--pos);border-color:color-mix(in srgb,var(--pos) 40%,transparent)}
.tag.neg{color:var(--neg);border-color:color-mix(in srgb,var(--neg) 40%,transparent)}
.evt{min-width:520px}
"""


def guesser_page():
    bs = bankrolls()
    slips = [slip(b) for b in bs]
    start = sum(b["bankroll"] for b in bs)
    staked = sum(s["staked"] for s in slips)
    open_stakes = sum(s["staked"] for s in slips if s["pending"])
    done_staked = staked - open_stakes
    back = sum(s["back"] for s in slips if not s["pending"])
    profit = back - done_staked
    graded = done_staked > 0
    prof_txt = (f"+{money(profit)}" if profit > 0 else f"-{money(-profit)}" if profit < 0 else money(0)) if graded else "Pending"
    prof_cls = ("pos" if profit > 0 else "neg" if profit < 0 else "") if graded else "wait"
    now = start - staked + back + open_stakes
    pend_note = ('<p class="fine">Bankroll now counts stakes on games not graded yet at face value.</p>' if open_stakes else "")
    body = ('<main class="wrap">\n' + theme.nav("guesser") +
            '<header class="hero"><span class="eyebrow">Paper bankroll · pretend money</span><h1>Guesser</h1>'
            '<p class="sub">Before a game, a pretend $20 goes on the picks that were locked in before kickoff, split into '
            'PrizePicks-style entries. After the game each leg is graded and this page keeps the running score.</p></header>\n'
            f'<div class="ledger-wrap"><dl class="ledger"><div><dt>Started with</dt><dd>{money(start)}</dd></div>'
            f'<div><dt>In play</dt><dd>{money(staked)}</dd></div>'
            f'<div><dt>Paid back</dt><dd class="{"" if graded else "wait"}">{money(back) if graded else "Pending"}</dd></div>'
            f'<div><dt>Profit</dt><dd class="{prof_cls}">{prof_txt}</dd></div>'
            f'<div><dt>Bankroll now</dt><dd>{money(now)}</dd></div></dl>{pend_note}</div>\n'
            + ("".join(s["html"] for s in slips) or '<p class="fine">No games yet.</p>') +
            '\n<footer class="notes"><p>Payouts use the standard PrizePicks tables in config/payouts.toml (Power pays only '
            'when every leg hits; Flex also pays when most legs hit). The app can offer lower multipliers on same-game '
            'entries; that would be noted on the slip. Pretend money, small samples, not betting advice.</p></footer>\n</main>\n')
    return theme.document("Guesser", body, GUESSER_CSS)


# ---- Lineups: cross-game PrizePicks lineups for the latest week ---------------------------------------

def latest_lineups():
    files = sorted(glob.glob(os.path.join(V0_DIR, "lineups-*-w*.json")))
    if not files:
        return None
    return max((json.load(open(f)) for f in files), key=lambda d: (d["season"], d["week"]))


LINEUPS_CSS = """
.more-models{display:grid;gap:24px;border-top:1px solid var(--line);padding-top:16px}
.more-models summary{cursor:pointer;color:var(--muted);font-size:.9rem;list-style:none}
.more-models summary::before{content:"+ ";color:var(--faint)}
.more-models[open] summary::before{content:"\2212 "}
.more-models summary:hover{color:var(--text)}
.more-models[open]{gap:40px}
.hero{display:grid;gap:8px}
.loaded{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:0 20px;border-top:1px solid var(--line-2)}
.loaded a{display:flex;align-items:center;gap:8px;padding:9px 0;border-bottom:1px solid var(--line);color:var(--text);text-decoration:none;font-size:.88rem}
.loaded b{font-weight:700;font-stretch:85%}
.loaded .n{margin-left:auto;color:var(--muted);white-space:nowrap}
.loaded .n.has{color:var(--text);font-weight:600}
.at2{color:var(--faint);font-size:.8rem}
"""


def lineups_page():
    from nfl_lab.report import picks_html
    d = latest_lineups()
    if d is None:
        inner = f'<p class="sub">No week has been run yet. {picks_html.HOW_TO_ADD}</p>'
        week_txt, built = "", ""
    else:
        games = d["games"]
        with_lines = sum(1 for g in games.values() if g["pp_lines"])
        rows = ""
        for gid, g in sorted(games.items()):
            _, _, a, h = gid.split("_")
            a, h = ("LAR" if a == "LA" else a), ("LAR" if h == "LA" else h)
            n = g["pp_lines"]
            rows += (f'<a href="games/v0-{g["slug"]}.html#lines">{theme.logo(a, 18)}<b>{e(a)}</b><span class="at2">at</span>'
                     f'{theme.logo(h, 18)}<b>{e(h)}</b><span class="n{" has" if n else ""}">{n} lines</span></a>')
        msg = ("No PrizePicks lines are loaded for this week yet." if not with_lines else
               "No lineup can be built from the PrizePicks lines loaded so far.")
        inner = ('<section aria-labelledby="h-best"><div><h2 id="h-best">Top lineups</h2><p class="sub">The best 3, 4, 5 and '
                 '6-pick entries this week, each built from a single game (picks are never mixed across games), ranked by '
                 'expected return per $1. Each pick\'s chance is half the game simulation and half recent form, plus a '
                 'small adjustment for this game\'s matchups (pass rush vs offensive line, what the defense allows to his '
                 'position, run defense, game script), with the reasons listed under each pick. Every game\'s own top '
                 'lineups are on its page.</p></div>'
                 + picks_html.lineup_cards(d["best"], prefix="", no_lines_msg=msg) + '</section>\n'
                 f'<section aria-labelledby="h-loaded"><div><h2 id="h-loaded">PrizePicks lines loaded</h2><p class="sub">'
                 f'{with_lines} of {len(games)} games have PrizePicks lines on file. Tap a game to see its lines.</p></div>'
                 f'<div class="loaded">{rows}</div></section>\n')
        week_txt = f' · Week {d["week"]}'
        built = f' Last run {e(d["built_at"][:16])} UTC.'
    body = ('<main class="wrap">\n' + theme.nav("lineups") +
            f'<header class="hero"><span class="eyebrow">PrizePicks{week_txt}</span><h1>Lineups</h1>'
            '<p class="sub">PrizePicks Power and Flex lineups, one game at a time. Only PrizePicks lines are used.'
            f'{built}</p>'
            '<p class="sub">An experiment, not proven picks: the model has not been graded against real PrizePicks lines '
            'yet, and the matchup adjustments are rules of thumb that have not been tested on past seasons.</p></header>\n' + inner +
            '<footer class="notes"><p>Payouts: PrizePicks standard tables (Power 2 to 6 picks, Flex 3 to 6). The app can pay '
            'less on some entries; skip a lineup if the app shows less than the listed multiplier. Estimates, not betting '
            'advice.</p></footer>\n</main>\n')
    return theme.document("Lineups", body, LINEUPS_CSS)


if __name__ == "__main__":
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "games"))
    theme.copy_logos(SITE)
    for g in G:
        if g.get("report"):
            open(os.path.join(SITE, "games", g["slug"] + ".html"), "w").write(game_page(g))
    for p in v0_pages():
        doc = open(os.path.join(V0_DIR, p["file"])).read()
        open(os.path.join(SITE, "games", "v0-" + p["file"]), "w").write(theme.refresh(doc))
    open(os.path.join(SITE, "index.html"), "w").write(index_page())
    open(os.path.join(SITE, "guesser.html"), "w").write(guesser_page())
    open(os.path.join(SITE, "lineups.html"), "w").write(lineups_page())
    print("wrote", SITE)
