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
GAMES_DIR = os.path.join(ROOT, "output", "games")
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


# ---- this week's games (Guesser method, output/games) ---------------------------------------------

def et(ts_utc: str) -> str:
    """'2026-10-16 00:15 UTC' (or ISO) -> 'Thu Oct 15, 8:15 PM ET'."""
    t = ts_utc.replace(" UTC", "+00:00")
    d = datetime.fromisoformat(t if "T" in t or "+" in t else t + "+00:00").astimezone(ET)
    return d.strftime("%a %b %-d, %-I:%M %p ET")


def game_recs():
    """Guesser-method game records (output/games/<season>-w<week>-<away>-<home>.json), newest week first, then kickoff."""
    out = []
    for path in sorted(glob.glob(os.path.join(GAMES_DIR, "*.json"))):
        m = re.match(r"(\d{4})-w(\d{2})-([a-z]+)-([a-z]+)\.json$", os.path.basename(path))
        if m:
            rec = json.load(open(path))
            out.append(dict(file=os.path.basename(path)[:-5] + ".html", season=int(m[1]), week=int(m[2]),
                            away=rec["away"], home=rec["home"], meta=rec))
    return sorted(out, key=lambda d: (-d["season"], -d["week"], d["meta"].get("kickoff_utc", ""), d["file"]))


def implied(m):
    """(away points, home points) from the spread and total, or (None, None)."""
    if m.get("home_spread") is None or m.get("total") is None:
        return None, None
    return (m["total"] + m["home_spread"]) / 2, (m["total"] - m["home_spread"]) / 2


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
    if not m or m.get("home_spread") is None or m.get("total") is None:
        return ""
    fav = h if m["home_spread"] < 0 else a
    spread = f'{fav} {-abs(m["home_spread"]):+g}' if m["home_spread"] else "Pick'em"
    return f'{e(spread)}<span class="muted"> · </span>O/U {m["total"]:g}'


def game_row(p, now):
    m, a, h = p["meta"], p["away"], p["home"]
    k = kickoff(p)
    st = status(p, now)
    ia, ih = implied(m)
    sides = [(a, ia), (h, ih)]
    trs = "".join(
        f'<span class="tr{" fav" if ip is not None and ip >= max(ia, ih) else ""}">{theme.logo(t, 24)}<b>{e(t)}</b>'
        f'<span class="tn">{e(nickname(t))}</span><span class="ip">{"" if ip is None else f"{ip:.1f}"}</span></span>'
        for t, ip in sides)
    n = len(m.get("picks") or [])
    ready = bool((m.get("plan") or {}).get("entries"))
    picks = (f'<span class="np">{n} line{"" if n == 1 else "s"}</span>'
             + ('<span class="np has">Play ready</span>' if ready else '<span class="np">No play yet</span>'))
    return (f'<a class="gm" href="games/{p["file"]}"{kick_attr(p)} data-status="{st}"><span class="time">'
            f'{k.strftime("%-I:%M %p") if k else ""}<span class="st">{STATUS_TXT[st]}</span></span>'
            f'<span class="trs">{trs}</span><span class="pk">{picks}</span><span class="ln">{line_text(m, a, h)}</span>'
            f'<span class="go" aria-hidden="true">&rsaquo;</span></a>')


def chip(p, now):
    m, a, h = p["meta"], p["away"], p["home"]
    k = kickoff(p)
    st = status(p, now)
    when = k.strftime("%a %-I:%M %p") if k else ""
    return (f'<a class="chip" href="games/{p["file"]}"{kick_attr(p)} data-status="{st}">'
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
        f'<div class="day"><div class="dayh"><span>{e(d)}</span><span class="cols"><span>Pts</span><span>PrizePicks</span>'
        f'<span class="lnh">Line</span><span></span></span></div><div class="list">{"".join(game_row(p, now) for p in dps)}</div></div>'
        for d, dps in days.items())


def games_sections(now):
    pages = game_recs()
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
.cols{display:grid;grid-template-columns:44px 110px 150px 16px;gap:0 10px;text-align:right;flex:none}
.list{display:grid}
.gm{display:grid;grid-template-columns:72px 1fr 110px 150px 16px;align-items:center;gap:0 10px;padding:12px 0;border-bottom:1px solid var(--line);
  color:var(--text);text-decoration:none}
.gm:hover{background:var(--surface)}
.gm[data-status="final"]{opacity:.62}
.gm .time{color:var(--muted);font-size:.82rem}
.trs{display:grid;gap:6px;min-width:0}
.tr{display:grid;grid-template-columns:24px 40px 1fr 44px;align-items:center;gap:0 10px;color:var(--muted)}
.tr b{font-weight:700;font-stretch:85%;color:var(--text)}
.tr.fav .ip{color:var(--text);font-weight:600}
.pk{display:grid;gap:4px;text-align:right;font-size:.8rem;color:var(--muted)}
.np.has{color:var(--text)}
.tr .tn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.88rem}
.tr .ip{text-align:right;font-size:.88rem}
.gm .ln{text-align:right;font-size:.85rem}
.gm .go{color:var(--faint);font-size:1.2rem;text-align:right}
@media (max-width:640px){
  .cols{grid-template-columns:44px}.cols span:nth-child(n+2){display:none}
  .gm{grid-template-columns:56px 1fr;row-gap:6px}.gm .ln,.gm .pk{grid-column:2;text-align:left;color:var(--muted);font-size:.8rem}
  .gm .pk{display:flex;gap:12px}.gm .go{display:none}}
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
b.pos{color:var(--pos)} b.neg{color:var(--neg)}
.lock p b{font-weight:650}
"""


def best_plays():
    recs = week_recs()
    rows = play_rows(recs, limit=3)
    if not rows:
        return ""
    return ('<section aria-labelledby="h-best"><div class="secth"><h2 id="h-best">Best plays this week</h2>'
            '<a class="more" href="lineups.html">All plays &rarr;</a></div>'
            '<p class="sub">The three games where the Guesser\'s $20 play expects the most back. Each line is one entry to '
            'enter in PrizePicks, with its stake. Green: expected to return more than the $20.</p>'
            f'<div class="plays">{rows}</div></section>\n')


def last_game(link_prefix=""):
    """The most recent graded game: what the Guesser's bet did, and the current method's before-kickoff test on it."""
    done = [b for b in bankrolls() if b.get("result")]
    if not done:
        return ""
    b = max(done, key=lambda b: b["kickoff_utc"])
    _, wk, away, home = b["game"].split("_")
    a_d, h_d = ("LAR" if away == "LA" else away), ("LAR" if home == "LA" else home)
    back = sum(grade_entry(en, b["result"])[1] or 0 for en in b["entries"])
    staked = sum(en["stake"] for en in b["entries"])
    method = "original hand-built picks" if "legacy" in b.get("picks_source", "") else "current method"
    test = ""
    rp = os.path.join(LIVE_DIR, os.path.basename(b["_path"]).replace("-bankroll.json", "-guesser-retro.json"))
    if os.path.exists(rp):
        v = json.load(open(rp))["variants"]["no_prices"]
        parts = []
        for n, lst in sorted({**v["lineups"], **(v.get("filled") or {})}.items(), key=lambda kv: int(kv[0])):
            o = lst[0].get("outcome") or {}
            parts.append(f'{n}-pick: {o.get("hits", "?")} of {o.get("of", n)}'
                         + (f' (${o["returned"]:.2f} per $1)' if o.get("returned") else " (lost)"))
        test = (f'<p>Current method, tested with picks made before kickoff: {e("; ".join(parts))}. '
                f'<a class="more" href="{link_prefix}guesser.html#last">Details &rarr;</a></p>')
    page = f'games/{os.path.basename(b["_path"]).replace("-bankroll.json", ".html")}'
    rep = (f'<a class="more" href="{link_prefix}{page}">Original report &rarr;</a>'
           if os.path.exists(os.path.join(SITE, page)) else "")
    final = (b["result"] or {}).get("final") or ""
    return (f'<section aria-labelledby="h-last"><div class="secth"><h2 id="h-last">Last game</h2></div>'
            f'<article class="lock"><header>{matchup(a_d, h_d)}<span class="tag">{e("Final " + final if final else "Graded")}</span></header>'
            f'<p>Guesser ({e(method)}): <b class="{"pos" if back > staked else "neg"}">${back:,.2f} back on ${staked:.0f}</b>.</p>'
            f'{test}{rep}</article></section>\n')


def index_page():
    now = datetime.now(timezone.utc)
    pages = game_recs()
    open_weeks = [p["week"] for p in pages if status(p, now) != "final"] or [p["week"] for p in pages]
    week = f"Week {max(open_weeks)}" if pages else "Games"
    from nfl_lab.report import picks_html
    return (theme.head("NFL Matchup Lab") + f"<style>{INDEX_CSS}{PLAYS_CSS}{picks_html.GUESSER_CSS}</style>\n" + '<main class="wrap">\n' + theme.nav("home") +
            f'<header class="hero"><span class="eyebrow">2026 season</span><h1>{week}</h1>'
            '<p class="sub">For each game the Guesser puts $20 on PrizePicks entries built from that game\'s best props. '
            'Pts is the score the betting line implies.</p></header>\n'
            + best_plays()
            + last_game()
            + games_sections(now)
            + '<footer class="notes"><p>Data: nflverse (CC-BY 4.0), team logos from nflverse. Built by <code>src/site.py</code>. '
            'Estimates with real uncertainty, not betting advice.</p></footer>\n</main>\n'
            f'<script>{INDEX_JS}</script>\n')


# ---- the Guesser: paper bankroll on the locked picks ------------------------------------------------

def bankrolls():
    out = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "*-bankroll.json"))):
        b = json.load(open(path))
        b["_path"] = os.path.relpath(path, ROOT)
        out.append(b)
    todo = sorted([b for b in out if not b.get("result")], key=lambda b: b["kickoff_utc"])
    done = sorted([b for b in out if b.get("result")], key=lambda b: b["kickoff_utc"], reverse=True)
    return todo + done


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


def leg_html(leg, result, shown=None):
    g = ((result or {}).get("legs") or {}).get(leg)
    if g is None:
        mark, actual = '<span class="mk" title="Not graded yet">&#9675;</span>', ""
    else:
        mark = '<span class="mk hit" title="Hit">&#10003;</span>' if g["hit"] else '<span class="mk miss" title="Missed">&#10005;</span>'
        actual = f'<span class="act">{e(str(g.get("actual")))}</span>' if g.get("actual") is not None else ""
    return f"<li>{mark}<span>{shown or e(leg)}</span>{actual}</li>"


EV_COLS = {"estimate": "Guesser estimate", "legacy_estimates": "Locked estimates", "market_no_vig": "No-vig market",
           "model_v0": "Model v0"}


def kind_label(k):
    """'flex3' -> '3-pick Flex'."""
    m = re.match(r"([a-z]+)(\d+)$", k)
    return f"{m[2]}-pick {m[1].title()}" if m else k


def slip(b):
    from nfl_lab import guesser
    from nfl_lab.report import picks_html
    res = b.get("result")
    _, wk, away, home = b["game"].split("_")
    legacy = "legacy" in b.get("picks_source", "")
    ev = {k.split("_")[0]: v for k, v in b.get("expected_return_per_dollar", {}).items() if isinstance(v, dict)}
    entries_html, staked, back, pending = [], 0.0, 0.0, False
    for en in b["entries"]:
        status, r = grade_entry(en, res)
        staked += en["stake"]
        pending |= status == "pending"
        back += r or 0.0
        n = len(en["legs"])
        pays = " · ".join(f'{k} of {n}: {money(en["stake"] * float(v))}'
                          for k, v in sorted(en["pays"].items(), key=lambda kv: -int(kv[0])) if float(v))
        est = (ev.get(en["id"]) or {}).get("estimate")
        exp = f' · expect {money(en["stake"] * est)}' if est is not None and status == "pending" else ""
        outcome = {"pending": '<span class="tag">Pending</span>',
                   "won": f'<span class="tag pos">Returned {money(r or 0)}</span>',
                   "lost": '<span class="tag neg">Lost</span>'}[status]
        if en.get("legs_detail"):
            legs = "".join(
                leg_html(text, res, f'{theme.logo(x["team"], 16)} {e(x["player"])} {picks_html.side_chip(x["side"])} '
                                    f'<b>{x["line"]:g}</b> {e(picks_html.stat_name(x["stat"]))}')
                for text, x in zip(en["legs"], en["legs_detail"]))
        else:
            legs = "".join(leg_html(l, res) for l in en["legs"])
        entries_html.append(f'<article class="entry"><header><div><h3>Entry {e(en["id"])}</h3><span class="kind">{n}-pick '
                            f'{en["type"].title()}</span></div><span class="stake">{money(en["stake"])}</span></header>'
                            f'<ul class="legs">{legs}</ul>'
                            f'<footer><span class="fine">Pays {pays}{exp}</span>{outcome}</footer></article>')
    profit = None if pending else back - staked
    page = os.path.basename(b["_path"]).replace("-bankroll.json", ".html")
    open_link = (f'<a class="more" href="games/{e(page)}">{"Original report" if legacy else "Open game"} &rarr;</a>'
                 if os.path.exists(os.path.join(SITE, "games", page)) else "")
    tag = ('<span class="tag">Waiting on the game</span>' if pending else
           f'<span class="tag pos">Up {money(profit)}</span>' if profit > 0 else
           f'<span class="tag neg">Down {money(-profit)}</span>' if profit < 0 else '<span class="tag">Even</span>')
    final = f' · Final {e(res["final"])}' if res and res.get("final") else ""
    if res and res.get("provisional"):
        final += " · graded from web recaps, waiting on official stats"
    a_d, h_d = ("LAR" if away == "LA" else away), ("LAR" if home == "LA" else home)
    who = ("Original hand-built picks (the first Guesser bet)" if legacy else
           f"Expected back {money(b['expected_back'])} on {money(staked)}" if b.get("expected_back") is not None else "")
    return dict(staked=staked, back=back, pending=pending, legacy=legacy, html=f"""
<section class="slip" aria-labelledby="h-{e(b['game'])}">
<header class="slip-head"><div class="sh"><h3 class="mh" id="h-{e(b['game'])}">{matchup(a_d, h_d, 28)}</h3>
<span class="fine">Week {int(wk)} · {e(et(b['kickoff_utc']))}{final}</span><span class="fine">{e(who)}</span>{open_link}</div>{tag}</header>
<div class="entries">{''.join(entries_html)}</div>
</section>""")


def retro_tests():
    """Before-kickoff tests of the current Guesser method on finished games (backtests/live/*-guesser-retro.json)."""
    from nfl_lab import guesser
    from nfl_lab.report import picks_html
    out = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "*-guesser-retro.json"))):
        rec = json.load(open(path))
        if not rec.get("graded"):
            continue
        _, wk, away, home = rec["game_id"].split("_")
        a_d, h_d = ("LAR" if away == "LA" else away), ("LAR" if home == "LA" else home)
        def rows(v):
            r = ""
            for n, lst in sorted({**v["lineups"], **(v.get("filled") or {})}.items(), key=lambda kv: int(kv[0])):
                L = lst[0]
                o = L.get("outcome") or {}
                legs = "".join(
                    f'<li><span class="mk {"hit" if x["result"]["hit"] else "miss"}">{"&#10003;" if x["result"]["hit"] else "&#10005;"}</span>'
                    f'<span>{e(x["player"])} {picks_html.side_chip(x["side"])} {x["line"]:g} {e(picks_html.stat_name(x["stat"]))}'
                    f' <span class="muted">({x["p_model"]:.0%})</span></span><span class="act">{e(str(x["result"]["actual"]))}</span></li>'
                    for x in L["legs"] if x.get("result"))
                ret = o.get("returned", 0)
                tag = (f'<span class="tag pos">Won ${ret:.2f} per $1</span>' if ret > 1 else
                       f'<span class="tag">Got ${ret:.2f} back per $1</span>' if ret > 0 else '<span class="tag neg">Lost</span>')
                r += (f'<article class="entry"><header><div><h3>{n}-pick {e(L["play_as"])}</h3><span class="kind">'
                      f'{o.get("hits", "?")} of {o.get("of", n)} hit' + (" · filled with coin flips" if L.get("filled") else "")
                      + f'</span></div>{tag}</header><ul class="legs">{legs}</ul></article>')
            return r
        v1, v2 = rec["variants"]["no_prices"], rec["variants"]["with_prices"]
        def total(v):
            ls = [lst[0] for lst in {**v["lineups"], **(v.get("filled") or {})}.values()]
            return len(ls), sum((L.get("outcome") or {}).get("returned", 0) for L in ls)
        def brief(v):
            parts = []
            for n, lst in sorted({**v["lineups"], **(v.get("filled") or {})}.items(), key=lambda kv: int(kv[0])):
                o = lst[0].get("outcome") or {}
                parts.append(f'{n}-pick {lst[0]["play_as"]}{" (filled with coin flips)" if lst[0].get("filled") else ""} '
                             f'{o.get("hits", "?")} of {o.get("of", n)}, ' + (f'${o["returned"]:.2f} per $1' if o.get("returned") else "lost"))
            return "; ".join(parts)
        n1, r1 = total(v1)
        n2, r2 = total(v2)
        out.append(f"""<section class="slip" aria-labelledby="h-rt-{e(rec['game_id'])}">
<header class="slip-head"><div class="sh"><span class="eyebrow">Before-kickoff test · current method</span>
<h3 class="mh" id="h-rt-{e(rec['game_id'])}">{matchup(a_d, h_d, 28)}</h3>
<span class="fine">Week {int(wk)} · picks rated with only what was known before kickoff and committed ({e(rec.get('frozen_commit') or 'see git log')})
before any result was read · {e(rec['graded'].get('final') or '')}</span></div><span class="tag neg">$1 on each: ${r1:.2f} back from ${n1:.0f}</span></header>
<p class="fine">Same code as this week's lineups, run on the 13 lines saved before kickoff, with every line's market at 50% as on a
PrizePicks board. Not part of the bankroll above. Sizes the method would not build (fewer than 5 or 6 players at Lean or better) were
filled with the next-best coin flips and are marked.</p>
<div class="entries">{rows(v1)}</div>
<p class="fine">With the sportsbook prices the original report collected (instead of 50%): ${r2:.2f} back from ${n2:.0f} at $1
each ({e(brief(v2))}). Record: <code>{e(os.path.relpath(path, ROOT))}</code>.
The matchup rules were written after this game was played, so this is a fair test of the data, not of the rules.</p>
</section>""")
    return "".join(out)


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
b.pos{color:var(--pos)} b.neg{color:var(--neg)}
.ledger dd small{font-size:.8rem;font-weight:400;color:var(--muted)}
.slip .mh{font-size:1rem}
section>h2{margin-bottom:4px}
.more{justify-self:start;font-size:.88rem;text-decoration:none;color:var(--text)}
.more:hover{text-decoration:underline}
"""


def guesser_page():
    bs = bankrolls()
    slips = [slip(b) for b in bs]
    graded = [s for s in slips if not s["pending"]]
    waiting = [s for s in slips if s["pending"]]
    staked = sum(s["staked"] for s in graded)
    back = sum(s["back"] for s in graded)
    profit = back - staked
    prof_txt = (f"+{money(profit)}" if profit > 0 else f"-{money(-profit)}" if profit < 0 else money(0)) if graded else "Pending"
    prof_cls = ("pos" if profit > 0 else "neg" if profit < 0 else "") if graded else "wait"
    body = ('<main class="wrap">\n' + theme.nav("guesser") +
            '<header class="hero"><span class="eyebrow">Pretend money</span><h1>Guesser</h1>'
            '<p class="sub">$20 per game, no more. Before kickoff the Guesser splits $20 across up to 3 PrizePicks entries '
            'built from that game\'s best picks (the same play shown on the Lineups tab) and locks it. After the game '
            'every leg is graded from the official stats.</p></header>\n'
            f'<div class="ledger-wrap"><dl class="ledger"><div><dt>Each game</dt><dd>$20</dd></div>'
            f'<div><dt>Games graded</dt><dd>{len(graded)}</dd></div>'
            f'<div><dt>Back from graded games</dt><dd class="{"" if graded else "wait"}">'
            f'{money(back) + "<small> on " + money(staked) + "</small>" if graded else "Pending"}</dd></div>'
            f'<div><dt>Profit</dt><dd class="{prof_cls}">{prof_txt}</dd></div>'
            f'<div><dt>Waiting</dt><dd>{len(waiting)} game{"" if len(waiting) == 1 else "s"}</dd></div></dl></div>\n'
            + ('<section aria-labelledby="h-last2" id="last"><h2 id="h-last2">Last game</h2>'
               + "".join(s["html"] for s in graded[:1]) + retro_tests() + '</section>\n' if graded else "")
            + ('<section aria-labelledby="h-up"><h2 id="h-up">This week</h2>' + "".join(s["html"] for s in waiting)
               + '</section>\n' if waiting else "")
            + ("".join(s["html"] for s in graded[1:]))
            + '\n<footer class="notes"><p>Payouts use the standard PrizePicks tables in config/payouts.toml (Power pays only '
            'when every leg hits; Flex also pays when most legs hit). The app can offer lower multipliers on same-game '
            'entries. How the $20 is split: <code>nfl_lab/guesser.py</code> plan(). Pretend money, small samples, not '
            'betting advice.</p></footer>\n</main>\n')
    from nfl_lab.report import picks_html
    return theme.document("Guesser", body, GUESSER_CSS + picks_html.GUESSER_CSS)


# ---- Lineups: cross-game PrizePicks lineups for the latest week ---------------------------------------

PLAYS_CSS = """
.plays{display:grid;border-top:1px solid var(--line-2)}
.prow{display:grid;grid-template-columns:minmax(150px,200px) 1fr auto;gap:6px 18px;align-items:start;padding:12px 0;
  border-bottom:1px solid var(--line);color:var(--text);text-decoration:none}
.prow:hover{background:var(--surface)}
.prow .pg{display:grid;gap:3px}
.prow .pg .fine{font-size:.76rem}
.prow ul{list-style:none;margin:0;padding:0;display:grid;gap:8px;font-size:.9rem}
.prow li{display:grid;gap:3px}
.prow .eh2{font-size:.8rem;color:var(--muted)}
.prow .eh2 b{color:var(--text);font-size:.95rem}
.prow .lgs{display:flex;flex-wrap:wrap;gap:4px 14px}
.prow .lg1{white-space:nowrap}
.prow .pev{font-size:.8rem;color:var(--muted);text-align:right;white-space:nowrap;display:grid;gap:3px}
.prow .pev b{color:var(--text)}
.prow .pev b.ok{color:var(--pos)} .prow .pev b.no{color:var(--neg)}
@media (max-width:640px){.prow{grid-template-columns:1fr}.prow .pev{text-align:left;display:flex;gap:12px}}
"""


def play_rows(recs, limit=None):
    """One row per game: each entry of the Guesser's $20 play with its stake and picks, and what the play expects back."""
    from nfl_lab.report import picks_html
    rows = []
    for p in recs:
        pl = p["meta"].get("plan") or {}
        if not pl.get("entries"):
            continue
        k = kickoff(p)
        ents = "".join(
            f'<li><span class="eh2"><b>${L["stake"]}</b> Entry {L["entry"]} · {L["n"]}-pick {e(L["play_as"])}</span><span class="lgs">'
            + "".join(f'<span class="lg1">{theme.logo(x["team"], 14)} {e(x["player"])} {picks_html.side_chip(x["side"])} '
                      f'{x["line"]:g} {e(picks_html.stat_name(x["stat"]))}</span>' for x in L["legs"])
            + '</span></li>' for L in pl["entries"])
        ok = pl["expected_back"] > pl["budget"]
        rows.append((pl["expected_back"], f'<a class="prow" href="lineups.html#h-{p["file"][:-5]}"><span class="pg">'
                     f'{matchup(p["away"], p["home"], 20)}<span class="fine">{k.strftime("%a %-I:%M %p") if k else ""} ET</span></span>'
                     f'<ul>{ents}</ul><span class="pev"><span>expect <b class="{"ok" if ok else "no"}">${pl["expected_back"]:.2f}</b></span>'
                     f'<span class="fine">back on $20</span></span></a>'))
    if limit:
        rows = sorted(rows, key=lambda t: -t[0])[:limit]
    return "".join(r for _, r in rows)


def with_results(p):
    """The game record with the Guesser's graded results (backtests/live/<slug>-bankroll.json) copied onto its play."""
    from nfl_lab import guesser
    rec = json.loads(json.dumps(p["meta"]))
    path = os.path.join(LIVE_DIR, p["file"][:-5] + "-bankroll.json")
    if not os.path.exists(path) or not (rec.get("plan") or {}).get("entries"):
        return rec
    b = json.load(open(path))
    res = b.get("result")
    if not res:
        return rec
    for L, en in zip(rec["plan"]["entries"], b["entries"]):
        status, r = grade_entry(en, res)
        if status != "pending":
            hits = sum(bool(res["legs"][t]["hit"]) for t in en["legs"])
            L["outcome"] = dict(returned=r, hits=hits, of=len(en["legs"]))
        for x, t in zip(L["legs"], en["legs"]):
            if t in res.get("legs", {}):
                x["result"] = res["legs"][t]
    return rec


def week_recs():
    """This week's game records: the latest week with a game not yet final (else the latest week)."""
    recs, now = game_recs(), datetime.now(timezone.utc)
    if not recs:
        return []
    open_wk = [(p["season"], p["week"]) for p in recs if status(p, now) != "final"]
    wk = min(open_wk) if open_wk else max((p["season"], p["week"]) for p in recs)
    return sorted([p for p in recs if (p["season"], p["week"]) == wk], key=lambda p: p["meta"].get("kickoff_utc", ""))


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
.gsec{display:grid;gap:14px;padding-top:8px;border-top:1px solid var(--line-2)}
.gsh{display:grid;gap:4px}
.gsh h2{font-size:1rem}
.mu2{display:inline-flex;align-items:center;gap:8px;font-weight:750;font-stretch:85%;font-size:1.25rem}
.fine{color:var(--muted);font-size:.82rem}
.more{justify-self:start;font-size:.88rem;text-decoration:none;color:var(--text)}
.more:hover{text-decoration:underline}
.st{font-weight:650;margin-left:6px}
[data-status="live"].st{color:var(--pos)}
"""


def lineups_page():
    from nfl_lab.report import picks_html
    recs = week_recs()
    now = datetime.now(timezone.utc)
    if not recs:
        inner, week_txt, built = f'<p class="sub">No week has been run yet. {picks_html.HOW_TO_ADD}</p>', "", ""
    else:
        with_lu = [p for p in recs if (p["meta"].get("plan") or {}).get("entries")]
        without = [p for p in recs if not (p["meta"].get("plan") or {}).get("entries")]
        inner = ('<section aria-labelledby="h-plays"><div><h2 id="h-plays">This week\'s plays</h2><p class="sub">What to enter for '
                 'each game: every line is one PrizePicks entry with its stake, $20 per game, exactly what the Guesser plays. '
                 'Green: the play is expected to return more than the $20; red: less. Tap a game for the reasons.</p></div>'
                 f'<div class="plays">{play_rows(with_lu)}</div></section>\n')
        for p in with_lu:
            m, k = p["meta"], kickoff(p)
            slug = p["file"][:-5]
            inner += (f'<section class="gsec" aria-labelledby="h-{slug}"><div class="gsh"><h2 id="h-{slug}">'
                      f'{matchup(p["away"], p["home"], 26)}</h2><span class="fine">{k.strftime("%a %b %-d, %-I:%M %p") if k else ""} ET'
                      f' · {line_text(m, p["away"], p["home"])}<span class="st"{kick_attr(p)} data-status="{status(p, now)}"></span>'
                      f'</span><a class="more" href="games/{p["file"]}">Props board and matchups &rarr;</a></div>'
                      + picks_html.guesser_play(m.get("plan"), m.get("lineups"), prefix="") + '</section>\n')
        if without:
            def why(p):
                ps = p["meta"].get("picks") or []
                lean = {x["player_id"] for x in ps if x["p_model"] >= 0.525 and x["games"] >= 3}
                return (f'{len(ps)} line{"" if len(ps) == 1 else "s"}, {len(lean)} good pick{"" if len(lean) == 1 else "s"}')
            rows = "".join(f'<a href="games/{p["file"]}#props">{theme.logo(p["away"], 18)}<b>{e(p["away"])}</b><span class="at2">at</span>'
                           f'{theme.logo(p["home"], 18)}<b>{e(p["home"])}</b><span class="n">{why(p)}</span></a>'
                           for p in without)
            inner += ('<section aria-labelledby="h-wait"><div><h2 id="h-wait">No play yet</h2><p class="sub">A play needs at '
                      'least 2 good picks (rated 52.5% or better, from both teams). These games don\'t have that on the '
                      'PrizePicks board loaded so far; PrizePicks posts most lines a day or two before kickoff, so a fresh board '
                      f'may add some.</p></div><div class="loaded">{rows}</div></section>\n')
        week_txt = f' · Week {recs[0]["week"]}'
        built = f' Last run {e(max(p["meta"]["built_at"] for p in recs)[:16])} UTC.'
    body = ('<main class="wrap">\n' + theme.nav("lineups") +
            f'<header class="hero"><span class="eyebrow">PrizePicks{week_txt}</span><h1>Lineups</h1>'
            '<p class="sub">$20 per game, split into PrizePicks entries built from that game\'s best picks. Each pick comes from '
            'the player\'s last 4 games, the market and this game\'s matchups (open a game for the details).'
            f'{built}</p></header>\n' + inner +
            '<footer class="notes"><p>Payouts: PrizePicks standard tables (Power 2 to 6 picks, Flex 3 to 6). The app can pay '
            'less on some entries; skip a lineup if the app shows less than the listed multiplier. A lineup\'s chance '
            'multiplies its picks\' estimates, as the original did. Estimates, not betting advice.</p></footer>\n</main>\n'
            f'<script>{INDEX_JS}</script>\n')
    return theme.document("Lineups", body, LINEUPS_CSS + PLAYS_CSS + picks_html.GUESSER_CSS)


if __name__ == "__main__":
    from pathlib import Path
    from nfl_lab.report import gamepage
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "games"))
    theme.copy_logos(SITE)
    for g in G:
        if g.get("report"):
            open(os.path.join(SITE, "games", g["slug"] + ".html"), "w").write(game_page(g))
    for p in game_recs():
        gamepage.render(with_results(p), Path(SITE) / "games" / p["file"])
    open(os.path.join(SITE, "index.html"), "w").write(index_page())
    open(os.path.join(SITE, "guesser.html"), "w").write(guesser_page())
    open(os.path.join(SITE, "lineups.html"), "w").write(lineups_page())
    print("wrote", SITE)
