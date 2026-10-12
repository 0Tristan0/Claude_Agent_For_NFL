"""Shared look for every page on the site: the CSS, the top navigation and the team pennants.

Pages inline the CSS in a <style id="lab-theme"> block so each HTML file works on its own; src/site.py
swaps that block for the current one when it copies pages into site/, so a restyle never needs a re-run
of the simulations.
"""
from __future__ import annotations

import csv
import html
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
e = html.escape

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..800;'
         '1,6..96,400..800&family=Graduate&family=Libre+Franklin:ital,wght@0,400..800;1,400..800&display=swap">\n')

CSS = r"""
/* Layout: a members' club game program at night. Felt-green ground, ivory type, brass rules; varsity
   pennants on the scoreboard, a lace-stitch rule over each section head, agate-style stat tables below. */
:root{
  color-scheme:dark;
  --ground:#0B1611; --panel:#11201A; --panel-hi:#172A21; --rule:#24382E; --rule-hi:#38503F;
  --ivory:#ECE5D2; --sage:#9AA99D; --brass:#C8A55E; --brass-dim:#7F6A3E; --pine:#8CC9A0; --claret:#E39189;
  --serif:"Bodoni Moda","Bodoni 72",Didot,Georgia,serif;
  --varsity:"Graduate",Rockwell,"Courier New",serif;
  --sans:"Libre Franklin","Franklin Gothic Medium","Helvetica Neue",Arial,sans-serif;
  /* token names the older report pages use */
  --bg:var(--ground); --surface:var(--panel); --fg:var(--ivory); --muted:var(--sage); --line:var(--rule);
  --track:#22362C; --display:var(--sans); --body:var(--sans);
  --t1:#8CC9A0; --t1bg:#163426; --t2:#BCCB8A; --t2bg:#212F1C; --t3:#E2BE6E; --t3bg:#33291A; --t4:#E39189; --t4bg:#3A1F1F;
  --buf:#E8909A; --lar:#D6B05C; --lar-strong:#E2BD68;
}
*{box-sizing:border-box}
html{background:var(--ground)}
body{margin:0;background:var(--ground);color:var(--ivory);font-family:var(--sans);font-size:16px;line-height:1.55;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
.wrap{max-width:1080px;margin:0 auto;padding-inline:16px;padding-block:18px 64px;display:grid;gap:48px}
@media (min-width:760px){.wrap{padding-inline:28px}}
h1,h2,h3{font-family:var(--serif);font-weight:600;line-height:1.08;margin:0;text-wrap:balance;letter-spacing:-.005em}
h2{font-size:clamp(1.65rem,4vw,2.15rem)}
h2::before{content:"";display:block;width:92px;height:11px;margin-bottom:14px;
  background:linear-gradient(var(--brass),var(--brass)) 0 50%/100% 1.5px no-repeat,
             repeating-linear-gradient(90deg,var(--brass) 0 2px,transparent 2px 10px) 7px 0/80px 100% no-repeat}
p{margin:0}
.sub{color:var(--sage);margin-top:10px;max-width:68ch}
.eyebrow{font-weight:600;font-size:.72rem;letter-spacing:.17em;text-transform:uppercase;color:var(--brass)}
section{display:grid;gap:18px;min-width:0}
a{color:var(--brass);text-underline-offset:3px}
a:hover{color:var(--ivory)}
code{font-size:.9em;color:var(--ivory)}
:focus-visible{outline:2px solid var(--brass);outline-offset:3px}
.num{font-variant-numeric:tabular-nums lining-nums}

/* top navigation */
.site{display:flex;justify-content:space-between;align-items:center;gap:10px 20px;flex-wrap:wrap;padding-bottom:12px;border-bottom:1px solid var(--rule)}
.brand{display:flex;align-items:center;gap:10px;color:var(--ivory);text-decoration:none;font-family:var(--serif);font-size:1.1rem;font-weight:600}
.brand:hover{color:var(--ivory)}
.brand svg{width:30px;height:30px;color:var(--brass);flex:none}
.tabs{display:flex;gap:2px;flex-wrap:wrap}
.tabs a{font-size:.74rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--sage);text-decoration:none;
  padding:8px 12px;border-bottom:2px solid transparent}
.tabs a:hover{color:var(--ivory)}
.tabs a[aria-current="page"]{color:var(--ivory);border-bottom-color:var(--brass)}

/* scoreboard */
.board{background:repeating-linear-gradient(90deg,transparent 0 calc(10% - 1px),color-mix(in srgb,var(--ivory) 5%,transparent) calc(10% - 1px) 10%),
  radial-gradient(ellipse at 50% -10%,color-mix(in srgb,var(--brass) 10%,transparent),transparent 65%),var(--panel);
  border:1px solid var(--rule-hi);border-radius:3px;padding:24px clamp(16px,3.5vw,32px);display:grid;gap:20px;
  box-shadow:inset 0 0 0 6px var(--panel),inset 0 0 0 7px color-mix(in srgb,var(--brass) 40%,transparent)}
.teams{display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:12px}
.team{display:grid;gap:6px;justify-items:start;min-width:0}
.team.home{justify-items:end;text-align:right}
.team .ab{font-family:var(--varsity);font-size:clamp(2.4rem,8vw,4rem);line-height:.95;letter-spacing:.02em}
.team.away .ab{color:var(--buf)} .team.home .ab{color:var(--lar-strong)}
.team .nm{font-family:var(--serif);font-size:clamp(1.05rem,2.6vw,1.4rem);font-weight:500;line-height:1.2}
.team .rec{color:var(--sage);font-size:.86rem;font-variant-numeric:tabular-nums}
.at{font-family:var(--serif);font-style:italic;font-size:1.6rem;color:var(--brass);align-self:center}
.pn{display:block;max-width:100%;filter:drop-shadow(0 0 .6px var(--brass)) drop-shadow(0 0 .6px var(--brass))}
.pennant{--w:clamp(120px,30vw,224px);width:var(--w);max-width:100%;aspect-ratio:5/2;display:flex;align-items:center;
  padding-left:calc(var(--w) * .1);color:var(--tt);font-family:var(--varsity);font-size:calc(var(--w) * .165);letter-spacing:.03em;line-height:1;
  background:linear-gradient(90deg,var(--brass) 0 5px,var(--tc) 5px);clip-path:polygon(0 0,100% 50%,0 100%)}
.team.home .pennant{justify-content:flex-end;padding-left:0;padding-right:calc(var(--w) * .1);
  background:linear-gradient(270deg,var(--brass) 0 5px,var(--tc) 5px);clip-path:polygon(100% 0,0 50%,100% 100%)}
.facts{display:flex;flex-wrap:wrap;gap:12px 30px;border-top:1px solid color-mix(in srgb,var(--brass) 30%,transparent);padding-top:16px;font-variant-numeric:tabular-nums}
.facts dl{display:contents}
.facts div{display:grid;gap:2px;min-width:0}
.facts dt{font-size:.66rem;letter-spacing:.15em;text-transform:uppercase;color:var(--sage);font-weight:600}
.facts dd{margin:0;font-weight:600}
.v0tag{display:inline-block;border:1px solid var(--brass-dim);color:var(--brass);border-radius:2px;padding:0 7px;margin-left:4px;
  font-size:.64rem;letter-spacing:.15em;vertical-align:1px}
@media (max-width:520px){.teams{grid-template-columns:1fr 1fr}.at{display:none}.team .rec{font-size:.8rem}}

/* tables */
.tbl-wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:3px}
table{border-collapse:collapse;width:100%;min-width:600px;font-variant-numeric:tabular-nums lining-nums}
caption{text-align:left;padding:18px 18px 8px;font-family:var(--serif);font-weight:600;font-size:1.3rem}
caption .sw{display:inline-block;width:22px;height:12px;margin-right:10px;background:var(--tc);clip-path:polygon(0 0,100% 50%,0 100%);vertical-align:1px}
thead th{font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;color:var(--sage);font-weight:600;text-align:left;padding:10px 16px;
  border-bottom:1px solid var(--brass-dim);white-space:nowrap}
tbody th{text-align:left;font-weight:600;padding:10px 16px;vertical-align:top}
table:not(.pt) tbody th{width:28%}
tbody td{padding:10px 16px;vertical-align:middle}
tbody tr+tr>*{border-top:1px solid var(--rule)}
tbody tr:hover>*{background:var(--panel-hi)}
.vs-lbl{display:block;font-weight:400;font-size:.76rem;color:var(--sage);letter-spacing:.02em}
.pt td,.pt th{white-space:nowrap}
.pt .num{text-align:right}
.pt thead th:not(:first-child){text-align:right}
.pt thead th.l,.pt td.l{text-align:left}
.rng{color:var(--sage);font-size:.8rem}
.q-yes{color:var(--pine);font-weight:700}
.q-no{color:var(--sage)}
.small{font-size:.85rem;color:var(--sage)}

/* report pieces shared with the older Bills-Rams page */
.side{white-space:nowrap}
.side .v{display:inline-block;min-width:4.6em;font-weight:600}
.pill{display:inline-block;min-width:3.2em;text-align:center;font-size:.76rem;font-weight:700;padding:1px 8px;border-radius:2px;margin-right:10px}
.t1{color:var(--t1);background:var(--t1bg)} .t2{color:var(--t2);background:var(--t2bg)}
.t3{color:var(--t3);background:var(--t3bg)} .t4{color:var(--t4);background:var(--t4bg)}
.bar{display:inline-block;width:72px;height:5px;background:var(--track);vertical-align:middle;overflow:hidden}
.fill{display:block;height:100%}
.fill.buf{background:var(--buf)} .fill.lar{background:var(--lar)}
.edge{display:inline-block;font-weight:700;font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;padding:3px 10px;border-radius:2px;border:1px solid currentColor}
.edge.buf{color:var(--buf)} .edge.lar{color:var(--lar)} .edge.even{color:var(--sage)}
.tchip{font-family:var(--varsity);letter-spacing:.03em}
.tchip.buf{color:var(--buf)} .tchip.lar{color:var(--lar)}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:.84rem;color:var(--sage);align-items:center}
.legend .pill{margin-right:4px}
.takes{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:18px}
.take{border-left:2px solid var(--rule-hi);padding:2px 0 2px 16px;min-width:0}
.take.buf{border-color:var(--buf)} .take.lar{border-color:var(--lar)}
.take b{display:block;font-family:var(--serif);font-size:1.2rem;font-weight:600;line-height:1.2}
.take p{margin-top:6px;color:color-mix(in srgb,var(--ivory) 88%,var(--sage))}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:18px;display:grid;gap:10px;align-content:start;min-width:0}
.card header{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.card h3{font-size:1.45rem}
.card .pos{font-size:.74rem;color:var(--sage);letter-spacing:.1em;text-transform:uppercase}
.line{font-variant-numeric:tabular-nums;font-weight:600;font-size:.94rem}
.vsbox{display:grid;gap:2px;padding:10px 12px;background:var(--ground);border-left:2px solid var(--brass-dim);font-size:.94rem}
.vsbox .k{font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;color:var(--sage);font-weight:600}
.verdict{font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.78rem}
.verdict.big{color:var(--t1)} .verdict.mid{color:var(--t3)} .verdict.low{color:var(--t4)}

/* lineups */
.lus{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.lu{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:18px;display:grid;gap:12px;align-content:start;min-width:0}
.lu header{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.lu h3{font-size:1.5rem}
.lu .rec{font-size:.88rem;color:var(--sage)} .lu .rec b{color:var(--ivory)}
.lu ol{margin:0;padding-left:1.3em;display:grid;gap:4px}
.lu .pp{color:var(--sage);font-variant-numeric:tabular-nums;font-size:.84rem}
.lustats{display:grid;grid-template-columns:1fr 1fr;gap:8px 12px;margin:0;border-top:1px solid var(--rule);padding-top:12px;font-variant-numeric:tabular-nums}
.lustats div{display:grid} .lustats dt{font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:var(--sage)} .lustats dd{margin:0;font-weight:700}

.notes{color:var(--sage);font-size:.86rem;display:grid;gap:8px;max-width:78ch;border-top:1px solid var(--rule);padding-top:20px}
.notes a{color:inherit}
.notes ul{margin:0;padding-left:1.2em}
@media (max-width:520px){.bar{width:44px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

CREST = ('<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" stroke-width="1"/>'
         '<circle cx="16" cy="16" r="12.6" fill="none" stroke="currentColor" stroke-width=".6"/>'
         '<g transform="rotate(-35 16 16)"><ellipse cx="16" cy="16" rx="8.6" ry="5.1" fill="currentColor"/>'
         '<g style="stroke:var(--ground)" stroke-width=".9" stroke-linecap="round" fill="none">'
         '<path d="M12.4 16h7.2M13.6 14.7v2.6M15.2 14.7v2.6M16.8 14.7v2.6M18.4 14.7v2.6"/></g></g></svg>')

TABS = [("games", "Games", "index.html"), ("guesser", "Guesser", "guesser.html")]


def style_block() -> str:
    return f'<style id="lab-theme">{CSS}</style>\n'


def head(title: str) -> str:
    """<title>, fonts and theme CSS (for the index fragment, which the publisher wraps)."""
    return f"<title>{e(title)}</title>\n{FONTS}{style_block()}"


def document(title: str, body: str, extra_css: str = "") -> str:
    """A full HTML document (game pages and the Guesser page are published as extra files)."""
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            + head(title) + (f"<style>{extra_css}</style>\n" if extra_css else "")
            + "</head>\n<body>\n" + body + "</body>\n</html>\n")


def nav(active: str, prefix: str = "") -> str:
    cur = ' aria-current="page"'
    links = "".join(f'<a href="{prefix}{href}"{cur if key == active else ""}>{label}</a>' for key, label, href in TABS)
    return (f'<nav class="site" aria-label="Site"><a class="brand" href="{prefix}index.html">{CREST}NFL Matchup Lab</a>'
            f'<div class="tabs">{links}</div></nav>\n')


def refresh(doc: str) -> str:
    """Replace the theme block (or, for pages built before the theme existed, the whole old head style)."""
    if 'id="lab-theme"' in doc:
        return re.sub(r'<style id="lab-theme">.*?</style>\n?', lambda m: style_block(), doc, count=1, flags=re.S)
    return doc


# ---- team colors ------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _colors() -> dict[str, str]:
    with open(ROOT / "config" / "teams.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[r["team"]] = r["color"]
        out[r["display"]] = r["color"]
    return out


def _rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _lum(rgb) -> float:
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a, b) -> float:
    la, lb = sorted([_lum(a), _lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0, min(1, x)) * 255):02X}" for x in rgb)


IVORY, GROUND, PANEL = _rgb("#ECE5D2"), _rgb("#0B1611"), _rgb("#11201A")


def team_vars(team: str) -> str:
    """CSS custom properties for a team: --tc fill, --tt text on the fill, --ti the color as text on the panel."""
    c = _rgb(_colors().get(team, "#C8A55E"))
    tt = IVORY if _contrast(c, IVORY) >= _contrast(c, GROUND) else GROUND
    ink, k = c, 0.0
    while _contrast(ink, PANEL) < 4.5 and k < 1:
        k += 0.05
        ink = [x + (i - x) * k for x, i in zip(c, IVORY)]
    return f"--tc:{_hex(c)};--tt:{_hex(tt)};--ti:{_hex(ink)}"


def pennant(team_display: str) -> str:
    return f'<span class="pn"><span class="pennant" style="{team_vars(team_display)}">{e(team_display)}</span></span>'


def swatch(team_display: str) -> str:
    return f'<span class="sw" style="{team_vars(team_display)}" aria-hidden="true"></span>'
