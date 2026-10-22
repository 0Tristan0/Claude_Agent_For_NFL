import json, os
D = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(D, "ev.json")))
be = E["be"]
BE_LO, BE_HI = be["power6"], be["power3"]

rows = []
for k, p in sorted(E["props"].items(), key=lambda kv: -kv[1]["est"]):
    est = p["est"]
    if est >= BE_HI: v, c = "Worth it in any size", "c-hi"
    elif est >= be["power4"]: v, c = "Worth it in 4+ pick", "c-mh"
    elif est >= BE_LO: v, c = "Only in 5-6 pick", "c-md"
    else: v, c = "Below break-even", "c-lo"
    w = round(est * 100)
    rows.append(f'''<tr><th scope="row">{p["label"]}<span class="vs-lbl">Market: {p["note"]}</span></th>
<td class="num">{est:.0%}</td><td class="num">{p["mkt"]:.0%}</td>
<td><span class="meter"><span class="mfill" style="width:{w}%"></span><span class="mline" style="left:{BE_LO*100:.1f}%"></span><span class="mline" style="left:{BE_HI*100:.1f}%"></span></span></td>
<td><span class="lean {c}">{v}</span></td></tr>''')

names = {k: v["label"] for k, v in E["props"].items()}
notes = {
 "3": "Allen scores, Stafford airs it out, and Kyren catches checkdowns. Stafford and Kyren receiving rise together when the Rams throw a lot.",
 "4": "Adds Kincaid under. The Rams have allowed the fewest TE yards in the league and his role is shrinking.",
 "5": "Adds Allen under 249.5 pass yds. It pairs well with his TD leg: if the Rams take away the pass, Allen runs it in.",
 "6": "Adds Kyren over 69.5 rush yds, the weakest leg. If Kyren TD shows as a standard line in the app, swap it in (about 62%).",
}
cards = []
for n, L in E["lineups"].items():
    legs = "".join(f"<li>{names[k]} <span class='pp'>{E['props'][k]['est']:.0%}</span></li>" for k in L["legs"])
    best = "Power" if L["power_ev"] >= L["flex_ev"] + 0.03 else "Flex"
    cards.append(f'''<article class="lu">
<header><h3>{n}-pick</h3><span class="rec">Play it as <b>{best}</b></span></header>
<ol>{legs}</ol>
<dl class="lustats">
<div><dt>All hit</dt><dd>{L["p_all"]:.1%}</dd></div>
<div><dt>Power pays</dt><dd>{L["power_mult"]:g}x</dd></div>
<div><dt>Return per $1 (Power)</dt><dd>${L["power_ev"]:.2f}</dd></div>
<div><dt>Return per $1 (Flex)</dt><dd>${L["flex_ev"]:.2f}</dd></div>
<div><dt>Skip if app shows under</dt><dd>{L["min_mult"]:g}x</dd></div>
</dl>
<p>{notes[n]}</p></article>''')

css = '''<style>
.meter{position:relative;display:block;width:160px;height:10px;background:var(--track);border-radius:5px}
.mfill{position:absolute;left:0;top:0;bottom:0;background:var(--lar);border-radius:5px}
.mline{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--fg)}
.lus{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px}
.lu{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;display:grid;gap:10px;align-content:start;min-width:0}
.lu header{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.lu h3{font-size:1.6rem}
.lu .rec{font-size:.9rem;color:var(--muted)} .lu .rec b{color:var(--fg)}
.lu ol{margin:0;padding-left:1.3em;display:grid;gap:4px}
.lu .pp{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.85rem}
.lustats{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin:0;border-top:1px solid var(--line);padding-top:10px;font-variant-numeric:tabular-nums}
.lustats div{display:grid} .lustats dt{font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)} .lustats dd{margin:0;font-weight:700}
.lu p{margin:0;font-size:.93rem}
.warn{border:1.5px solid var(--t3);background:var(--t3bg);color:var(--fg);border-radius:8px;padding:12px 14px;max-width:80ch}
.warn b{color:var(--t3)}
</style>'''

sec = f'''{css}
  <section aria-labelledby="h-pay" id="payouts">
    <div>
      <h2 id="h-pay">Payout check</h2>
      <p class="sub">On PrizePicks every pick needs to hit more often than the payout implies. With standard Power payouts (3-pick 5x, 4-pick 10x, 5-pick 20x, 6-pick 37.5x) a pick has to hit about {be["power3"]:.1%} of the time in a 3-pick, {be["power4"]:.1%} in a 4-pick, and {be["power6"]:.1%} in a 6-pick. My hit estimate is half this season's game log (pulled toward 50% because it's only 4 games) plus the matchup, and half what the betting market says. The two black lines on each bar mark those break-evens.</p>
    </div>
    <div class="warn"><b>Check the multiplier in the app first.</b> PrizePicks says lineups with several players from the same game can get lower, non-standard payouts, and every pick here is from one game. Each lineup below lists the lowest multiplier where it still makes sense.</div>
    <div class="tbl-wrap"><table class="pt">
      <thead><tr><th scope="col">Pick</th><th scope="col">My est.</th><th scope="col">Market</th><th scope="col">vs break-even</th><th scope="col">Worth the payout?</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table></div>
  </section>

  <section aria-labelledby="h-lu" id="lineups">
    <div>
      <h2 id="h-lu">Lineups</h2>
      <p class="sub">Built from the picks that clear break-even, strongest first. "Return per $1" is the average payback if my estimates are right; above $1.00 means the payout is worth the risk. Even so, the 6-pick only hits about 1 time in 28.</p>
    </div>
    <div class="lus">{"".join(cards)}</div>
    <p class="caveat">These estimates come from four games of data and could easily be off by several points per pick. Only put in what you're fine losing.</p>
  </section>

  <footer class="notes">'''

f = os.path.join(D, "..", "output", "bills-rams-mnf.html")
html = open(f).read()
html = html.replace('  <footer class="notes">', sec, 1)
html = html.replace('<li>Prop lines:', '<li>Payouts: <a href="https://intercom.help/prizepicks/en/articles/9047668-payouts-explained">PrizePicks payouts explained</a></li>\n      <li>Prop lines:', 1)
open(f, "w").write(html)
print("ok")
