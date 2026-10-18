import os, json
D = os.path.dirname(os.path.abspath(__file__))

# Game logs (FootballDB, 2026 regular season, in order). None = did not play.
P = [
 # player, team, prop, line, src, log values, side we evaluate, lean, conf, why
 ("Josh Allen","BUF","Anytime TD","Yes","Books (no PP line posted)",[2,2,2,1],"td","Yes","High",
  "Scored a rushing TD in all 4 games. 6 TDs on 9 red zone carries. Rams allow a TD on 58.3% of red zone trips and are only 12th in yards per carry allowed."),
 ("Matthew Stafford","LAR","Pass yards",274.5,"Books / Covers",[155,327,390,317],"over","Over","High",
  "Over in 3 of 4, the only miss was the opener vs SF. Bills are 29th in pass yards allowed (274 gross a game). Rams throw 41 times a game. Risk: Rousseau pressure."),
 ("Kyren Williams","LAR","Receiving yards",19.5,"Books / Yahoo",[24,12,70,67],"over","Over","High",
  "Over in 3 of 4 and 67+ the last two weeks. Bills allow the 6th most catches to RBs, and their pass rush forces quick checkdowns."),
 ("Dalton Kincaid","BUF","Receiving yards",57.5,"Books / Yahoo",[130,95,38,7],"under","Under","High",
  "Under the last 2 games (38, 7) with his route share dropping. Rams allow the fewest TE yards in the NFL: 85 total on 12 catches."),
 ("Josh Allen","BUF","Pass yards",249.5,"Books / Covers",[334,248,204,253],"under","Under","Med-High",
  "Under in 2 of 4. Rams are #1 in pass yards allowed (145.5 net a game) and #1 in yards per attempt allowed (4.84). Covers' model says over; the ranks say under."),
 ("Kyren Williams","LAR","Rush yards",69.5,"Books / Covers",[41,85,88,80],"over","Over","Medium",
  "Over in 3 of 4. Bills are 18th in rush yards allowed, give up 3.05 yards after contact a carry, and are missing DT Ed Oliver. Risk: Corum takes about 38% of carries, and two models lean under."),
 ("Kyren Williams","LAR","Anytime TD","Yes","Books",[1,1,0,2],"td","Yes","Medium",
  "Scored in 3 of 4. Rams are 3.5-point favorites at home. The price is steep (around -170)."),
 ("Puka Nacua","LAR","Receiving yards",94.5,"PrizePicks",[74,None,None,125],"over","Over","Medium",
  "Best matchup on the board (Bills 29th vs the pass), but the line is high, he has only 2 games played, and Adams splits targets. Over in 1 of 2."),
 ("James Cook","BUF","Rush yards",75.5,"PrizePicks",[57,135,154,75],"over","Slight over","Coin flip",
  "Over in 2 of 4 and missed by half a yard in Week 4. Rams run D is 13th, much softer than their pass D, so Buffalo should lean on him. Covers projects 66. Aaron Donald returning hurts."),
 ("Khalil Shakir","BUF","Receiving yards",49.5,"Books / Covers",[50,38,6,82],"under","Slight under","Coin flip",
  "Under in 2 of 4 against the #1 pass D. If DJ Moore sits, Shakir's targets go up, which is why this is close."),
 ("James Cook","BUF","Receptions",1.5,"Books / Covers",[3,1,0,2],"over","Pass","Low",
  "Over in 2 of 4. No clear edge either way."),
 ("Matthew Stafford","LAR","Rush yards",9.5,"Books / Predictem",[-1,-1,13,10],"under","Pass","Low",
  "Under in 2 of 4. Two kneel-downs or one scramble decides it."),
 ("Colby Parkinson","LAR","Receiving yards",14.5,"Books / Covers",[6,15,None,None],"over","Pass","Low",
  "Only 2 games played this year. Not enough to go on."),
]

def hit(v, line, side):
    if v is None: return None
    if side == "td": return v >= 1
    return v > line if side == "over" else v < line

rows = []
for name, team, prop, line, src, log, side, lean, conf, why in P:
    res = [hit(v, line, side) for v in log]
    played = [r for r in res if r is not None]
    n, h = len(played), sum(played)
    dots = "".join(
        f'<span class="dot {"hit" if r else "miss"}" title="{("TD" if side=="td" else "")}{log[i] if log[i] is not None else ""}"></span>' if r is not None
        else '<span class="dot dnp" title="Did not play"></span>' for i, r in enumerate(res))
    vals = " · ".join("DNP" if v is None else (f"{v} TD" if side == "td" else str(v)) for v in log)
    side_lbl = "Yes" if side == "td" else side.capitalize()
    cc = {"High":"c-hi","Med-High":"c-mh","Medium":"c-md","Coin flip":"c-cf","Low":"c-lo"}[conf]
    line_txt = line if isinstance(line, str) else f"{line}"
    rows.append(f'''<tr>
<th scope="row"><span class="tchip {team.lower()}">{team}</span> {name}<span class="vs-lbl">{prop} · {src}</span></th>
<td class="num">{line_txt}</td>
<td><span class="dots">{dots}</span><span class="hr">{side_lbl} {h}/{n}</span><span class="vs-lbl">{vals}</span></td>
<td><span class="lean {cc}">{lean}</span><span class="conf">{conf}</span></td>
<td class="why">{why}</td></tr>''')

props_html = f'''
  <section aria-labelledby="h-props" id="props">
    <div>
      <h2 id="h-props">Props board</h2>
      <p class="sub">PrizePicks only published two lines for this game outside the app (Cook 75.5 rush yds and Nacua 94.5 rec yds). The rest are sportsbook lines from the same day; PrizePicks usually sits within a point or two of these, so check the app before you lock anything in. Dots show each game this season, green if that side would have hit.</p>
    </div>
    <div class="tbl-wrap"><table class="pt">
      <thead><tr><th scope="col">Player / prop</th><th scope="col">Line</th><th scope="col">This season</th><th scope="col">Lean</th><th scope="col">Why</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table></div>
    <div class="takes">
      <div class="take"><b>Strongest plays</b><p>Allen anytime TD, Stafford over 274.5 pass yds, Kyren Williams over 19.5 rec yds, Kincaid under 57.5 rec yds. Each has both the game logs and the matchup ranks pointing the same way.</p></div>
      <div class="take"><b>Where ranks beat the models</b><p>Allen under 249.5 pass yds. The Rams have the #1 pass D by yards and yards per attempt, so the over projection from Covers looks too high.</p></div>
      <div class="take"><b>Game total check</b><p>Averaging each offense's points with what the other D allows gives about BUF 26, LAR 24, or 50 total. That leans under 54.5 and makes Bills +3.5 interesting, though CBS and AP both pick the Rams.</p></div>
    </div>
    <p class="caveat">Four games is a small sample, and none of this is a guarantee. Treat it as a read on the matchup, not a sure thing.</p>
  </section>'''

M = [
 ("Puka Nacua","LAR","Bills slot / nickel (Dee Alford questionable)","BUF",
  "LAR","Nacua moves around and plays inside, so Buffalo can't shadow him. Bills are 29th in pass yds allowed and 30th in success rate allowed."),
 ("Davante Adams","LAR","CBs Christian Benford & Maxwell Hairston","BUF",
  "LAR","Adams works outside and is the red zone target. Benford returns from a toe injury; if he's limited, rookie Davison Igbinosun plays. Adams has 10.3 yds per target, a career best."),
 ("Keon Coleman","BUF","CB Trent McDuffie","LAR",
  "LAR","CBS expects McDuffie to travel with Coleman. Rams allow 53.5% completions (2nd) and 4.84 yds per attempt (1st)."),
 ("Dalton Kincaid","BUF","Rams LBs / TE coverage","LAR",
  "LAR","Rams have allowed the fewest TE yards in the league. Kincaid's usage has dropped two weeks in a row."),
 ("Greg Rousseau","BUF","Rams pass protection","LAR",
  "EVEN","Rousseau leads the NFL in sacks (6) with 29 pressures. Rams allow only 1.5 sacks a game (6th), but Stafford was sacked 4 times at Denver. Ed Oliver being out helps the Rams."),
 ("James Cook","BUF","Rams run defense (Aaron Donald back)","LAR",
  "BUF","Cook leads the NFL with 15 runs of 10+ yards. Rams use light boxes and rank 15th in EPA per run vs 2nd per pass."),
 ("Kyren Williams & Blake Corum","LAR","Bills tackling, no Ed Oliver","BUF",
  "LAR","Bills give up 3.05 yards after contact per carry. Williams leads the NFL with a 64.8% rushing success rate."),
 ("Josh Allen in the red zone","BUF","Rams red zone D","LAR",
  "BUF","6 TDs on 9 red zone carries. Rams give up a TD on 58.3% of red zone trips."),
]
cards = []
for off, oteam, dfn, dteam, win, why in M:
    wl = {"LAR":'<span class="win lar">Edge: Rams</span>', "BUF":'<span class="win buf">Edge: Bills</span>', "EVEN":'<span class="win even">Toss-up</span>'}[win]
    cards.append(f'''<article class="mu"><div class="mu-top"><span><span class="tchip {oteam.lower()}">{oteam}</span> {off}</span><span class="vs">vs</span><span><span class="tchip {dteam.lower()}">{dteam}</span> {dfn}</span></div>{wl}<p>{why}</p></article>''')
oneone_html = f'''
  <section aria-labelledby="h-1v1">
    <div>
      <h2 id="h-1v1">1-on-1 matchups</h2>
      <p class="sub">Who should win each key battle, based on the ranks above and this week's previews. Rams win 5, Bills win 2, one is even.</p>
    </div>
    <div class="mus">{"".join(cards)}</div>
  </section>'''

css = '''
<style>
.pt{min-width:860px}
.pt td{vertical-align:top}
.pt .num{font-weight:700;font-size:1.05rem;white-space:nowrap}
.dots{display:inline-flex;gap:4px;margin-right:8px;vertical-align:middle}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block;border:1.5px solid var(--line)}
.dot.hit{background:var(--t1);border-color:var(--t1)}
.dot.miss{background:var(--t4bg);border-color:var(--t4)}
.dot.dnp{background:transparent;border-style:dashed}
.hr{font-weight:700;white-space:nowrap}
.lean{display:block;font-family:var(--display);font-weight:700;font-size:1.05rem;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
.conf{display:block;font-size:.8rem;color:var(--muted)}
.c-hi{color:var(--t1)} .c-mh{color:var(--t2)} .c-md{color:var(--t3)} .c-cf,.c-lo{color:var(--muted)}
.why{font-size:.92rem;max-width:44ch;min-width:28ch}
.caveat{color:var(--muted);font-size:.9rem;margin:0}
.mus{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:12px}
.mu{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:grid;gap:8px;align-content:start;min-width:0}
.mu-top{display:grid;gap:2px;font-weight:600}
.mu .vs{font-family:var(--display);color:var(--muted);font-size:.85rem;letter-spacing:.1em;text-transform:uppercase}
.mu p{margin:0;font-size:.95rem}
.win{justify-self:start;font-family:var(--display);font-weight:700;letter-spacing:.05em;text-transform:uppercase;font-size:.95rem;padding:2px 10px;border-radius:4px;border:1.5px solid currentColor}
.win.lar{color:var(--lar)} .win.buf{color:var(--buf)} .win.even{color:var(--muted)}
</style>'''

html = open(os.path.join(D, "..", "output", "bills-rams-mnf.html")).read()
html = html.replace("{{ONEONE}}", css + oneone_html).replace("{{PROPS}}", props_html)
open(os.path.join(D, "..", "output", "bills-rams-mnf.html"), "w").write(html)
print("ok", "{{" in html)
