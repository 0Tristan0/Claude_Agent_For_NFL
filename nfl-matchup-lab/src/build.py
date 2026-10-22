import json, os
D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(D, "stats.json")))

def tier(r):
    return "t1" if r <= 8 else "t2" if r <= 16 else "t3" if r <= 24 else "t4"

def pill(team, key):
    c = S[key][team]
    t = "T" if c["tie"] else ""
    return f'<span class="pill {tier(c["rank"])}">#{c["rank"]}{t}</span>'

def bar(team, key, cls):
    r = S[key][team]["rank"]
    pct = round((33 - r) / 32 * 100)
    return f'<span class="bar"><span class="fill {cls}" style="width:{pct}%"></span></span>'

def row(o_team, o_key, d_team, d_key):
    ro, rd = S[o_key][o_team]["rank"], S[d_key][d_team]["rank"]
    if abs(ro - rd) <= 3:
        edge = '<span class="edge even">Even</span>'
    elif ro < rd:
        edge = f'<span class="edge {o_team.lower()}">{o_team} O</span>'
    else:
        edge = f'<span class="edge {d_team.lower()}">{d_team} D</span>'
    oc, dc = o_team.lower(), d_team.lower()
    return f'''<tr>
<th scope="row">{S[o_key]["label"]}<span class="vs-lbl">vs {S[d_key]["label"].lower()}</span></th>
<td class="side"><span class="v">{S[o_key][o_team]["val"]}</span>{pill(o_team,o_key)}{bar(o_team,o_key,oc)}</td>
<td class="side"><span class="v">{S[d_key][d_team]["val"]}</span>{pill(d_team,d_key)}{bar(d_team,d_key,dc)}</td>
<td class="e">{edge}</td></tr>'''

PAIRS = [("o_ppg","d_ppg"),("o_ypg","d_ypg"),("o_pypg","d_pypg"),("o_ypa","d_ypa"),
         ("o_cmp","d_cmp"),("o_rate","d_rate"),("o_sk","d_sk"),("o_int","d_int"),
         ("o_rypg","d_rypg"),("o_ypc","d_ypc")]

def table(o, d, cap):
    rows = "\n".join(row(o, a, d, b) for a, b in PAIRS)
    return f'''<div class="tbl-wrap"><table class="mt">
<caption>{cap}</caption>
<thead><tr><th scope="col">Stat</th><th scope="col"><span class="tchip {o.lower()}">{o}</span> offense</th><th scope="col"><span class="tchip {d.lower()}">{d}</span> defense</th><th scope="col">Edge</th></tr></thead>
<tbody>{rows}</tbody></table></div>'''

COV = ["d_pypg","d_ypa","d_cmp","d_rate","d_ptd","d_int","d_sk"]
cov_rows = "\n".join(f'''<tr><th scope="row">{S[k]["label"]}</th>
<td class="side"><span class="v">{S[k]["BUF"]["val"]}</span>{pill("BUF",k)}{bar("BUF",k,"buf")}</td>
<td class="side"><span class="v">{S[k]["LAR"]["val"]}</span>{pill("LAR",k)}{bar("LAR",k,"lar")}</td></tr>''' for k in COV)

html = open(os.path.join(D, "template.html")).read()
html = html.replace("{{BUF_TABLE}}", table("BUF","LAR","Bills offense against the Rams defense"))
html = html.replace("{{LAR_TABLE}}", table("LAR","BUF","Rams offense against the Bills defense"))
html = html.replace("{{COV_ROWS}}", cov_rows)
open(os.path.join(D, "..", "output", "bills-rams-mnf.html"), "w").write(html)
print("ok", len(html))
