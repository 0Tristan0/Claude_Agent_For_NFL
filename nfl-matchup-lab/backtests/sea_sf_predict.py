"""SF at SEA, Oct 11 2026. Same fixed method as the Bills-Rams page.
Inputs are pre-game only (Weeks 1-4 logs, pre-game lines). Written before pulling the result."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))

def shrink(h, n, prior=5): return (h + prior * 0.5) / (n + prior)
def fair(american, vig=0.025):
    p = 100 / (american + 100) if american > 0 else -american / (-american + 100)
    return p - vig

# Game model: each side's points = average of its PF/g and the opponent's PA/g (Weeks 1-4)
SF_PF, SF_PA = (27+35+36+24)/4, (7+13+30+14)/4
SEA_PF, SEA_PA = (13+31+31+30)/4, (10+7+33+23)/4
sf = (SF_PF + SEA_PA)/2; sea = (SEA_PF + SF_PA)/2
game = {"sf_pts": round(sf,1), "sea_pts": round(sea,1), "total": round(sf+sea,1),
        "winner": "SF" if sf > sea else "SEA", "line": "SEA -3", "ou": 46.5,
        "spread_pick": "SF +3" if (sf - sea) > -3 else "SEA -3",
        "total_pick": "Under 46.5" if sf+sea < 46.5 else "Over 46.5"}

# Props: (label, side, line, log values that count, hit test, market fair prob, matchup adj, note)
props = [
 ("McCaffrey rush yds", "under", 49.5, [68,23,75,52], 0.54, +0.04, "SEA 3rd in rush success rate D"),
 ("McCaffrey anytime TD", "yes", None, [0,2,1,1], fair(-135), 0, ""),
 ("Kittle anytime TD", "yes", None, [0,1,2,1], fair(185), 0, ""),
 ("Evans anytime TD", "yes", None, [1,0,1,0], fair(200), 0, ""),
 ("Deebo anytime TD", "yes", None, [1,0,1,1], fair(240), 0, "Week 3 log row looks off"),
 ("Deebo rush yds", "over", 9.5, [12,4,14,44], 0.49, 0, ""),
 ("JSN anytime TD", "yes", None, [1,3,2,0], fair(-120), 0, ""),
 ("Wilson receptions", "over", 1.5, [0,0,3], fair(-147), +0.05, "SF allowed most RB catches; Week 3 missing"),
 ("Black rush yds", "under", 15.5, [65,25,13], fair(-112), +0.03, "Week 3 missing"),
 ("Barner rec yds", "under", 24.5, None, 0.56, +0.04, "under in 3 of 4 per Covers"),
 ("Darnold pass TDs", "over", 1.5, [0,4,2], fair(-115), 0, "left Week 1 early, missed Week 2"),
]
out = []
for lab, side, line, log, mkt, adj, note in props:
    if log is None: h, n = 3, 4
    elif side == "yes": h, n = sum(1 for v in log if v >= 1), len(log)
    elif side == "over": h, n = sum(1 for v in log if v > line), len(log)
    else: h, n = sum(1 for v in log if v < line), len(log)
    est = round((shrink(h, n) + adj + mkt) / 2, 3)
    # pick whichever side our estimate beats the market on (value, not just "more likely")
    pick_side, p = (side, est) if est >= mkt else ({"over":"under","under":"over","yes":"no"}[side], round(1-est,3))
    edge = round(p - (mkt if pick_side == side else 1-mkt), 3)
    out.append({"prop": lab, "line": line, "listed_side": side, "hits": f"{h}/{n}", "pick": pick_side,
                "est": p, "market": round(mkt if pick_side == side else 1-mkt, 3), "edge": edge, "note": note})
json.dump({"game": game, "props": out}, open(os.path.join(D, "sea_sf_predictions.json"), "w"), indent=1)
print(json.dumps(game, indent=1))
for o in out: print(f'{o["prop"]:22} {str(o["line"]):5} {o["pick"]:6} est {o["est"]:.0%}  mkt {o["market"]:.0%}  edge {o["edge"]:+.0%}  ({o["hits"]})')
