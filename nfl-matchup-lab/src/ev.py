import json, os, itertools
from math import comb
D = os.path.dirname(os.path.abspath(__file__))

def shrink(h, n, prior=5):  # pull small samples toward 50%
    return (h + prior * 0.5) / (n + prior)

# key: (label, hits, games, matchup_adj, market_fair_prob, market_note)
P = {
 "allen_td":  ("Allen Rush+Rec TDs O 0.5", 4, 4, +0.04, 0.52, "+100 to -124 at books, 52¢ Polymarket"),
 "staff_py":  ("Stafford Pass Yds O 274.5", 3, 4, +0.03, 0.545, "54¢ Polymarket, 55% Kalshi"),
 "kyren_rec": ("K. Williams Rec Yds O 19.5", 3, 4, +0.03, 0.52, "-114 FanDuel"),
 "kinc_u":    ("Kincaid Rec Yds U 57.5", 2, 4, +0.08, 0.52, "-115 bet365"),
 "allen_pu":  ("Allen Pass Yds U 249.5", 2, 4, +0.08, 0.52, "48% over on Kalshi"),
 "kyren_ru":  ("K. Williams Rush Yds O 69.5", 3, 4, -0.01, 0.475, "under 52-53% on Kalshi/Polymarket"),
 "kyren_td":  ("K. Williams Rush+Rec TDs O 0.5", 3, 4, +0.02, 0.60, "-170 at books"),
 "nacua":     ("Nacua Rec Yds O 94.5", 1, 2, +0.03, 0.52, "o90 at -140 Predictem"),
 "cook_ry":   ("Cook Rush Yds O 75.5", 2, 4, +0.02, 0.50, "Covers projects 66"),
 "shakir_u":  ("Shakir Rec Yds U 49.5", 2, 4, +0.01, 0.55, "45% over on Kalshi"),
 "cook_rec":  ("Cook Receptions O 1.5", 2, 4, 0.0, 0.53, "53% on Kalshi"),
 "staff_ru":  ("Stafford Rush Yds U 9.5", 2, 4, 0.0, 0.50, "no price found"),
 "park":      ("Parkinson Rec Yds O 14.5", 1, 2, 0.0, 0.56, "56% on Kalshi"),
}
est = {}
for k, (lab, h, n, adj, mkt, note) in P.items():
    model = shrink(h, n) + adj
    est[k] = round((model + mkt) / 2, 3)   # half our read, half the market

POWER = {2: 3, 3: 5, 4: 10, 5: 20, 6: 37.5}
FLEX = {3: {3: 2.25, 2: 1.25}, 4: {4: 5, 3: 1.5}, 5: {5: 10, 4: 2, 3: 0.4}, 6: {6: 25, 5: 2, 4: 0.4}}

def be_power(n):  # per-leg break-even if all legs equal
    return (1 / POWER[n]) ** (1 / n)

def flex_ev_equal(n, p):
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) * m for k, m in FLEX[n].items())

def be_flex(n):
    lo, hi = 0.3, 0.9
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if flex_ev_equal(n, mid) < 1 else (lo, mid)
    return lo

def flex_ev(ps):
    n = len(ps); ev = 0
    for combo in itertools.product([0, 1], repeat=n):
        pr = 1
        for c, p in zip(combo, ps): pr *= p if c else (1 - p)
        ev += pr * FLEX[n].get(sum(combo), 0)
    return ev

out = {"props": {}, "be": {}, "lineups": {}}
for n in range(2, 7):
    out["be"][f"power{n}"] = round(be_power(n), 4)
for n in range(3, 7):
    out["be"][f"flex{n}"] = round(be_flex(n), 4)
for k, (lab, h, n, adj, mkt, note) in P.items():
    out["props"][k] = {"label": lab, "est": est[k], "mkt": mkt, "note": note, "hits": f"{h}/{n}"}

L = {
 3: ["allen_td", "staff_py", "kyren_rec"],
 4: ["allen_td", "staff_py", "kyren_rec", "kinc_u"],
 5: ["allen_td", "staff_py", "kyren_rec", "kinc_u", "allen_pu"],
 6: ["allen_td", "staff_py", "kyren_rec", "kinc_u", "allen_pu", "kyren_ru"],
}
for n, legs in L.items():
    ps = [est[k] for k in legs]
    p_all = 1
    for p in ps: p_all *= p
    out["lineups"][n] = {
        "legs": legs, "p_all": round(p_all, 4),
        "power_mult": POWER[n], "power_ev": round(p_all * POWER[n], 3),
        "min_mult": round(1 / p_all, 1),
        "flex_ev": round(flex_ev(ps), 3),
    }
json.dump(out, open(os.path.join(D, "ev.json"), "w"), indent=1)
for k, v in out["props"].items(): print(f'{v["label"]:34} est {v["est"]:.1%}  mkt {v["mkt"]:.1%}')
print(out["be"])
for n, v in out["lineups"].items(): print(n, v)
