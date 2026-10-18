"""
Nebius Group (NASDAQ: NBIS) - independent valuation model.
Valuation date: 2026-10-18 (price = 2026-10-16 close, $165.47).

Run:  python3 nbis_model.py
Writes: NBIS_model.xlsx and model_output.md next to this file.

Every hard-coded input is tagged in INPUT_SOURCES with its provenance:
  [F]  confirmed fact (company filing / release, cross-checked)
  [G]  management guidance
  [C]  analyst consensus / third-party data
  [A]  analyst (my) assumption
All money in US$ billions unless noted; shares in millions.
"""
from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = list(range(2026, 2036))  # explicit forecast 2026-2035

# ----------------------------------------------------------------------------
# 1. MARKET / BALANCE-SHEET INPUTS
# ----------------------------------------------------------------------------
PRICE = 165.47                       # [F] close 2026-10-16 (GuruFocus/MarketBeat)

# Share count
BASIC_SHARES_JUN30 = 271.855         # [F] Class A+B outstanding 30-Jun-26 (= AGM record count)
EXCHANGE_SHARES = 15.8               # [F] Aug-26 exchange of $400m 2029s + $400m 2031s
NVDA_PREFUNDED_WARRANT = 21.066      # [F] pre-funded warrant, $0.0001 strike (exercisable from 11-Sep-26)
UNVESTED_EQUITY_AWARDS = 8.0         # [A] RSUs/options; SBC ~$100m/qtr at ~$200/sh, 4-yr vest
BASE_DILUTED = BASIC_SHARES_JUN30 + EXCHANGE_SHARES + NVDA_PREFUNDED_WARRANT + UNVESTED_EQUITY_AWARDS

# Convertible notes outstanding after Aug-2026 transactions (principal, conversion price)
CONVERTS = [
    # name,                   principal $bn, conv. price $, coupon
    ("1.00% 2029 (Jun-25)",    0.100,  51.45, 0.0100),   # [F] $500m less $400m exchanged
    ("2.75% 2031 (Jun-25)",    0.100,  51.45, 0.0275),   # [F] $500m less $400m exchanged
    ("1.00% 2030 (Sep-25)",    1.58125, 138.75, 0.0100), # [F] $1.375bn + $206.25m option
    ("2.75% 2032 (Sep-25)",    1.58125, 138.75, 0.0275), # [F]
    ("1.25% 2031 (Mar-26)",    2.590, 183.22, 0.0125),   # [F]
    ("2.625% 2033 (Mar-26)",   1.750, 180.31, 0.02625),  # [F]
    ("0.50% 2030 (Aug-26)",    3.450, 313.46, 0.0050),   # [F] total $5.75bn; split between series [A]
    ("4.50% 2034 (Aug-26)",    2.300, 324.65, 0.0450),   # [F] conv. price; split [A]
]
SECURED_FACILITY = 0.775             # [F] Jul-26 GPU-backed facility, SOFR+250, due Oct-30
LEASES_OTHER_DEBT = 1.50             # [A] "debt + leases" ~$10.06bn (TIKR) less converts carrying ~$8.5bn

CASH_JUN30 = 8.042                   # [F] cash & equivalents 30-Jun-26
AUG26_CONVERT_NET_PROCEEDS = 5.60    # [A] $5.75bn gross less ~2.5% fees
CASH_PROFORMA = CASH_JUN30 + AUG26_CONVERT_NET_PROCEEDS + SECURED_FACILITY

DEFERRED_REV_CURRENT = 0.9794        # [F] 30-Jun-26
DEFERRED_REV_NONCURRENT = 4.9958     # [F] 30-Jun-26

# H1-2026 actuals (to carve out the H2 stub)
H1_REVENUE = 0.3990 + 0.5823         # [F] Q1 + Q2
H1_ADJ_EBITDA = 0.1295 + 0.2362      # [F]
H1_CAPEX = 2.47 + 5.70               # [F] approx. as reported
H1_SBC = 0.0900 + 0.1025             # [A] Q1 approx; Q2 $102.5m [F]

HIST_CAPEX = {2023: 0.3, 2024: 0.9, 2025: 4.1}   # 2025 [F]; 2023-24 [A] (immaterial)
EXIT_ARR_2025 = 1.25                 # [F]
NOL_START = 2.0                      # [A] usable tax losses at 1-Jan-26

# ----------------------------------------------------------------------------
# 2. DISCOUNT RATE
# ----------------------------------------------------------------------------
RF = 0.0465        # [A] 10y UST ~4.6-4.7% (4.68% on 30-Jul-26, FRED)
ERP = 0.050        # [A]
BETA = 1.80        # [A] relevered; NBIS realised beta well above 2 but includes meme/flow vol
KD_PRE_TAX = 0.075 # [A] secured GPU debt ~6.8% (SOFR+250); unsecured HY marginal cost higher
TAX = 0.23         # [A] blended NL 25.8% / US 21% / FI 20%
D_V = 0.30         # [A] long-run target leverage
KE = RF + BETA * ERP
WACC = (1 - D_V) * KE + D_V * KD_PRE_TAX * (1 - TAX)
# Terminal stage: a mature, 3%-growth GPU cloud should not carry a 1.8 beta.
BETA_MATURE = 1.30  # [A] equipment-rental / data-centre analogues
KE_MATURE = RF + BETA_MATURE * ERP
WACC_TERMINAL = (1 - D_V) * KE_MATURE + D_V * KD_PRE_TAX * (1 - TAX)
# Equity-as-option floor (Merton): within-scenario asset volatility and average claim maturity
ASSET_VOL = 0.35    # [A]
CLAIM_T = 4.5       # [A] converts mature 2029-2034

# ----------------------------------------------------------------------------
# 3. SCENARIOS  (operating drivers)
# ----------------------------------------------------------------------------
BASE = dict(
    name="Base",
    prob=0.45,
    # Exit ARR, $bn, end of year 2026..2035
    exit_arr=[7.5, 15.5, 24.0, 31.5, 38.0, 43.0, 47.0, 50.0, 52.5, 54.5],
    rev_2026=3.15,
    realisation=0.92,             # revenue / average(open,close ARR): adds are back-end loaded
    capex_2026=22.0,
    capex_per_arr=2.7,            # $ capex per $1 of new exit ARR (GPU-hour arithmetic: owned ~2.75x, leased ~1.9x + rent)
    capex_lead=0.35,              # share of capacity capex spent the year before it goes live
    gpu_share=0.75,               # compute share of capex (rest = shell/power/cooling)
    gpu_econ_life=6,              # years before replacement capex (book life 5y; industry debate 4-7y)
    refresh_cost=0.85,            # replacement cost vs original (perf/W gains)
    ebitda_margin=[0.40, 0.50, 0.55, 0.58, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
    sbc_pct=[0.12, 0.06, 0.045, 0.035, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    terminal_g=0.03,
    ronic=0.12,
    # Equity stakes (Nebius share, $bn)
    clickhouse=3.4,               # ~27% x $15bn (Jan-26 Series D) less 15% illiquidity haircut
    avride=1.0,
    tripleten_toloka=0.4,
)

BULL = deepcopy(BASE)
BULL.update(
    name="Bull",
    prob=0.25,
    exit_arr=[8.5, 20.0, 31.0, 41.0, 50.0, 57.0, 63.0, 68.0, 72.0, 75.0],
    rev_2026=3.35,
    capex_2026=24.0,
    capex_per_arr=2.4,
    gpu_econ_life=7,
    refresh_cost=0.80,
    ebitda_margin=[0.41, 0.53, 0.59, 0.62, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64],
    ronic=0.16,
    clickhouse=6.0,
    avride=2.0,
    tripleten_toloka=0.5,
)

BEAR = deepcopy(BASE)
BEAR.update(
    name="Bear",
    prob=0.30,
    # Pricing resets and funding tightens: management finishes contracted builds (Meta firm
    # orders deploy 2027-28) then stops growth capex and does NOT refresh compute (refresh IRR
    # is negative at these economics) -> fleet runs off. Terminal = residual value of powered
    # sites / shells sold to a hyperscaler.
    exit_arr=[6.8, 11.5, 13.0, 12.5, 11.0, 9.0, 6.5, 4.0, 2.5, 1.5],
    no_refresh=True,
    terminal_residual=6.0,        # [A] ~2 GW of powered/built capacity at ~$3m/MW in 2035
    rev_2026=2.95,
    capex_2026=21.0,
    capex_per_arr=3.2,
    gpu_econ_life=5,
    refresh_cost=0.90,
    ebitda_margin=[0.38, 0.45, 0.48, 0.48, 0.46, 0.44, 0.42, 0.40, 0.40, 0.40],
    ronic=0.09,
    clickhouse=2.5,
    avride=0.4,
    tripleten_toloka=0.2,
)

BASE["prob"], BULL["prob"], BEAR["prob"] = 0.50, 0.25, 0.25
SCENARIOS = [BEAR, BASE, BULL]

# ----------------------------------------------------------------------------
# 4. ENGINE
# ----------------------------------------------------------------------------

def build_forecast(s: dict, wacc: float = WACC) -> pd.DataFrame:
    arr = dict(zip(YEARS, s["exit_arr"]))
    arr[2025] = EXIT_ARR_2025
    arr_next = {y: arr.get(y + 1, arr[y] * (1 + s["terminal_g"])) for y in YEARS}

    rows = []
    capex_hist = dict(HIST_CAPEX)
    gpu_book = []        # (year, gpu capex) for straight-line book depreciation (5 yrs, policy)
    dc_book = []         # (year, dc capex) 20 yrs
    nol = NOL_START
    for i, y in enumerate(YEARS):
        prev_arr = arr[y - 1]
        rev = s["rev_2026"] if y == 2026 else s["realisation"] * (prev_arr + arr[y]) / 2
        margin = s["ebitda_margin"][i]
        ebitda = rev * margin
        sbc = rev * s["sbc_pct"][i]

        d_arr = max(arr[y] - prev_arr, 0.0)
        d_arr_next = max(arr_next[y] - arr[y], 0.0)
        if y == 2026:
            growth_capex = s["capex_2026"]
        else:
            growth_capex = s["capex_per_arr"] * ((1 - s["capex_lead"]) * d_arr + s["capex_lead"] * d_arr_next)
        # Replacement of compute bought `life` years earlier
        old = capex_hist.get(y - s["gpu_econ_life"], 0.0)
        refresh_capex = 0.0 if s.get("no_refresh") else old * s["gpu_share"] * s["refresh_cost"]
        capex = growth_capex + refresh_capex
        capex_hist[y] = capex

        gpu_book.append((y, capex * s["gpu_share"]))
        dc_book.append((y, capex * (1 - s["gpu_share"])))
        # Book D&A (policy: servers/network 5y since Q1-26; DC 20y), half-year convention
        def sl(book, life):
            tot = 0.0
            for (yy, c) in book:
                age = y - yy
                if age == 0:
                    tot += c / life / 2
                elif 0 < age < life:
                    tot += c / life
                elif age == life:
                    tot += c / life / 2
            return tot
        # include pre-2026 capex in the book
        pre = [(yy, c * s["gpu_share"]) for yy, c in HIST_CAPEX.items()]
        pre_dc = [(yy, c * (1 - s["gpu_share"])) for yy, c in HIST_CAPEX.items()]
        da = sl(pre + gpu_book, 5) + sl(pre_dc + dc_book, 20)

        ebit = ebitda - sbc - da
        taxable = ebit
        use = min(max(taxable, 0), nol)
        nol = nol - use + max(-taxable, 0)
        tax = max(taxable - use, 0) * TAX
        fcf = ebitda - sbc - tax - capex
        rows.append(dict(year=y, exit_arr=arr[y], revenue=rev, rev_growth=None, adj_ebitda=ebitda,
                         adj_ebitda_margin=margin, sbc=sbc, ebitda_post_sbc=ebitda - sbc,
                         d_and_a=da, ebit=ebit, ebit_margin=ebit / rev, cash_tax=tax,
                         growth_capex=growth_capex, refresh_capex=refresh_capex, capex=capex,
                         capex_to_rev=capex / rev, ufcf=fcf))
    df = pd.DataFrame(rows).set_index("year")
    df["rev_growth"] = df["revenue"].pct_change()
    df.loc[2026, "rev_growth"] = s["rev_2026"] / 0.5298 - 1   # FY25 revenue $529.8m [F]

    # ---- H2-2026 stub (valuation date ~ Oct-26; balance sheet = 30-Jun-26 pro forma)
    h2 = df.loc[2026].copy()
    h2_ebitda = h2["adj_ebitda"] - H1_ADJ_EBITDA
    h2_sbc = h2["sbc"] - H1_SBC
    h2_capex = h2["capex"] - H1_CAPEX
    df.loc[2026, "ufcf_valued"] = h2_ebitda - h2_sbc - h2_capex
    for y in YEARS[1:]:
        df.loc[y, "ufcf_valued"] = df.loc[y, "ufcf"]

    # discount factors: valuation date 2026-10-18; H2-26 stub mid-point ~ 2026-11-15
    t = {2026: 0.08}
    for y in YEARS[1:]:
        t[y] = (y - 2026) + 0.5 - 0.80     # mid-year of y measured from 18-Oct-26
    df["t"] = pd.Series(t)
    df["disc_factor"] = 1 / (1 + wacc) ** df["t"]
    df["pv_ufcf"] = df["ufcf_valued"] * df["disc_factor"]
    return df


def terminal_value(df: pd.DataFrame, s: dict, wacc: float = WACC, g: float | None = None) -> dict:
    """Value-driver TV: steady-state NOPAT uses ECONOMIC depreciation (replacement capex of the
    installed compute base + DC maintenance), not book D&A."""
    g = s["terminal_g"] if g is None else g
    last = df.loc[2035]
    # installed compute base = compute capex over last `life` years
    life = s["gpu_econ_life"]
    installed_gpu = sum(df.loc[y, "capex"] for y in range(2036 - life, 2036) if y in df.index) * s["gpu_share"]
    installed_dc = sum(df["capex"]) * (1 - s["gpu_share"]) + sum(HIST_CAPEX.values()) * (1 - s["gpu_share"])
    econ_dep = installed_gpu / life * s["refresh_cost"] + installed_dc / 25
    rev_t1 = last["revenue"] * (1 + g)
    ebitda_post_sbc_t1 = last["ebitda_post_sbc"] * (1 + g)
    nopat_t1 = (ebitda_post_sbc_t1 - econ_dep * (1 + g)) * (1 - TAX)
    reinvest_rate = g / s["ronic"]
    fcf_t1 = nopat_t1 * (1 - reinvest_rate)
    w_t = WACC_TERMINAL + (wacc - WACC)   # sensitivities shift both stages together
    tv = fcf_t1 / (w_t - g)
    pv_tv = tv * last["disc_factor"]
    return dict(rev_t1=rev_t1, econ_dep=econ_dep, nopat_t1=nopat_t1,
                nopat_margin=nopat_t1 / rev_t1, fcf_t1=fcf_t1, tv=tv, pv_tv=pv_tv,
                tv_ev_ebitda=tv / (last["adj_ebitda"] * (1 + g)))


def pv_deferred_revenue(wacc: float = WACC) -> float:
    cur = DEFERRED_REV_CURRENT / (1 + wacc) ** 0.5
    nc = sum(DEFERRED_REV_NONCURRENT / 4 / (1 + wacc) ** t for t in (1.5, 2.5, 3.5, 4.5))
    return cur + nc


def equity_per_share(ev_core: float, stakes: float, wacc: float = WACC) -> dict:
    """Iterate on conversion: a note converts if value/share > conversion price."""
    straight_debt = SECURED_FACILITY + LEASES_OTHER_DEBT
    dr = pv_deferred_revenue(wacc)
    vps = PRICE
    for _ in range(50):
        conv_sh, conv_debt = 0.0, 0.0
        for _, p, cp, _ in CONVERTS:
            if vps > cp:
                conv_sh += p * 1000 / cp
            else:
                conv_debt += p
        eq = ev_core + stakes + CASH_PROFORMA - straight_debt - conv_debt - dr
        shares = BASE_DILUTED + conv_sh
        new = eq / shares * 1000
        if abs(new - vps) < 0.01:
            break
        vps = new
    return dict(equity=eq, shares=shares, vps=vps, convert_debt=conv_debt, convert_shares=conv_sh,
                deferred_rev_pv=dr)


def value_scenario(s: dict, wacc: float = WACC, g: float | None = None,
                   exit_multiple: float | None = None) -> dict:
    """exit_multiple=None -> value-driver TV (economic depreciation); else EV/EBITDA(2035) exit.
    Equity value per share is floored at zero (limited liability)."""
    df = build_forecast(s, wacc)
    tv = terminal_value(df, s, wacc, g)
    if s.get("terminal_residual") is not None:
        tv = dict(tv)
        tv["tv"] = s["terminal_residual"]
        tv["pv_tv"] = tv["tv"] * df.loc[2035, "disc_factor"]
    elif exit_multiple is not None:
        tv = dict(tv)
        tv["tv"] = exit_multiple * df.loc[2035, "adj_ebitda"]
        tv["pv_tv"] = tv["tv"] * df.loc[2035, "disc_factor"]
    pv_fcf = df["pv_ufcf"].sum()
    ev = pv_fcf + tv["pv_tv"]
    stakes = s["clickhouse"] + s["avride"] + s["tripleten_toloka"]
    eq = equity_per_share(ev, stakes, wacc)
    eq["vps_raw"] = eq["vps"]
    eq["option_floor"] = merton_floor(ev + stakes, wacc)
    # Option floor only matters where intrinsic equity is ~nil (distress branch); applying it
    # to solvent branches would double-count dispersion already captured by the scenarios.
    eq["vps"] = eq["vps"] if eq["vps"] > 0 else max(eq["option_floor"], 0.0)
    return dict(df=df, tv=tv, pv_fcf=pv_fcf, ev=ev, stakes=stakes, **eq)


def merton_floor(asset_ex_cash: float, wacc: float = WACC) -> float:
    """Value/share of equity as a call on (EV + stakes + cash) struck at total claims
    (all converts at par + other debt + PV deferred revenue), on pre-convert share count."""
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    V = asset_ex_cash + CASH_PROFORMA
    K = sum(p for _, p, _, _ in CONVERTS) + SECURED_FACILITY + LEASES_OTHER_DEBT + pv_deferred_revenue(wacc)
    if V <= 0:
        return 0.0
    sig, T = ASSET_VOL, CLAIM_T
    d1 = (log(V / K) + (RF + sig ** 2 / 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    call = V * N(d1) - K * exp(-RF * T) * N(d2)
    return call / BASE_DILUTED * 1000


def implied_irr(s: dict, exit_multiple: float | None = None, target: float = PRICE) -> float:
    """Discount rate at which scenario value/share equals the current price (= expected
    annual return from buying today if the scenario plays out)."""
    lo, hi = 0.0, 0.60
    for _ in range(60):
        mid = (lo + hi) / 2
        v = value_scenario(s, mid, exit_multiple=exit_multiple)["vps_raw"]
        if v > target:
            lo = mid
        else:
            hi = mid
    return mid


# ----------------------------------------------------------------------------
# 5. MARKET-IMPLIED (reverse DCF) + comps + unit economics
# ----------------------------------------------------------------------------

def market_ev() -> dict:
    conv_sh = sum(p * 1000 / cp for _, p, cp, _ in CONVERTS if PRICE > cp)
    conv_debt = sum(p for _, p, cp, _ in CONVERTS if PRICE <= cp)
    shares = BASE_DILUTED + conv_sh
    mcap = PRICE * shares / 1000
    debt = conv_debt + SECURED_FACILITY + LEASES_OTHER_DEBT
    ev = mcap + debt - CASH_PROFORMA
    return dict(shares=shares, mcap=mcap, debt=debt, cash=CASH_PROFORMA, ev=ev,
                ev_core=ev + pv_deferred_revenue() - (BASE["clickhouse"] + BASE["avride"] + BASE["tripleten_toloka"]))


def scale_arr(s: dict, k: float) -> dict:
    s2 = deepcopy(s)
    a0 = s["exit_arr"][0]
    s2["exit_arr"] = [a0 + (a - a0) * k for a in s["exit_arr"]]
    return s2


def reverse_dcf(target_vps: float = PRICE) -> dict:
    """Scale Base ARR additions until value/share = price (all else Base)."""
    lo, hi = 0.2, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = value_scenario(scale_arr(BASE, mid))["vps"]
        if v < target_vps:
            lo = mid
        else:
            hi = mid
    s = scale_arr(BASE, mid)
    r = value_scenario(s)
    return dict(k=mid, exit_arr_2030=s["exit_arr"][4], exit_arr_2035=s["exit_arr"][-1],
                rev_2030=r["df"].loc[2030, "revenue"], vps=r["vps"])


def reverse_dcf_margin_capex(target_vps: float = PRICE) -> dict:
    """Holding Base ARR, which (margin, capex/ARR) combos justify the price?"""
    out = {}
    for cpa in (2.2, 2.4, 2.6, 2.9):
        lo, hi = 0.40, 0.85
        for _ in range(50):
            mid = (lo + hi) / 2
            s = deepcopy(BASE)
            s["capex_per_arr"] = cpa
            s["ebitda_margin"] = [0.40] + [m / 0.60 * mid for m in BASE["ebitda_margin"][1:]]
            v = value_scenario(s)["vps"]
            if v < target_vps:
                lo = mid
            else:
                hi = mid
        out[cpa] = mid
    return out


def project_irr(capex_per_arr: float, margin_post_sbc: float, life: int, dc_share: float = 0.25,
                dc_life: int = 25, ramp_year1: float = 0.6) -> float:
    """Unlevered pre-tax IRR of $1 of exit ARR: capex at t0, revenue ramps, compute retired
    after `life` years, residual DC book value returned at the end."""
    cfs = [-capex_per_arr]
    for yr in range(1, life + 1):
        f = ramp_year1 if yr == 1 else 1.0
        cfs.append(margin_post_sbc * f)
    cfs[-1] += capex_per_arr * dc_share * max(dc_life - life, 0) / dc_life
    return _irr(cfs)


def _irr(cfs):
    lo, hi = -0.99, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        npv = sum(c / (1 + mid) ** i for i, c in enumerate(cfs))
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return mid


# Comparable companies ($bn) - see report for sources/dates
COMPS = pd.DataFrame([
    # name, EV, rev 2026E, rev 2027E, EBITDA 2027E, net debt/EBITDA comment
    dict(company="CoreWeave (CRWV)", ev=79.9, rev26=12.8, rev27=26.5, ebitda27=14.0,
         note="EV StockAnalysis ~16-Oct-26; '26 rev = guide midpoint; '27 = Needham/consensus range"),
    dict(company="Lambda (private)", ev=14.5, rev26=1.5, rev27=None, ebitda27=None,
         note="Reported pre-money of up to $14.5bn (WSJ, unconfirmed); '26 rev 'on track to top $1.5bn'"),
])


# ----------------------------------------------------------------------------
# 6. RUN + OUTPUT
# ----------------------------------------------------------------------------

def fmt(x, d=1):
    return f"{x:,.{d}f}"


def main():
    out = []
    p = out.append
    p(f"# NBIS model output (auto-generated by nbis_model.py)\n")
    p(f"Cost of equity {KE:.2%}; WACC (explicit period) {WACC:.2%}; terminal-stage WACC {WACC_TERMINAL:.2%} "
      f"(mature beta {BETA_MATURE}); price ${PRICE}\n")

    m = market_ev()
    p("## Market-implied EV\n")
    p(f"| Item | Value |\n|---|---|\n| Diluted shares at ${PRICE} (m) | {fmt(m['shares'])} |\n"
      f"| Market cap ($bn) | {fmt(m['mcap'])} |\n| Debt incl. OTM converts + leases ($bn) | {fmt(m['debt'])} |\n"
      f"| Pro-forma cash ($bn) | {fmt(m['cash'])} |\n| Enterprise value ($bn) | {fmt(m['ev'])} |\n"
      f"| Core AI-cloud EV (adds PV deferred rev, ex stakes) ($bn) | {fmt(m['ev_core'])} |\n")

    results = {}
    for s in SCENARIOS:
        r = value_scenario(s)
        results[s["name"]] = r
    pw = sum(results[s["name"]]["vps"] * s["prob"] for s in SCENARIOS)

    p("## Scenario summary\n")
    p("| Scenario | Prob. | PV FCF 2026H2-35 | PV TV | Core EV | Stakes | Equity (intrinsic) | Diluted sh (m) | Intrinsic/share | Option floor/share | Value/share | vs price |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SCENARIOS:
        r = results[s["name"]]
        p(f"| {s['name']} | {s['prob']:.0%} | {fmt(r['pv_fcf'])} | {fmt(r['tv']['pv_tv'])} | {fmt(r['ev'])} | "
          f"{fmt(r['stakes'])} | {fmt(r['equity'])} | {fmt(r['shares'])} | ${fmt(r['vps_raw'],0)} | ${fmt(r['option_floor'],0)} | "
          f"${fmt(r['vps'],0)} | {r['vps']/PRICE-1:+.0%} |")
    p(f"| **Probability-weighted** | 100% | | | | | | | | | **${fmt(pw,0)}** | **{pw/PRICE-1:+.0%}** |\n")

    for s in SCENARIOS:
        r = results[s["name"]]
        df = r["df"]
        p(f"## {s['name']} case forecast ($bn)\n")
        cols = ["exit_arr", "revenue", "rev_growth", "adj_ebitda", "adj_ebitda_margin", "sbc", "d_and_a",
                "ebit", "ebit_margin", "cash_tax", "growth_capex", "refresh_capex", "capex", "ufcf"]
        t = df[cols].T
        lines = ["| Line | " + " | ".join(str(y) for y in t.columns) + " |",
                 "|---|" + "---|" * len(t.columns)]
        for c in cols:
            vals = []
            for y in t.columns:
                v = t.loc[c, y]
                if c in ("rev_growth", "adj_ebitda_margin", "ebit_margin"):
                    vals.append(f"{v:.0%}")
                else:
                    vals.append(fmt(v))
            lines.append(f"| {c} | " + " | ".join(vals) + " |")
        p("\n".join(lines) + "\n")
        tv = r["tv"]
        p(f"Terminal (2036): revenue {fmt(tv['rev_t1'])}, economic depreciation {fmt(tv['econ_dep'])}, "
          f"NOPAT {fmt(tv['nopat_t1'])} ({tv['nopat_margin']:.1%} margin), FCF {fmt(tv['fcf_t1'])}, "
          f"TV {fmt(tv['tv'])} (= {tv['tv_ev_ebitda']:.1f}x 2036 adj. EBITDA), PV TV {fmt(tv['pv_tv'])}. "
          f"Cumulative 2027-30 UFCF {fmt(df.loc[2027:2030,'ufcf'].sum())}; peak cumulative funding need "
          f"{fmt(-df['ufcf'].cumsum().min())}.\n")

    # Sensitivities on Base: intrinsic value (floored at 0, no option floor) so grids are monotonic
    def iv(sc, w=WACC, g=None):
        return max(value_scenario(sc, w, g)["vps_raw"], 0.0)
    p("Sensitivity grids show intrinsic value per share (floored at $0; the Merton option floor is excluded).\n")
    p("## Sensitivity: Base value/share - WACC vs terminal growth\n")
    waccs = [0.095, 0.105, WACC, 0.12, 0.13]
    gs = [0.02, 0.03, 0.04]
    p("| WACC \\ g | " + " | ".join(f"{g:.0%}" for g in gs) + " |\n|---|" + "---|" * len(gs))
    for w in waccs:
        p(f"| {w:.1%} | " + " | ".join(f"${iv(BASE, w, g):,.0f}" for g in gs) + " |")
    p("")

    p("## Sensitivity: Base value/share - LT adj. EBITDA margin vs capex per $ of new ARR\n")
    margins = [0.50, 0.55, 0.60, 0.65, 0.70]
    cpas = [2.1, 2.4, 2.7, 3.0, 3.3]
    p("| Margin \\ capex/ARR | " + " | ".join(f"{c:.1f}x" for c in cpas) + " |\n|---|" + "---|" * len(cpas))
    for mg in margins:
        row = []
        for c in cpas:
            s = deepcopy(BASE)
            s["capex_per_arr"] = c
            s["ebitda_margin"] = [0.40] + [m_ / 0.60 * mg for m_ in BASE["ebitda_margin"][1:]]
            row.append(f"${iv(s):,.0f}")
        p(f"| {mg:.0%} | " + " | ".join(row) + " |")
    p("")

    p("## Sensitivity: Base value/share - GPU economic life vs 2035 exit ARR scale\n")
    lives = [5, 6, 7, 8]
    ks = [0.6, 0.8, 1.0, 1.25, 1.5]
    p("| GPU life \\ ARR scale | " + " | ".join(f"{k:.2f}x (2035 ARR ${7.5+(54.5-7.5)*k:.0f}bn)" for k in ks) + " |\n|---|" + "---|" * len(ks))
    for L in lives:
        row = []
        for k in ks:
            s = scale_arr(BASE, k)
            s["gpu_econ_life"] = L
            row.append(f"${iv(s):,.0f}")
        p(f"| {L} yrs | " + " | ".join(row) + " |")
    p("")

    p("## Sensitivity: Base value/share - RONIC in terminal vs WACC\n")
    ronics = [0.09, 0.11, 0.12, 0.15, 0.20]
    p("| WACC \\ RONIC | " + " | ".join(f"{x:.0%}" for x in ronics) + " |\n|---|" + "---|" * len(ronics))
    for w in [0.105, WACC, 0.12]:
        row = []
        for ro in ronics:
            s = deepcopy(BASE)
            s["ronic"] = ro
            row.append(f"${iv(s, w):,.0f}")
        p(f"| {w:.1%} | " + " | ".join(row) + " |")
    p("")

    # Reverse DCF
    p("## Reverse DCF (what does $165 imply?)\n")
    rmc = reverse_dcf_margin_capex()
    p("Holding Base ARR, LT adj. EBITDA margin needed to justify the price at each capex intensity: " +
      "; ".join(f"{c:.1f}x -> {mg:.0%}" for c, mg in rmc.items()) + "\n")

    # Unit economics
    p("## Unit economics: unlevered pre-tax IRR per $1 of new ARR\n")
    p("Capex at t0, 60% of run-rate in year 1, compute retired at end of economic life, "
      "undepreciated data-centre value (25% of capex, 25-yr life) recovered at the end.\n")
    ms = [0.50, 0.55, 0.60, 0.65]
    cp = [2.0, 2.4, 2.7, 3.0, 3.4]
    for L in (5, 6, 7):
        p(f"GPU economic life {L} years\n")
        p("| Margin post-SBC \\ capex/ARR | " + " | ".join(f"{c:.1f}x" for c in cp) + " |\n|---|" + "---|" * len(cp))
        for mm in ms:
            p(f"| {mm:.0%} | " + " | ".join(f"{project_irr(c, mm, L):.1%}" for c in cp) + " |")
        p("")

    # Comps
    p("## Comparable companies (EV in $bn)\n")
    p("| Company | EV | EV/2026E rev | EV/2027E rev | EV/2027E EBITDA | Note |\n|---|---|---|---|---|---|")
    for _, c in COMPS.iterrows():
        e26 = f"{c.ev / c.rev26:.1f}x" if c.rev26 else "n/a"
        e27 = f"{c.ev / c.rev27:.1f}x" if c.rev27 and not pd.isna(c.rev27) else "n/a"
        eb = f"{c.ev / c.ebitda27:.1f}x" if c.ebitda27 and not pd.isna(c.ebitda27) else "n/a"
        p(f"| {c.company} | {c.ev:.1f} | {e26} | {e27} | {eb} | {c.note} |")
    nb_cons = dict(rev26=3.2, rev27=12.3, ebitda27=6.6)  # [C] consensus (Yahoo/MarketScreener, Oct-26)
    p(f"| NBIS @ ${PRICE} (consensus) | {m['ev']:.1f} | {m['ev']/nb_cons['rev26']:.1f}x | {m['ev']/nb_cons['rev27']:.1f}x | "
      f"{m['ev']/nb_cons['ebitda27']:.1f}x | Consensus '27 rev $12.3bn / EBITDA $6.6bn |")
    b27 = results["Base"]["df"].loc[2027]
    p(f"| NBIS @ ${PRICE} (my Base) | {m['ev']:.1f} | {m['ev']/BASE['rev_2026']:.1f}x | {m['ev']/b27.revenue:.1f}x | "
      f"{m['ev']/b27.adj_ebitda:.1f}x | Base '27 rev ${b27.revenue:.1f}bn / EBITDA ${b27.adj_ebitda:.1f}bn |\n")

    # Comps-implied value
    p("### Comps-implied value per share (multiple x my Base 2027E adj. EBITDA, discounted one year at WACC)\n")
    p("| EV/2027E EBITDA | Implied core EV | Value/share |\n|---|---|---|")
    for mult in (5.5, 7.0, 8.5, 10.0):
        ev_core = mult * b27.adj_ebitda / (1 + WACC)   # discount the 2027 value one year
        eq = equity_per_share(ev_core, BASE["clickhouse"] + BASE["avride"] + BASE["tripleten_toloka"])
        p(f"| {mult:.1f}x | {ev_core:.1f} | ${eq['vps']:,.0f} |")
    p("")

    # SOTP (Base)
    rb = results["Base"]
    p("## Sum-of-the-parts (Base)\n")
    p("| Component | $bn | $/share |\n|---|---|---|")
    sh = rb["shares"] / 1000
    comps_ = [("AI cloud (DCF EV)", rb["ev"]), ("ClickHouse stake (~27%, $15bn round less 15%)", BASE["clickhouse"]),
              ("Avride (~83%)", BASE["avride"]), ("TripleTen + Toloka stake", BASE["tripleten_toloka"]),
              ("Pro-forma cash", CASH_PROFORMA), ("Converts treated as debt", -rb["convert_debt"]),
              ("Secured facility + leases", -(SECURED_FACILITY + LEASES_OTHER_DEBT)),
              ("PV of deferred revenue (already-collected prepayments)", -rb["deferred_rev_pv"])]
    for n, v in comps_:
        p(f"| {n} | {v:,.1f} | {v/sh:,.0f} |")
    p(f"| **Equity value** | **{rb['equity']:,.1f}** | **{rb['vps']:,.0f}** |\n")

    # Exit-multiple cross-check + implied IRR
    p("## Cross-check: terminal value on EV/EBITDA exit multiple (2035)\n")
    p("| Scenario | Value-driver TV | 5x exit | 7x exit | 9x exit | Implied IRR at $165 (value-driver) | Implied IRR (7x exit) |")
    p("|---|---|---|---|---|---|---|")
    for s_ in SCENARIOS:
        vd = results[s_["name"]]["vps"]
        row = [f"${value_scenario(s_, exit_multiple=m_)['vps']:,.0f}" for m_ in (5, 7, 9)]
        irr_fmt = lambda x: "< 0% (n.m.)" if x < 0.001 else f"{x:.1%}"
        p(f"| {s_['name']} | ${vd:,.0f} | " + " | ".join(row) +
          f" | {irr_fmt(implied_irr(s_))} | {irr_fmt(implied_irr(s_, 7))} |")
    pw7 = sum(value_scenario(s_, exit_multiple=7)["vps"] * s_["prob"] for s_ in SCENARIOS)
    p(f"| **Prob-weighted** | **${pw:,.0f}** | | **${pw7:,.0f}** | | | |\n")
    p(f"Blended fair value (50% value-driver DCF / 50% 7x-exit DCF): **${(pw + pw7) / 2:,.0f}**\n")

    # Funding need / dilution illustration
    p("## Funding need and dilution illustration (Base)\n")
    dfb = results["Base"]["df"]
    p("| Year | UFCF | Cum. funding need | Equity-funded share @50% at $165 (m sh, cumulative) |\n|---|---|---|---|")
    cum = 0.0
    for y in YEARS[:6]:
        f_ = dfb.loc[y, "ufcf_valued"] if y == 2026 else dfb.loc[y, "ufcf"]
        cum += f_
        p(f"| {y}{' (H2)' if y == 2026 else ''} | {f_:,.1f} | {max(-cum, 0):,.1f} | {max(-cum, 0) * 0.5 / PRICE * 1000:,.0f} |")
    p("")

    # 2030 value with explicit funding/dilution (3-5 year outlook)
    p("## 2030 equity value per share with explicit funding (3-5 year outlook)\n")
    p("Funding gap through 2030 financed 50% new equity (issued at $165) / 50% new debt; converts converted "
      "if 2030 value exceeds their conversion price; EV at end-2030 = multiple x 2031E adj. EBITDA.\n")
    p("| Scenario | Peak funding need ($bn) | New shares (m) | 2031E EBITDA | 5x | 7x | 9x | CAGR to 7x value |")
    p("|---|---|---|---|---|---|---|---|")
    for s_ in SCENARIOS:
        r = results[s_["name"]]
        d = r["df"]
        flows = [d.loc[2026, "ufcf_valued"]] + [d.loc[y, "ufcf"] for y in range(2027, 2031)]
        cum = np.cumsum(flows)
        need = max(-cum.min(), 0.0)
        end_cum = cum[-1]                       # cumulative FCF to end-2030
        new_eq = need * 0.5
        new_sh = new_eq / PRICE * 1000
        new_debt = need * 0.5
        # cash at end-2030 = pro-forma cash + cumulative FCF + new money
        cash30 = CASH_PROFORMA + end_cum + need
        vals = []
        for mult in (5, 7, 9):
            ev30 = mult * d.loc[2031, "adj_ebitda"]
            if s_.get("terminal_residual") is not None:
                # run-off fleet: EV = PV (at end-2030) of remaining UFCF + residual site value
                ev30 = sum(d.loc[y, "ufcf"] / (1 + WACC) ** (y - 2030) for y in range(2031, 2036)) \
                    + s_["terminal_residual"] / (1 + WACC) ** 5
            vps = PRICE
            for _ in range(50):
                conv_sh = sum(pp * 1000 / cp for _, pp, cp, _ in CONVERTS if vps > cp)
                conv_debt = sum(pp for _, pp, cp, _ in CONVERTS if vps <= cp)
                eq = ev30 + r["stakes"] + cash30 - conv_debt - new_debt - SECURED_FACILITY - LEASES_OTHER_DEBT
                sh = BASE_DILUTED + new_sh + conv_sh
                nv = max(eq, 0) / sh * 1000
                if abs(nv - vps) < 0.01:
                    break
                vps = nv
            vals.append(vps)
        cagr = (vals[1] / PRICE) ** (1 / 4.2) - 1 if vals[1] > 0 else -1
        p(f"| {s_['name']} | {need:,.1f} | {new_sh:,.0f} | {d.loc[2031, 'adj_ebitda']:,.1f} | " +
          " | ".join(f"${v:,.0f}" for v in vals) + f" | {cagr:+.0%} |")
    p("")

    # 12-month view
    p("## 12-month expected value\n")
    blend = {s_["name"]: (results[s_["name"]]["vps"] + value_scenario(s_, exit_multiple=7)["vps"]) / 2 for s_ in SCENARIOS}
    pt = {k: v * (1 + KE) for k, v in blend.items()}
    p("Blended scenario values (50/50 terminal methods) rolled forward one year at the cost of equity:\n")
    p("| Scenario | Blended fair value today | 12m target | Prob. |\n|---|---|---|---|")
    for s in SCENARIOS:
        p(f"| {s['name']} | ${blend[s['name']]:,.0f} | ${pt[s['name']]:,.0f} | {s['prob']:.0%} |")
    pw12 = sum(pt[s["name"]] * s["prob"] for s in SCENARIOS)
    pwb = sum(blend[s["name"]] * s["prob"] for s in SCENARIOS)
    p(f"| Weighted | ${pwb:,.0f} | ${pw12:,.0f} | |\n")
    p(f"Implied 12-month return to weighted target: {pw12 / PRICE - 1:+.1%} "
      f"(weighted fair value ${pwb:,.2f}; 12m target ${pw12:,.2f})\n")

    text = "\n".join(out)
    with open(os.path.join(HERE, "model_output.md"), "w") as f:
        f.write(text)
    print(text)

    # Excel export
    with pd.ExcelWriter(os.path.join(HERE, "NBIS_model.xlsx"), engine="openpyxl") as xw:
        inputs = pd.DataFrame([
            ("Price (16-Oct-26 close)", PRICE, "F"), ("Basic shares 30-Jun-26 (m)", BASIC_SHARES_JUN30, "F"),
            ("Aug-26 exchange shares (m)", EXCHANGE_SHARES, "F"), ("NVIDIA pre-funded warrant (m)", NVDA_PREFUNDED_WARRANT, "F"),
            ("Unvested awards (m)", UNVESTED_EQUITY_AWARDS, "A"), ("Cash 30-Jun-26", CASH_JUN30, "F"),
            ("Aug-26 convert net proceeds", AUG26_CONVERT_NET_PROCEEDS, "A"), ("Secured facility", SECURED_FACILITY, "F"),
            ("Leases/other debt", LEASES_OTHER_DEBT, "A"), ("Deferred revenue current", DEFERRED_REV_CURRENT, "F"),
            ("Deferred revenue non-current", DEFERRED_REV_NONCURRENT, "F"), ("Risk-free", RF, "A"), ("ERP", ERP, "A"),
            ("Beta", BETA, "A"), ("Kd pre-tax", KD_PRE_TAX, "A"), ("Tax", TAX, "A"), ("D/V", D_V, "A"),
            ("Cost of equity", KE, "calc"), ("WACC", WACC, "calc"),
        ], columns=["input", "value", "source_tag"])
        inputs.to_excel(xw, sheet_name="Inputs", index=False)
        pd.DataFrame(CONVERTS, columns=["series", "principal_bn", "conv_price", "coupon"]).to_excel(
            xw, sheet_name="Convertibles", index=False)
        for s in SCENARIOS:
            r = results[s["name"]]
            r["df"].T.to_excel(xw, sheet_name=f"{s['name']}_forecast")
            drv = {k: (str(v) if isinstance(v, list) else v) for k, v in s.items()}
            pd.Series(drv).to_frame("value").to_excel(xw, sheet_name=f"{s['name']}_drivers")
        summ = pd.DataFrame([dict(scenario=s["name"], prob=s["prob"], pv_fcf=results[s["name"]]["pv_fcf"],
                                  pv_tv=results[s["name"]]["tv"]["pv_tv"], ev=results[s["name"]]["ev"],
                                  stakes=results[s["name"]]["stakes"], equity=results[s["name"]]["equity"],
                                  shares=results[s["name"]]["shares"], value_per_share=results[s["name"]]["vps"])
                             for s in SCENARIOS])
        summ.to_excel(xw, sheet_name="Valuation_summary", index=False)
        COMPS.to_excel(xw, sheet_name="Comps", index=False)
    return results


if __name__ == "__main__":
    main()
