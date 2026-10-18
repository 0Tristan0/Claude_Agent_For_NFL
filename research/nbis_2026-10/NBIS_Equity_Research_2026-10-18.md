# Nebius Group N.V. (NASDAQ: NBIS) — Initiation of Coverage

**Rating: SELL (Underweight) · Fair value: $120 · 12-month price target: $135 (−18%) · Conviction: Medium**

| Snapshot (as of 18-Oct-2026) | |
|---|---|
| Price (16-Oct-26 close) | **$165.47** (−12% on the day; 52-wk range $73.52–$299.86) |
| Fully diluted shares (my count, at $165) | **343m** (aggregators show ~272m — see §7.4) |
| Diluted market cap / Enterprise value | **$56.8bn / $54.8bn** (pro forma for Jul–Aug 2026 financings) |
| EV / consensus 2027E revenue / EBITDA | 4.5x / 8.3x (CoreWeave: 3.0x / 5.7x) |
| FY26 guidance (reaffirmed 12-Aug-26) | Revenue $3.0–3.4bn; exit ARR $7–9bn; adj. EBITDA margin ~40%; capex $20–25bn |
| Street | ~Moderate Buy; average PT $283–300 (TipRanks/ChartMill); range $145–$430 |
| Next catalyst | Q3-26 results — date not yet confirmed by the company (Earnings Whispers estimates ~11-Nov) |

Labels used throughout: **[F]** confirmed fact (filing / company release, cross-checked) · **[G]** management guidance · **[C]** consensus / third-party data · **[A]** my assumption · **[I]** my interpretation.

---

## 1. Executive summary

Nebius is the best-capitalized independent "neocloud", and one of the fastest-growing companies in public markets. Q2-26 revenue rose 454% YoY to $582m [F]. ARR reached $3.0bn [F]. More than $40bn of committed backlog sits with Microsoft and Meta [F/G], and customer prepayments are funding a large part of the build-out [F]. The operating story is excellent. The equity story is not, and the gap between the two is the crux of this report.

**My central finding [I]:** the value of a GPU cloud depends almost entirely on three numbers that the market rarely discusses:

1. Capex per dollar of ARR.
2. Mature EBITDA margin.
3. The economic life of the GPUs.

At the levels implied by GPU-hour arithmetic and Nebius's own 2026 capex/ARR ratio (~2.7x capex per $1 of ARR, ~60% EBITDA margin, 6-year life), each new megawatt earns an unlevered pre-tax IRR of ~8–10%. That is below an ~11% WACC. Growth at these returns adds revenue but little per-share value. **At $165, the market is underwriting either ~68–71% long-run EBITDA margins at 2.4–2.6x capex intensity, which no GPU cloud has yet demonstrated, or an implied discount rate of ~9%.**

| Valuation method | Bear | Base | Bull | Weighted (25/50/25) |
|---|---|---|---|---|
| DCF, value-driver terminal (economic depreciation) | $11 | $42 | $256 | **$88** |
| DCF, 7x EV/EBITDA exit (2035) | $11 | $142 | $303 | **$150** |
| **Blended fair value (50/50)** | $11 | $92 | $280 | **$119 → $120** |
| 12-month target (blended, rolled forward at 13.65% cost of equity) | $12 | $105 | $318 | **$135** |
| Comps cross-check: 5.5–10x my 2027E EBITDA | | $79–145 | | |

**Three strongest reasons for SELL**

1. **The unit economics do not clear the cost of capital at base-case assumptions.** At 2.7x capex/ARR and a 6-year life, project IRRs are ~8–10% pre-tax (§8). The "22-month payback" cited for Q2 contracts and a BNP-linked "1.3-year payback" both count customer prepayments as cash inflows. Prepayments are financing, not return.
2. **The headline valuation understates dilution and the claims ahead of equity.** On a fully diluted basis (NVIDIA warrant, the August exchange shares, in-the-money converts), there are 343m shares, not 272m. There are $13.45bn of convertibles with conversion prices of $51–$325, so dilution grows as the stock rises. There is also $6.0bn of deferred revenue: cash already collected for service still owed. Base-case funding needs reach ~$49bn by 2029 (§9.3).
3. **Consensus is above my base case, and the sector's cost of capital is rising.** Consensus is ~$12.3bn revenue for 2027E and ~$45bn for 2030E. My base is $10.6bn and $32bn, bounded by connected power, commissioning lags and funding. CoreWeave's 5-year CDS hit ~1,195bp on 15-Oct-26, and Gartner (14-Oct-26) expects >80% of neoclouds to lose independence by 2030. Equity multiples for the group are de-rating toward credit reality.

**Most important risks to a SELL (upside risks)**

- Nebius proves superior economics. Examples: the +15% capacity-auction pricing persists, software/inference (Token Factory, Inferize) lifts margins, or GPUs earn for 7–8 years.
- A 2027 guide in BNP's "$22bn ARR" territory.
- A short squeeze. Short interest is ~21–22% of float.
- A strategic bid. NVIDIA, Microsoft and Meta are all deeply involved, though Volozh-controlled voting makes a bid harder.

**What would make me a buyer:**

- Price below ~$95 (a 20% discount to fair value), **or**
- Disclosed evidence of ≤2.4x capex/ARR with ≥62% mature margins on 2026–27 vintages, **or**
- A hyperscaler re-contracting H100/H200 capacity at ≥70% of original rates (evidence of a ≥7-year economic life).

---

## 2. Research integrity note

- **Sources:** SEC 6-K/20-F exhibits, Nebius newsroom releases, earnings-call coverage, and reputable financial media. They are linked in §15 with dates.
- **How data was retrieved:** this environment's network policy blocked direct downloads from sec.gov and nebius.com. Filing figures were therefore retrieved through search-engine extracts of those filings. Every critical figure was cross-checked against at least two independent extracts or reports. Where sources conflicted, I say so.
- **Not independently verified (flagged where used):**
  - Q3-26 earnings date.
  - Oct-1 price increase of 17–21% (single source).
  - Consensus 2028–30 estimates (aggregators disagree).
  - Exact split of the Aug-26 converts between 2030 and 2034 series (total $5.75bn confirmed).
  - Lease liabilities (derived).
  - Avride/TripleTen valuations.
- **Model:** fully reproducible in `nbis_model.py`. Running `python3 nbis_model.py` regenerates every table in `model_output.md` and the workbook `NBIS_model.xlsx`. Each input is tagged [F]/[G]/[C]/[A] in code.

---

## 3. Business analysis

### 3.1 What Nebius is

Nebius is the international, non-Russian remainder of Yandex N.V. after the 2024 divestment of Russian assets. It is headquartered in Amsterdam, with large R&D hubs in Europe and Israel. The business segments are:

| Segment / asset | Description | Economic weight |
|---|---|---|
| **Nebius AI Cloud** | Full-stack GPU cloud covering bare metal, managed Kubernetes/Slurm, storage, AI Studio, and Token Factory (inference). Sells both on-demand/reserved capacity and multi-year dedicated clusters to hyperscalers and AI labs. | ~98% of revenue; $574.9m in Q2-26 [F]; ~50% segment adj. EBITDA margin [F] |
| **Avride** (~83% owned [C]) | Autonomous driving and delivery robots. Partnered with Uber; up to $375m of Uber + Nebius commitments (Oct-25) [F] | Adj. EBITDA loss of $82.7m in 2025 and $34.1m in Q1-26 [F] |
| **TripleTen** | Ed-tech / developer bootcamps | Small; ~$54m 2025 revenue [C, unverified] |
| **ClickHouse stake** (~25–28% [C]) | Equity stake in the open-source OLAP database company. Series D of $400m at a $15bn valuation (Jan-26) [C]; ARR reportedly >$350m [C] | Worth ~$3.4–4.2bn, roughly $10–13/share |
| **Toloka** (deconsolidated) | Data labelling; minority economic stake | Immaterial |

**Acquisitions and strategic moves:**
- Inferize (inference stack), acquired 1-Oct-26, terms undisclosed [F].
- NVIDIA invested $2bn (Mar-26) through pre-funded warrants for 21.07m shares, ~$95/share implied by my arithmetic [F/I]. This came with a partnership targeting >5 GW of NVIDIA-based capacity by 2030 [F].

### 3.2 Competitive advantages: real, but narrower than the narrative

**Real advantages [I]:**
- **Engineering depth.** About 1,000+ ex-Yandex infrastructure engineers have built a proprietary software stack and their own server and rack designs. This shows in fast time-to-revenue and in hyperscaler wins: Microsoft and Meta both chose Nebius.
- **Balance sheet.** Roughly net-cash at the corporate level, against CoreWeave's ~$52bn of debt.
- **Customer-funded capex.** Microsoft is paying ~$6.96bn of upfront payments [F]. H1-26 advance payments were $4.4bn [F].
- **Power pipeline.** 5 GW of contracted power targeted by YE26 [G].
- **NVIDIA alignment.** Allocation priority.

**Limits [I]:**
- GPU capacity is a largely **commoditised** input. Nebius buys the same NVIDIA systems as everyone else.
- The largest customers (Microsoft, Meta) are also the largest **competitors**. Meta's in-house cloud ambitions knocked ~14% off the stock on 1-Jul-26 [C].
- Switching costs are low for training workloads, which move to wherever capacity is cheapest.
- The software layer is a genuine differentiator for mid-market and AI-native customers. It is not yet a material share of revenue.

### 3.3 Management and governance

- **Leadership:** Co-founder/CEO Arkady Volozh, who also founded Yandex, and CFO Dado Alonso [F].
- **Voting control:** Class B shares carry 10 votes each. At the Aug-26 AGM there were 33.46m Class B shares (334.6m votes) against 238.4m Class A votes [F]. Volozh's trust held 28.66m Class B shares per his Mar-26 Form 3 [F]. That gives him **roughly half of all votes** [I].
- **Assessment:** execution has been outstanding. YE25 active power was ~170 MW against a 100 MW target, and ARR was $1.25bn against $0.9–1.1bn guidance [F]. Capital-markets execution has also been outstanding: ~$13bn of converts at 0.5–4.5% coupons in 12 months [F].
- **Governance discounts:**
  - Dual-class control.
  - Heavy SBC: $102.5m in Q2-26, 18% of revenue [F].
  - An accounting-life extension that flatters EBIT. Server and network equipment moved from 4 to 5 years in Q1-26, lowering Q2 D&A by ~$43m [F/C].
  - A residual Yandex/geopolitical overhang [I].

---

## 4. Industry analysis: AI infrastructure / GPU cloud

**Demand drivers:**
- Frontier-model training: the multi-GW hyperscaler and lab contracts.
- Rapidly growing inference demand: tokens served.
- Enterprise and sovereign AI.
- Hyperscalers outsourcing capacity to neoclouds because their own power and build timelines are the bottleneck.

NVIDIA frames each GW as a ~$25bn (Blackwell) to ~$40bn (Vera Rubin) revenue opportunity for itself [C]. NVIDIA also expects neocloud capacity to reach ~8 GW by late 2026 [C]. On that basis Nebius's 0.8–1.0 GW of connected power would be roughly 10–12% of neocloud capacity [I].

**Supply and pricing:**
- 2026 has been a seller's market. A SemiAnalysis 1-year H100 rental index was reported up ~40% from its Oct-25 low by Mar-26 [C].
- Nebius's first mid-term capacity auction cleared ~15% above any previous rate [G].
- Nebius raised on-demand prices on 1-Oct-26 (reported at 17–21%, single source) [C].
- Tracked medians are ~$3–4 per GPU-hour for H100 and ~$6–8 for B200 [C].

**This is a cyclical peak signal, not a structural one [I].** GPU rental prices fell sharply in 2024–25 when supply caught up, and the same will happen once the 2026–27 build wave (CoreWeave capex $35–39bn [F], Nebius $20–25bn [G], IREN $25–30bn FY27 [C], plus Lambda and Crusoe) is energised.

**Competition:**

| Competitor | Notes |
|---|---|
| Hyperscalers (AWS, Azure, GCP, OCI) | Simultaneously Nebius's customers |
| CoreWeave | Q2-26 revenue $2.6bn; backlog $104bn, or $129bn including early-Q3 commitments [F] |
| Lambda | Reported pre-IPO round at $12–14.5bn; 2026 revenue ~$1.5bn [C] |
| IREN | Microsoft $9.7bn contract [C] |
| Crusoe, Together and other specialists | |

**Structural risk:** Gartner (14-Oct-26) estimates that AI data-centre owners need ~$8.5tn of cumulative revenue through 2030 to earn a 12% ROIC on announced capex, against ≤$3.6tn of forecast AI revenue. It puts >$3.5tn of infrastructure at risk of impairment and expects >80% of neoclouds to cease independent operation by 2030 [C]. I treat this as directionally important, not as a forecast for Nebius specifically. Nebius is more likely an acquirer or a survivor than a casualty.

---

## 5. Major contracts, partnerships and financing: economics and my assessment

| Agreement | Terms (fact) | My assessment [I] |
|---|---|---|
| **Microsoft** (Sep-25), Vineland NJ (leased site; 300 MW initial) | 5-year dedicated GPU capacity. Committed fees **up to $17.4bn** through 2031, or ~$19.4bn with options. **~$6.96bn upfront payments** tied to deployment milestones. Nine tranches over 2025–26. Microsoft has service credits and tranche-termination rights for delays or uptime misses [F] | Worth ~$3.5bn/yr at run-rate. The upfront payments roughly halve the equity needed. My rough estimate is that 5-year contract EBITDA ≈ capex, so profit relies on residual GPU value after 2031. Tranche-termination rights make delivery execution the key risk. Latest planned tranche delivered (Jul-26) [F] |
| **Meta I** (Nov-25) | ~$3bn over 5 years [F] | Small; validates the platform |
| **Meta II** (signed 13-Mar-26) | **$12bn firm** dedicated cluster orders, deployments from 2027. Plus Meta's obligation to buy **up to $15bn** of specified residual capacity that Nebius fails to sell, for the remainder of 5 years after deployment. Headline "up to ~$27bn" [F] | Only the $12bn is firm. The $15bn backstop is a valuable **put** that reduces Nebius's merchant risk, but it is not backlog. It also deepens concentration on a customer building competing capacity |
| **NVIDIA** (Mar-26) | $2bn for pre-funded warrants (21.07m shares; exercisable from 11-Sep-26; 6-month lock-up) plus a >5 GW partnership [F] | Strategic validation and allocation priority. The lock-up has now expired (overhang is low; NVIDIA rarely sells) |
| **$775m secured facility** (17-Jul-26) | Due Oct-2030; SOFR+250 (~6.8%). Secured on GPUs and contracted cash flows from an investment-grade customer. MUFG-led; oversubscribed [F] | Opens the cheaper asset-backed debt market. This is the template for funding the ~$49bn base-case gap |
| **AIB Data Centers** (30-Sep-26) | Nebius leases 50 MW of critical IT capacity in the US Southeast; 12-year term [F] | A long lease liability against shorter GPU contracts creates duration mismatch |
| **Customer prepayments** | >$9bn expected in 2026 [G]; deferred revenue $5.98bn at 30-Jun-26 [F] | Cheap funding, but it is a **liability**. 98% of H1-26 operating cash flow ($4.40bn of $4.50bn) was advance payments [F]. Do not capitalise OCF as earnings power |
| **Concentration** | Three unnamed customers were 24%, 21% and 14% of Q2-26 revenue [C]. Microsoft + Meta firm commitments are ~$32bn of ~$40bn IG backlog [C] | High. A Microsoft or Meta in-sourcing decision is the single largest risk to 2028+ renewals |

---

## 6. Capacity, power, GPU deployment and capex

| Metric | Value |
|---|---|
| Active power, YE25 | ~170 MW [F] (target was 100 MW) |
| Connected power, YE26 | 800 MW–1 GW (reaffirmed Aug-26) [G] |
| Contracted power, YE26 | **5 GW** (raised from >4 GW, earlier >3 GW, earlier 2.5 GW) [G] |
| Pace of new capacity | ">1 GW of new capacity a year starting 2027"; "almost all [5 GW] online over the next 2–3.5 years" [G] |
| Capex | FY25 $4.1bn [F]; Q1-26 $2.47bn [F]; Q2-26 ~$5.7bn [F]; FY26 guide $20–25bn (raised from $16–20bn) [G] |
| Key sites | Independence MO (up to 1.2 GW, first US GW campus); Schuylkill County PA (1.2 GW; 260 MW phase by Oct-27); Birmingham AL (~300 MW); Vineland NJ (leased, Microsoft); Lappeenranta FI (up to 310 MW) and Mäntsälä FI (75 MW); Béthune FR (240 MW); UK (Ark DC) and Israel (leased) [F/C] |

**Calibration [I]:** ARR per active MW was ~$7.4m at YE25 ($1.25bn / 170 MW), on a Hopper-heavy fleet. GB300-class economics run at about $12–16m per IT-MW at full utilisation. The arithmetic: ~470 GPUs per IT-MW × ~$4.0/GPU-hr × 8,760 hours × 95% utilisation ≈ $15.6m. All-in capex is ~$29m/IT-MW for compute plus ~$12–15m for an owned shell, power and cooling. That gives:
- **~2.7–2.8x capex/ARR for owned sites**, and
- **~1.9x for leased sites**, with rent of ~13% of revenue moving into opex.

Nebius's own numbers point the same way: ~$27bn of cumulative capex against $7.5bn of YE26 ARR (~3.6x gross, ~2.7–3.0x after stripping 2027 pre-build). So I use 2.7x in the base case, 2.4x bull and 3.2x bear.

**Scalability constraints, in order of tightness [I]:**
1. Commissioning lag. Connected power takes "several months" to become revenue [G].
2. Capital: ~$40bn+ of external funding over 2026H2–2029 in the base case.
3. Power interconnects at the gigawatt campuses.
4. NVIDIA allocation, which NVIDIA's equity stake largely de-risks.

---

## 7. Financial analysis

### 7.1 Historical P&L ($m)

| | Q1-25 | Q2-25 | Q3-25* | Q4-25 | FY25 | Q1-26 | Q2-26 |
|---|---|---|---|---|---|---|---|
| Revenue | 50.9 | 105.1 | 146.1 | 227.7 | 529.8 | 399.0 | 582.3 |
| YoY growth | | | | 547% | 479% | 684% | 454% |
| Adj. EBITDA | (53.7) | (21.0) | (5.2) | 15.0 | (64.9) | 129.5 | 236.2 |
| Adj. EBITDA margin | n.m. | n.m. | n.m. | 7% | n.m. | 32% | 41% |
| D&A | | 75.2 | | | | 212.0 | 259.7 |
| SBC | | 14.7 | | | | | 102.5 |
| GAAP net income / (loss), cont. ops | (104.3) | 502.5** | | | | 621.2** | (190.4) |
| Adjusted net loss | | (91.5) | | | | (100.3) | (33.2) |
| ARR (period end) | | | | 1,250 | 1,250 | 1,900 | 3,000 |

\* Q3-25 is derived as FY25 minus the other quarters. \*\* Includes large equity-investment revaluation gains (ClickHouse): $780.6m in Q1-26. Source: company releases and 6-K exhibits [F].

**Reading the numbers [I]:**
- Revenue quality is high: contracted, with blue-chip counterparties.
- Adj. EBITDA margin is ramping as expected: 41% group, ~50% AI cloud.
- **Operating profitability has not arrived.** D&A ($260m) exceeded adj. EBITDA ($236m) in Q2, even after the life extension. SBC adds another 18% of revenue.
- GAAP net income has been driven by **non-operating** ClickHouse mark-ups.

### 7.2 Balance sheet and liquidity (30-Jun-26, pro forma for Jul–Aug 2026)

| $bn | 30-Jun-26 | Pro forma |
|---|---|---|
| Cash & equivalents | 8.04 [F] | **14.42** (+$5.60bn net Aug converts [A on fees] +$0.775bn secured facility) |
| Convertible notes (principal) | ~8.5 [F] | **13.45** (−$0.8bn exchanged for 15.8m shares; +$5.75bn Aug-26) |
| Secured GPU facility | – | 0.775 [F] |
| Leases / other | ~1.5 [A, derived from "debt + leases ≈ $10.06bn" (TIKR)] | 1.5 |
| **Net debt** | | **~1.3** |
| Deferred revenue (prepaid service obligations) | 5.98 [F] | 5.98 |
| ATM capacity remaining | 12.3m shares [F] (12.7m sold through Jun-26 at $223.6 avg for ~$2.8bn [F]) | |

### 7.3 Convertible stack: a "success tax" [F terms; I analysis]

| Series | Principal ($bn) | Conversion price | Shares if converted (m) |
|---|---|---|---|
| 1.00% 2029 / 2.75% 2031 (Jun-25), post-exchange | 0.10 + 0.10 | $51.45 | 3.9 |
| 1.00% 2030 / 2.75% 2032 (Sep-25) | 1.58 + 1.58 | $138.75 | 22.8 |
| 1.25% 2031 (Mar-26) | 2.59 | $183.22 | 14.1 |
| 2.625% 2033 (Mar-26) | 1.75 | $180.31 | 9.7 |
| 0.50% 2030 / 4.50% 2034 (Aug-26)† | 3.45 / 2.30 | $313.46 / $324.65 | 18.1 |
| **Total** | **13.45** | | **68.6** |

† The total of $5.75bn is confirmed; the split between the two series is my assumption.

The stack dilutes in three steps. At $165, ~27m shares are in the money. Above ~$183, another ~24m are added. Above ~$315, another ~18m. **The bull case is diluted by up to ~20%. The bear case is protected by long-dated, low-coupon debt with no maintenance covenants** [I]. That is a genuine strength relative to CoreWeave.

### 7.4 Share count reconciliation

| Shares (m) | |
|---|---|
| Class A + B outstanding, 30-Jun-26 | 271.9 [F] |
| + August exchange shares | 15.8 [F] |
| + NVIDIA pre-funded warrant ($0.0001 strike) | 21.1 [F] |
| + unvested awards (estimate) | 8.0 [A] |
| = Pre-convert diluted | 316.7 |
| + in-the-money converts at $165 | 26.7 |
| **= Fully diluted at $165** | **343.4** |

Market-data sites quoting a ~$45bn market cap use ~272m shares. **True diluted equity value is ~26% higher, at $56.8bn** [I].

---

## 8. Unit economics and ROIC: the heart of the debate

The table shows unlevered pre-tax IRR per $1 of new ARR. Assumptions: capex at t0; 60% of run-rate revenue in year 1; compute retired at end of life; undepreciated data-centre value (25% of capex, 25-year life) recovered at the end. Margins are after SBC.

| Margin post-SBC \ capex/ARR | 2.0x | 2.4x | **2.7x** | 3.0x | 3.4x |
|---|---|---|---|---|---|
| **5-yr GPU life:** 55% | 12.2% | 6.9% | 3.8% | 1.2% | −1.6% |
| 5-yr: 60% | 15.0% | 9.4% | 6.1% | 3.3% | 0.3% |
| **6-yr GPU life:** 55% | 15.9% | 10.7% | **7.6%** | 5.1% | 2.3% |
| 6-yr: 60% | 18.6% | 13.1% | 9.8% | 7.1% | 4.2% |
| **7-yr GPU life:** 60% | 21.0% | 15.6% | 12.4% | 9.8% | 6.9% |
| 7-yr: 65% | 23.6% | 17.9% | 14.6% | 11.8% | 8.8% |

Full 5/6/7-year grids are in `model_output.md`.

**Interpretation [I]:**

- **Base case.** At ~2.7x capex/ARR, 57% post-SBC margin and a 6-year life, the project IRR is ~8–9%, below the ~11.3% WACC.
- **What creates value.** Value is created only with roughly ≤2.4x capex intensity and ≥60% margins, or with a ≥7-year economic life.
- **The bullish payback claims are misleading.**
  - Management cites ~22-month paybacks on Q2 contracts [C/G], with prepayments covering ~50–60% of capex on ~70% of deals [C].
  - A BNP-linked analysis cites a "66% ROIC and 1.3-year payback" [C].
  - Both treat prepayments as returns. A prepayment is a customer loan repaid in service. It reduces the equity needed and raises levered equity IRR, but it does not change project economics. The customer is compensated through a lower effective price.
  - The BNP figure implies ~$30bn of annual revenue per GW. That is roughly 2x what GPU-hour arithmetic supports even at today's elevated prices.
- **Peer evidence.** CoreWeave, at ~$13bn of revenue, guides FY26 adjusted operating income to only $0.96–1.15bn (~8% margin) [F] on an asset base well above $50bn. This is the clearest real-world read of mature neocloud ROIC: low single digits pre-tax.
- **Nebius's levers to do better:**
  - Owned campuses, which cut rent.
  - The software and inference mix: Token Factory, AI Studio, Inferize.
  - Auction pricing.
  - Longer GPU lives, given that A100s and H100s still rent profitably in 2026.

  I credit these in the bull case. I do not assume them in the base case.

---

## 9. Independent forecasts (2026–2035)

### 9.1 Methodology

- **Revenue.** Revenue is driven by exit ARR, constrained by connected power. Revenue = 0.92 × the average of opening and closing ARR, reflecting back-end-loaded commissioning.
- **Capex.** Growth capex = capex/ARR × new ARR, with 35% spent a year ahead. Replacement capex re-buys the compute share (75%) of each vintage at 85–90% of original cost once its economic life ends.
- **Depreciation.** Book D&A follows company policy: 5 years for compute, 20 years for data centres.
- **Taxes.** 23% blended cash tax after a $2bn NOL [A].
- **SBC.** Treated as a real cost.

### 9.2 Base case vs consensus ($bn)

| | 2026E | 2027E | 2028E | 2029E | 2030E | 2032E | 2035E |
|---|---|---|---|---|---|---|---|
| Exit ARR | 7.5 | 15.5 | 24.0 | 31.5 | 38.0 | 47.0 | 54.5 |
| **Revenue** | **3.15** | **10.6** | **18.2** | **25.5** | **32.0** | **41.4** | **49.2** |
| Consensus revenue [C] | ~3.2 | 12.3 | ~21 | | ~45 | | |
| Adj. EBITDA | 1.3 | 5.3 | 10.0 | 14.8 | 19.2 | 24.8 | 29.5 |
| Consensus EBITDA [C] | | 6.6 | 13.7–15.5 | | | | |
| Adj. EBITDA margin | 40% | 50% | 55% | 58% | 60% | 60% | 60% |
| EBIT (after SBC, book D&A) | (1.8) | (1.6) | (0.6) | 0.8 | 2.6 | 7.5 | 10.9 |
| EBIT margin | n.m. | −15% | −3% | 3% | 8% | 18% | 22% |
| Capex (growth + refresh) | 22.0 | 22.1 | 22.0 | 19.5 | 16.7 | 23.9 | 17.5 |
| Unlevered FCF | (21.1) | (17.4) | (12.8) | (5.6) | 1.5 | (2.0) | 8.1 |

Bear and bull tables are in `model_output.md`, with summary drivers below.

| Driver | Bear (25%) | Base (50%) | Bull (25%) |
|---|---|---|---|
| Exit ARR 2027 / 2030 / 2035 ($bn) | 11.5 / 11.0 / 1.5 (run-off) | 15.5 / 38 / 54.5 | 20 / 50 / 75 |
| 2030 revenue ($bn) | 10.8 | 32.0 | 41.9 |
| Capex per $ of new ARR | 3.2x | 2.7x | 2.4x |
| LT adj. EBITDA margin | 48% → 40% | 60% | 64% |
| GPU economic life | 5 years, no refresh (run-off) | 6 years | 7 years |
| Terminal RONIC | n/a (residual site value $6bn) | 12% | 16% |
| Stakes (ClickHouse / Avride / other, $bn) | 2.5 / 0.4 / 0.2 | 3.4 / 1.0 / 0.4 | 6.0 / 2.0 / 0.5 |

**Narrative of each case:**

- **Bear.** Pricing resets as the 2026–27 build wave lands, and CoreWeave-style credit stress spreads. Nebius completes its contracted Microsoft and Meta builds and then stops growth and refresh capex. The fleet runs off, and terminal value is the sale of powered sites.
- **Base.** Nebius delivers ~1 GW/yr of connected power through 2028, then slows as funding costs and competition bite. Margins match CoreWeave's.
- **Bull.** Nebius hits management's >1 GW/yr pace, sells software-rich capacity at premium prices, and earns a ≥7-year GPU life. This is roughly where consensus 2030 revenue (~$45bn) sits.

**Why I am below consensus [I]:**

1. Connected-to-revenue lag. YE26 exit ARR of $7–9bn [G] needs a steep H2 ramp. Finsee notes ~$2.2bn of H2 revenue is required for the $3.2bn midpoint. I sit at the lower half: $3.15bn revenue and $7.5bn exit ARR.
2. Hopper-era contracts re-price lower as they roll.
3. Funding capacity: $45–50bn of external capital by 2029 at a rising cost of capital.
4. Consensus 2030 (~$45bn) assumes ~4.5 GW of revenue-generating IT capacity. That is essentially all of today's contracted power, delivered on time.

### 9.3 Funding needs and dilution (base)

| | 2026H2 | 2027 | 2028 | 2029 | 2030 |
|---|---|---|---|---|---|
| Unlevered FCF ($bn) | (13.1) | (17.4) | (12.8) | (5.6) | 1.5 |
| Cumulative funding need ($bn) | 13.1 | 30.5 | 43.4 | **49.0** | 47.4 |
| New shares if 50% equity-funded at $165 (m) | 40 | 92 | 131 | **148** | 143 |

Mitigants:
- $14.4bn of pro forma cash.
- H2-26 prepayments (~$4.6bn implied by the >$9bn FY26 guide [G]).
- Asset-backed debt at ~6.8%.
- Meta's $12bn firm orders, which will likely carry prepayments.

Even so, **I assign an 85% probability that Nebius raises ≥$5bn of new equity or convertibles within 12 months.**

---

## 10. Valuation

### 10.1 Discount rate

| Input | Value | Basis |
|---|---|---|
| Risk-free rate | 4.65% | 10-yr UST ~4.6–4.7% (4.68% on 30-Jul-26 per FRED) [C/A] |
| Equity risk premium | 5.0% | [A] |
| Beta (explicit period / terminal) | 1.8 / 1.3 | [A]. Realised beta is higher but distorted by flows. Mature analogues: equipment rental, data centres |
| Cost of equity | 13.65% / 11.15% | |
| Pre-tax cost of debt | 7.5% | Secured facility ~6.8%; HY marginal cost higher (CRWV 2030s yield ~16%) [F/C] |
| Target debt / value | 30% | [A] |
| **WACC** | **11.3% explicit / 9.5% terminal** | |

### 10.2 DCF results ($bn except per share)

| | Bear | Base | Bull |
|---|---|---|---|
| PV of UFCF, 2026H2–2035 | (5.8) | (34.9) | (18.4) |
| PV of terminal value (value-driver) | 2.4 | 49.3 | 102.0 |
| Terminal value / 2036 adj. EBITDA | residual | 4.1x | 5.8x |
| **Core EV** | (3.4) | 14.4 | 83.6 |
| + Stakes (ClickHouse, Avride, TripleTen/Toloka) | 3.1 | 4.8 | 8.5 |
| + Pro forma cash | 14.4 | 14.4 | 14.4 |
| − Converts treated as debt | (13.5) | (13.5) | (5.8)* |
| − Secured facility + leases | (2.3) | (2.3) | (2.3) |
| − PV of deferred revenue | (4.6) | (4.6) | (4.6) |
| Equity value | (6.2) | 13.3 | 93.9 |
| Diluted shares (m) | 316.7 | 316.7 | 367.2 |
| **Value per share** | **$11**† | **$42** | **$256** |

\* In the bull case the $51.45–$183.22 series convert (+50.5m shares). † The bear's intrinsic value is negative (−$20). Equity is valued as a Merton call on assets struck at total claims (σ 35%, T 4.5 years), because the converts don't mature until 2029–34.

**Why the explicit-period PV is so negative [I].** Roughly $60bn of growth capex in 2026H2–2028 is spent before the matching revenue arrives. At a 6-year GPU life, a ~$14bn refresh wave hits in 2032–34. Investors valuing NBIS on EV/EBITDA ignore both.

**Terminal value is the swing factor.** The value-driver method values the business on **economic** depreciation, meaning the cost of continually replacing GPUs, at roughly half of EBITDA. That yields ~4x EBITDA. The exit-multiple method at 7x (a premium to CoreWeave's ~5.7x, and in the range where capital-intensive equipment-rental businesses have historically traded — an analogy I have not re-verified for this report) yields ~$100 more per share in the base case. I weight the two 50/50 because the terminal economics are the key unknown.

| Terminal method | Bear | Base | Bull | Weighted |
|---|---|---|---|---|
| Value-driver (RONIC) | $11 | $42 | $256 | $88 |
| 5x EV/EBITDA exit | $11 | $70 | $210 | |
| 7x EV/EBITDA exit | $11 | $142 | $303 | $150 |
| 9x EV/EBITDA exit | $11 | $208 | $393 | |
| **50/50 blend** | **$11** | **$92** | **$280** | **$119** |

### 10.3 Sensitivity analysis (base case, value per share)

All grids show value-driver DCF intrinsic value per share, floored at $0. They exclude the option floor so the grids stay monotonic. Base = $42.

**WACC vs terminal growth** (explicit and terminal WACC shifted together):

| WACC \ g | 2% | 3% | 4% |
|---|---|---|---|
| 9.5% | $111 | $133 | $164 |
| 10.5% | $63 | $75 | $91 |
| **11.3%** | $35 | **$42** | $51 |
| 12.0% | $15 | $19 | $25 |
| 13.0% | $0 | $0 | $0 |

**Long-term EBITDA margin vs capex intensity:**

| Margin \ capex/ARR | 2.1x | 2.4x | **2.7x** | 3.0x | 3.3x |
|---|---|---|---|---|---|
| 50% | $39 | $0 | $0 | $0 | $0 |
| 55% | $87 | $41 | $0 | $0 | $0 |
| **60%** | $136 | $89 | **$42** | $0 | $0 |
| 65% | $181 | $137 | $90 | $44 | $0 |
| 70% | $223 | **$182** | $139 | $92 | $45 |

**GPU economic life vs scale (2035 exit ARR):**

| Life \ 2035 ARR | $36bn | $45bn | $54bn | $66bn | $78bn |
|---|---|---|---|---|---|
| 5 yrs | $1 | $2 | $1 | $1 | $1 |
| **6 yrs** | $29 | $36 | **$42** | $50 | $58 |
| 7 yrs | $51 | $64 | $76 | $91 | $106 |
| 8 yrs | $71 | $88 | $106 | $127 | $148 |

**Terminal RONIC vs WACC:**

| WACC \ RONIC | 9% | 11% | 12% | 15% | 20% |
|---|---|---|---|---|---|
| 10.5% | $54 | $69 | $75 | $87 | $99 |
| **11.3%** | $25 | $37 | **$42** | $53 | $63 |
| 12.0% | $4 | $15 | $19 | $28 | $37 |

**Key insights [I]:**
- **At a 5-year economic life the equity is worth roughly nothing at any scale.**
- At 6 years, doubling the business adds only ~$30/share.
- Scale becomes valuable only at ≥7-year lives.

So the "ARR beat" narrative is far less important than the unanswered question of how long a GPU earns.

### 10.4 Reverse DCF: what $165 implies

Holding my base ARR path and WACC, the price is justified only by **long-term adj. EBITDA margins of 65% at 2.2x capex/ARR, 68% at 2.4x, 71% at 2.6x, or 76% at 2.9x**. For reference, CoreWeave runs a 59% adj. EBITDA margin [F]. Alternatively, holding all my base assumptions, the stock **discounts the base case at ~9.1%** (implied IRR), below any reasonable cost of capital for a 1.8-beta, pre-profit, cash-burning company.

### 10.5 Comparable companies

| Company | EV ($bn) | EV / 2026E revenue | EV / 2027E revenue | EV / 2027E EBITDA | Comment |
|---|---|---|---|---|---|
| CoreWeave (CRWV) | 79.9 | 6.2x | 3.0x | ~5.7x | Distressed credit (CDS ~1,195bp); gross leverage ~6.3x [C] |
| Lambda (private) | ≤14.5 (pre-money, reported) | ~9.7x | n/a | n/a | Unconfirmed round; IPO targeted 2027 [C] |
| **NBIS at $165 (consensus)** | **54.8** | 17.1x | 4.5x | **8.3x** | |
| NBIS at $165 (my base) | 54.8 | 17.4x | 5.2x | 10.4x | |

**Peer selection [I].** CoreWeave is the only liquid pure-play comparable. IREN, Applied Digital and other ex-miners have mixed business models, and I could not verify their current valuations. Hyperscalers (AWS, Azure, Google) and Oracle carry far lower capital intensity and diversified cash flows, so I exclude them.

**Premium [I].** Nebius deserves a premium to CoreWeave for its balance sheet. I estimate 20–40%, i.e. ~7–8x 2027E EBITDA. Applying 5.5–10x to my base 2027E EBITDA, discounted one year, gives **$79–145/share**. That range is consistent with the blended DCF.

### 10.6 Sum of the parts (base)

| Component | $bn | $/share |
|---|---|---|
| AI cloud (DCF EV) | 14.4 | 46 |
| ClickHouse (~27% of a $15bn round, less a 15% illiquidity haircut) | 3.4 | 11 |
| Avride (~83%) | 1.0 | 3 |
| TripleTen + Toloka | 0.4 | 1 |
| Pro forma cash | 14.4 | 46 |
| Convertibles (all out of the money at value) | (13.5) | (42) |
| Secured facility + leases | (2.3) | (7) |
| PV of deferred revenue | (4.6) | (14) |
| **Equity (value-driver base)** | **13.3** | **42** |

The non-core assets are worth ~$15/share combined. ClickHouse could be worth ~$20/share at a ~$30bn private mark (speculated, not reported). That is not enough to change the conclusion.

### 10.7 Valuation summary (per share)

| Method | Low | Mid | High |
|---|---|---|---|
| DCF, value-driver (bear / weighted / bull) | $11 | $88 | $256 |
| DCF, 7x exit (bear / weighted / bull) | $11 | $150 | $303 |
| Comps (5.5–10x 2027E EBITDA) | $79 | $112 | $145 |
| Analyst PT range [C] | $145 | ~$290 | $430 |
| 52-week trading range | $74 | | $300 |
| **Blended fair value** | | **$119** | |
| **Current price** | | **$165** | |

---

## 11. Analyst consensus and where I differ

- **Ratings [C].**
  - TipRanks: "Moderate Buy", average PT $300 (14 analysts; high $410).
  - ChartMill: average $297 (25 analysts; range $145–$431).
  - Benzinga: consensus $234 (1 Strong Buy / 11 Buy / 4 Hold).
  - Koyfin: 7 of 9 at Buy or better, 1 Hold, 1 Sell; average $284.
- **Recent actions.**
  - BNP Paribas upgraded to Outperform, PT $260 → $399 (24-Sep-26), seeing ~$22bn of ARR by YE27.
  - Rosenblatt initiated at Buy, PT $304 (8-Oct-26).
- **Estimates [C].** 2027E revenue ~$12.3bn (Yahoo, 22 analysts; range $8.8–18.0bn), EBITDA ~$6.2–6.6bn. 2028E revenue ~$15–22.5bn. 2030E revenue ~$45bn (sell-side survey cited by Jonah Lupton).

**Key disagreements with the Street [I]:**

| Issue | Street | Me |
|---|---|---|
| Valuation anchor | EV/ARR, EV/backlog, EV/2027–28 EBITDA | FCF after replacement capex and SBC; value-driver terminal |
| Capex | Growth capex treated as optional and value-creating | Replacement capex ≈ 50% of mature EBITDA. Growth is value-neutral or negative unless IRR > WACC |
| Prepayments | Treated as validating demand and "payback" | A liability ($6bn) and a financing source; I subtract its PV ($4.6bn) |
| Share count | Many models use basic shares | 343m diluted at $165, rising with price |
| 2030 revenue | ~$45bn | $32bn base. The Street number is essentially my bull case |
| Discount rate | Often 9–10% | 11.3% explicit / 9.5% terminal; sector credit spreads are widening |

---

## 12. Risk assessment

| # | Risk | Likelihood | Impact | Comment |
|---|---|---|---|---|
| 1 | **GPU pricing reset** as the 2026–27 build wave lands | High | High | Spot rates are cyclical; Hopper roll-offs re-price lower |
| 2 | **Financing / dilution** | Very high | Med–High | ~$49bn base-case gap; converts dilute as the price rises; ATM still has 12.3m shares |
| 3 | **Customer concentration and in-sourcing** (Microsoft, Meta) | Medium | High | Top 3 customers = 59% of Q2 revenue; Meta is building its own cloud. Tranche-termination rights for delays |
| 4 | **Technology obsolescence** (Blackwell → Rubin → beyond) | High | High | Determines economic life, the key driver of value. Book life was extended 4 → 5 years in Q1-26 |
| 5 | **Execution / commissioning delays** at GW campuses | Medium | Medium | Revenue lags connected power by months; FY26 ARR guide needs a steep Q4 |
| 6 | **Sector contagion** (CoreWeave credit event) | Medium | High | NBIS fell 12% in sympathy on 16-Oct-26 |
| 7 | **Governance / control** (Volozh ~50% of votes; dual class) | Certain | Low–Med | Limits activism and bid premia |
| 8 | **Avride cash burn** | High | Low | ~$130–150m/yr of losses [I] |
| 9 | **Interest rates** | Medium | Medium | Long-duration equity; high short interest (~22%) creates squeeze risk both ways |
| 10 | **Upside risk to my SELL:** superior economics, a strategic bid, or a large 2027 guide | Medium | High | See §13 |

---

## 13. Independent thesis, catalysts and predictions

### 13.1 Bull thesis (the case I could be wrong about)

1. **Capacity scarcity is structural.** Power, not GPUs, is the bottleneck through 2028. Nebius's 5 GW contracted power is a scarce asset, and the +15% auction and Oct-26 price increases show pricing power.
2. **Hyperscalers are de-risking the model.** Microsoft and Meta contracts with prepayments and IG-backed debt mean Nebius can grow 3–4x with limited equity dilution, and the Meta backstop puts a floor under merchant capacity.
3. **Full-stack software turns rented GPUs into a platform.** Token Factory, Inferize and AI Studio could push mature margins to 65%+, with inference workloads extending GPU lives to 7–8 years.
4. **Strategic value.** NVIDIA owns ~7% of the company, and a hyperscaler could acquire Nebius for power plus talent.

### 13.2 Bear thesis

1. **Commodity returns.** GPU rental earns roughly its cost of capital over a full cycle. CoreWeave's ~8% adjusted operating margin at $13bn of revenue is the tell.
2. **A treadmill of capex.** Every 5–7 years the compute base must be re-bought, consuming ~half of mature EBITDA.
3. **Customer power.** The largest buyers set prices, own the demand, and are building their own capacity.
4. **Funding reflexivity.** Growth requires ~$50bn of external capital. Equity issuance works at $165 but becomes punitive at $80, and sector credit stress is already visible.

### 13.3 Overlooked factors

- **Deferred revenue is debt-like.** About 98% of H1-26 operating cash flow was customer prepayments. Valuing NBIS on "positive OCF" double-counts.
- **True share count is ~26% above what quote screens show**, and the convertible stack adds ~68m more shares as the stock rises.
- **The 4 → 5-year depreciation change** boosted EBIT by ~$170m in 2026 [F, company estimate for 2026: $167.6m]. Peers using 6 years (CoreWeave) are even more flattered. On an economic basis, EBIT across the sector is overstated.
- **The Meta backstop cuts both ways.** It insures unsold capacity, but it also means the incremental demand signal for Nebius's 2027–28 capacity is partly Meta's own capacity planning, from a customer that is also building in-house.
- **The power portfolio is a hidden floor.** Even in the bear case, 5 GW of contracted power and powered shells have option value to hyperscalers. This is why I don't model the equity at zero.
- **ClickHouse is a real asset** (~$11–20/share) but not a thesis driver. Its value shows up in GAAP EPS through revaluation gains, which flatters reported profitability.

### 13.4 Catalysts (next 12 months)

| Timing | Event | Expected direction |
|---|---|---|
| ~Nov-26 (date unconfirmed; ~11-Nov estimated) | **Q3-26 results**: ARR progress toward $7–9bn, Q4 implied ramp, 2027 capex commentary | Volatile. A miss on the ARR trajectory is the key downside risk to the stock |
| Q4-26 – Q1-27 | **New equity, convertible or ABS issuance** | Negative (dilution) unless done via secured ABS |
| Feb-27 (with Q4) | **FY27 guidance** (ARR, capex, margin) | Upside risk. An ARR guide ≥$20bn would challenge my base case |
| 2027 | Meta $12bn firm deployments; Rubin ramp; Microsoft tranche completion | Execution test |
| Rolling | CoreWeave credit (CDS, bond yields), Gartner-style capex scepticism | Negative skew |
| Rolling | ClickHouse funding round / IPO; Avride milestones | Modest positive |

### 13.5 Probabilistic predictions

| Prediction | Probability |
|---|---|
| FY26 revenue ≥ $3.0bn (guidance low end) | 75% |
| YE26 ARR ≥ $7.0bn / ≥ $9.0bn | 65% / 15% |
| FY27 revenue ≥ consensus $12.3bn | 35% |
| ≥ $5bn of new equity or convertibles issued within 12 months | 85% |
| Fully diluted share count ≥ 375m by YE27 | 60% |
| New ≥ $10bn contract with a hyperscaler or AI lab within 12 months | 50% |
| Positive GAAP operating income in FY28 | 35% |
| NBIS acquired or merged by 2030 | 15% |
| NBIS below $165 on 18-Oct-27 | 60% |

### 13.6 What would invalidate my thesis

1. Disclosed AI-cloud EBIT margins ≥ 25% on a 5-year depreciation basis by 2028. That would imply superior unit economics.
2. Evidence that 2022–24 GPU vintages re-contract at ≥ 70% of their original rate, implying ≥ 7-year economic lives.
3. A sustained mix shift to software and inference above ~20% of revenue at premium margins.
4. A ≥ $20bn ARR guide for YE27 funded predominantly by prepayments and secured debt rather than equity.
5. A takeover offer.

---

## 14. Final recommendation

**SELL (Underweight). Fair value $120. 12-month price target $135, about −18% from $165.47.**

**3–5 year outlook.** My base case has Nebius reaching ~$32bn of revenue and ~$19bn of adj. EBITDA in 2030, a remarkable business outcome. But after funding a ~$49bn gap (~148m new shares if half is equity), the 2030 value per share is:
- ~$194 at 5x EBITDA, a ~4% annualised return, or
- ~$281 at 7x, a ~13% annualised return.

Probability-weighted, the expected annual return is **~2–10%**, below the ~13.7% cost of equity [I]. The bear case (~25%) loses most of the investment. The bull case (~25%) compounds at ~23%.

**Why SELL, in three points:**

1. **The unit economics are roughly IRR ≈ WACC.** Growth does not create per-share value without proof of better pricing, margins or GPU lives.
2. **Dilution and debt-like items are understated.** These include the convertible "success tax", $6bn of prepaid obligations, and a ~$49bn funding gap.
3. **Consensus and price targets assume a bull case**, and sector credit stress is rising. The market prices NBIS at a ~45% EBITDA-multiple premium to CoreWeave and discounts my base case at ~9%.

**Key risks to the call:**
- Faster-than-expected ARR delivery with premium pricing.
- A strategic bid.
- A short squeeze: short interest is ~22% and the stock fell 45% from its high.
- Software-driven margin expansion.

This is a **medium-conviction** call on valuation, not on business quality. I would revisit below ~$95 or on the evidence listed in §13.6.

*This report reflects independent analysis for informational purposes. It is not personalised investment advice. Figures are as of 18-Oct-2026; verify live prices and the latest filings before acting.*

---

## 15. Sources (accessed 18-Oct-2026)

**Company filings and releases**

- Nebius Q2-26 shareholder letter, 12-Aug-26 (SEC 6-K ex-99.2): https://www.sec.gov/Archives/edgar/data/1513845/000110465926094568/tm2622968d1_ex99-2.htm
- Nebius Q2-26 results press release, 12-Aug-26: https://www.morningstar.com/news/business-wire/20260812963359/nebius-reports-second-quarter-2026-financial-results
- Nebius Q2-26 unaudited interim financial statements (6-K): https://www.sec.gov/Archives/edgar/data/0001513845/000110465926094844/nbis-20260812xex99d2.htm
- Nebius Q1-26 results, 13-May-26: https://www.businesswire.com/news/home/20260513568820/en/Nebius-reports-first-quarter-2026-financial-results
- Q1-26 interim statements (RPO $33.6bn at 31-Mar-26; share count): https://www.sec.gov/Archives/edgar/data/0001513845/000110465926064092/nbis-20260331xex99d2.htm
- Nebius Q4-25 results, 12-Feb-26: https://www.businesswire.com/news/home/20260212534404/en/
- Q4-25 letter (170 MW active power; $1.25bn ARR): https://www.sec.gov/Archives/edgar/data/1513845/000110465926013946/tm266173d1_ex99-2.htm
- 2025 Form 20-F (depreciation policy; Avride losses): https://www.sec.gov/Archives/edgar/data/1513845/000110465926052948/nbis-20251231x20f.htm
- Meta agreement, 16-Mar-26: https://nebius.com/newsroom/nebius-signs-new-ai-infrastructure-agreement-with-meta (6-K: https://www.sec.gov/Archives/edgar/data/1513845/000110465926027886/tm268879d1_6k.htm)
- Microsoft agreement, 8-Sep-25: https://nebius.com/newsroom/nebius-announces-multi-billion-dollar-agreement-with-microsoft-for-ai-infrastructure (6-K: https://www.sec.gov/Archives/edgar/data/1513845/000110465925088860/tm2525580d2_ex99-2.htm)
- Microsoft upfront payments (~$6.96bn), 4-May-26: https://www.theenergymag.com/news/2026-05-04/microsoft-nebius-ai
- Meta ~$3bn deal (Nov-25), DCD: https://www.datacenterdynamics.com/en/news/nebius-signs-3bn-deal-with-meta-says-current-available-capacity-is-sold-out-as-it-targets-25gw-by-end-of-2026/
- $4.0bn converts (Mar-26): https://nebius.com/newsroom/nebius-group-announces-pricing-of-upsized-private-offering-of-4-0-billion-of-convertible-senior-notes
- $5.0bn converts (Aug-26), options and exchange: https://nebius.com/newsroom/nebius-group-announces-pricing-of-upsized-private-offering-of-5-0-billion-of-convertible-senior-notes ; https://www.sec.gov/Archives/edgar/data/0001513845/000110465926098924/tm2623617d1_ex99-1.htm
- $2.75bn converts (Sep-25): https://www.streetinsider.com/Equity+Offerings/Nebius+Group+prices+%242.75+billion+convertible+notes+offering/25322193.html
- $1bn converts (Jun-25): https://www.sec.gov/Archives/edgar/data/1513845/000110465925055197/tm2516825d1_ex99-1.htm
- $775m secured facility, 17-Jul-26: https://nebius.com/newsroom/nebius-raises-775-million-in-first-secured-debt-financing-to-accelerate-global-buildout
- AGM results / share count, Aug-26: https://www.sec.gov/Archives/edgar/data/0001513845/000110465926101076/tm2624077d1_ex99-1.htm
- Volozh Form 3 (Mar-26): https://www.sec.gov/Archives/edgar/data/0001513845/000151384526000030/form3.xml
- Avride $375m (Oct-25): https://nebius.com/newsroom/avride-secures-strategic-investment-and-other-commitments-of-up-to-375-million-backed-by-uber-and-nebius
- Finland AI factory: https://nebius.com/newsroom/nebius-to-construct-310-mw-ai-factory-in-finland
- AIB Data Centers lease, 30-Sep-26: https://www.globenewswire.com/news-release/2026/09/30/3371855/0/en/aib-data-centers-signs-contract-with-nebius-for-ai-data-center-capacity.html

**Earnings-call coverage**

- Investing.com Q2-26 transcript: https://www.investing.com/news/transcripts/earnings-call-transcript-nebius-beats-q2-2026-forecasts-shares-jump-165-93CH-4855076
- Motley Fool Q2-26 transcript: https://www.fool.com/earnings/call-transcripts/2026/08/19/nebius-nbis-q2-2026-earnings-call-transcript/
- Yahoo Q2 call highlights (connected vs contracted power): https://finance.yahoo.com/technology/ai/articles/nebius-group-q2-earnings-call-150444472.html
- Finsee Q2-26 review: https://finsee.ai/earnings/nbis/2026/q2/en/
- TIKR Q1-26: https://www.tikr.com/blog/nebius-grew-revenue-684-in-q1-sold-out-its-entire-capacity-and-raised-its-capex-target-to-25-billion

**Market, consensus and peers**

- Price, 16-Oct-26: https://www.gurufocus.com/news/9128275/nebius-group-nbis-shares-plunge-nearly-12-amid-sector-selloff-on-october-16-2026 ; https://247wallst.com/investing/2026/10/16/coreweave-and-nebius-sink-9-as-sentiment-index-closes-in-on-extreme-fear/
- TipRanks: https://www.tipranks.com/stocks/nbis/forecast · ChartMill: https://www.chartmill.com/stock/quote/NBIS/analyst-ratings · Benzinga: https://www.benzinga.com/quote/NBIS/price-targets
- Yahoo estimates: https://finance.yahoo.com/quote/NBIS/analysis/ · MarketScreener: https://www.marketscreener.com/quote/stock/NEBIUS-GROUP-8037501/finances/
- BNP Paribas upgrade (24-Sep-26): https://cryptobriefing.com/nebius-stock-jumps-bnp-paribas-price-target/ ; https://stocktwits.com/news-articles/markets/equity/nebius-stock-surges-bnp-paribas-upgrades-sees-22-b-arr-by-2027/cZMaMEpRBa4
- Customer concentration: https://www.nasdaq.com/articles/do-nebius-mega-deals-increase-customer-concentration-risk ; https://www.webull.com/news/15694468284498944
- 22-month payback (Motley Fool, 17-Oct-26): https://www.fool.com/investing/2026/10/17/is-nebius-a-buy-its-22-month-payback-offers-a-clue/
- Balance-sheet comparison (TIKR): https://www.tikr.com/blog/nebius-stock-fell-12-in-a-day-as-coreweaves-debt-scare-spread-heres-why-its-balance-sheet-isnt-coreweaves-yet
- Short interest: https://www.benzinga.com/quote/NBIS/sentiment
- Earnings date estimate: https://beta.earningswhispers.com/go/w/NBIS
- CoreWeave Q2-26: https://investors.coreweave.com/news/news-details/2026/CoreWeave-Reports-Strong-Second-Quarter-2026-Results/default.aspx ; EV: https://stockanalysis.com/stocks/crwv/statistics/ ; credit stress: https://www.marketscreener.com/news/coreweave-shares-bonds-slide-further-as-ai-debt-fears-mount-ce785dd3dd8cf522 ; https://www.fool.com/investing/2026/10/17/coreweave-is-projected-to-have-a-63x-gross-leverag/
- Lambda funding: https://www.datacenterdynamics.com/en/news/lambda-seeks-4bn-funding-round-ahead-of-planned-ipo-report
- IREN: https://www.zacks.com/stock/news/2854563/iren-aims-for-34b-ai-cloud-arr-by-2026-can-it-hit-the-target

**Industry**

- Gartner, 14-Oct-26: https://www.gartner.com/en/newsroom/press-releases/2026-10-14-gartner-says-over-us-dollars-three-and-a-half-trillion-of-artificial-intelligence-infrastructure-spending-could-be-exposed-to-impairment-by-the-end-of-2030
- NVIDIA per-GW economics: https://www.techtimes.com/articles/328295/20260930/nvidia-per-gigawatt-revenue-hits-40b-vera-rubin-anthropic-locks-25-gw.htm
- Neocloud capacity: https://thebuildout.ai/intel/04bb5450-a326-43c0-a0a6-ce3cd03d192f
- GPU rental prices: https://www.mercatus-ai.com/blog/gpu-rental-prices ; https://gpusmith.com/articles/en/h100-rental-price-history-trends
- BNP-linked unit-economics analysis: https://exencialrp.substack.com/p/scarce-compute-is-repricing-nebius
- Nebius price increase (single source): https://shattered.io/nebius-ai-cloud-price-hike-memory-2026/
- Sites (DCD): Independence https://www.datacenterdynamics.com/en/news/nebius-breaks-ground-on-400-acre-data-center-campus-in-independence-missouri/ · Pennsylvania https://www.datacenterdynamics.com/en/news/nebius-details-plans-for-12gw-data-center-campus-in-pennsylvania/ · Birmingham https://www.datacenterdynamics.com/en/news/nebius-targets-data-center-development-in-birmingham-alabama/ · Béthune https://www.datacenterdynamics.com/en/news/nebius-plans-240mw-data-center-in-b%C3%A9thune-france/
- NVIDIA $2bn investment: https://thenextweb.com/news/nvidia-invests-2-billion-in-nebius
- Inferize acquisition: https://ad-hoc-news.de/boerse/news/corporate-news/nebius-acquires-inferize-nebius-stock-trades-at-eur-150-07/70355330
- ClickHouse: https://sacra.com/c/clickhouse/ ; https://www.nasdaq.com/articles/can-nebius-monetize-its-bets-clickhouse-avride-and-toloka

---

### Appendix: model files

| File | Contents |
|---|---|
| `nbis_model.py` | Auditable model: inputs tagged by source, scenario engine, DCF (value-driver and exit multiple), option floor, reverse DCF, unit-economics IRR, comps, SOTP, funding/dilution, sensitivities |
| `model_output.md` | Every table generated by the model, including full bear and bull forecasts |
| `NBIS_model.xlsx` | Inputs, convertible stack, scenario drivers, annual forecasts and valuation summary |
