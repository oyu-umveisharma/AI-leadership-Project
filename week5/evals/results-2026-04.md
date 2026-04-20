# Evaluation Results — April 2026

**Run Date:** 2026-04-19 18:49
**Platform:** CRE Intelligence Platform

---

## Summary

| Category | Pass | Fail | Rate |
|----------|------|------|------|
| Data Pipeline Integrity | 27 | 0 | 100% |

| LLM Facility Extraction | 10 | 0 | 100% |
| **Total** | **37** | **0** | **100%** |
| Schema Compliance | 8/8 | - | 100% |
| Freshness Compliance | 3/19 | - | STALE |

**Overall Status:** PASS

---

## Benchmark Test Cases

| ID | Agent | Description | Result | Details |
|-----|-------|-------------|--------|---------|
| MIG-01 | migration | Texas is a top-3 migration destination state | PASS | Top 3: ['TX', 'FL', 'AZ'] |
| MIG-02 | migration | Florida is a top-3 migration destination state | PASS | Top 3: ['TX', 'FL', 'AZ'] |
| MIG-03 | migration | California shows net outbound migration (low score) | PASS | CA composite_score = 21.2 |
| MIG-04 | migration | Migration data returns all 50 states + DC | PASS | State count: 51 |
| MIG-05 | migration | Composite score formula: 60% pop growth + 40% biz score | PASS | Actual=99.2, Expected=99.2, Diff=0.0 |
| PRC-01 | pricing | Industrial cap rate benchmark is 5.6% | PASS | Benchmark constant verified in src/cre_pricing.py |
| PRC-02 | pricing | Office has highest cap rate among all property types | PASS | Office cap_rate 0.085 is highest by code definition |
| PRC-03 | pricing | REIT universe contains expected tickers | PASS | All anchor REITs present |
| PRC-04 | pricing | REIT prices are positive and within reasonable range | PASS | All prices in valid range |
| RATE-01 | rates | Fed Funds Rate is within plausible range | PASS | Fed Funds = 3.64% |
| RATE-02 | rates | 10Y Treasury is populated and plausible | PASS | 10Y Treasury = 4.32% |
| RATE-03 | rates | Yield curve data has all 5 maturities | PASS | All 5 maturities present |
| RATE-04 | rates | Cap rate adjustments cover all 7 property types | PASS | Cap rate adjustments count: 7 |
| NRG-01 | energy | Construction cost signal is one of HIGH/MODERATE/LOW | PASS | Signal = 'MODERATE' |
| NRG-02 | energy | All 5 commodities are tracked with valid SMA data | PASS | All 5 commodities with valid SMA |
| NRG-03 | energy | Signal HIGH when avg momentum > +5% | PASS | avg_momentum=2.9%, signal='MODERATE' |
| LBR-01 | labor_market | Tenant demand signal score is 0-100 | PASS | Demand signal score = 91 |
| LBR-02 | labor_market | Demand signal label matches score thresholds | PASS | score=91, label='STRONG', expected='STRONG' |
| INF-01 | inflation | Inflation signal is one of HOT/MODERATE/COOLING | PASS | Inflation signal = 'HOT' |
| CRD-01 | credit | Credit conditions signal is one of LOOSE/NEUTRAL/TIGHT | PASS | Credit signal = 'LOOSE' |
| PUB-01 | published_benchmark | Industrial cap rate matches CBRE survey within 25bp | PASS | Our=5.6%, Benchmark=5.5%, Diff=10bp (tol=25bp) |
| PUB-02 | published_benchmark | Office cap rate within CBRE survey range (7.0-7.5%) | PASS | Our=7.2%, Range=7.0-7.5% |
| PUB-03 | published_benchmark | Multifamily cap rate matches NCREIF within 25bp | PASS | Our=5.3%, Benchmark=5.2%, Diff=10bp (tol=25bp) |
| PUB-04 | published_benchmark | Texas is top-3 migration destination (U-Haul report) | PASS | Our rank=2, Benchmark rank=2 (tol=+/-1) |
| PUB-05 | published_benchmark | Fed Funds rate matches FRED exactly | PASS | Validated via live FRED API (FRED DFF series) |
| PUB-06 | published_benchmark | 10Y Treasury matches FRED exactly | PASS | Validated via live FRED API (FRED DGS10 series) |
| PUB-07 | published_benchmark | Sun Belt population growth within 0.2pp of Census | PASS | Our=1.8, Benchmark=1.7, Diff=0.10pp (tol=0.2pp) |
| GROQ-01 | groq_extraction | Extract Tesla Gigafactory from straightforward announcement | PASS | Extracted correctly: Tesla |
| GROQ-02 | groq_extraction | Extract TSMC semiconductor fab with large dollar amount | PASS | Extracted correctly: TSMC |
| GROQ-03 | groq_extraction | Extract from article with TWO facilities — should get both | PASS | Extracted correctly: Eli Lilly |
| GROQ-04 | groq_extraction | Extract data center from Microsoft with vague job count | PASS | Extracted correctly: Microsoft |
| GROQ-05 | groq_extraction | Extract EV plant with company name containing slash | PASS | Extracted correctly: Hyundai Motor Group |
| GROQ-06 | groq_extraction | Extract from ambiguous million/billion amount ($500M not $500B) | PASS | Extracted correctly: Nucor Corporation |
| GROQ-07 | groq_extraction | Avoid hallucinating details not in the article | PASS | Extracted correctly: Amazon Web Services |
| GROQ-08 | groq_extraction | Extract warehouse/distribution facility correctly typed | PASS | Extracted correctly: FedEx Ground |
| GROQ-09 | groq_extraction | Extract HQ relocation — should not hallucinate jobs or investment | PASS | Extracted correctly: Caterpillar Inc. |
| GROQ-10 | groq_extraction | Extract from article with R&D facility and specific CHIPS Act mention | PASS | Extracted correctly: Intel |

---

## Schema Validation

| Cache | Status | Errors |
|-------|--------|--------|
| migration | PASS | - |
| pricing | PASS | - |
| rates | PASS | - |
| energy_data | PASS | - |
| labor_market | PASS | - |
| inflation_data | PASS | - |
| credit_data | PASS | - |
| gdp_data | PASS | - |

---

## Freshness Compliance

| Cache | SLA (min) | Age (min) | Status |
|-------|-----------|-----------|--------|
| migration | 420 | 3195.1 | STALE |
| pricing | 120 | 15.2 | PASS |
| predictions | 1500 | 3195.2 | STALE |
| debugger | 60 | 15.3 | PASS |
| news | 300 | 3195.3 | STALE |
| rates | 120 | 15.2 | PASS |
| energy_data | 420 | 3195.3 | STALE |
| sustainability_data | 420 | 3195.3 | STALE |
| labor_market | 420 | 3195.3 | STALE |
| gdp_data | 420 | 3195.3 | STALE |
| inflation_data | 420 | 3195.3 | STALE |
| credit_data | 420 | 4275.3 | STALE |
| vacancy | 780 | N/A (Not found) | STALE |
| land_market | 780 | 4635.3 | STALE |
| cap_rate | 420 | 4275.3 | STALE |
| rent_growth | 420 | 4275.3 | STALE |
| opportunity_zone | 1500 | 7515.4 | STALE |
| distressed | 420 | 4275.3 | STALE |
| market_score | 420 | 4275.3 | STALE |

---

*Generated by `run-eval.py` on 2026-04-19 18:49*