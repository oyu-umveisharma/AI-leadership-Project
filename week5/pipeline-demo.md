# Pipeline Demo — End-to-End Run

**CRE Intelligence Platform — Week 5**
**Date:** April 14, 2026

---

## Overview

This document demonstrates a complete end-to-end pipeline run: from app launch through all 19 agent executions, cache population, dashboard rendering, and evaluation.

---

## Step 1: Launch Application

```bash
python3 -m streamlit run app/cre_app.py --server.port 8503
```

**What happens on launch:**
1. APScheduler `BackgroundScheduler` starts as a daemon thread
2. All 19 agents are registered with their interval triggers
3. Each agent fires immediately in its own background thread
4. UI renders with "Loading..." states while agents populate cache

**Expected timeline:**
| Time | Event |
|------|-------|
| T+0s | App launches, scheduler starts |
| T+5s | Debugger agent completes (fastest — local checks only) |
| T+10-15s | Energy, Sustainability agents complete (yfinance calls) |
| T+15-20s | Migration agent completes (Census API) |
| T+20-30s | Pricing agent completes (27 REIT yfinance calls) |
| T+20-30s | Rate, Labor, GDP, Inflation, Credit agents complete (FRED API) |
| T+30-60s | Predictions agent completes (RSS + Groq LLM) |
| T+30-60s | Vacancy, Land, Cap Rate, Rent Growth, OZ, Distressed, Market Score agents complete |

---

## Step 2: Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    APScheduler (daemon)                         │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │  ...  │
│  │Migration │  │ Pricing  │  │Prediction│  │ Debugger │       │
│  │  6h      │  │  1h      │  │  24h     │  │  30min   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              cache/ (JSON files)                     │       │
│  │  migration.json  pricing.json  predictions.json ...  │       │
│  └─────────────────────────────────────────────────────┘       │
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           audit_log.csv (audit trail)                │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Streamlit UI        │
              │   read_cache() calls  │
              │   on each tab render  │
              └───────────────────────┘
```

---

## Step 3: Data Flow Per Agent

### Example: Migration Agent (Agent 1)

```
Census API  ──►  fetch_migration_scores()  ──►  DataFrame (51 rows)
                        │
                        ▼
              get_top_metros()  ──►  DataFrame (17 metros)
                        │
                        ▼
              write_cache("migration", {...})
                        │
                        ▼
              cache/migration.json (~13KB)
              {
                "updated_at": "2026-04-14T13:34:08",
                "data": {
                  "migration": [51 state records],
                  "metros": [17 metro records],
                  "top3_cities": ["Texas", "Florida", "Arizona"]
                }
              }
                        │
                        ▼
              UI Tab 1: read_cache("migration")
              → Choropleth map, ranking table, bubble chart
```

### Example: Rate Agent (Agent 6)

```
FRED API (10 series)  ──►  run_rate_agent()
     │
     ├──► DGS10, DGS2, DGS30, FEDFUNDS, ...
     │
     ▼
  Calculate:
     ├── yield_curve shape
     ├── environment signal (BULLISH/CAUTIOUS/BEARISH)
     ├── cap_rate_adjustments (7 property types)
     └── reit_debt_risk scores
     │
     ▼
  write_cache("rates", {...})  ──►  cache/rates.json (~149KB)
     │
     ▼
  UI Tab: Current rates table, yield curve chart,
          cap rate adjustment bars, debt risk heatmap
```

---

## Step 4: Cache Validation Pipeline

After agents populate cache, validation checks run:

```
cache/*.json  ──►  Schema validation  ──►  Pass/Fail per cache
                         │
                         ▼
                   Freshness check  ──►  Age vs SLA threshold
                         │
                         ▼
                   Outlier detection  ──►  Flag anomalies
                         │
                         ▼
                   Null/missing check  ──►  Data completeness
                         │
                         ▼
                   audit_log.csv  ──►  Full traceability
```

---

## Step 5: Run Evaluation

```bash
python week5/evals/run-eval.py --verbose
```

**Expected output:**
```
  CRE Platform Evaluation Runner
  ========================================
  Cases: 20  |  Pass threshold: 80%

  [PASS] MIG-01: Texas is a top-3 migration destination state
  [PASS] MIG-02: Florida is a top-3 migration destination state
  [PASS] MIG-03: California shows net outbound migration (low score)
  ...
  [PASS] CRD-01: Credit conditions signal is one of LOOSE/NEUTRAL/TIGHT

  Schema Validation
  ----------------------------------------
  [PASS] migration
  [PASS] pricing
  [PASS] rates
  [PASS] energy_data
  ...

  ========================================
  Benchmark: 20/20 passed (100%)
  Schema:    8/8 valid
  Freshness: 19/19 within SLA
  Overall:   PASS (threshold: 80%)
```

---

## Step 6: Verify in Dashboard

1. Open `http://localhost:8503`
2. Check System Monitor tab — all agents show "OK" status
3. Each tab shows "Last updated: Xm ago" caption
4. Expand "How This Is Calculated" on any tab — methodology is documented
5. Check audit trail in `cache/audit_log.csv` for full run history
