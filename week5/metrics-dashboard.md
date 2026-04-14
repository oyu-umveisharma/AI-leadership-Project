# Metrics Dashboard — Cost & Latency Per Agent

**CRE Intelligence Platform — Week 5**
**Date:** April 14, 2026

---

## Agent Performance Metrics

| Agent | Schedule | Avg Latency | API Calls/Run | Estimated Cost/Day | Data Volume |
|-------|----------|-------------|---------------|--------------------| ------------|
| Agent 1: Migration | 6h | ~8-12s | 1 Census API call | $0.00 (free) | ~13 KB |
| Agent 2: REIT Pricing | 1h | ~15-25s | 27 yfinance calls | $0.00 (free) | ~12 KB |
| Agent 3: Predictions | 24h | ~30-60s | 10+ RSS + 2 Groq calls | ~$0.01 (Groq free tier) | ~5 KB |
| Agent 4: Debugger | 30min | ~3-5s | 1 yfinance + 1 Census + local checks | $0.00 | ~2.5 KB |
| Agent 5: News | 4h | ~10-15s | 10+ RSS feeds | $0.00 (free) | ~5 KB |
| Agent 6: Rates | 1h | ~8-15s | 10 FRED series | $0.00 (free) | ~149 KB |
| Agent 7: Energy | 6h | ~8-12s | 5 yfinance calls | $0.00 (free) | ~1 KB |
| Agent 8: Sustainability | 6h | ~8-12s | 7 yfinance calls | $0.00 (free) | ~1 KB |
| Agent 9: Labor Market | 6h | ~10-18s | 7 FRED + 8 yfinance + BLS API | $0.00 (free) | ~12 KB |
| Agent 10: GDP | 6h | ~8-12s | 8 FRED series | $0.00 (free) | ~5 KB |
| Agent 11: Inflation | 6h | ~8-12s | 9 FRED series | $0.00 (free) | ~10 KB |
| Agent 12: Credit | 6h | ~8-12s | 9 FRED series | $0.00 (free) | ~10 KB |
| Agent 13: Vacancy | 12h | ~5-8s | Local computation | $0.00 | ~3 KB |
| Agent 14: Land Market | 12h | ~5-8s | Local computation | $0.00 | ~9 KB |
| Agent 15: Cap Rate | 6h | ~5-8s | Derived from rates cache | $0.00 | ~3 KB |
| Agent 16: Rent Growth | 6h | ~5-8s | Local computation | $0.00 | ~4 KB |
| Agent 17: OZ & Incentives | 24h | ~5-10s | Local + optional Groq | ~$0.005 | ~7 KB |
| Agent 18: Distressed | 6h | ~5-10s | Local + optional Groq | ~$0.005 | ~5 KB |
| Agent 19: Market Score | 6h | ~3-5s | Derived from other caches | $0.00 | ~4 KB |

---

## Daily Cost Summary

| Resource | Daily Runs | Cost/Run | Daily Cost |
|----------|-----------|----------|------------|
| Census API | 4 | Free | $0.00 |
| FRED API | ~60 (across 6 agents) | Free | $0.00 |
| yfinance | ~120 calls | Free | $0.00 |
| BLS API | 4 | Free | $0.00 |
| Groq API (free tier) | 1-3 calls | ~$0.005 | ~$0.02 |
| RSS feeds | ~30 fetches | Free | $0.00 |
| **Total** | | | **~$0.02/day** |

**Monthly estimate:** ~$0.60/month (essentially free)

---

## Latency Breakdown

### Slowest Operations (by agent)

| Operation | Typical Latency | Bottleneck |
|-----------|----------------|------------|
| yfinance bulk REIT fetch (27 tickers) | 15-25s | Yahoo Finance rate limiting |
| Groq LLM inference (facility extraction) | 8-15s | Model inference time |
| FRED API (10 series with history) | 8-15s | Network round-trips |
| Census API (population estimates) | 5-10s | Single API call |
| RSS feed scraping (10+ sources) | 8-12s | Sequential HTTP fetches |
| yfinance commodity/ETF fetch (5-8 tickers) | 5-10s | Yahoo Finance |
| BLS API (supersector payrolls) | 3-5s | Single API call |
| Local computation (derived agents) | 1-3s | JSON read/write |

### Parallelism

All 19 agents run concurrently in separate `threading.Thread` instances. Total wall-clock time from cold start to all caches populated: **30-60 seconds**.

---

## Cache Efficiency

| Metric | Value |
|--------|-------|
| Total cache size (all 19 files) | ~260 KB |
| Cache format | JSON (human-readable) |
| Persistence | Survives Streamlit reruns |
| Storage location | `cache/` directory |
| Rotation | Overwritten on each agent run |
| Audit trail | `cache/audit_log.csv` (append-only) |

---

## Monitoring

### Real-Time (System Monitor Tab)
- Agent status: OK / Running / Error / Idle
- Run count per agent
- Last run timestamp
- Last error message (truncated)

### Audit Trail (`cache/audit_log.csv`)
- Every agent run logged with timestamp, latency, status
- Queryable for historical performance analysis
- Used by `run-eval.py` for metrics reporting

### Evaluation (`week5/evals/run-eval.py`)
- 20 benchmark cases scored against ground truth
- Schema validation on 8 core caches
- Freshness compliance against SLA thresholds
- Results written to `week5/evals/results-2026-04.md`
