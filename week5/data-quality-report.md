# Data Quality Report

**CRE Intelligence Platform — Week 5**
**Date:** April 14, 2026

---

## 1. Freshness SLAs

Each agent has a defined maximum staleness threshold. If cache age exceeds the SLA, the System Monitor flags it and the UI shows a stale-data warning.

| Agent | Cache Key | Update Frequency | Freshness SLA | Stale Threshold |
|-------|-----------|-----------------|---------------|-----------------|
| Migration Analyst | `migration` | Every 6h | 7 hours | > 7h = stale |
| REIT Pricing | `pricing` | Every 1h | 2 hours | > 2h = stale |
| Facility Intelligence | `predictions` | Every 24h | 25 hours | > 25h = stale |
| System Monitor | `debugger` | Every 30min | 1 hour | > 1h = stale |
| News & Announcements | `news` | Every 4h | 5 hours | > 5h = stale |
| Rate Environment | `rates` | Every 1h | 2 hours | > 2h = stale |
| Energy & Construction | `energy_data` | Every 6h | 7 hours | > 7h = stale |
| Sustainability & ESG | `sustainability_data` | Every 6h | 7 hours | > 7h = stale |
| Labor Market | `labor_market` | Every 6h | 7 hours | > 7h = stale |
| GDP & Economic Growth | `gdp_data` | Every 6h | 7 hours | > 7h = stale |
| Inflation | `inflation_data` | Every 6h | 7 hours | > 7h = stale |
| Credit & Capital Markets | `credit_data` | Every 6h | 7 hours | > 7h = stale |
| Vacancy Monitor | `vacancy` | Every 12h | 13 hours | > 13h = stale |
| Land Market | `land_market` | Every 12h | 13 hours | > 13h = stale |
| Cap Rate Monitor | `cap_rate` | Every 6h | 7 hours | > 7h = stale |
| Rent Growth Tracker | `rent_growth` | Every 6h | 7 hours | > 7h = stale |
| Opportunity Zones | `opportunity_zone` | Every 24h | 25 hours | > 25h = stale |
| CMBS & Distressed | `distressed` | Every 6h | 7 hours | > 7h = stale |
| Market Opportunity Score | `market_score` | Every 6h | 7 hours | > 7h = stale |

---

## 2. Schema Validation Rules

Each cache file is validated against expected schema. Validation runs on every `write_cache()` call via `src/audit_logger.py`.

### Migration Cache (`migration.json`)
```
Required fields per record:
  - state_abbr: string, 2 chars, uppercase
  - state_name: string, non-empty
  - population: integer, > 0
  - pop_growth_pct: float, range [-5.0, 10.0]
  - biz_score: integer, range [0, 100]
  - composite_score: float, range [0, 100]
  - rank: integer, range [1, 51]
Record count: >= 50
```

### Pricing Cache (`pricing.json`)
```
Required fields per REIT record:
  - ticker: string, 1-5 uppercase chars
  - property_type: string, must be one of 7 valid types
  - price: float, > 0 and < 1000
  - market_cap: float, > 0
  - dividend_yield: float, >= 0
Record count: >= 20
```

### Rates Cache (`rates.json`)
```
Required top-level keys: rates, yield_curve, environment, cap_rate_adjustments
rates: dict with >= 6 series, each having 'current' (float)
yield_curve: dict with keys 3M, 2Y, 5Y, 10Y, 30Y (all floats > 0)
environment.signal: one of BULLISH, CAUTIOUS, BEARISH
cap_rate_adjustments: list of 7 records
```

### Energy Cache (`energy_data.json`)
```
Required keys: commodities, construction_cost_signal, avg_momentum_pct
commodities: list of 5 records, each with ticker, label, latest_price (>0), sma_60 (>0), pct_above_sma
construction_cost_signal: one of HIGH, MODERATE, LOW
```

### Labor Market Cache (`labor_market.json`)
```
Required keys: fred_labor, bls_sectors, sector_etfs, demand_signal
demand_signal.score: integer, range [0, 100]
demand_signal.label: one of STRONG, MODERATE, SOFT
bls_sectors: list of >= 8 records
```

### GDP Cache (`gdp_data.json`)
```
Required keys: series, cycle
cycle.label: one of EXPANSION, SLOWDOWN, CONTRACTION
cycle.score: integer, range [0, 100]
series: dict with >= 4 FRED series
```

### Inflation Cache (`inflation_data.json`)
```
Required keys: series, signal
signal.label: one of HOT, MODERATE, COOLING
signal.score: integer, range [0, 100]
series: dict with >= 6 FRED series
```

### Credit Cache (`credit_data.json`)
```
Required keys: series, signal
signal.label: one of LOOSE, NEUTRAL, TIGHT
signal.score: integer, range [0, 100]
series: dict with >= 5 FRED series
```

---

## 3. Outlier Detection Rules

Applied on every cache write to flag anomalous data:

| Check | Threshold | Action |
|-------|-----------|--------|
| REIT price change | > 20% in 1 hour | Log warning, flag in audit trail |
| FRED rate value | > 15% or < 0% | Log warning, retain previous value |
| Migration composite_score | Outside [0, 100] | Reject record, log error |
| Commodity pct_above_sma | > 50% or < -50% | Log warning (extreme deviation) |
| Demand signal score | Outside [0, 100] | Clamp to bounds, log warning |
| VIX value | > 80 | Log warning (extreme volatility) |
| Unemployment rate | > 15% | Log warning (potential data error) |

---

## 4. Null / Missing Data Limits

| Cache | Max Null Rate | Fail Condition |
|-------|--------------|----------------|
| Migration records | 0% for core fields | Any null in state_abbr, population, composite_score |
| REIT prices | 10% | > 10% of tickers return null price |
| FRED series | 20% per series | > 20% of data points are NaN |
| Commodity prices | 0% | Any commodity missing latest_price |
| BLS sectors | 20% | > 20% of sectors missing employment data |

---

## 5. Published Benchmark Validation

To verify our indicators are in the right ballpark, we tie key metrics to published industry sources:

| Metric | Our Value | Published Benchmark | Source | Variance |
|--------|-----------|---------------------|--------|----------|
| Industrial Cap Rate | 5.6% | 5.4% | CBRE North America Cap Rate Survey Q4 2025 | +20bp |
| Office Cap Rate | 8.5% | 8.0-9.0% | CBRE / Green Street Q4 2025 | Within range |
| Multifamily Cap Rate | 5.2% | 5.0-5.5% | NCREIF Property Index Q4 2025 | Within range |
| Data Center Cap Rate | 4.8% | 4.5-5.0% | JLL Data Center Outlook 2025 | Within range |
| Self-Storage Cap Rate | 5.4% | 5.0-5.5% | Yardi Matrix Self-Storage Report 2025 | Within range |

**Interpretation:** Our benchmark cap rates are within 20-30bp of published industry surveys. This is acceptable given that:
1. Published surveys lag by 1-3 months (our data refreshes hourly via yfinance)
2. Cap rates vary by market, vintage, and quality tier — published surveys report ranges
3. Our rates represent averages across the REIT universe for each property type, not a single market

**Note:** Published benchmark sources are referenced for validation context. Our platform uses live REIT market data (yfinance) for real-time cap rate estimation, not static survey values.

---

## 6. Current Data Quality Status

*Run `python week5/evals/run-eval.py` to generate a live data quality snapshot.*

The evaluation script checks all validation rules against live cache data and reports:
- Schema compliance per cache file
- Freshness compliance vs. SLA
- Outlier flags
- Null/missing data counts
- LLM extraction faithfulness (10 labeled test cases against Groq)
- Overall quality score (target: > 80% combined)
