# CRE Intelligence Platform

> Real-time commercial real estate intelligence powered by twenty-two autonomous AI agents — tracking migration flows, REIT pricing, interest rates, labor markets, GDP, inflation, credit conditions, facility announcements, energy costs, ESG momentum, vacancy rates, cap rates, rent growth, climate risk, opportunity zones, distressed assets, real property listings, and forward macro projections across US markets.

**Purdue Daniels School of Business · MGMT 690: AI Leadership**

---

## What It Does

Most CRE research tools require manual data pulls, static spreadsheets, and hours of synthesis. This platform replaces that workflow with a live dashboard backed by twenty-two background agents that update continuously — surfacing which markets to watch, which property types are most profitable, what macro conditions mean for CRE valuations, and delivering a personalized AI investment brief on demand.

The app opens with an AI-powered chatbox that asks what you are looking for. Type a simple query like `"industrial in Los Angeles"` and a **Quick Investment Brief** appears inline with your top 3 scored markets, cap rates, rent growth, and a Groq-generated insight — plus two CTAs to either explore the personalized dashboard or jump straight to a full P&L analysis. Type a full investment query like `"build a 50,000 sqft warehouse in Texas, $8M budget, 5-year hold"` and the platform routes you directly to the **Investment Advisor**, auto-generates a scored market recommendation, and returns a full financial brief — no navigation required.

The agents run on a scheduler. Open the dashboard and the data is already there.

---

## Industry Benchmark Comparison

| Feature | CRE Intelligence Platform | CoStar | Morningstar | CBRE EA |
|---------|---------------------------|--------|-------------|---------|
| Real-time REIT pricing | Yes (hourly) | Yes | Yes | Limited |
| Population migration data | Yes (Census API) | No | No | No |
| AI-powered facility tracking | Yes (Groq LLM, 70% extraction accuracy) | No | No | No |
| Interest rate integration | Yes (40+ FRED series, exact match) | Limited | Yes | Limited |
| Construction cost signals | Yes (commodities) | No | No | Yes |
| Labor market analysis | Yes (BLS + FRED) | No | Limited | Yes |
| ESG/Sustainability metrics | Yes | Limited | Yes | Limited |
| Personalized AI chatbox | Yes | No | No | No |
| AI Investment Advisor | Yes | No | No | No |
| Climate risk scoring | Yes (FEMA + NIFC + NOAA) | Limited | No | No |
| Vacancy & cap rate tracking | Yes (20 markets) | Yes | No | Yes |
| Industrial Cap Rate | 5.6% (±10bp vs CBRE Q1 2026) | 5.5% | N/A | 5.5% |
| Opportunity zone analysis | Yes | Limited | No | Limited |
| Metro neighborhood maps | Yes (17 metros) | Yes | No | Limited |
| Real property listings | Yes (RentCast API, 6 states) | Yes | No | Limited |
| Macro forecasting | Yes (FRED FOMC projections, Q2–Q4 2026) | No | Limited | Limited |
| Pricing | Free / Open Source | $$$$ | $$$ | $$$$ |

*Comparison based on publicly available feature documentation. Capabilities may vary by subscription tier.*

---

## Validation & Data Quality

This platform is designed to be **validated, auditable, and trustworthy** — meeting the same standards a bank's model risk group (SR 11-7) or investment committee would expect.

### Published Benchmark Validation

| Metric | Our Model | Published Benchmark | Source | Variance |
|--------|-----------|---------------------|--------|----------|
| Industrial Cap Rate | 5.6% | 5.5% | CBRE Americas Cap Rate Survey Q1 2026 | +10bp |
| Office Cap Rate | 7.2% | 7.0–7.5% | CBRE Americas Cap Rate Survey Q1 2026 | Within range |
| Multifamily Cap Rate | 5.3% | 5.2% | NCREIF Q4 2025 | +10bp |
| Texas Migration Rank | #2 | #2 | U-Haul Migration Report 2025 | Exact match |
| Fed Funds Rate | 4.50% | 4.50% | FRED DFF Series | Exact match |
| 10Y Treasury | 4.25% | 4.25% | FRED DGS10 Series | Exact match |

### Evaluation Results

| Test Category | Pass | Fail | Rate |
|---------------|------|------|------|
| Data Pipeline Integrity | 20 | 0 | 100% |
| Published Benchmark Match | 7 | 0 | 100% |
| LLM Facility Extraction | 7 | 3 | 70% |
| **Total** | **34** | **3** | **92%** |

### Known Limitations

- Groq facility extraction accuracy is ~70% — fails on articles with multiple facilities or ambiguous dollar amounts
- WRDS DealScan access pending institutional approval; loan-level validation planned for future iteration
- Climate risk scores use 2019–2024 historical data; forward projections not included
- Cap rates are REIT-implied averages, not transaction-level — published surveys may differ by ±25bp

See [`week5/`](week5/) for full evaluation methodology, benchmark cases, and data quality reports.

### Harness Guardrails (April 2026)

Two structural guardrails added in response to panelist feedback — both are harness-level (not model-level) checks that run on every agent cycle and log failures to `cache/audit_log.csv`.

| Guardrail | Where | What It Catches | Log Format |
|-----------|-------|----------------|------------|
| **Source-quote verification loop** | `src/cre_news.py:verify_source_quote` + Pass 1 of `_extract_confirmed_announcements` | Location hallucinations (GROQ-09 failure mode) — verifies the extracted `location` actually appears in the verbatim `source_quote` sentence the model cites. On mismatch, triggers ONE retry with the discrepancy flagged in the prompt; whichever pass has fewer failures wins. | `predictions, warning, source_quote_mismatch: location 'X' not found in quote 'Y…'` |
| **Pandera pre-flight validation** | `src/data_validator.py` — wired into `run_pricing_agent`, `run_migration_agent`, `run_cap_rate_agent` | Out-of-range numeric values BEFORE they hit cache or downstream AI steps. Schemas: REIT `Price` ∈ (0, 10 000), `cap_rate` ∈ [2.0, 15.0], `composite_score` ∈ [0, 100], `pop_growth_pct` ∈ [-10, 20]. Non-destructive — bad rows are flagged, not dropped. | `pricing, warning, pandera_validation: greater_than(0), failure_case=-5.0` |

Both guardrails are active on the live pipeline. Every validation failure lands in the audit log alongside the regular run rows, so the System Monitor tab surfaces them immediately.

---

## Agent Architecture

Twenty-two specialized agents operate independently on fixed schedules, writing to a shared JSON cache that survives Streamlit reruns.

| # | Agent | Responsibility | Schedule |
|---|-------|---------------|----------|
| 1 | Migration Analyst | Population flows, state migration scores, metro rankings | Every 6 hours |
| 2 | REIT Analyst | Live REIT prices, cap rates, NOI margins, profit matrix | Every 1 hour |
| 3 | Facility Intelligence | Confirmed plant & facility announcements extracted from live news | Every 24 hours |
| 4 | System Monitor | API health, cache freshness, agent status | Every 30 minutes |
| 5 | Market Intelligence | RSS news feeds — manufacturing, government, industry press | Every 4 hours |
| 6 | Rate Environment | Fed Funds, Treasuries, yield curve, cap rate adjustments, REIT debt risk | Every 1 hour |
| 7 | Energy Analyst | Oil, gas, copper, steel prices vs. 60-day moving average | Every 6 hours |
| 8 | Sustainability Analyst | Clean energy ETFs, green REIT performance vs. S&P 500 | Every 6 hours |
| 9 | Labor Market Analyst | Job growth by metro, sector payrolls, tenant demand signal, hiring momentum | Every 6 hours |
| 10 | GDP & Economic Growth | Real GDP, industrial production, retail sales, consumer sentiment, CFNAI | Every 6 hours |
| 11 | Inflation Analyst | CPI (headline/core/shelter/rent), PPI, breakeven inflation expectations | Every 6 hours |
| 12 | Credit & Capital Markets | Corporate spreads, VIX, bank lending standards, CRE loan conditions | Every 6 hours |
| 13 | Vacancy Monitor | Commercial vacancy rates by market × property type, trend arrows | Every 12 hours |
| 14 | Land & Development | Land parcel prices, zoning types, entitlement status, availability scores | Every 12 hours |
| 15 | Cap Rate Monitor | Market-level cap rates by property type across 19 canonical markets | Every 6 hours |
| 16 | Rent Growth | Rent growth trends (PSF and %) by market and property type | Every 6 hours |
| 17 | Opportunity Zone | OZ designations, investment scores, and key zone listings by market | Every 24 hours |
| 18 | CMBS & Distressed | Distressed asset tracking, CMBS delinquency signals, market stress scores | Every 6 hours |
| 19 | Market Score | Composite opportunity score per market aggregating all agent signals | Every 6 hours |
| 20 | Climate Risk | Physical climate hazard scoring (flood, wildfire, heat, wind, sea level rise) across 51 states and 41 metros | Every 24 hours |
| 21 | RentCast Property Database | Real for-sale and for-rent property listings from the RentCast API, covering TX, FL, AZ, NC, TN, GA (50 free calls/month; falls back to mock data when quota is exhausted) | Every 24 hours |
| 22 | Economic Forecast | Forward-looking FRED projections — Atlanta Fed GDPNow (`GDPNOW`), FOMC Summary of Economic Projections real GDP (`GDPC1MD`) and Fed funds (`FEDTARMD`), and 10-year breakeven inflation (`T10YIE`) — producing Q2 / Q3 / Q4 2026 forecasts that feed the Investment Advisor's forecasting section and investment rationale | Every 6 hours |

All agents start automatically when the app launches. No manual triggers needed.

---

## AI Investment Advisor

The Investment Advisor is a top-level tab that synthesizes all twenty-two agent data streams into a single personalized investment brief — including Agent 22's forward macro projections that feed directly into the narrative prompt. It can be reached by navigating to the tab directly, or triggered automatically by typing a full investment query into the welcome screen or persistent chat bar.

### How It Works

```
Prompt  →  Parse  →  Resolve Markets  →  Gather Data  →  AI Weights
                                                              ↓
Report  ←  Narrative  ←  Financials  ←  Score Markets  ←  6 Factors
```

1. **Parse** — Extracts property type, location, budget, sqft, hold period, and risk tolerance from free-text using Groq (regex fallback). Missing fields surface as follow-up inputs.
2. **Resolve Markets** — Maps the location string to candidate markets from 19 canonical metros using a region/keyword lookup (`southern texas` → `Houston, TX`; `sunbelt` → 10 markets).
3. **Gather Data** — Pulls all relevant cache data for each candidate market: market score, cap rates, rent growth, vacancy, climate risk, opportunity zones, credit conditions, GDP cycle, energy signal, migration score.
4. **AI Weights** — Groq determines factor weights (summing to 1.0) based on property type and risk tolerance, with rationale per factor. Falls back to platform-standard defaults by asset class.
5. **Score Markets** — Each market receives a composite 0–100 opportunity score across six weighted factors: market fundamentals, rent growth, cap rate attractiveness, migration, climate risk (inverted), and macro environment.
6. **Financials** — Estimates land cost (18% of budget), construction (PSF × energy signal multiplier), soft costs (15%), annual NOI, cumulative NOI, exit value, total profit, ROI, and IRR over the hold period.
7. **Narrative** — Groq generates a 3-paragraph investment rationale covering market thesis, key risks, and strategic outlook. Falls back to a data-driven template when Groq is unavailable.

### Report Sections

| Section | Contents |
|---------|----------|
| Summary cards | Opportunity score, est. total cost, estimated ROI, buildout timeline, est. exit value |
| Factor score chart | Horizontal bar chart — raw score + weighted contribution per factor |
| Market signals | Cap rate, rent growth, climate risk, GDP cycle, credit conditions, migration score |
| Financial estimates | Land / construction / soft costs / total cost, annual NOI, cumulative NOI, IRR, total profit |
| Investment rationale | 3-paragraph Groq narrative (or template fallback) |
| Climate risk alert | Conditional — shown only when primary market is High or Severe risk |
| Runner-up comparison | Table + bar chart comparing primary vs. top 2 runner-up markets |
| All candidates | Collapsed table of every market scored |
| Forecasting | Macro projections table (Current / Q2 / Q3 / Q4 2026) for four FRED series, Plotly line chart with solid historical data and dotted forecast extensions, color-coded directional deltas |
| Methodology | Factor weights, raw scores, weighted contributions, and rationales |

### Chat Bar Routing

The platform auto-detects investment advisor queries in the welcome screen and persistent chat bar. A query is routed to the Investment Advisor (instead of filtering the dashboard) when it contains:

- A budget amount + an action word (`build`, `invest`, `develop`, `acquire`, etc.), **or**
- A budget amount + square footage, **or**
- Three or more of: budget / sqft / timeline / action word

Examples that route to the Advisor:
- `"build a 50k sqft warehouse in Texas with $8M budget, 5-year hold"`
- `"invest $15M in multifamily in the Southeast, 7 years"`
- `"where should I develop a data center? $30M budget"`

Examples that filter the dashboard (no routing):
- `"industrial in Austin"`
- `"multifamily Chicago"`

---

## AI Chatbox and Personalization

When the app launches, a welcome screen asks: "What are you looking to invest in today?" Users can type a natural-language query or pick a quick-select category.

**What the parser understands:**

- Property types: Industrial, Office, Retail, Multifamily, Data Center, Healthcare, Hospitality, Self-Storage, Mixed-Use
- Synonyms: "warehouse" and "logistics" map to Industrial, "car wash" and "restaurant" map to Retail, "apartment" maps to Multifamily
- Locations: 200+ US cities with state resolution (typing "Chicago" resolves to IL, "LA" resolves to CA)
- Full state names: "north carolina" resolves to NC
- Regions: "west coast", "midwest", "sun belt", "northeast" filter to the relevant state group
- Filler word stripping: "potential warehouse investment opportunities in Austin" parses to Industrial in Austin, TX

**How personalization works across tabs:**

| Tab | What Changes |
|-----|-------------|
| Migration Intelligence | Map zooms to your state or metro. County-level and neighborhood-level drill-down maps available via radio selector. Matching state/metro stats highlighted at top. |
| Pricing and Profit | Your property type highlighted in the profit margin chart. Rate-adjusted cap rate table highlights the matching row. |
| Company Predictions | Facility announcements sorted by relevance to your property type and location. |
| Cheapest Buildings | Listings filtered to your city and property type. On-demand generation for states not in the default top-3 migration set. |
| Rate Environment | Cap rate adjustment table highlights your property type row. |
| Energy and Construction | Contextual insight for how commodity costs affect your target investment. |
| Sustainability and ESG | Maps your property type to the most relevant green REIT (Industrial to Prologis, Data Center to Equinix). |
| Climate Risk | Highlights your market's risk score and relevant hazard factors for your property type. |

A persistent chat bar at the top of the dashboard allows changing the query at any time without returning to the welcome screen. AI-generated insights powered by Groq appear at the top of each tab when a focus is set.

### Unified AI Experience — Quick Investment Brief

Any non-advisor query — whether typed on the welcome screen or the persistent chat bar — now produces a **Quick Investment Brief** panel before navigation. This closes the panelist feedback that "the AI feels disconnected" by making every query show AI-driven intelligence immediately.

| Brief Element | What It Shows |
|--------------|--------------|
| Header | Property type and location extracted from the query (e.g., `OFFICE IN CHICAGO — QUICK BRIEF`) |
| Top 3 markets | Best-scored candidate markets for the query — each row shows composite score (color-coded), letter grade, cap rate (for the inferred property type), and rent growth (green for positive, red for negative). When the query names a specific market, it surfaces first regardless of rank. |
| AI Insight | 2-sentence Groq-generated insight grounded in the top-3 row data |
| CTAs | `Explore Dashboard →` (personalizes all tabs), `Get Full P&L Analysis →` (routes to Investment Advisor with prompt pre-filled and auto-generate triggered), `Dismiss` |

Queries that already contain budget + square footage + timeline signals auto-route straight to the full Investment Advisor (bypassing the brief), preserving the fast-path for serious investment queries. The brief is also surfaced as a banner on the dashboard whenever a new query comes in through the top chat bar — so the AI layer is always one input away.

---

## Migration Map Drill-Down

The Migration Intelligence tab supports three levels of geographic resolution:

| Level | View | Data |
|-------|------|------|
| National | US choropleth by state | 51 states/territories colored by composite migration score |
| State (Counties) | County choropleth with FIPS codes | 8–18 counties per state with population, growth rate, key economic drivers |
| Metro (Neighborhoods) | Scattermapbox with dark tiles | 10–18 neighborhoods per metro with lat/lon, migration score, rent growth, zone type |

The map auto-selects the appropriate level based on the user's query. Typing "Texas" zooms to the state view with counties. Typing "Chicago" zooms to the metro view with neighborhood dots. 17 major metros are supported with accurate coordinates.

---

## Climate Risk Scoring

The Climate Risk agent (Agent 20) scores physical climate exposure across 51 states and 41 US metros using free public data sources — no commercial API required.

| Factor | Weight | Data Source |
|--------|--------|-------------|
| Flood | 25% | OpenFEMA Disaster Declarations API |
| Wildfire | 20% | NIFC WFIGS ArcGIS REST API (GIS acres burned since 2019) |
| Extreme Heat | 20% | NOAA 1991–2020 Climate Normals (static) |
| Wind / Hurricane | 20% | OpenFEMA (straight-line winds + hurricane declarations) |
| Sea Level Rise | 15% | NOAA 2022 Sea Level Rise Technical Report (static) |

Scores are normalized 0–100 (Low / Moderate / High / Severe). The composite score feeds directly into the Investment Advisor's climate risk factor.

---

## Dashboard

Four top-level tabs — **Real Estate**, **Energy**, **Macro Environment**, and **Investment Advisor**.

### Real Estate

| Tab | What You See |
|-----|-------------|
| Migration Intelligence | US choropleth map, top 10 states by migration score, business vs. population bubble chart, metro table |
| Pricing & Profit | Live REIT prices, profit margin heatmap by market × property type, top 10 opportunities, property type comparison |
| Company Predictions | Confirmed corporate facility announcements (plants, fabs, warehouses, data centers, HQ moves) with investment, jobs, and CRE impact |
| Cheapest Buildings | Lowest-price commercial listings in the top 3 migration destination states |
| Industry Announcements | AI-structured brief from 10+ news and government RSS sources |
| System Monitor | Agent status, cache health, API connectivity |
| Vacancy Monitor | Commercial vacancy rates by market and property type, trend indicators, national benchmarks |
| Land & Development | Land parcel listings by zoning type, entitlement status, price per acre, and availability score |
| Cap Rate Monitor | Market cap rates by property type, national benchmarks, spread vs. 10Y Treasury |
| Rent Growth | Rent growth (% YoY and PSF) by market and property type, heat map, trend chart |
| Opportunity Zones | OZ-designated markets, investment scores, key zone listings, tax incentive summary |
| Market Score | Composite opportunity score per market aggregating all agent signals, ranked leaderboard |
| Climate Risk | State and metro climate hazard scores, US choropleth, factor heatmap, FEMA trend chart |

### Energy

| Tab | What You See |
|-----|-------------|
| Energy & Construction Costs | Commodity prices vs. 60-day SMA, construction cost signal (HIGH / MODERATE / LOW) |
| Sustainability | Clean energy ETF performance (ICLN, TAN, QCLN), green REIT performance (PLD, EQIX, ARE) vs. SPY |

### Macro Environment

| Tab | What You See |
|-----|-------------|
| Rate Environment | Current rates table, yield curve shape, 12-month trend, cap rate adjustments by property type, REIT refinancing risk scores |
| Labor Market | Tenant demand signal, nonfarm payrolls, JOLTS job openings, BLS sector payrolls by CRE property type, sector ETF momentum, state unemployment for top CRE destination markets |
| GDP & Economic Growth | Economic cycle phase (Expansion / Slowdown / Contraction), real GDP growth, industrial production, consumer sentiment, Chicago Fed Activity Index, CRE outlook by cycle phase |
| Inflation | Inflation regime (Hot / Moderate / Cooling), CPI headline/core/shelter/rent YoY trends, PPI construction cost pressure, market breakeven inflation expectations (5Y/10Y) |
| Credit & Capital Markets | Credit conditions (Loose / Neutral / Tight), IG/HY/BBB corporate spreads, VIX volatility, Fed bank lending standards (C&I and CRE), Moody's BAA–AAA spread |
| CMBS & Distressed | Distressed asset signals, CMBS delinquency trends, market stress scores |

### Investment Advisor

See [AI Investment Advisor](#ai-investment-advisor) section above.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/oyu-umveisharma/AI-leadership-Project.git
cd AI-leadership-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys (see API Keys section below)
export GROQ_API_KEY=your_key_here
export FRED_API_KEY=your_key_here

# 4. Run (use python3 -m streamlit to ensure the correct Python environment)
python3 -m streamlit run app/cre_app.py
# or on a specific port:
python3 -m streamlit run app/cre_app.py --server.port 8503
```

The app opens at `http://localhost:8501` by default. All twenty-two agents start in the background immediately. Data populates within 30–60 seconds.

---

## API Keys

| Key | Required | Purpose | Get It |
|-----|----------|---------|--------|
| `GROQ_API_KEY` | Recommended | Company facility announcements, Investment Advisor narrative and weight reasoning (Llama 3.3-70B) | [console.groq.com](https://console.groq.com) — free tier available |
| `FRED_API_KEY` | Recommended | Interest rates, yield curve, labor market, GDP, inflation, credit markets, and Agent 22 macro forecasts | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — free |
| `RENTCAST_API_KEY` | Optional | Agent 21 real property listings for TX, FL, AZ, NC, TN, GA. Free tier = 50 calls/month. Falls back to mock listings when absent or quota is exhausted. | [rentcast.io](https://www.rentcast.io) — free tier available |

Store keys in a `.env` file at the project root (no spaces around `=`):

```
GROQ_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

The app runs without both keys — migration, REIT pricing, energy, sustainability, climate risk, vacancy, cap rate, rent growth, opportunity zone, and market score tabs work on public data alone. FRED-powered tabs (Rate Environment, Labor Market, GDP, Inflation, Credit) degrade gracefully with a status message. The Investment Advisor uses template fallbacks for narrative and default weights when Groq is unavailable.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit |
| Agent Scheduling | APScheduler — BackgroundScheduler with IntervalTrigger |
| Data Cache | File-based JSON (`/cache/`) — persists across Streamlit reruns |
| Market Data | yfinance — REIT prices, dividends, commodities, sector ETFs |
| Population Data | US Census Bureau Population Estimates API (2023) |
| Macro Data | FRED API — 40+ series across rates, labor, GDP, inflation, and credit markets |
| Labor Data | BLS Public API — supersector payroll data (10 industry groups) |
| Climate Data | OpenFEMA Disaster Declarations API, NIFC WFIGS ArcGIS REST API, NOAA static normals |
| AI / LLM | Groq API — llama-3.3-70b-versatile |
| Structured Outputs | Groq JSON mode with schema validation |
| News Sources | Reuters, Manufacturing.net, IndustryWeek, PR Newswire, Business Wire, Dept. of Energy, Dept. of Commerce, EDA, Expansion Solutions, Site Selection Magazine |
| Property Listings | RentCast API — for-sale and for-rent listings across TX, FL, AZ, NC, TN, GA (Agent 21) |
| Forecasting | FRED FOMC / market series — `GDPNOW`, `GDPC1MD`, `FEDTARMD`, `T10YIE` (Agent 22) |
| Data Validation | Pandera — pre-flight schema checks on REIT prices, cap rates, migration scores, population growth |
| Charts | Plotly — choropleth maps, heatmaps, bar, scatter, line |
| Language | Python 3.10+ |

---

## Project Structure

```
AI-leadership-Project/
├── app/
│   └── cre_app.py                    # Streamlit dashboard — all tabs and visualization
├── src/
│   ├── cre_agents.py                 # Agent runner, scheduler, cache helpers (22 agents)
│   ├── recommendation_engine.py      # Investment Advisor engine (parse → score → financials → narrative; macro forecast feeds into narrative prompt)
│   ├── data_validator.py             # Pandera pre-flight schemas — REIT prices, cap rates, migration scores, population growth (logs failures to audit_log.csv)
│   ├── cre_population.py             # Census API — migration scores, metro data
│   ├── cre_pricing.py                # REIT universe, cap rates, profit matrix
│   ├── cre_news.py                   # RSS feed scraper, facility keyword filter, source_quote verification loop
│   ├── cre_listings.py               # Commercial property listings by state (28 states, 200+ cities)
│   ├── rate_agent.py                 # FRED API — interest rates, yield curve
│   ├── energy_analyst.py             # Commodity prices, construction cost signal
│   ├── sustainability_analyst.py     # Clean energy ETFs, green REIT tracking
│   ├── labor_market_agent.py         # BLS + FRED + yfinance — labor market & tenant demand
│   ├── gdp_agent.py                  # FRED — GDP, industrial production, consumer sentiment
│   ├── inflation_agent.py            # FRED — CPI, PPI, breakeven inflation expectations
│   ├── credit_markets_agent.py       # FRED — corporate spreads, VIX, lending standards
│   ├── vacancy_agent.py              # Vacancy rates by market and property type
│   ├── land_market_agent.py          # Land parcel data, zoning, entitlement status
│   ├── cap_rate_agent.py             # Market-level cap rates by property type
│   ├── rent_growth_agent.py          # Rent growth trends by market and property type
│   ├── opportunity_zone_agent.py     # OZ designations and investment scores
│   ├── distressed_asset_agent.py     # CMBS delinquency and distressed asset signals
│   ├── market_score_agent.py         # Composite market opportunity scores
│   ├── climate_risk_agent.py         # Physical climate hazard scoring (FEMA, NIFC, NOAA)
│   ├── property_tax_agent.py         # Property tax rates by market
│   ├── rentcast_agent.py             # Agent 21 — RentCast API property listings (TX, FL, AZ, NC, TN, GA) with mock-data fallback
│   ├── forecast_agent.py             # Agent 22 — FRED macro projections (GDPNOW, GDPC1MD, T10YIE, FEDTARMD) for Q2–Q4 2026
│   ├── county_migration.py           # County-level migration data (FIPS codes, 12 states seeded)
│   └── zip_migration.py              # Neighborhood-level data (17 metros, real lat/lon)
├── week5/                            # Evaluation artifacts and data quality reports
│   ├── response-to-reviews.md        # Panelist feedback responses
│   ├── pipeline-demo.md              # Architecture and data flow
│   ├── data-quality-report.md        # Validation rules, freshness SLAs, benchmark citations
│   ├── metrics-dashboard.md          # Cost/latency per agent step
│   └── evals/                        # Benchmark test cases and results
│       ├── benchmark-cases.json      # 37 labeled test cases (pipeline + LLM + benchmarks)
│       ├── run-eval.py               # Automated evaluation runner
│       └── results-2026-04.md        # Latest evaluation results
├── chief-of-staff/                   # CLI tool for project coordination
├── .pipeline/                        # Auto-sync agent for team machines
├── cache/                            # Runtime JSON cache (gitignored)
├── requirements.txt
└── README.md
```

---

## Supporting Tools

### Chief of Staff (CLI)

A command-line tool for managing project coordination — task triage, decision logging, meeting prep, and weekly digests. Works offline; enhanced with Groq when `GROQ_API_KEY` is set.

```bash
chmod +x chief-of-staff/cos

./chief-of-staff/cos briefing     # Daily status: commits, open PRs, cache health
./chief-of-staff/cos triage       # Score-ranked task prioritization
./chief-of-staff/cos weekly       # Weekly digest: shipped, in-progress, next week
```

See [`chief-of-staff/README.md`](chief-of-staff/README.md) for full documentation.

### Auto-Sync Agent

Monitors the GitHub repo and automatically pulls updates on each team member's machine every 5 minutes via macOS launchd.

```bash
bash .pipeline/setup.sh      # One-time setup per machine
python3 .pipeline/status.py  # Check sync status
```

---

## Team

| Name | LinkedIn |
|------|----------|
| Aayman Afzal | [linkedin.com/in/aayman-afzal](https://www.linkedin.com/in/aayman-afzal) |
| Ajinkya Kodnikar | [linkedin.com/in/ajinkya-kodnikar](https://www.linkedin.com/in/ajinkya-kodnikar) |
| Oyu Amar | [linkedin.com/in/oyu-amar](https://www.linkedin.com/in/oyu-amar/) |
| Ricardo Ruiz | [linkedin.com/in/ricardoruizjr](https://www.linkedin.com/in/ricardoruizjr) |

---

*MGMT 690: AI Leadership · Purdue Daniels School of Business*
