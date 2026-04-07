# CRE Intelligence Platform

> Real-time commercial real estate intelligence powered by twelve autonomous AI agents — tracking migration flows, REIT pricing, interest rates, labor markets, GDP, inflation, credit conditions, facility announcements, energy costs, and ESG momentum across US markets.

**Purdue Daniels School of Business · MGMT 690: AI Leadership**

---

## What It Does

Most CRE research tools require manual data pulls, static spreadsheets, and hours of synthesis. This platform replaces that workflow with a live dashboard backed by twelve background agents that update continuously — surfacing which markets to watch, which property types are most profitable, which companies are building new facilities, and what macro conditions mean for CRE valuations.

The app opens with an AI-powered chatbox that asks what you are looking for. Type a query like "industrial in Los Angeles" or "office in Chicago" and the entire dashboard personalizes to that search — tab titles rewrite, maps zoom to your metro, listings filter to your city, and AI-generated insights appear for your specific property type and location. Over 200 US cities are recognized with accurate metro-level neighborhood maps.

The agents run on a scheduler. Open the dashboard and the data is already there.

---

## Agent Architecture

Twelve specialized agents operate independently on fixed schedules, writing to a shared JSON cache that survives Streamlit reruns.

| Agent | Responsibility | Schedule |
|-------|---------------|----------|
| Migration Analyst | Population flows, state migration scores, metro rankings | Every 6 hours |
| REIT Analyst | Live REIT prices, cap rates, NOI margins, profit matrix | Every 1 hour |
| Facility Intelligence | Confirmed plant & facility announcements extracted from live news | Every 24 hours |
| System Monitor | API health, cache freshness, agent status | Every 30 minutes |
| Market Intelligence | RSS news feeds — manufacturing, government, industry press | Every 4 hours |
| Rate Environment | Fed Funds, Treasuries, yield curve, cap rate adjustments, REIT debt risk | Every 1 hour |
| Energy Analyst | Oil, gas, copper, steel prices vs. 60-day moving average | Every 6 hours |
| Sustainability Analyst | Clean energy ETFs, green REIT performance vs. S&P 500 | Every 6 hours |
| Labor Market Analyst | Job growth by metro, sector payrolls, tenant demand signal, hiring momentum | Every 6 hours |
| GDP & Economic Growth | Real GDP, industrial production, retail sales, consumer sentiment, CFNAI | Every 6 hours |
| Inflation Analyst | CPI (headline/core/shelter/rent), PPI, breakeven inflation expectations | Every 6 hours |
| Credit & Capital Markets | Corporate spreads, VIX, bank lending standards, CRE loan conditions | Every 6 hours |

All agents start automatically when the app launches. No manual triggers needed.

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

A persistent chat bar at the top of the dashboard allows changing the query at any time without returning to the welcome screen. AI-generated insights powered by Groq appear at the top of each tab when a focus is set.

---

## Migration Map Drill-Down

The Migration Intelligence tab supports three levels of geographic resolution:

| Level | View | Data |
|-------|------|------|
| National | US choropleth by state | 51 states/territories colored by composite migration score |
| State (Counties) | County choropleth with FIPS codes | 8-18 counties per state with population, growth rate, key economic drivers |
| Metro (Neighborhoods) | Scattermapbox with dark tiles | 10-18 neighborhoods per metro with lat/lon, migration score, rent growth, zone type |

The map auto-selects the appropriate level based on the user's query. Typing "Texas" zooms to the state view with counties. Typing "Chicago" zooms to the metro view with neighborhood dots. 17 major metros are supported with accurate coordinates.

---

## Dashboard

Three top-level tabs — **Real Estate**, **Energy**, and **Macro Environment** — each with focused sub-tabs.

### Real Estate

| Tab | What You See |
|-----|-------------|
| Migration Intelligence | US choropleth map, top 10 states by migration score, business vs. population bubble chart, metro table |
| Pricing & Profit | Live REIT prices, profit margin heatmap by market × property type, top 10 opportunities, property type comparison |
| Company Predictions | Confirmed corporate facility announcements (plants, fabs, warehouses, data centers, HQ moves) with investment, jobs, and CRE impact |
| Cheapest Buildings | Lowest-price commercial listings in the top 3 migration destination states |
| Industry Announcements | AI-structured brief from 10+ news and government RSS sources |
| System Monitor | Agent status, cache health, API connectivity |

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

The app opens at `http://localhost:8501` by default. All twelve agents start in the background immediately. Data populates within 30–60 seconds.

---

## API Keys

| Key | Required | Purpose | Get It |
|-----|----------|---------|--------|
| `GROQ_API_KEY` | Recommended | Company facility announcements (Llama 3.3-70B) and news summaries | [console.groq.com](https://console.groq.com) — free tier available |
| `FRED_API_KEY` | Recommended | Interest rates, yield curve, labor market, GDP, inflation, and credit market data from the Federal Reserve | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — free |

Store keys in a `.env` file at the project root (no spaces around `=`):

```
GROQ_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

The app runs without both keys — migration, REIT pricing, energy, and sustainability tabs work on public data alone. FRED-powered tabs (Rate Environment, Labor Market, GDP, Inflation, Credit) degrade gracefully with a status message.

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
| AI / LLM | Groq API — llama-3.3-70b-versatile |
| News Sources | Reuters, Manufacturing.net, IndustryWeek, PR Newswire, Business Wire, Dept. of Energy, Dept. of Commerce, EDA, Expansion Solutions, Site Selection Magazine |
| Charts | Plotly — choropleth maps, heatmaps, bar, scatter, line |
| Language | Python 3.10+ |

---

## Project Structure

```
AI-leadership-Project/
├── app/
│   └── cre_app.py                  # Streamlit dashboard — all tabs and visualization
├── src/
│   ├── cre_agents.py               # Agent runner, scheduler, cache helpers (12 agents)
│   ├── cre_population.py           # Census API — migration scores, metro data
│   ├── cre_pricing.py              # REIT universe, cap rates, profit matrix
│   ├── cre_news.py                 # RSS feed scraper, facility keyword filter
│   ├── cre_listings.py             # Commercial property listings by state (28 states, 200+ cities)
│   ├── rate_agent.py               # FRED API — interest rates, yield curve
│   ├── energy_analyst.py           # Commodity prices, construction cost signal
│   ├── sustainability_analyst.py   # Clean energy ETFs, green REIT tracking
│   ├── labor_market_agent.py       # BLS + FRED + yfinance — labor market & tenant demand
│   ├── gdp_agent.py                # FRED — GDP, industrial production, consumer sentiment
│   ├── inflation_agent.py          # FRED — CPI, PPI, breakeven inflation expectations
│   ├── credit_markets_agent.py     # FRED — corporate spreads, VIX, lending standards
│   ├── county_migration.py         # County-level migration data (FIPS codes, 12 states seeded)
│   └── zip_migration.py            # Neighborhood-level data (17 metros, real lat/lon)
├── chief-of-staff/                 # CLI tool for project coordination
├── .pipeline/                      # Auto-sync agent for team machines
├── cache/                          # Runtime JSON cache (gitignored)
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
| Oyu Amar | [linkedin.com/in/oyuamar](https://www.linkedin.com/in/oyuamar) |
| Ricardo Ruiz | [linkedin.com/in/ricardoruizjr](https://www.linkedin.com/in/ricardoruizjr) |

---

*MGMT 690: AI Leadership · Purdue Daniels School of Business*
