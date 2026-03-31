# CRE Intelligence Platform
**MGMT 690 AI Leadership Project | Purdue Daniels School of Business**

A professional Streamlit application that provides **real-time Commercial Real Estate intelligence** powered by eight independent AI agents running in the background.

## 🤖 AI Workforce Architecture

This platform operates under a **Human-in-Command model** with 8 specialized AI agents:

| Agent | Role | Frequency | Output |
|-------|------|-----------|--------|
| 🏃 Migration Analyst | Population flows, corporate relocations | Every 6h | `migration.json` |
| 💰 REIT Analyst | Live pricing, cap rates, NOI margins | Every 1h | `pricing.json` |
| 🔮 Macro Strategist | LLM-powered company relocation predictions | Every 24h | `predictions.json` |
| 🛠️ System Monitor | Health checks, API status | Every 30min | `debugger.json` |
| 📰 Market Intelligence | News & government facility tracking | Every 4h | `news.json` |
| 📈 Rate Environment | Interest rates, yield curve, cap rate adjustments, REIT debt risk (FRED) | Every 1h | `rates.json` |
| 🛢️ Energy Analyst | Oil, gas, copper, steel → construction costs | Every 6h | `energy_data.json` |
| 🌱 Sustainability Analyst | Clean energy ETFs, green REITs → ESG momentum | Every 6h | `sustainability_data.json` |

```
                    ┌─────────────────────┐
                    │   HUMAN COMMANDER   │
                    │  (Review & Approve) │
                    └────────┬────────────┘
                             │
       ┌─────────────┬──────┴──────┬─────────────┐
       │             │             │             │
 ┌─────┴─────┐ ┌────┴────┐ ┌─────┴─────┐ ┌─────┴─────┐
 │ Migration │ │  REIT   │ │  Macro    │ │   Rate    │
 │  Analyst  │ │ Analyst │ │Strategist │ │Environment│
 └───────────┘ └─────────┘ └───────────┘ └───────────┘
       │             │             │             │
 ┌─────┴─────┐ ┌────┴────┐ ┌─────┴─────┐ ┌─────┴─────┐
 │  System   │ │ Market  │ │  Energy   │ │Sustainab. │
 │  Monitor  │ │  Intel  │ │ Analyst   │ │  Analyst  │
 └───────────┘ └─────────┘ └───────────┘ └───────────┘
```

**Human-in-Command Protocol:** All agent outputs are reviewed by the human commander. Nothing reaches users without explicit approval.

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **🗺️ Migration Intelligence** | Choropleth map, migration scores, metro rankings, bubble chart |
| **💰 Pricing & Profit** | Live REIT prices, heatmap of margin by market × property type, top 10 opportunities |
| **📈 Rate Environment** | Fed Funds/Treasury rates, yield curve, cap rate adjustments by property type, REIT refinancing risk |
| **🔮 Company Predictions** | AI-generated corporate relocation forecast (Llama 3.3-70B via Groq) |
| **🏗️ Cheapest Buildings** | Lowest-price commercial listings in the top 3 migration destination states |
| **📰 Industry Announcements** | News & government facility announcements from 10+ sources |
| **🛢️ Energy & Construction Costs** | Oil, gas, copper, steel prices; Construction Cost Signal (HIGH/MODERATE/LOW) |
| **🌱 Sustainability & ESG** | Clean energy ETF performance, green REITs, ESG Momentum Signal (STRONG/NEUTRAL/WEAK) |
| **🛠️ System Monitor** | Agent status, cache health, force-run controls, API health checks |

## Setup

```bash
pip install -r requirements.txt

# Create .env with API keys
cp .env.example .env   # or create manually

# Required for Rate Environment agent
FRED_API_KEY=your_fred_key_here   # free at https://fred.stlouisfed.org/docs/api/api_key.html

# Optional for AI predictions
GROQ_API_KEY=your_groq_key_here

streamlit run app/cre_app.py
```

## Tech Stack

- **UI**: Streamlit (Purdue gold/black branding)
- **Background Agents**: APScheduler (BackgroundScheduler)
- **Cache**: File-based JSON (`/cache/*.json`) — survives Streamlit reruns
- **Market Data**: yfinance (live REIT prices, dividends, market cap, commodities, ETFs)
- **Population Data**: US Census Bureau Population Estimates API
- **Interest Rate Data**: FRED API (10 series: Treasuries, SOFR, Fed Funds, mortgage rates, credit spreads)
- **AI Predictions**: Groq API (llama-3.3-70b-versatile)
- **Charts**: Plotly (choropleth maps, heatmaps, scatter, bar)

## Rate Environment Agent

Agent 6 pulls live data from the Federal Reserve (FRED) to provide macro context for CRE investment decisions:

- **Rate Signals**: 10Y/2Y/3M Treasuries, Fed Funds Rate, SOFR, 30Y mortgage, IG credit spreads
- **Yield Curve**: Shape classification (normal / flat / inverted) with historical trend
- **Environment Signal**: Composite BULLISH / CAUTIOUS / BEARISH rating based on rate levels, curve shape, and credit conditions
- **Cap Rate Adjustments**: Sector-specific cap rate estimates for 7 property types, adjusted dynamically to the current 10Y Treasury using sector betas
- **REIT Debt Risk**: Near-term refinancing exposure scored across major REITs using yfinance balance sheet data

Requires `FRED_API_KEY` in `.env`.

---

## Chief of Staff — AI Executive Assistant

A CLI tool for managing and coordinating work across the codebase. Tracks tasks, decisions, follow-ups, and generates briefings — with optional Groq LLM enhancement.

### Setup

```bash
chmod +x chief-of-staff/cos

# Run directly
./chief-of-staff/cos briefing

# Or add to PATH for global access
export PATH="$PATH:$(pwd)/chief-of-staff"
cos briefing
```

### Commands

| Command | Description |
|---------|-------------|
| `cos briefing` | Daily status: recent commits, open PRs, TODOs, cache health |
| `cos triage` | Score-ranked task prioritization from `tasks.md` |
| `cos prep "<topic>"` | Meeting prep doc with codebase context and talking points |
| `cos decide "<title>"` | Log an architectural/product decision |
| `cos decisions` | List recent decision log entries |
| `cos follow-up add "<item>" --owner --due` | Add a tracked action item |
| `cos follow-up list [--filter open\|overdue\|done\|all]` | List follow-ups |
| `cos follow-up done <id>` | Mark a follow-up complete |
| `cos weekly` | Weekly digest: shipped, in-progress, blocked, next week |

### State Files

All state is plain markdown in `chief-of-staff/state/`:

- `tasks.md` — edit manually to manage your task list
- `decisions.md` — auto-maintained decision log
- `follow-ups.md` — auto-maintained action item tracker

### AI Mode

Works fully offline without an API key. With `GROQ_API_KEY` set in `.env`, commands are enhanced with LLM summaries, strategic recommendations, and talking points.

See [`chief-of-staff/README.md`](chief-of-staff/README.md) for full documentation.

---

## Autonomous Sync Agent

Monitors the GitHub repo for new commits and automatically pulls updates on each team member's machine.

```bash
# One-time setup per machine
bash .pipeline/setup.sh

# Check sync status
python3 .pipeline/status.py
```

The agent runs every 5 minutes via macOS launchd and re-installs dependencies on each pull.

---
*MGMT 690: AI Leadership | Purdue Daniels School of Business*
