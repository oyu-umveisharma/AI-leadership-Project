# CRE Intelligence Platform
**MGMT 690 AI Leadership Project | Purdue Daniels School of Business**

A professional Streamlit application that provides **real-time Commercial Real Estate intelligence** powered by four independent AI agents running in the background.

## What It Does

Four background agents update automatically on schedules:

| Agent | Description | Frequency |
|-------|-------------|-----------|
| **Agent 1 · Migration** | US state population growth + corporate HQ relocations (Census API) | Every 6h |
| **Agent 2 · Pricing** | Live REIT prices, cap rates, NOI margins, profit rankings (yfinance) | Every 1h |
| **Agent 3 · Predictions** | LLM-predicted company relocations + cheapest buildings in top 3 cities | Every 24h |
| **Agent 4 · Debugger** | Health checks: APIs, cache freshness, connectivity | Every 30min |

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Migration Intelligence** | Choropleth map, migration scores, metro rankings, bubble chart |
| **Pricing & Profit** | Live REIT prices, heatmap of margin by market × property type, top 10 opportunities |
| **Company Predictions** | AI-generated corporate relocation forecast (Llama 3.3-70B via Groq) |
| **Cheapest Buildings** | Lowest-price commercial listings in the top 3 migration destination states |
| **System Monitor** | Agent status, cache health, force-run controls, API health checks |

## Setup

```bash
pip install -r requirements.txt

# Optional: add GROQ_API_KEY to .env for AI predictions
echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app/cre_app.py
```

## Tech Stack

- **UI**: Streamlit (Purdue gold/black branding)
- **Background Agents**: APScheduler (BackgroundScheduler)
- **Cache**: File-based JSON (`/cache/*.json`) — survives Streamlit reruns
- **Market Data**: yfinance (live REIT prices, dividends, market cap)
- **Population Data**: US Census Bureau Population Estimates API
- **AI Predictions**: Groq API (llama-3.3-70b-versatile)
- **Charts**: Plotly (choropleth maps, heatmaps, scatter, bar)

---
*MGMT 690: AI Leadership | Purdue Daniels School of Business*
