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
