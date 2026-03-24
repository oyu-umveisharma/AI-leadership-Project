# AI-leadership-Project — M&A Deal Analyzer
**MGMT 690 AI Leadership Project | Purdue Daniels School of Business**

A professional Streamlit application that performs a **full step-by-step M&A analysis** for any two public companies, following the same methodology as the course Excel model.

## What It Does

Enter any two ticker symbols (Acquirer + Target) and the app walks through:

| Step | Analysis |
|------|----------|
| **Overview** | Company profiles, sector, market cap |
| **Step 1** | Financial Statements — IS, BS, CF for both companies |
| **Step 2** | NWC Analysis — NWC/Sales ratios, historical trends |
| **Step 3** | Beta Regression — OLS vs S&P 500 (60 months) |
| **Step 4** | WACC — Capital structure, cost of equity/debt |
| **Step 5** | Standalone DCF — FCF projections, terminal value, EV |
| **Step 6** | Synergy Analysis — Revenue + cost synergies, PV |
| **Step 7** | Deal Summary — Offer price, total deal value, recommendation |
| **Step 8** | Suggested Deals — AI-scored alternative acquisition targets |
| **Step 9** | Cannibalization Analysis — Revenue at risk, adjusted deal value |

## Setup

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Tech Stack

- **UI**: Streamlit (Purdue gold/black branding)
- **Data**: yfinance (live financial statements + prices)
- **Math**: scipy, numpy, statsmodels (beta regression, DCF)
- **Charts**: Plotly (interactive waterfall, scatter, bar, football field)

## Example

- Acquirer: `PG` (Procter & Gamble)
- Target: `PHSI` (Prestige Consumer Healthcare)

---
*MGMT 690: AI Leadership | Purdue Daniels School of Business*
