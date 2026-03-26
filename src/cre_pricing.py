"""
CRE Pricing Agent — fetches daily commercial real estate market data
via REIT tickers (yfinance) and FRED economic indicators.
Computes cap rate proxies, NOI estimates, and profit margin rankings.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── REIT Universe by Property Type ──────────────────────────────────────────
REIT_UNIVERSE = {
    "Industrial / Logistics": [
        {"ticker": "PLD",  "name": "Prologis",               "market": "National"},
        {"ticker": "STAG", "name": "STAG Industrial",         "market": "Secondary"},
        {"ticker": "EGP",  "name": "EastGroup Properties",    "market": "Sunbelt"},
        {"ticker": "FR",   "name": "First Industrial Realty",  "market": "National"},
    ],
    "Multifamily / Residential": [
        {"ticker": "EQR",  "name": "Equity Residential",      "market": "Urban"},
        {"ticker": "AVB",  "name": "AvalonBay Communities",    "market": "Coastal"},
        {"ticker": "MAA",  "name": "Mid-America Apartment",    "market": "Sunbelt"},
        {"ticker": "CPT",  "name": "Camden Property Trust",    "market": "Sunbelt"},
        {"ticker": "NMD",  "name": "NexPoint Residential",     "market": "Sunbelt"},
    ],
    "Retail": [
        {"ticker": "O",    "name": "Realty Income",           "market": "National"},
        {"ticker": "NNN",  "name": "NNN REIT",                "market": "National"},
        {"ticker": "SPG",  "name": "Simon Property Group",    "market": "Premium Malls"},
        {"ticker": "KIM",  "name": "Kimco Realty",            "market": "Open-Air Centers"},
    ],
    "Office": [
        {"ticker": "BXP",  "name": "BXP (Boston Properties)", "market": "Trophy Office"},
        {"ticker": "HIW",  "name": "Highwoods Properties",    "market": "Sunbelt Office"},
        {"ticker": "CUZ",  "name": "Cousins Properties",      "market": "Sunbelt Office"},
        {"ticker": "PDM",  "name": "Piedmont Office",         "market": "Sunbelt Office"},
    ],
    "Healthcare / Medical Office": [
        {"ticker": "WELL", "name": "Welltower",               "market": "Senior Care"},
        {"ticker": "VTR",  "name": "Ventas",                  "market": "Senior Care"},
        {"ticker": "HR",   "name": "Healthcare Realty",       "market": "Medical Office"},
        {"ticker": "DOC",  "name": "Healthpeak Properties",   "market": "Life Science"},
    ],
    "Self-Storage": [
        {"ticker": "PSA",  "name": "Public Storage",          "market": "National"},
        {"ticker": "EXR",  "name": "Extra Space Storage",     "market": "National"},
        {"ticker": "CUBE", "name": "CubeSmart",               "market": "Urban"},
        {"ticker": "LSI",  "name": "Life Storage",            "market": "Sunbelt"},
    ],
    "Data Centers": [
        {"ticker": "EQIX", "name": "Equinix",                 "market": "Global"},
        {"ticker": "DLR",  "name": "Digital Realty",          "market": "Global"},
        {"ticker": "QTS",  "name": "QTS Realty",              "market": "National"},
    ],
}

# Typical cap rates & margin benchmarks by property type (market data 2024-2025)
CAP_RATE_BENCHMARKS = {
    "Industrial / Logistics":      {"cap_rate": 0.056, "noi_margin": 0.72, "vacancy": 0.045, "rent_growth": 0.08},
    "Multifamily / Residential":   {"cap_rate": 0.052, "noi_margin": 0.62, "vacancy": 0.055, "rent_growth": 0.05},
    "Retail":                      {"cap_rate": 0.068, "noi_margin": 0.58, "vacancy": 0.075, "rent_growth": 0.02},
    "Office":                      {"cap_rate": 0.085, "noi_margin": 0.48, "vacancy": 0.185, "rent_growth": -0.03},
    "Healthcare / Medical Office": {"cap_rate": 0.058, "noi_margin": 0.65, "vacancy": 0.060, "rent_growth": 0.04},
    "Self-Storage":                {"cap_rate": 0.054, "noi_margin": 0.70, "vacancy": 0.085, "rent_growth": 0.03},
    "Data Centers":                {"cap_rate": 0.048, "noi_margin": 0.55, "vacancy": 0.020, "rent_growth": 0.12},
}

# Market-level cap rate adjustments (sunbelt premium vs. gateway discount)
MARKET_CAP_ADJUSTMENTS = {
    "Dallas-Fort Worth, TX":  -0.006,
    "Austin, TX":             -0.008,
    "Phoenix, AZ":            -0.005,
    "Miami, FL":              -0.004,
    "Nashville, TN":          -0.005,
    "Charlotte, NC":          -0.004,
    "Raleigh-Durham, NC":     -0.006,
    "Atlanta, GA":            -0.003,
    "Tampa, FL":              -0.003,
    "Salt Lake City, UT":     -0.004,
    "New York, NY":           +0.008,
    "San Francisco, CA":      +0.012,
    "Chicago, IL":            +0.007,
    "Los Angeles, CA":        +0.006,
}


def fetch_reit_prices() -> pd.DataFrame:
    """Fetch daily price, yield, and performance for all REITs."""
    all_rows = []
    for prop_type, reits in REIT_UNIVERSE.items():
        tickers = [r["ticker"] for r in reits]
        try:
            raw = yf.download(tickers, period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        except Exception:
            closes = pd.DataFrame()

        for reit in reits:
            t = reit["ticker"]
            info = {}
            try:
                info = yf.Ticker(t).info
            except Exception:
                pass

            price     = info.get("currentPrice") or info.get("regularMarketPrice")
            div_yield = info.get("dividendYield") or 0
            div_rate  = info.get("dividendRate") or 0
            mktcap    = info.get("marketCap") or 0
            pe        = info.get("trailingPE")
            pffo      = info.get("priceToBook")   # proxy for P/FFO

            # 1-day return
            try:
                col_closes = closes[t] if t in closes.columns else closes.squeeze()
                daily_ret = float(col_closes.pct_change().iloc[-1]) if len(col_closes) >= 2 else 0.0
            except Exception:
                daily_ret = 0.0

            bench = CAP_RATE_BENCHMARKS.get(prop_type, {})
            cap_r = bench.get("cap_rate", 0.06)
            noi_m = bench.get("noi_margin", 0.60)

            all_rows.append({
                "Property Type":  prop_type,
                "Ticker":         t,
                "Company":        reit["name"],
                "Market Focus":   reit["market"],
                "Price":          price,
                "Div Yield":      div_yield,
                "Daily Return":   daily_ret,
                "Market Cap":     mktcap,
                "Cap Rate":       cap_r,
                "NOI Margin":     noi_m,
                "Vacancy Rate":   bench.get("vacancy", 0.07),
                "Rent Growth":    bench.get("rent_growth", 0.03),
            })
    return pd.DataFrame(all_rows)


def compute_profit_matrix() -> pd.DataFrame:
    """
    Ranks property types × markets by estimated gross profit margin.
    Profit Margin = NOI Margin × (1 - Vacancy) × (1 + Rent Growth)
    """
    rows = []
    markets = list(MARKET_CAP_ADJUSTMENTS.keys())
    for prop_type, bench in CAP_RATE_BENCHMARKS.items():
        for market in markets:
            adj      = MARKET_CAP_ADJUSTMENTS[market]
            eff_cap  = bench["cap_rate"] + adj
            noi_m    = bench["noi_margin"]
            vacancy  = bench["vacancy"]
            rent_g   = bench["rent_growth"]

            # Effective profit margin accounting for vacancy and rent trajectory
            eff_margin = noi_m * (1 - vacancy) * (1 + rent_g)
            # Score: higher cap rate relative to market = more income per dollar invested
            profit_score = (eff_cap / 0.06) * eff_margin * 100

            rows.append({
                "Market":         market,
                "Property Type":  prop_type,
                "Cap Rate":       round(eff_cap, 4),
                "NOI Margin":     round(noi_m, 3),
                "Vacancy Rate":   round(vacancy, 3),
                "Rent Growth":    round(rent_g, 3),
                "Eff Profit Margin": round(eff_margin, 3),
                "Profit Score":   round(profit_score, 1),
            })
    df = pd.DataFrame(rows).sort_values("Profit Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


def get_top_opportunities(n: int = 10) -> pd.DataFrame:
    """Returns top N market × property type combinations by profit score."""
    df = compute_profit_matrix()
    top = df.head(n).copy()
    top["Cap Rate"]       = top["Cap Rate"].apply(lambda x: f"{x*100:.2f}%")
    top["NOI Margin"]     = top["NOI Margin"].apply(lambda x: f"{x*100:.1f}%")
    top["Vacancy Rate"]   = top["Vacancy Rate"].apply(lambda x: f"{x*100:.1f}%")
    top["Rent Growth"]    = top["Rent Growth"].apply(lambda x: f"{x*100:.1f}%")
    top["Eff Profit Margin"] = top["Eff Profit Margin"].apply(lambda x: f"{x*100:.1f}%")
    return top[["Rank", "Market", "Property Type", "Cap Rate", "NOI Margin",
                "Vacancy Rate", "Rent Growth", "Eff Profit Margin", "Profit Score"]]


def get_property_type_summary() -> pd.DataFrame:
    """Summary table of property types ranked by attractiveness."""
    rows = []
    for pt, b in CAP_RATE_BENCHMARKS.items():
        eff = b["noi_margin"] * (1 - b["vacancy"]) * (1 + b["rent_growth"])
        rows.append({
            "Property Type":      pt,
            "Cap Rate":           b["cap_rate"],
            "NOI Margin":         b["noi_margin"],
            "Vacancy Rate":       b["vacancy"],
            "Rent Growth YoY":    b["rent_growth"],
            "Eff Profit Margin":  round(eff, 3),
        })
    df = pd.DataFrame(rows).sort_values("Eff Profit Margin", ascending=False).reset_index(drop=True)
    return df
