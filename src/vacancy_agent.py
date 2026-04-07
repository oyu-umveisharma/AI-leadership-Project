"""
Vacancy Rate Monitor — tracks commercial vacancy rates by property type and market.

Data sources:
  - FRED API: residential rental vacancy (national benchmark / leading indicator)
  - Market-informed estimates: office/industrial/retail/multifamily vacancy by
    top CRE markets, calibrated to CoStar/CBRE Q1 2025 benchmarks.

Returns cache-ready dict with national rates, market table, and trend signals.
"""

import os
import requests
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── National vacancy benchmarks by property type (Q1 2025) ───────────────────
# Sources: CBRE, JLL, CoStar market reports
NATIONAL_VACANCY = {
    "Office":      {"rate": 19.8, "prior_year": 18.2, "trend": "rising",   "note": "Remote work headwinds; suburban outperforming CBD"},
    "Industrial":  {"rate":  6.1, "prior_year":  4.9, "trend": "rising",   "note": "Supply surge normalizing; still historically tight"},
    "Retail":      {"rate": 10.3, "prior_year": 10.7, "trend": "falling",  "note": "Experiential and grocery-anchored outperforming"},
    "Multifamily": {"rate":  6.8, "prior_year":  5.8, "trend": "rising",   "note": "Record supply deliveries pressuring Sunbelt markets"},
    "Mixed-Use":   {"rate":  8.4, "prior_year":  8.1, "trend": "stable",   "note": "Transit-oriented and walkable locations outperform"},
}

# ── Market-level vacancy by property type ─────────────────────────────────────
# Format: market -> property type -> (vacancy_pct, trend, absorption_note)
MARKET_VACANCY = {
    "Austin, TX":        {"Office": (24.1, "rising"),  "Industrial": (8.2,  "rising"),  "Retail": (5.8,  "stable"),  "Multifamily": (11.2, "rising")},
    "Dallas, TX":        {"Office": (22.8, "rising"),  "Industrial": (6.4,  "rising"),  "Retail": (7.2,  "stable"),  "Multifamily": (9.8,  "rising")},
    "Houston, TX":       {"Office": (25.3, "rising"),  "Industrial": (5.9,  "stable"),  "Retail": (8.1,  "stable"),  "Multifamily": (10.4, "rising")},
    "Phoenix, AZ":       {"Office": (20.1, "stable"),  "Industrial": (8.8,  "rising"),  "Retail": (7.5,  "falling"), "Multifamily": (9.1,  "rising")},
    "Nashville, TN":     {"Office": (17.2, "stable"),  "Industrial": (5.1,  "stable"),  "Retail": (6.3,  "falling"), "Multifamily": (8.7,  "rising")},
    "Charlotte, NC":     {"Office": (18.4, "rising"),  "Industrial": (6.8,  "stable"),  "Retail": (7.8,  "falling"), "Multifamily": (7.9,  "rising")},
    "Raleigh, NC":       {"Office": (15.9, "stable"),  "Industrial": (5.6,  "stable"),  "Retail": (6.1,  "stable"),  "Multifamily": (8.2,  "rising")},
    "Atlanta, GA":       {"Office": (21.3, "rising"),  "Industrial": (5.8,  "stable"),  "Retail": (8.9,  "stable"),  "Multifamily": (10.1, "rising")},
    "Denver, CO":        {"Office": (23.7, "rising"),  "Industrial": (7.2,  "rising"),  "Retail": (8.3,  "stable"),  "Multifamily": (8.8,  "rising")},
    "Las Vegas, NV":     {"Office": (14.8, "falling"), "Industrial": (4.2,  "stable"),  "Retail": (6.7,  "falling"), "Multifamily": (7.4,  "stable")},
    "Salt Lake City, UT":{"Office": (16.2, "rising"),  "Industrial": (5.4,  "rising"),  "Retail": (6.8,  "stable"),  "Multifamily": (7.1,  "stable")},
    "Jacksonville, FL":  {"Office": (13.1, "falling"), "Industrial": (4.8,  "stable"),  "Retail": (7.3,  "stable"),  "Multifamily": (9.3,  "rising")},
    "Tampa, FL":         {"Office": (15.6, "stable"),  "Industrial": (5.2,  "stable"),  "Retail": (6.9,  "falling"), "Multifamily": (10.8, "rising")},
    "Orlando, FL":       {"Office": (14.9, "falling"), "Industrial": (6.1,  "stable"),  "Retail": (7.1,  "falling"), "Multifamily": (11.4, "rising")},
    "Indianapolis, IN":  {"Office": (16.8, "stable"),  "Industrial": (4.1,  "stable"),  "Retail": (8.2,  "stable"),  "Multifamily": (6.8,  "stable")},
    "Columbus, OH":      {"Office": (17.3, "stable"),  "Industrial": (4.6,  "rising"),  "Retail": (7.9,  "stable"),  "Multifamily": (7.2,  "stable")},
    "Kansas City, MO":   {"Office": (18.9, "rising"),  "Industrial": (3.8,  "stable"),  "Retail": (8.8,  "stable"),  "Multifamily": (7.6,  "stable")},
    "Los Angeles, CA":   {"Office": (22.4, "rising"),  "Industrial": (4.9,  "rising"),  "Retail": (10.1, "stable"),  "Multifamily": (5.8,  "stable")},
    "Seattle, WA":       {"Office": (21.8, "rising"),  "Industrial": (5.6,  "rising"),  "Retail": (7.4,  "stable"),  "Multifamily": (6.9,  "rising")},
    "Chicago, IL":       {"Office": (20.6, "rising"),  "Industrial": (4.3,  "rising"),  "Retail": (11.2, "stable"),  "Multifamily": (6.4,  "stable")},
    "New York, NY":      {"Office": (17.9, "rising"),  "Industrial": (3.2,  "stable"),  "Retail": (12.8, "stable"),  "Multifamily": (4.1,  "falling")},
    "Boston, MA":        {"Office": (16.7, "rising"),  "Industrial": (5.1,  "rising"),  "Retail": (9.3,  "stable"),  "Multifamily": (5.2,  "stable")},
    "Miami, FL":         {"Office": (14.2, "falling"), "Industrial": (4.6,  "stable"),  "Retail": (5.4,  "falling"), "Multifamily": (7.8,  "rising")},
}

TREND_ARROW = {"rising": "↑", "falling": "↓", "stable": "→"}
TREND_COLOR = {"rising": "#ef5350", "falling": "#66bb6a", "stable": "#CFB991"}


def _fetch_fred_vacancy(api_key: str) -> list[dict]:
    """Fetch US rental vacancy rate from FRED (RRVRUSQ156N)."""
    if not api_key:
        return []
    try:
        params = {
            "series_id": "RRVRUSQ156N",
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": "20",
        }
        import urllib.parse
        url = f"{FRED_BASE}?{urllib.parse.urlencode(params)}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return [
            {"date": o["date"], "value": float(o["value"])}
            for o in obs
            if o.get("value") not in (".", None, "")
        ]
    except Exception:
        return []


def run_vacancy_agent() -> dict:
    """Main entry point — returns cache-ready vacancy data dict."""
    api_key = os.getenv("FRED_API_KEY", "")
    fred_series = _fetch_fred_vacancy(api_key)

    # Build market rows for table display
    market_rows = []
    for market, ptypes in MARKET_VACANCY.items():
        for ptype, (rate, trend) in ptypes.items():
            national = NATIONAL_VACANCY.get(ptype, {}).get("rate", 10.0)
            vs_national = round(rate - national, 1)
            market_rows.append({
                "market":       market,
                "property_type": ptype,
                "vacancy_rate": rate,
                "trend":        trend,
                "vs_national":  vs_national,
            })

    return {
        "national":       NATIONAL_VACANCY,
        "market_rows":    market_rows,
        "fred_rental":    fred_series,
        "fetched_at":     datetime.now().isoformat(),
        "data_as_of":     "Q1 2025",
    }
