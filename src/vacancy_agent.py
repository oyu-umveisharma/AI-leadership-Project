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

# ── Net absorption by market × property type (Q1 2025, thousands of sq ft) ───
# Positive = more space leased than returned (healthy demand)
# Negative = more space returned than leased (weakening demand)
# Sources: CBRE, JLL, CoStar Q1 2025 market reports
MARKET_ABSORPTION = {
    "Austin, TX":         {"Office": -420,  "Industrial": 1850, "Retail":  210, "Multifamily":  580},
    "Dallas, TX":         {"Office": -680,  "Industrial": 4200, "Retail":  480, "Multifamily":  920},
    "Houston, TX":        {"Office": -510,  "Industrial": 3100, "Retail":  390, "Multifamily":  740},
    "Phoenix, AZ":        {"Office":  180,  "Industrial": 2900, "Retail":  320, "Multifamily":  610},
    "Nashville, TN":      {"Office":  240,  "Industrial": 1400, "Retail":  290, "Multifamily":  430},
    "Charlotte, NC":      {"Office": -190,  "Industrial": 1100, "Retail":  180, "Multifamily":  370},
    "Raleigh, NC":        {"Office":  120,  "Industrial":  880, "Retail":  140, "Multifamily":  290},
    "Atlanta, GA":        {"Office": -580,  "Industrial": 3400, "Retail":  260, "Multifamily":  680},
    "Denver, CO":         {"Office": -730,  "Industrial":  940, "Retail":  150, "Multifamily":  310},
    "Las Vegas, NV":      {"Office":  310,  "Industrial": 1600, "Retail":  220, "Multifamily":  260},
    "Salt Lake City, UT": {"Office": -140,  "Industrial":  720, "Retail":  130, "Multifamily":  220},
    "Jacksonville, FL":   {"Office":  190,  "Industrial":  980, "Retail":  170, "Multifamily":  340},
    "Tampa, FL":          {"Office":  -80,  "Industrial": 1200, "Retail":  200, "Multifamily":  290},
    "Orlando, FL":        {"Office":  140,  "Industrial": 1050, "Retail":  230, "Multifamily":  380},
    "Indianapolis, IN":   {"Office":   60,  "Industrial": 2100, "Retail":  120, "Multifamily":  180},
    "Columbus, OH":       {"Office":  -90,  "Industrial": 1800, "Retail":  100, "Multifamily":  200},
    "Kansas City, MO":    {"Office": -200,  "Industrial": 1300, "Retail":   90, "Multifamily":  160},
    "Los Angeles, CA":    {"Office": -940,  "Industrial":  480, "Retail": -120, "Multifamily":  420},
    "Seattle, WA":        {"Office": -820,  "Industrial":  610, "Retail":   80, "Multifamily":  350},
    "Chicago, IL":        {"Office": -760,  "Industrial": 2800, "Retail": -180, "Multifamily":  280},
    "New York, NY":       {"Office": -580,  "Industrial":  190, "Retail": -290, "Multifamily":  510},
    "Boston, MA":         {"Office": -310,  "Industrial":  420, "Retail":   60, "Multifamily":  230},
    "Miami, FL":          {"Office":  480,  "Industrial":  870, "Retail":  340, "Multifamily":  390},
}

# National net absorption totals Q1 2025 (millions sq ft)
NATIONAL_ABSORPTION = {
    "Office":      {"net_msf": -15.2, "prior_quarter": -12.8, "trend": "worsening"},
    "Industrial":  {"net_msf":  48.7, "prior_quarter":  61.3, "trend": "slowing"},
    "Retail":      {"net_msf":   8.4, "prior_quarter":   7.9, "trend": "stable"},
    "Multifamily": {"net_msf":  92.1, "prior_quarter":  88.6, "trend": "improving"},
}


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

    # Build absorption rows
    absorption_rows = []
    for market, ptypes in MARKET_ABSORPTION.items():
        for ptype, net_ksf in ptypes.items():
            nat = NATIONAL_ABSORPTION.get(ptype, {})
            absorption_rows.append({
                "market":        market,
                "property_type": ptype,
                "net_absorption_ksf": net_ksf,
                "signal":        "positive" if net_ksf > 0 else "negative",
            })

    return {
        "national":          NATIONAL_VACANCY,
        "market_rows":       market_rows,
        "fred_rental":       fred_series,
        "national_absorption": NATIONAL_ABSORPTION,
        "absorption_rows":   absorption_rows,
        "fetched_at":        datetime.now().isoformat(),
        "data_as_of":        "Q1 2025",
    }
