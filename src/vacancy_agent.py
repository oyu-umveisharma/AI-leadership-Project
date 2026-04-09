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
    from pathlib import Path as _P
    load_dotenv(_P(__file__).parent.parent / ".env", override=True)
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

# ── Developable / vacant land availability by market (Q1 2025) ───────────
# Entitled or shovel-ready acres actively listed / available for development.
# Sources: CoStar Land, Xceligent, state planning databases.
# pipeline_trend: direction of new entitlements coming online
LAND_AVAILABILITY = {
    "Austin, TX":       {"lat": 30.27, "lon":  -97.74, "industrial_ac":  8_400, "mixed_use_ac": 2_100, "residential_ac": 12_500, "avg_ppa":  45_000, "entitlement_mo": 18, "pipeline_trend": "rising"},
    "Dallas, TX":       {"lat": 32.78, "lon":  -96.80, "industrial_ac": 22_000, "mixed_use_ac": 4_800, "residential_ac": 35_000, "avg_ppa":  32_000, "entitlement_mo": 14, "pipeline_trend": "rising"},
    "Houston, TX":      {"lat": 29.76, "lon":  -95.37, "industrial_ac": 18_500, "mixed_use_ac": 3_200, "residential_ac": 28_000, "avg_ppa":  28_000, "entitlement_mo": 12, "pipeline_trend": "stable"},
    "Phoenix, AZ":      {"lat": 33.45, "lon": -112.07, "industrial_ac": 14_000, "mixed_use_ac": 2_900, "residential_ac": 22_000, "avg_ppa":  38_000, "entitlement_mo": 16, "pipeline_trend": "rising"},
    "Nashville, TN":    {"lat": 36.17, "lon":  -86.78, "industrial_ac":  6_500, "mixed_use_ac": 1_400, "residential_ac":  9_800, "avg_ppa":  42_000, "entitlement_mo": 20, "pipeline_trend": "stable"},
    "Charlotte, NC":    {"lat": 35.23, "lon":  -80.84, "industrial_ac":  7_800, "mixed_use_ac": 1_800, "residential_ac": 11_200, "avg_ppa":  35_000, "entitlement_mo": 18, "pipeline_trend": "rising"},
    "Atlanta, GA":      {"lat": 33.75, "lon":  -84.39, "industrial_ac": 12_000, "mixed_use_ac": 2_500, "residential_ac": 18_000, "avg_ppa":  30_000, "entitlement_mo": 15, "pipeline_trend": "rising"},
    "Denver, CO":       {"lat": 39.74, "lon": -104.99, "industrial_ac":  4_200, "mixed_use_ac":   900, "residential_ac":  6_500, "avg_ppa":  65_000, "entitlement_mo": 24, "pipeline_trend": "falling"},
    "Las Vegas, NV":    {"lat": 36.17, "lon": -115.14, "industrial_ac":  9_500, "mixed_use_ac": 2_100, "residential_ac": 14_000, "avg_ppa":  48_000, "entitlement_mo": 14, "pipeline_trend": "stable"},
    "Raleigh, NC":      {"lat": 35.78, "lon":  -78.64, "industrial_ac":  5_600, "mixed_use_ac": 1_200, "residential_ac":  8_400, "avg_ppa":  38_000, "entitlement_mo": 20, "pipeline_trend": "rising"},
    "Tampa, FL":        {"lat": 27.95, "lon":  -82.46, "industrial_ac":  5_200, "mixed_use_ac": 1_100, "residential_ac":  8_100, "avg_ppa":  52_000, "entitlement_mo": 22, "pipeline_trend": "stable"},
    "Orlando, FL":      {"lat": 28.54, "lon":  -81.38, "industrial_ac":  6_800, "mixed_use_ac": 1_500, "residential_ac": 10_500, "avg_ppa":  48_000, "entitlement_mo": 20, "pipeline_trend": "rising"},
    "Indianapolis, IN": {"lat": 39.77, "lon":  -86.16, "industrial_ac": 11_000, "mixed_use_ac": 1_800, "residential_ac": 15_000, "avg_ppa":  18_000, "entitlement_mo": 12, "pipeline_trend": "stable"},
    "Los Angeles, CA":  {"lat": 34.05, "lon": -118.24, "industrial_ac":  1_200, "mixed_use_ac":   380, "residential_ac":  2_100, "avg_ppa": 280_000, "entitlement_mo": 42, "pipeline_trend": "falling"},
    "Seattle, WA":      {"lat": 47.61, "lon": -122.33, "industrial_ac":  1_800, "mixed_use_ac":   420, "residential_ac":  3_200, "avg_ppa": 180_000, "entitlement_mo": 36, "pipeline_trend": "falling"},
    "Chicago, IL":      {"lat": 41.88, "lon":  -87.63, "industrial_ac":  8_500, "mixed_use_ac": 1_600, "residential_ac": 12_000, "avg_ppa":  42_000, "entitlement_mo": 28, "pipeline_trend": "stable"},
    "New York, NY":     {"lat": 40.71, "lon":  -74.01, "industrial_ac":    420, "mixed_use_ac":   180, "residential_ac":    680, "avg_ppa": 420_000, "entitlement_mo": 60, "pipeline_trend": "falling"},
    "Miami, FL":        {"lat": 25.77, "lon":  -80.19, "industrial_ac":  2_100, "mixed_use_ac":   640, "residential_ac":  3_800, "avg_ppa":  95_000, "entitlement_mo": 30, "pipeline_trend": "stable"},
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
        "land_availability": LAND_AVAILABILITY,
        "fetched_at":        datetime.now().isoformat(),
        "data_as_of":        "Q1 2025",
    }
