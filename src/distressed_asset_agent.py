"""
CMBS & Distressed Asset Monitor — tracks CRE loan delinquency rates,
CMBS spread conditions, known distressed deals, and distress signals
by property type.

Data sources:
  - FRED API: DRCRELEXBS (CRE loan delinquency rate),
               BAMLC0A4CBBB (BBB corporate spread as CMBS proxy)
  - Baseline benchmarks: Trepp, MSCI Real Capital Analytics Q1 2025
  - Groq LLM: market intelligence on distress opportunity/risk
"""

import os
import requests
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

CMBS_DELINQUENCY = {
    "Office":      {"rate_pct": 8.6,  "prior_year": 5.2, "trend": "rising",  "note": "WFH driving maturity defaults on pre-2020 vintage loans"},
    "Retail":      {"rate_pct": 7.1,  "prior_year": 6.8, "trend": "stable",  "note": "Stabilizing after COVID wave; enclosed malls still elevated"},
    "Multifamily": {"rate_pct": 3.2,  "prior_year": 1.8, "trend": "rising",  "note": "Sunbelt oversupply pushing value-add sponsors into distress"},
    "Industrial":  {"rate_pct": 0.4,  "prior_year": 0.3, "trend": "stable",  "note": "Near-zero delinquency — best performing CMBS collateral"},
    "Hotel":       {"rate_pct": 5.8,  "prior_year": 4.9, "trend": "rising",  "note": "Business travel uneven; extended-stay outperforming leisure"},
    "Mixed-Use":   {"rate_pct": 4.1,  "prior_year": 3.2, "trend": "rising",  "note": "Office component dragging otherwise stable mixed portfolios"},
}

DISTRESSED_PIPELINE = [
    {"asset": "1740 Broadway, NYC",            "type": "Office",      "loan_amount": 308_000_000,   "status": "REO",               "market": "New York, NY",    "opportunity": "Deep value repositioning / residential conversion"},
    {"asset": "Park Place Mall, Tucson AZ",    "type": "Retail",      "loan_amount":  89_000_000,   "status": "Special Servicing", "market": "Phoenix, AZ",     "opportunity": "Mixed-use redevelopment / anchor repositioning"},
    {"asset": "200 Paul Ave, San Francisco",   "type": "Office",      "loan_amount": 142_000_000,   "status": "Maturity Default",  "market": "Los Angeles, CA", "opportunity": "Data center / lab conversion"},
    {"asset": "One Market Plaza, SF",          "type": "Office",      "loan_amount": 975_000_000,   "status": "Special Servicing", "market": "Los Angeles, CA", "opportunity": "Trophy office at discount to replacement cost"},
    {"asset": "Parkmerced Apartments, SF",     "type": "Multifamily", "loan_amount": 1_500_000_000, "status": "Maturity Default",  "market": "Los Angeles, CA", "opportunity": "Large-scale MF at below-replacement-cost pricing"},
    {"asset": "111 Sutter St, SF",             "type": "Office",      "loan_amount":  97_000_000,   "status": "REO",               "market": "Los Angeles, CA", "opportunity": "Sold at 75% discount — residential conversion case study"},
    {"asset": "Southridge Mall, Greendale WI", "type": "Retail",      "loan_amount":  56_000_000,   "status": "Special Servicing", "market": "Chicago, IL",     "opportunity": "Partial demo / mixed-use repositioning"},
    {"asset": "Austin Domain Office Tower",    "type": "Office",      "loan_amount": 210_000_000,   "status": "Watchlist",         "market": "Austin, TX",      "opportunity": "Sublease overhang; potential conversion or deep discount"},
    {"asset": "Denver Tech Center Tower",      "type": "Office",      "loan_amount": 165_000_000,   "status": "Maturity Default",  "market": "Denver, CO",      "opportunity": "Class A office at 40-50% below peak valuation"},
    {"asset": "Forum at Carlsbad, CA",         "type": "Retail",      "loan_amount":  78_000_000,   "status": "Special Servicing", "market": "Los Angeles, CA", "opportunity": "Open-air retrofit / experiential anchor"},
]

DISTRESS_SIGNALS = {
    "office_maturity_wall":     {"amount_bn": 94.4, "peak_year": 2025, "trend": "rising",  "note": "$94B in office CMBS maturing 2024-2026 — largest distress wave since GFC"},
    "retail_special_servicing": {"rate_pct": 9.8,   "trend": "stable", "note": "Improving from 2021 peak; Class B/C malls still elevated"},
    "multifamily_watchlist":    {"rate_pct": 12.4,  "trend": "rising", "note": "Austin, Phoenix, Denver value-add watchlisted due to rate cap expirations"},
    "cre_loan_delinquency":     {"rate_pct": 4.8,   "prior_year": 3.1, "trend": "rising", "note": "Bank CRE delinquency rising but below GFC peak of 9.5%"},
    "cmbs_new_issuance":        {"amount_bn": 68.2, "prior_year": 44.1,"trend": "rising", "note": "New CMBS issuance recovering in 2025 as spreads tighten"},
}

TREND_ARROW = {"rising": "↑", "falling": "↓", "stable": "→"}
TREND_COLOR = {"rising": "#ef5350", "falling": "#66bb6a", "stable": "#CFB991"}
STATUS_COLOR = {
    "REO":               "#ef5350",
    "Maturity Default":  "#ef5350",
    "Special Servicing": "#CFB991",
    "Watchlist":         "#42a5f5",
    "Modified":          "#66bb6a",
}


def _fetch_fred_series(api_key: str, series_id: str, limit: int = 20) -> list:
    if not api_key:
        return []
    try:
        import urllib.parse
        params = {"series_id": series_id, "api_key": api_key,
                  "file_type": "json", "sort_order": "desc", "limit": str(limit)}
        r = requests.get(f"{FRED_BASE}?{urllib.parse.urlencode(params)}", timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return [{"date": o["date"], "value": float(o["value"])}
                for o in obs if o.get("value") not in (".", None, "")]
    except Exception:
        return []


def _groq_distressed_intel() -> dict:
    key = os.getenv("GROQ_API_KEY", "")
    fallback = {
        "summary": (
            "Office CMBS faces its largest maturity wall since the GFC — $94B in loans maturing "
            "2024-2026 against markets with 20%+ vacancy. Opportunity: acquire distressed office "
            "at 30-60 cents on the dollar for residential/lab/data center conversion. Multifamily "
            "distress is emerging in Sunbelt markets as floating-rate caps expire on 2021-2022 "
            "vintage acquisitions. Industrial CMBS remains near-zero delinquency."
        ),
        "top_opportunity": "Distressed office-to-residential conversion in gateway cities — 30-60% discount to 2019 peak values",
        "key_risk": "Sunbelt multifamily rate cap expirations — $40B+ in floating-rate loans face significant payment shock in 2025",
        "groq_used": False,
    }
    if not key:
        return fallback
    try:
        from groq import Groq
        client = Groq(api_key=key)
        prompt = (
            f"Today is {datetime.now().strftime('%B %d, %Y')}.\n\n"
            "You are a CMBS and distressed CRE analyst. Write a concise brief (3-4 sentences) on: "
            "where CMBS delinquency is most concentrated, best acquisition opportunities in "
            "distressed CRE, and key triggers to watch. Also provide top_opportunity (one sentence) "
            "and key_risk (one sentence). Respond as valid JSON with keys: summary, top_opportunity, key_risk. "
            "Raw JSON only."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Return only valid JSON."},
                      {"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.3,
        )
        import json as _j
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = _j.loads(raw.strip())
        result["groq_used"] = True
        return result
    except Exception as e:
        fallback["error"] = str(e)
        return fallback


def run_distressed_asset_agent() -> dict:
    api_key         = os.getenv("FRED_API_KEY", "")
    cre_delinquency = _fetch_fred_series(api_key, "DRCRELEXBS", 20)
    bbb_spread      = _fetch_fred_series(api_key, "BAMLC0A4CBBB", 20)
    market_intel    = _groq_distressed_intel()

    return {
        "cmbs_delinquency":     CMBS_DELINQUENCY,
        "distressed_pipeline":  DISTRESSED_PIPELINE,
        "distress_signals":     DISTRESS_SIGNALS,
        "market_intelligence":  market_intel,
        "fred_cre_delinquency": cre_delinquency,
        "fred_bbb_spread":      bbb_spread,
        "current_delinq_rate":  cre_delinquency[0]["value"] if cre_delinquency else None,
        "current_bbb_spread":   bbb_spread[0]["value"] if bbb_spread else None,
        "fetched_at":           datetime.now().isoformat(),
        "data_as_of":           "Q1 2025",
    }
