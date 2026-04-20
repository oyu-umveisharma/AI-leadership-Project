"""
Agent 9 · Labor Market & Tenant Demand
=======================================
Part of the Human-in-Command AI Workforce for the CRE Intelligence Platform.

Answers: Which metros have the strongest job growth → highest tenant demand?
         Which industries are expanding → driving leasing activity by property type?

Data Sources:
  - BLS API       : unemployment rate, nonfarm payrolls, employment by sector (national + state)
  - FRED API      : job openings (JOLTS), quits rate, layoffs, labor force participation
  - Census / ACS  : commuter flows (used for metro demand weighting)
  - yfinance      : sector ETFs as tenant-demand proxies (tech, finance, industrial, healthcare)
  - Hiring signals: large-employer hiring via sector ETF momentum + BLS sector data

CRE Property Type → Demand Driver mapping:
  Office          : Finance, Professional Services, Tech, Information
  Industrial      : Manufacturing, Trade/Transport/Warehousing
  Retail          : Leisure/Hospitality, Retail Trade
  Multifamily     : Total nonfarm (population proxy), wage growth
  Healthcare/Life : Education & Health Services
  Data Centers    : Information, Tech

Cache: cache/labor_market.json
Schedule: every 6 hours
"""

import json
import os
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
import requests as _req

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from pathlib import Path as _P; load_dotenv(_P(__file__).parent.parent / ".env", override=True)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

# ── BLS API ───────────────────────────────────────────────────────────────────
BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# FRED series for labor market
FRED_LABOR_SERIES = [
    ("UNRATE",    "Unemployment Rate",          "%",   1.0),
    ("PAYEMS",    "Nonfarm Payrolls",            "K",   1.0),
    ("JTSJOL",    "Job Openings (JOLTS)",        "K",   1.0),
    ("JTSQUR",    "Quits Rate",                 "%",   1.0),
    ("CIVPART",   "Labor Force Participation",  "%",   1.0),
    ("CES0500000003", "Avg Hourly Earnings",       "$",   1.0),
    ("LNS14000006","Unemployment 25-54 (Prime)", "%",  1.0),
]

# Sector ETFs as tenant-demand proxies
SECTOR_ETFS = {
    "XLK":  {"label": "Technology (XLK)",              "property_type": "Office / Data Center"},
    "XLF":  {"label": "Financials (XLF)",              "property_type": "Office"},
    "XLV":  {"label": "Healthcare (XLV)",              "property_type": "Healthcare / Life Science"},
    "XLI":  {"label": "Industrials (XLI)",             "property_type": "Industrial / Logistics"},
    "XLRE": {"label": "Real Estate (XLRE)",            "property_type": "All CRE"},
    "XLP":  {"label": "Consumer Staples (XLP)",        "property_type": "Retail / Grocery"},
    "XLY":  {"label": "Consumer Discretionary (XLY)", "property_type": "Retail / Mixed-Use"},
    "XLC":  {"label": "Communication Svcs (XLC)",     "property_type": "Data Center / Office"},
}

# BLS supersector codes for sector-level payrolls (national)
BLS_SUPERSECTORS = {
    "CEU0500000001": {"label": "Total Private",               "property_type": "All"},
    "CEU1000000001": {"label": "Mining & Logging",            "property_type": "Industrial"},
    "CEU2000000001": {"label": "Construction",                "property_type": "Industrial"},
    "CEU3000000001": {"label": "Manufacturing",               "property_type": "Industrial / Logistics"},
    "CEU4000000001": {"label": "Trade, Transport & Utilities","property_type": "Industrial / Retail"},
    "CEU5000000001": {"label": "Information",                 "property_type": "Office / Data Center"},
    "CEU5500000001": {"label": "Financial Activities",        "property_type": "Office"},
    "CEU6000000001": {"label": "Professional & Business Svcs","property_type": "Office"},
    "CEU6500000001": {"label": "Education & Health Services", "property_type": "Healthcare / Life Science"},
    "CEU7000000001": {"label": "Leisure & Hospitality",       "property_type": "Retail / Hospitality"},
}

# Top migration-destination states mapped to FRED state unemployment rate series
# These are state-level proxies for the key CRE metro markets
METRO_UNEMPLOYMENT = {
    "Texas (Austin / Dallas / Houston)": "TXUR",
    "Florida (Tampa / Jacksonville / Miami)": "FLUR",
    "Arizona (Phoenix)":                 "AZUR",
    "Tennessee (Nashville)":             "TNUR",
    "North Carolina (Charlotte / Raleigh)": "NCUR",
    "Georgia (Atlanta)":                 "GAUR",
    "Colorado (Denver)":                 "COUR",
    "Nevada (Las Vegas)":                "NVUR",
    "South Carolina (Greenville)":       "SCUR",
    "Utah (Salt Lake City)":             "UTUR",
}


def _fred_fetch(series_id: str, lookback_years: int = 2) -> list[dict]:
    """Fetch a FRED series and return list of {date, value} dicts."""
    if not FRED_API_KEY:
        return []
    start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "series_id":        series_id,
        "api_key":          FRED_API_KEY,
        "file_type":        "json",
        "observation_start": start,
        "sort_order":       "asc",
    })
    try:
        r = _req.get(f"{FRED_BASE}?{params}", timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return [{"date": o["date"], "value": float(o["value"])}
                for o in obs if o["value"] not in (".", "")]
    except Exception as e:
        print(f"  [LaborAgent] FRED error for {series_id}: {e}")
        return []


def _latest_and_delta(series: list[dict]) -> dict:
    """Return current value and 1m/1y deltas from a series."""
    if not series:
        return {"current": None, "delta_1m": None, "delta_1y": None, "series": []}
    current = series[-1]["value"]
    today   = datetime.fromisoformat(series[-1]["date"])
    d1m = next((o["value"] for o in reversed(series)
                if (today - datetime.fromisoformat(o["date"])).days >= 28), None)
    d1y = next((o["value"] for o in series
                if (today - datetime.fromisoformat(o["date"])).days >= 330), None)
    return {
        "current":   round(current, 3),
        "delta_1m":  round(current - d1m, 3) if d1m is not None else None,
        "delta_1y":  round(current - d1y, 3) if d1y is not None else None,
        "series":    series[-24:],  # last 24 observations for sparklines
    }


def fetch_fred_labor() -> dict:
    """Fetch all FRED labor series."""
    print("[LaborAgent] Fetching FRED labor series ...")
    result = {}
    for series_id, label, unit, _ in FRED_LABOR_SERIES:
        print(f"  -> {label}")
        series = _fred_fetch(series_id)
        r = _latest_and_delta(series)
        r["label"] = label
        r["unit"]  = unit
        result[label] = r
    return result


def fetch_bls_sectors() -> list[dict]:
    """
    Fetch BLS national employment by supersector.
    Uses public API (no key required for v1, limited to 25 series).
    """
    print("[LaborAgent] Fetching BLS sector payrolls ...")
    series_ids = list(BLS_SUPERSECTORS.keys())
    payload = {
        "seriesid": series_ids,
        "startyear": str(datetime.now().year - 1),
        "endyear":   str(datetime.now().year),
    }
    rows = []
    try:
        resp = _req.post(BLS_BASE, json=payload,
                         headers={"Content-Type": "application/json"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for series in data.get("Results", {}).get("series", []):
            sid   = series["seriesID"]
            meta  = BLS_SUPERSECTORS.get(sid, {})
            pts   = series.get("data", [])
            if not pts:
                continue
            # BLS returns newest first
            latest_pt  = pts[0]
            prev_pt    = pts[1] if len(pts) > 1 else pts[0]
            latest_val = float(latest_pt["value"].replace(",", ""))
            prev_val   = float(prev_pt["value"].replace(",", ""))
            mom        = round((latest_val - prev_val) / prev_val * 100, 2) if prev_val else 0
            rows.append({
                "series_id":     sid,
                "label":         meta.get("label", sid),
                "property_type": meta.get("property_type", ""),
                "employment_k":  round(latest_val, 1),
                "mom_pct":       mom,
                "period":        f"{latest_pt['periodName']} {latest_pt['year']}",
            })
    except Exception as e:
        print(f"  [LaborAgent] BLS API error: {e}")
        # Fallback: return empty list — page will handle gracefully
    return rows


def fetch_sector_etfs() -> list[dict]:
    """
    Fetch sector ETF 6-month returns as tenant-demand momentum signals.
    Rising sector ETF → expanding employment → rising leasing demand.
    """
    print("[LaborAgent] Fetching sector ETF momentum ...")
    rows = []
    for ticker, meta in SECTOR_ETFS.items():
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if hist.empty:
                continue
            start  = float(hist["Close"].iloc[0])
            latest = float(hist["Close"].iloc[-1])
            ret    = round((latest - start) / start * 100, 2)
            sma60  = float(hist["Close"].rolling(60, min_periods=20).mean().iloc[-1])
            vs_sma = round((latest - sma60) / sma60 * 100, 2) if sma60 else 0.0
            rows.append({
                "ticker":        ticker,
                "label":         meta["label"],
                "property_type": meta["property_type"],
                "latest_price":  round(latest, 2),
                "return_6mo":    ret,
                "sma_60":        round(sma60, 2),
                "pct_vs_sma":    vs_sma,
                "signal":        "EXPANDING" if ret > 2 else ("CONTRACTING" if ret < -2 else "FLAT"),
            })
        except Exception as e:
            print(f"  [LaborAgent] ETF error {ticker}: {e}")
    return sorted(rows, key=lambda x: x["return_6mo"], reverse=True)


def fetch_metro_unemployment() -> list[dict]:
    """
    Fetch unemployment rates for top CRE destination states via FRED.
    Uses state-level FRED series as proxies for key migration-destination metros.
    """
    print("[LaborAgent] Fetching state unemployment rates (CRE destination markets) ...")
    rows = []
    for label, series_id in METRO_UNEMPLOYMENT.items():
        series = _fred_fetch(series_id, lookback_years=1)
        if not series:
            continue
        current = series[-1]["value"]
        prev    = series[-2]["value"] if len(series) > 1 else current
        rows.append({
            "market":      label,
            "unemp_rate":  round(current, 1),
            "delta_1m":    round(current - prev, 1),
            "period":      series[-1]["date"],
            "signal":      "TIGHT" if current < 4.0 else ("LOOSE" if current > 6.0 else "BALANCED"),
        })
    return sorted(rows, key=lambda x: x["unemp_rate"])


def derive_demand_signal(fred_data: dict, sector_etfs: list[dict]) -> dict:
    """
    Synthesize a Tenant Demand Signal from labor market inputs.

    Logic:
      - Payroll growth MoM > 0      → positive
      - Job openings high / rising  → positive
      - Unemployment falling        → positive
      - Sector ETFs net positive    → positive
    Score 0-100; classify STRONG / MODERATE / SOFT.
    """
    score = 50  # neutral base

    payrolls = fred_data.get("Nonfarm Payrolls", {})
    if payrolls.get("delta_1m") is not None:
        if payrolls["delta_1m"] > 100:   score += 15
        elif payrolls["delta_1m"] > 0:   score += 8
        else:                            score -= 10

    openings = fred_data.get("Job Openings (JOLTS)", {})
    if openings.get("current") is not None:
        if openings["current"] > 8000:   score += 10
        elif openings["current"] > 6000: score += 5
        else:                            score -= 5

    unemp = fred_data.get("Unemployment Rate", {})
    if unemp.get("delta_1m") is not None:
        if unemp["delta_1m"] < -0.1:     score += 10
        elif unemp["delta_1m"] > 0.1:    score -= 8

    if sector_etfs:
        expanding = sum(1 for e in sector_etfs if e["signal"] == "EXPANDING")
        contracting = sum(1 for e in sector_etfs if e["signal"] == "CONTRACTING")
        score += (expanding - contracting) * 3

    score = max(0, min(100, score))
    label = "STRONG" if score >= 65 else ("SOFT" if score <= 40 else "MODERATE")

    return {"score": score, "label": label}


def run_labor_market_agent() -> dict:
    """Main entry point — fetch, analyze, return data dict."""
    print("=" * 60)
    print("[LaborAgent] Starting run ...")
    print("=" * 60)

    fred_data    = fetch_fred_labor()
    bls_sectors  = fetch_bls_sectors()
    sector_etfs  = fetch_sector_etfs()
    metro_unemp  = fetch_metro_unemployment()
    demand_signal = derive_demand_signal(fred_data, sector_etfs)

    data = {
        "fred_labor":      fred_data,
        "bls_sectors":     bls_sectors,
        "sector_etfs":     sector_etfs,
        "metro_unemployment": metro_unemp,
        "demand_signal":   demand_signal,
        "cached_at":       datetime.now().isoformat(),
        "error":           None,
    }

    print(f"[LaborAgent] Tenant Demand Signal: {demand_signal['label']} (score {demand_signal['score']})")
    print("=" * 60)
    return data


if __name__ == "__main__":
    result = run_labor_market_agent()
    print("\n--- National Labor ---")
    for k, v in result["fred_labor"].items():
        cur = v.get("current")
        d1m = v.get("delta_1m")
        print(f"  {k:35s}  {cur}  (1m: {d1m:+.3f})" if d1m is not None else f"  {k:35s}  {cur}")
    print("\n--- Sector ETF Demand Signals ---")
    for e in result["sector_etfs"]:
        print(f"  {e['label']:35s}  {e['return_6mo']:+.1f}%  [{e['signal']}]  → {e['property_type']}")
    print(f"\n  Tenant Demand Signal: {result['demand_signal']['label']} (score {result['demand_signal']['score']})")
