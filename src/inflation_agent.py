"""
Agent 11 · Inflation & Construction Costs
==========================================
Tracks consumer inflation, rent inflation, producer prices, and market-implied
inflation expectations to assess their impact on CRE valuations and development costs.

FRED Series:
  CPIAUCSL        — CPI All Items (monthly, YoY computed)
  CPILFESL        — Core CPI (less food & energy)
  CUSR0000SAH1    — CPI Shelter
  CUSR0000SEHA    — CPI Rent of Primary Residence
  PPIACO          — PPI All Commodities (proxy for input cost pressure)
  PCUOMFGOMFG     — PPI Manufacturing (construction input proxy)
  T5YIE           — 5-Year Breakeven Inflation (market expectation)
  T10YIE          — 10-Year Breakeven Inflation
  MICH            — U of Michigan 1-Year Inflation Expectations

CRE Implications:
  High CPI Shelter / Rent Inflation → Multifamily rent growth; supports NOI
  High PPI                          → Rising replacement cost; supports asset values
  High breakeven inflation          → Cap rate pressure as real yields stay elevated
  Core CPI falling                  → Fed rate cut path opens → cap rate relief

Cache: cache/inflation_data.json
Schedule: every 6 hours
"""

import json
import os
import urllib.parse
import requests as _req
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from pathlib import Path as _P; load_dotenv(_P(__file__).parent.parent / ".env", override=True)

CACHE_DIR    = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = [
    ("CPIAUCSL",      "CPI All Items",              "%idx", 1.0),
    ("CPILFESL",      "Core CPI",                   "%idx", 1.0),
    ("CUSR0000SAH1",  "CPI Shelter",                "%idx", 1.0),
    ("CUSR0000SEHA",  "CPI Rent",                   "%idx", 1.0),
    ("PPIACO",        "PPI All Commodities",         "idx",  1.0),
    ("PCUOMFGOMFG",   "PPI Manufacturing",           "idx",  1.0),
    ("T5YIE",         "5Y Breakeven Inflation",      "%",    1.0),
    ("T10YIE",        "10Y Breakeven Inflation",     "%",    1.0),
    ("MICH",          "1Y Inflation Expectations",   "%",    1.0),
]


def _fred_fetch(series_id: str, lookback_years: int = 3, retries: int = 3) -> list[dict]:
    if not FRED_API_KEY:
        return []
    import time
    start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start,
        "sort_order":        "asc",
    })
    for attempt in range(retries):
        try:
            resp = _req.get(f"{FRED_BASE}?{params}", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return [{"date": o["date"], "value": float(o["value"])}
                    for o in data.get("observations", []) if o["value"] not in (".", "")]
        except Exception as e:
            print(f"  [InflationAgent] FRED error {series_id} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


def _compute_yoy(series: list[dict]) -> float | None:
    """Compute YoY % change for index series (CPI, PPI)."""
    if len(series) < 12:
        return None
    latest   = series[-1]["value"]
    year_ago = series[-12]["value"] if len(series) >= 12 else None
    if year_ago and year_ago != 0:
        return round((latest - year_ago) / year_ago * 100, 2)
    return None


def _summarize(series: list[dict], unit: str, compute_yoy: bool = False) -> dict:
    if not series:
        return {"current": None, "yoy_pct": None, "delta_1m": None, "delta_1y": None,
                "series": [], "unit": unit}
    current = series[-1]["value"]
    today   = datetime.fromisoformat(series[-1]["date"])
    d1m = next((o["value"] for o in reversed(series)
                if (today - datetime.fromisoformat(o["date"])).days >= 25), None)
    d1y = next((o["value"] for o in series
                if (today - datetime.fromisoformat(o["date"])).days >= 330), None)
    yoy = _compute_yoy(series) if compute_yoy else None
    return {
        "current":  round(current, 3),
        "yoy_pct":  yoy,
        "delta_1m": round(current - d1m, 3) if d1m is not None else None,
        "delta_1y": round(current - d1y, 3) if d1y is not None else None,
        "series":   series[-24:],
        "unit":     unit,
    }


def _classify_inflation(data: dict) -> dict:
    """Classify the inflation environment and CRE implications."""
    score = 50
    bullets = []

    # Core CPI trend
    core = data.get("Core CPI", {})
    core_yoy = core.get("yoy_pct")
    if core_yoy is not None:
        if core_yoy > 3.5:
            score += 15
            bullets.append(f"Core CPI elevated at {core_yoy:.1f}% YoY — Fed rate cuts delayed, cap rate pressure persists")
        elif core_yoy > 2.5:
            score += 5
            bullets.append(f"Core CPI at {core_yoy:.1f}% YoY — above Fed 2% target, policy still restrictive")
        else:
            score -= 10
            bullets.append(f"Core CPI at {core_yoy:.1f}% YoY — near target, rate relief path opening")

    # Shelter/Rent CPI — positive for multifamily NOI
    shelter = data.get("CPI Shelter", {})
    rent    = data.get("CPI Rent", {})
    shelter_yoy = shelter.get("yoy_pct")
    rent_yoy    = rent.get("yoy_pct")
    if shelter_yoy and shelter_yoy > 4:
        bullets.append(f"CPI Shelter at {shelter_yoy:.1f}% YoY — strong multifamily rent growth environment")
    if rent_yoy and rent_yoy > 4:
        bullets.append(f"CPI Rent at {rent_yoy:.1f}% YoY — landlord pricing power intact")

    # PPI — construction cost pressure
    ppi = data.get("PPI All Commodities", {})
    ppi_yoy = ppi.get("yoy_pct")
    if ppi_yoy is not None:
        if ppi_yoy > 5:
            score += 5
            bullets.append(f"PPI at {ppi_yoy:.1f}% YoY — rising replacement costs support existing asset values")
        elif ppi_yoy < 0:
            score -= 5
            bullets.append(f"PPI declining ({ppi_yoy:.1f}% YoY) — easing construction cost pressure")

    # Breakeven inflation (market expectations)
    be5 = data.get("5Y Breakeven Inflation", {}).get("current")
    be10 = data.get("10Y Breakeven Inflation", {}).get("current")
    if be5 and be10:
        bullets.append(f"Market inflation expectations: 5Y={be5:.2f}%, 10Y={be10:.2f}%")
        if be5 > 2.5:
            score += 5
            bullets.append("Breakeven inflation elevated — real yields stay high, weighing on cap rate multiples")

    score = max(0, min(100, score))
    if score >= 65:
        label = "HOT"
        summary = "High inflation pressuring real returns. Rising replacement costs support existing asset values but cap rate relief is delayed."
    elif score >= 40:
        label = "MODERATE"
        summary = "Inflation moderating toward Fed target. Balanced environment — shelter inflation supports multifamily while input costs ease."
    else:
        label = "COOLING"
        summary = "Inflation cooling. Rate cut path opening. Cap rate compression likely — favorable for existing CRE owners."

    return {
        "label":   label,
        "score":   score,
        "bullets": bullets,
        "summary": summary,
    }


def run_inflation_agent() -> dict:
    print("=" * 60)
    print("[InflationAgent] Starting run ...")
    print("=" * 60)

    series_data = {}
    # CPI/PPI index series need YoY computed; breakevens/expectations are already %
    yoy_series = {"CPI All Items", "Core CPI", "CPI Shelter", "CPI Rent",
                  "PPI All Commodities", "PPI Manufacturing"}
    for series_id, label, unit, _ in FRED_SERIES:
        print(f"  -> {label}")
        raw = _fred_fetch(series_id)
        series_data[label] = _summarize(raw, unit, compute_yoy=(label in yoy_series))

    signal = _classify_inflation(series_data)
    print(f"[InflationAgent] Inflation Signal: {signal['label']} (score {signal['score']})")
    print("=" * 60)

    return {
        "series":    series_data,
        "signal":    signal,
        "cached_at": datetime.now().isoformat(),
        "error":     None,
    }


if __name__ == "__main__":
    result = run_inflation_agent()
    for k, v in result["series"].items():
        yoy = f"  YoY: {v['yoy_pct']:+.2f}%" if v.get("yoy_pct") is not None else ""
        print(f"  {k:35s}  {v.get('current')}{yoy}")
    print(f"\n  Inflation Signal: {result['signal']['label']} (score {result['signal']['score']})")
