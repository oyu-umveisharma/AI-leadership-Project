"""
Agent 10 · GDP & Economic Growth
==================================
Tracks real GDP growth, industrial production, retail sales, consumer sentiment,
and the Conference Board Leading Index to gauge the economic cycle phase
and its implications for CRE demand.

FRED Series:
  GDPC1           — Real GDP (quarterly, chained 2017 $B)
  A191RL1Q225SBEA — Real GDP growth rate % (quarterly, annualized)
  INDPRO          — Industrial Production Index (monthly)
  RSAFS           — Advance Retail & Food Services Sales (monthly, $M)
  UMCSENT         — U of Michigan Consumer Sentiment (monthly)
  USSLIND         — US Leading Index (monthly)
  PCEC96          — Real Personal Consumption Expenditures (monthly)
  DGORDER         — Durable Goods Orders (monthly, $M)

CRE Cycle Signal:
  Expansion  (GDP > 2%, IPI rising, sentiment > 80) → All property types benefit
  Slowdown   (GDP 0–2%, mixed signals)              → Caution; favor defensive types
  Contraction (GDP < 0% or 2 neg quarters)          → Industrial/multifamily defensive

Cache: cache/gdp_data.json
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
    ("GDPC1",            "Real GDP",                     "$B",  1.0),
    ("A191RL1Q225SBEA",  "Real GDP Growth Rate",         "%",   1.0),
    ("INDPRO",           "Industrial Production Index",  "idx", 1.0),
    ("RSAFS",            "Retail Sales",                 "$M",  1.0),
    ("UMCSENT",          "Consumer Sentiment",           "idx", 1.0),
    ("CFNAIMA3",         "Chicago Fed Activity Index",   "idx", 1.0),
    ("PCEC96",           "Real PCE",                     "$B",  1.0),
    ("DGORDER",          "Durable Goods Orders",         "$M",  1.0),
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
            print(f"  [GDPAgent] FRED error {series_id} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


def _summarize(series: list[dict], unit: str) -> dict:
    if not series:
        return {"current": None, "delta_1q": None, "delta_1y": None, "series": [], "unit": unit}
    current = series[-1]["value"]
    today   = datetime.fromisoformat(series[-1]["date"])
    d1q = next((o["value"] for o in reversed(series)
                if (today - datetime.fromisoformat(o["date"])).days >= 80), None)
    d1y = next((o["value"] for o in series
                if (today - datetime.fromisoformat(o["date"])).days >= 330), None)
    return {
        "current":  round(current, 3),
        "delta_1q": round(current - d1q, 3) if d1q is not None else None,
        "delta_1y": round(current - d1y, 3) if d1y is not None else None,
        "series":   series[-12:],
        "unit":     unit,
    }


def _classify_cycle(data: dict) -> dict:
    """
    Classify the economic cycle phase based on available indicators.
    Returns label, score (0-100), bullets.
    """
    score = 50
    bullets = []

    gdp_growth = data.get("Real GDP Growth Rate", {}).get("current")
    if gdp_growth is not None:
        if gdp_growth >= 2.5:
            score += 20; bullets.append(f"GDP expanding strongly at {gdp_growth:.1f}% annualized")
        elif gdp_growth >= 1.0:
            score += 8;  bullets.append(f"GDP growing moderately at {gdp_growth:.1f}%")
        elif gdp_growth >= 0:
            score -= 5;  bullets.append(f"GDP growth slowing to {gdp_growth:.1f}%")
        else:
            score -= 20; bullets.append(f"GDP contracting at {gdp_growth:.1f}% — recession risk elevated")

    ipi = data.get("Industrial Production Index", {})
    if ipi.get("delta_1q") is not None:
        if ipi["delta_1q"] > 1:
            score += 10; bullets.append("Industrial production rising — positive for logistics/industrial CRE")
        elif ipi["delta_1q"] < -1:
            score -= 8;  bullets.append("Industrial production falling — watch industrial vacancy")

    sentiment = data.get("Consumer Sentiment", {}).get("current")
    if sentiment is not None:
        if sentiment >= 85:
            score += 8;  bullets.append(f"Consumer sentiment strong ({sentiment:.0f}) — supportive of retail")
        elif sentiment <= 65:
            score -= 8;  bullets.append(f"Consumer sentiment weak ({sentiment:.0f}) — retail headwinds")

    cfnai = data.get("Chicago Fed Activity Index", {})
    if cfnai.get("current") is not None:
        if cfnai["current"] > 0.2:
            score += 5;  bullets.append(f"Chicago Fed Activity Index at {cfnai['current']:.2f} — above-trend growth")
        elif cfnai["current"] < -0.7:
            score -= 10; bullets.append(f"Chicago Fed Activity Index at {cfnai['current']:.2f} — recession signal")
        else:
            bullets.append(f"Chicago Fed Activity Index at {cfnai['current']:.2f} — near trend growth")

    score = max(0, min(100, score))
    if score >= 65:
        label, icon = "EXPANSION", ""
    elif score >= 45:
        label, icon = "SLOWDOWN", ""
    else:
        label, icon = "CONTRACTION", ""

    cre_implication = {
        "EXPANSION":   "All CRE types benefit. Prioritize growth markets: industrial, multifamily, office in tight-labor metros.",
        "SLOWDOWN":    "Favor defensive property types: essential retail, industrial, suburban multifamily. Monitor office vacancy.",
        "CONTRACTION": "Defensive posture. Multifamily and necessity retail hold up best. Industrial may soften on lower trade volumes.",
    }[label]

    return {
        "label":           label,
        "icon":            icon,
        "score":           score,
        "bullets":         bullets,
        "cre_implication": cre_implication,
    }


def run_gdp_agent() -> dict:
    print("=" * 60)
    print("[GDPAgent] Starting run ...")
    print("=" * 60)

    series_data = {}
    for series_id, label, unit, _ in FRED_SERIES:
        print(f"  -> {label}")
        raw = _fred_fetch(series_id)
        series_data[label] = _summarize(raw, unit)

    cycle = _classify_cycle(series_data)
    print(f"[GDPAgent] Economic Cycle: {cycle['label']} (score {cycle['score']})")
    print("=" * 60)

    return {
        "series":     series_data,
        "cycle":      cycle,
        "cached_at":  datetime.now().isoformat(),
        "error":      None,
    }


if __name__ == "__main__":
    result = run_gdp_agent()
    for k, v in result["series"].items():
        print(f"  {k:35s}  {v.get('current')}  (1y Δ: {v.get('delta_1y')})")
    print(f"\n  Cycle: {result['cycle']['label']} (score {result['cycle']['score']})")
