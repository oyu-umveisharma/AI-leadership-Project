"""
Agent 22 · Economic Forecast Agent
Forward-looking macro projections used by the Investment Advisor.

Series tracked (FRED):
  GDPNOW   — Atlanta Fed GDPNow real-GDP nowcast (updated multiple times per quarter)
  GDPC1MD  — FOMC Summary of Economic Projections: Real GDP, median
  T10YIE   — 10-Year Breakeven Inflation Rate (daily, market-implied)
  FEDTARMD — FOMC SEP: Federal Funds Rate, median projection

Produces Q2 / Q3 / Q4 2026 projections alongside a historical window
suitable for a solid-historical + dotted-forecast line chart.

Projection model:
  • FOMC series (GDPC1MD, FEDTARMD) are already forward-looking — hold flat at current.
  • Market series (GDPNOW, T10YIE) extend via a 4-period rolling drift.
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime
from pathlib import Path

import requests as _requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (series_id, display_name, unit, multiply_factor, treat_as_fomc)
FORECAST_SERIES = [
    ("GDPNOW",   "GDP Nowcast (Atlanta Fed)",   "%", 1.0, False),
    ("GDPC1MD",  "GDP Projection (FOMC)",        "%", 1.0, True),
    ("T10YIE",   "10Y Breakeven Inflation",      "%", 1.0, False),
    ("FEDTARMD", "Fed Funds Projection (FOMC)",  "%", 1.0, True),
]

PROJECTION_QUARTERS = ["Q2 2026", "Q3 2026", "Q4 2026"]


def _load_fred_key() -> str:
    """Load FRED API key from environment or the project .env."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()
    return os.getenv("FRED_API_KEY", "")


def _fetch_fred(series_id: str, api_key: str, limit: int = 200) -> list[dict]:
    params = {
        "series_id":  series_id,
        "api_key":    api_key,
        "file_type":  "json",
        "limit":      limit,
        "sort_order": "desc",
    }
    try:
        r = _requests.get(FRED_BASE, params=params, timeout=12)
        r.raise_for_status()
        obs = [
            {"date": o["date"], "value": float(o["value"])}
            for o in r.json().get("observations", [])
            if o.get("value") not in (".", "", None)
        ]
        return sorted(obs, key=lambda x: x["date"])   # oldest → newest
    except Exception:
        return []


def _rolling_drift(values: list[float], window: int = 4) -> float:
    """Drift per step, estimated from two adjacent rolling windows."""
    if len(values) < window * 2:
        return 0.0
    earlier = statistics.mean(values[-window * 2:-window])
    recent  = statistics.mean(values[-window:])
    return (recent - earlier) / float(window)


def run_forecast_agent() -> dict:
    """Entry point. Returns a dict with projections + historical windows."""
    api_key = _load_fred_key()
    if not api_key:
        return {
            "error": "FRED_API_KEY not set. Add to .env to enable forecasting.",
            "projections": {},
            "historical":  {},
            "quarters":    PROJECTION_QUARTERS,
            "generated_at": datetime.now().isoformat(),
        }

    projections: dict[str, dict] = {}
    historical:  dict[str, dict] = {}

    for series_id, display_name, unit, mul, is_fomc in FORECAST_SERIES:
        observations = _fetch_fred(series_id, api_key, limit=200)
        if not observations:
            continue

        scaled = [{"date": o["date"], "value": o["value"] * mul} for o in observations]
        current = scaled[-1]["value"]
        values  = [s["value"] for s in scaled]

        if is_fomc:
            # FOMC projections are already year-end forecasts — hold flat
            drift = 0.0
            forecast_values = [current, current, current]
        else:
            drift = _rolling_drift(values, window=4)
            forecast_values = [round(current + drift * n, 2) for n in (1, 2, 3)]

        projections[display_name] = {
            "series_id":         series_id,
            "unit":              unit,
            "current":           round(current, 2),
            "q2_2026":           forecast_values[0],
            "q3_2026":           forecast_values[1],
            "q4_2026":           forecast_values[2],
            "drift_per_quarter": round(drift, 3),
            "is_fomc":           is_fomc,
        }

        # Keep last 60 observations for the historical half of the chart
        historical[display_name] = {
            "series_id": series_id,
            "unit":      unit,
            "points":    [
                {"date": s["date"], "value": round(s["value"], 2)}
                for s in scaled[-60:]
            ],
        }

    return {
        "projections":  projections,
        "historical":   historical,
        "quarters":     PROJECTION_QUARTERS,
        "generated_at": datetime.now().isoformat(),
        "error":        None,
    }


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(run_forecast_agent(), indent=2, default=str))
