"""
Agent · CRE Rent Growth Tracker
Fetches CPI shelter indices from FRED and combines them with static
Zillow / CoStar Q1 2025 benchmarks to surface rent-growth trends across
multifamily and commercial property types at both the national and
major-market level.

Series tracked:
  CUSR0000SEHA  — CPI Rent of Primary Residence (last 24 obs)
  CUSR0000SEHA2 — CPI Owners' Equivalent Rent of Residences (last 24 obs)

Benchmarks sourced from Zillow Research / CoStar Q1 2025 reports.
"""

import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Market-level rent growth YoY % (Q1 2025, Zillow/CoStar benchmarks)
MARKET_RENT_GROWTH = {
    "Austin, TX":       {"multifamily": -3.2, "industrial_psf": 8.4,  "office_psf": -2.1, "retail_psf": 1.2},
    "Dallas, TX":       {"multifamily": -1.8, "industrial_psf": 9.1,  "office_psf": -1.4, "retail_psf": 1.8},
    "Houston, TX":      {"multifamily": -0.9, "industrial_psf": 7.8,  "office_psf": -1.8, "retail_psf": 1.4},
    "Phoenix, AZ":      {"multifamily": -2.4, "industrial_psf": 10.2, "office_psf": -0.8, "retail_psf": 2.1},
    "Nashville, TN":    {"multifamily": -1.2, "industrial_psf": 8.6,  "office_psf":  0.4, "retail_psf": 2.4},
    "Charlotte, NC":    {"multifamily":  1.4, "industrial_psf": 9.8,  "office_psf":  0.2, "retail_psf": 2.8},
    "Atlanta, GA":      {"multifamily": -0.6, "industrial_psf": 8.2,  "office_psf": -1.2, "retail_psf": 1.6},
    "Denver, CO":       {"multifamily": -2.8, "industrial_psf": 6.4,  "office_psf": -2.4, "retail_psf": 0.8},
    "Las Vegas, NV":    {"multifamily":  2.1, "industrial_psf": 11.4, "office_psf":  1.8, "retail_psf": 3.2},
    "Raleigh, NC":      {"multifamily":  1.8, "industrial_psf": 10.6, "office_psf":  1.2, "retail_psf": 3.0},
    "Tampa, FL":        {"multifamily": -0.4, "industrial_psf": 9.2,  "office_psf": -0.6, "retail_psf": 2.6},
    "Orlando, FL":      {"multifamily":  0.8, "industrial_psf": 8.8,  "office_psf": -0.2, "retail_psf": 2.2},
    "Indianapolis, IN": {"multifamily":  2.4, "industrial_psf": 12.1, "office_psf":  0.6, "retail_psf": 1.8},
    "Los Angeles, CA":  {"multifamily":  3.2, "industrial_psf": 5.4,  "office_psf": -3.8, "retail_psf": -0.4},
    "Seattle, WA":      {"multifamily":  1.6, "industrial_psf": 6.2,  "office_psf": -4.2, "retail_psf": 0.6},
    "Chicago, IL":      {"multifamily":  2.8, "industrial_psf": 7.6,  "office_psf": -2.8, "retail_psf": 0.2},
    "New York, NY":     {"multifamily":  4.6, "industrial_psf": 4.8,  "office_psf": -1.6, "retail_psf": -1.2},
    "Boston, MA":       {"multifamily":  3.8, "industrial_psf": 5.8,  "office_psf": -0.8, "retail_psf": 1.0},
    "Miami, FL":        {"multifamily":  5.2, "industrial_psf": 7.2,  "office_psf":  2.4, "retail_psf": 4.8},
}

# National rent growth benchmarks
NATIONAL_RENT_GROWTH = {
    "Multifamily":  {"yoy_pct":  0.8, "prior_year":  5.2, "trend": "falling", "note": "Record supply deliveries in Sunbelt cooling asking rents"},
    "Industrial":   {"yoy_pct":  8.1, "prior_year": 14.2, "trend": "falling", "note": "Normalizing from pandemic highs but still strong"},
    "Office":       {"yoy_pct": -1.8, "prior_year": -0.4, "trend": "falling", "note": "Negative effective rent growth as landlords offer concessions"},
    "Retail":       {"yoy_pct":  2.1, "prior_year":  2.8, "trend": "stable",  "note": "Positive growth driven by limited new supply and experiential demand"},
}


# ── FRED helpers ───────────────────────────────────────────────────────────────

def _load_fred_key() -> str:
    """Load FRED API key from the environment or the project .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()
    return os.getenv("FRED_API_KEY", "")


def _fetch_single_fred_series(api_key: str, series_id: str, limit: int = 24) -> list[dict]:
    """
    Fetch up to *limit* most-recent observations for *series_id* from FRED.

    Parameters
    ----------
    api_key   : FRED API key string.
    series_id : FRED series identifier.
    limit     : Maximum observations to return (default 24).

    Returns
    -------
    list of ``{"date": "YYYY-MM-DD", "value": float}`` sorted oldest → newest.
    Returns an empty list on any network or parsing error.
    """
    params = urllib.parse.urlencode({
        "series_id":  series_id,
        "api_key":    api_key,
        "file_type":  "json",
        "limit":      limit,
        "sort_order": "desc",
    })
    url = f"{FRED_BASE}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cre-rent-growth-agent/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        obs = [
            {"date": o["date"], "value": float(o["value"])}
            for o in data.get("observations", [])
            if o.get("value") not in (".", "", None)
        ]
        return sorted(obs, key=lambda x: x["date"])  # oldest first
    except Exception:
        return []


def _fetch_fred_rent_series(api_key: str) -> dict:
    """
    Fetch CPI shelter indices from FRED.

    Retrieves the last 24 observations for:
      - ``CUSR0000SEHA``  — CPI Rent of Primary Residence
      - ``CUSR0000SEHA2`` — CPI Owners' Equivalent Rent of Residences

    Parameters
    ----------
    api_key : FRED API key string.

    Returns
    -------
    dict with keys:
      ``cpi_rent`` — list of ``{"date", "value"}`` for CUSR0000SEHA
      ``oer``      — list of ``{"date", "value"}`` for CUSR0000SEHA2
    """
    cpi_rent = _fetch_single_fred_series(api_key, "CUSR0000SEHA",  limit=24)
    time.sleep(0.1)
    oer      = _fetch_single_fred_series(api_key, "CUSR0000SEHA2", limit=24)
    return {"cpi_rent": cpi_rent, "oer": oer}


# ── Ranking helpers ────────────────────────────────────────────────────────────

def _top_markets(metric_key: str, n: int = 5) -> list[dict]:
    """
    Return the top *n* markets sorted descending by *metric_key* YoY growth.

    Parameters
    ----------
    metric_key : One of the MARKET_RENT_GROWTH sub-keys
                 (``"multifamily"``, ``"industrial_psf"``, etc.).
    n          : Number of markets to return (default 5).

    Returns
    -------
    list of ``{"market": str, "yoy_pct": float}`` sorted highest first.
    """
    ranked = sorted(
        [
            {"market": market, "yoy_pct": values[metric_key]}
            for market, values in MARKET_RENT_GROWTH.items()
            if metric_key in values
        ],
        key=lambda x: x["yoy_pct"],
        reverse=True,
    )
    return ranked[:n]


# ── Main agent runner ──────────────────────────────────────────────────────────

def run_rent_growth_agent() -> dict:
    """
    Fetch live CPI shelter indices from FRED and compile a full rent-growth
    picture combining national benchmarks, market-level data, and ranked
    market leaders for multifamily and industrial sectors.

    Returns
    -------
    dict with keys:
      national               — NATIONAL_RENT_GROWTH static benchmarks
      market_rent_growth     — MARKET_RENT_GROWTH static benchmarks
      fred_rent_series       — live FRED CPI shelter observations
      top_markets_multifamily — top 5 markets by multifamily YoY rent growth
      top_markets_industrial  — top 5 markets by industrial PSF YoY rent growth
      fetched_at             — ISO timestamp of this run
      data_as_of             — benchmark vintage label
      error                  — None on success, error message string on failure
    """
    api_key = _load_fred_key()

    if not api_key:
        return {
            "national":                NATIONAL_RENT_GROWTH,
            "market_rent_growth":      MARKET_RENT_GROWTH,
            "fred_rent_series":        {"cpi_rent": [], "oer": []},
            "top_markets_multifamily": _top_markets("multifamily"),
            "top_markets_industrial":  _top_markets("industrial_psf"),
            "fetched_at":              datetime.now().isoformat(),
            "data_as_of":              "Q1 2025",
            "error":                   "FRED_API_KEY not set. Add it to .env to enable live CPI shelter fetching.",
        }

    # ── Fetch FRED rent series ─────────────────────────────────────────────────
    fred_rent_series = _fetch_fred_rent_series(api_key)

    # ── Rank markets ──────────────────────────────────────────────────────────
    top_markets_multifamily = _top_markets("multifamily")
    top_markets_industrial  = _top_markets("industrial_psf")

    return {
        "national":                NATIONAL_RENT_GROWTH,
        "market_rent_growth":      MARKET_RENT_GROWTH,
        "fred_rent_series":        fred_rent_series,
        "top_markets_multifamily": top_markets_multifamily,
        "top_markets_industrial":  top_markets_industrial,
        "fetched_at":              datetime.now().isoformat(),
        "data_as_of":              "Q1 2025",
        "error":                   None,
    }
