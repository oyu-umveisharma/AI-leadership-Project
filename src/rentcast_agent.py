"""
RentCast Property Database Agent
=================================
Fetches real property listings from the RentCast API (free tier: 50 calls/month).
Caches aggressively to stay within rate limits. Falls back to mock listings
from cre_listings.py when API quota is exhausted or key is not set.

RentCast API docs: https://developers.rentcast.io/reference

Usage:
    from src.rentcast_agent import run_rentcast_agent
    result = run_rentcast_agent()
"""

import json
import os
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Track API usage to stay within 50 calls/month
_USAGE_FILE = CACHE_DIR / "rentcast_usage.json"

# RentCast API base
_API_BASE = "https://api.rentcast.io/v1"

# States we query — top migration destinations (keep calls minimal)
_TARGET_STATES = ["TX", "FL", "AZ", "NC", "TN", "GA"]

# Property types RentCast accepts
_PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Multi-Family"]

# Map RentCast property types to our CRE types
_TYPE_MAP = {
    "Single Family": "Retail",
    "Condo": "Multifamily",
    "Townhouse": "Multifamily",
    "Multi-Family": "Multifamily",
    "Apartment": "Multifamily",
    "Commercial": "Office",
    "Industrial": "Industrial",
    "Land": "Mixed-Use",
}

# Cities to query per state (top metros)
_STATE_CITIES = {
    "TX": ["Austin", "Dallas", "Houston", "San Antonio"],
    "FL": ["Miami", "Tampa", "Orlando", "Jacksonville"],
    "AZ": ["Phoenix", "Scottsdale", "Tucson"],
    "NC": ["Charlotte", "Raleigh", "Durham"],
    "TN": ["Nashville", "Memphis", "Chattanooga"],
    "GA": ["Atlanta", "Savannah"],
}


def _load_usage() -> dict:
    """Load API usage tracker."""
    if _USAGE_FILE.exists():
        with open(_USAGE_FILE) as f:
            data = json.load(f)
        # Reset counter if month changed
        saved_month = data.get("month", "")
        current_month = datetime.now().strftime("%Y-%m")
        if saved_month != current_month:
            return {"month": current_month, "calls": 0, "log": []}
        return data
    return {"month": datetime.now().strftime("%Y-%m"), "calls": 0, "log": []}


def _save_usage(usage: dict):
    """Persist API usage tracker."""
    with open(_USAGE_FILE, "w") as f:
        json.dump(usage, f, default=str)


def _remaining_calls(usage: dict) -> int:
    """How many API calls are left this month."""
    return max(0, 50 - usage.get("calls", 0))


def fetch_rentcast_listings(city: str, state: str, limit: int = 5) -> list:
    """
    Fetch property listings from RentCast API for a city/state.
    Returns list of dicts in our standard listing format, or empty list on failure.
    """
    api_key = os.getenv("RENTCAST_API_KEY", "")
    if not api_key:
        return []

    usage = _load_usage()
    if _remaining_calls(usage) <= 0:
        return []

    try:
        resp = requests.get(
            f"{_API_BASE}/listings/sale",
            headers={"X-Api-Key": api_key, "Accept": "application/json"},
            params={
                "city": city,
                "state": state,
                "status": "Active",
                "limit": limit,
            },
            timeout=10,
        )

        # Track the call
        usage["calls"] += 1
        usage.setdefault("log", []).append({
            "time": datetime.now().isoformat(),
            "endpoint": "listings/sale",
            "city": city,
            "state": state,
            "status_code": resp.status_code,
        })
        _save_usage(usage)

        if resp.status_code == 401:
            return []  # Invalid key
        if resp.status_code == 429:
            return []  # Rate limited
        if resp.status_code != 200:
            return []

        raw = resp.json()
        if not isinstance(raw, list):
            raw = raw.get("listings", raw.get("data", []))
            if not isinstance(raw, list):
                return []

        return [_normalize_listing(r, city, state) for r in raw if r]

    except Exception:
        return []


def _normalize_listing(raw: dict, city: str, state: str) -> dict:
    """Convert a RentCast API response to our standard listing format."""
    price = raw.get("price") or raw.get("listPrice") or 0
    sqft = raw.get("squareFootage") or raw.get("sqft") or 0
    beds = raw.get("bedrooms") or 0
    baths = raw.get("bathrooms") or 0
    prop_type = raw.get("propertyType", "")
    address = raw.get("formattedAddress") or raw.get("addressLine1") or ""
    year_built = raw.get("yearBuilt") or 0
    days_on_market = raw.get("daysOnMarket") or raw.get("dom") or 0
    lat = raw.get("latitude")
    lon = raw.get("longitude")

    # Map to our property types
    cre_type = _TYPE_MAP.get(prop_type, "Mixed-Use")

    # Estimate CRE metrics
    price_per_sqft = round(price / sqft, 2) if sqft and price else 0
    # Estimate cap rate from price and area norms
    est_cap_rate = _estimate_cap_rate(cre_type, state, price_per_sqft)
    est_noi = int(price * est_cap_rate / 100) if price and est_cap_rate else 0

    features = []
    if beds:
        features.append(f"{beds}BR")
    if baths:
        features.append(f"{baths}BA")
    if year_built:
        features.append(f"Built {year_built}")
    if lat and lon:
        features.append(f"GPS: {lat:.4f}, {lon:.4f}")

    return {
        "address": address or f"{city}, {state}",
        "city": raw.get("city") or city,
        "state": raw.get("state") or state,
        "property_type": cre_type,
        "price": int(price),
        "sqft": int(sqft),
        "price_per_sqft": price_per_sqft,
        "cap_rate": est_cap_rate,
        "noi_annual": est_noi,
        "year_built": int(year_built) if year_built else 0,
        "days_on_market": int(days_on_market),
        "highlights": ", ".join(features),
        "_source": "rentcast",
        "_rentcast_id": raw.get("id", ""),
    }


def _estimate_cap_rate(prop_type: str, state: str, ppsf: float) -> float:
    """Estimate cap rate based on property type and location."""
    base_rates = {
        "Industrial": 5.6, "Multifamily": 5.2, "Retail": 6.5,
        "Office": 7.2, "Mixed-Use": 6.0, "Self-Storage": 5.4,
    }
    base = base_rates.get(prop_type, 6.0)
    # Sun Belt states get a slight compression
    if state in ("TX", "FL", "AZ", "NC", "TN", "GA", "SC", "NV"):
        base -= 0.3
    return round(base, 2)


def run_rentcast_agent() -> dict:
    """
    Main agent entry point. Fetches listings for target cities,
    caches results, tracks API usage. Falls back to mock data when
    API is unavailable or quota exhausted.
    """
    api_key = os.getenv("RENTCAST_API_KEY", "")
    usage = _load_usage()
    remaining = _remaining_calls(usage)

    all_listings = {}
    live_count = 0
    mock_count = 0
    source_states = {}  # track which states got live vs mock data

    for state_abbr in _TARGET_STATES:
        cities = _STATE_CITIES.get(state_abbr, [])
        state_listings = []

        for city in cities:
            if remaining <= 2:  # Reserve 2 calls as buffer
                break

            fetched = fetch_rentcast_listings(city, state_abbr, limit=5)
            if fetched:
                state_listings.extend(fetched)
                live_count += len(fetched)
                remaining = _remaining_calls(_load_usage())

        if state_listings:
            # Sort by price, keep cheapest
            state_listings.sort(key=lambda x: x.get("price", 0))
            all_listings[state_abbr] = state_listings[:10]
            source_states[state_abbr] = "live"
        else:
            # Fall back to mock data
            try:
                from src.cre_listings import get_cheapest_buildings
                mock = get_cheapest_buildings(state_abbr, n=10)
                # Tag as mock
                for m in mock:
                    m["_source"] = "mock"
                all_listings[state_abbr] = mock
                mock_count += len(mock)
                source_states[state_abbr] = "mock"
            except Exception:
                source_states[state_abbr] = "error"

    usage = _load_usage()

    # ── Build price-per-sqft trend for each state ────────────────────────
    # Simulate 6 monthly data points with slight variance around current avg
    price_per_sqft_trend = {}
    rng = random.Random(42)  # deterministic seed for reproducibility
    for state_abbr, state_lst in all_listings.items():
        ppsf_values = [
            l.get("price_per_sqft", 0) for l in state_lst if l.get("price_per_sqft")
        ]
        if not ppsf_values:
            continue
        avg_ppsf = sum(ppsf_values) / len(ppsf_values)
        months = []
        base_date = datetime.now()
        for i in range(5, -1, -1):
            month_date = (base_date - timedelta(days=30 * i)).strftime("%Y-%m")
            # Add slight trend drift (+0.5% per month) and small random noise
            trend_val = avg_ppsf * (1 + 0.005 * (5 - i)) + rng.uniform(-avg_ppsf * 0.03, avg_ppsf * 0.03)
            months.append({"month": month_date, "ppsf": round(trend_val, 2)})
        price_per_sqft_trend[state_abbr] = months

    return {
        "listings": all_listings,
        "source_states": source_states,
        "api_calls_used": usage.get("calls", 0),
        "api_calls_remaining": _remaining_calls(usage),
        "api_month": usage.get("month", ""),
        "live_listing_count": live_count,
        "mock_listing_count": mock_count,
        "has_api_key": bool(api_key),
        "fetched_at": datetime.now().isoformat(),
        "price_per_sqft_trend": price_per_sqft_trend,
    }
