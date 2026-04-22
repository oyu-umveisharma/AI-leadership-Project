"""
Commercial Real Estate Intelligence Platform
Purdue MSF | Group AI Project

Nine background agents update independently on schedules:
  Agent 1 — Population & Migration    (every 6h)
  Agent 2 — CRE Pricing & Profit      (every 1h)
  Agent 3 — Company Predictions       (every 24h, LLM)
  Agent 4 — Debugger / Monitor        (every 30min)
  Agent 5 — News & Announcements      (every 4h)
  Agent 6 — Interest Rate & Debt      (every 1h, requires FRED_API_KEY)
  Agent 7 — Energy & Construction     (every 6h)
  Agent 8 — Sustainability & ESG      (every 6h)
  Agent 9 — Labor Market & Tenant Demand (every 6h)

Run: streamlit run app/cre_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page config MUST be first Streamlit command ──────────────────────────────
st.set_page_config(
    page_title="CRE Intelligence | Purdue MSF",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Start background agents ──────────────────────────────────────────────────
from src.cre_agents import (
    start_scheduler, read_cache, cache_age_label, get_status,
)
from src.cre_listings import get_cheapest_buildings, format_listing_card, estimate_property_tax

# Land functions — added after initial release; guard against stale bytecode
# on Python 3.9 installations that haven't re-compiled the module yet.
try:
    from src.cre_listings import get_land_parcels, ZONING_TYPES, ENTITLEMENT_STATUS, LAND_PRICE_PER_ACRE
except ImportError:
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _cre_path = str(_Path(__file__).resolve().parent.parent / "src" / "cre_listings.py")
    _spec = _ilu.spec_from_file_location("_cre_listings_fresh", _cre_path)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    get_land_parcels    = _m.get_land_parcels
    ZONING_TYPES        = _m.ZONING_TYPES
    ENTITLEMENT_STATUS  = _m.ENTITLEMENT_STATUS
    LAND_PRICE_PER_ACRE = _m.LAND_PRICE_PER_ACRE

try:
    from src.vacancy_agent import NATIONAL_VACANCY, MARKET_VACANCY, TREND_ARROW, TREND_COLOR, LAND_AVAILABILITY
except ImportError:
    import importlib.util as _ilu2
    from pathlib import Path as _Path2
    _vac_path = str(_Path2(__file__).resolve().parent.parent / "src" / "vacancy_agent.py")
    _spec2 = _ilu2.spec_from_file_location("_vacancy_agent_fresh", _vac_path)
    _m2 = _ilu2.module_from_spec(_spec2)
    _spec2.loader.exec_module(_m2)
    NATIONAL_VACANCY = _m2.NATIONAL_VACANCY
    MARKET_VACANCY   = _m2.MARKET_VACANCY
    TREND_ARROW      = _m2.TREND_ARROW
    TREND_COLOR      = _m2.TREND_COLOR
    LAND_AVAILABILITY = _m2.LAND_AVAILABILITY

@st.cache_resource
def _init_scheduler():
    """Called once per server process — keeps one scheduler alive regardless of reruns or new sessions."""
    start_scheduler()
    return True

_init_scheduler()

# ── Brand ────────────────────────────────────────────────────────────────────
GOLD       = "#d4a843"
GOLD_LIGHT = "#e8c060"
GOLD_DARK  = "#a07830"
BLACK      = "#000000"

# ── Session State ────────────────────────────────────────────────────────────
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "user_intent" not in st.session_state:
    st.session_state.user_intent = {"property_type": None, "location": None, "city": None, "state": None, "raw_input": ""}
if "adv_home_prompt" not in st.session_state:
    st.session_state.adv_home_prompt = None
if "adv_auto_generate" not in st.session_state:
    st.session_state.adv_auto_generate = False
if "adv_navigate" not in st.session_state:
    st.session_state.adv_navigate = False
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []

# US state name/abbreviation lookup
_US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "Dist. of Columbia",
}
_STATE_NAME_TO_ABBR = {v.lower(): k for k, v in _US_STATES.items()}


# Common city abbreviations → proper display name
_CITY_ALIASES = {
    "nyc": "New York City", "la": "Los Angeles", "sf": "San Francisco",
    "dc": "Washington DC", "philly": "Philadelphia", "nola": "New Orleans",
    "lv": "Las Vegas", "slc": "Salt Lake City", "atl": "Atlanta",
    "chi": "Chicago", "det": "Detroit", "stl": "St. Louis",
}


def _normalize_input(text: str) -> str:
    """Normalize user input to proper title case with special handling."""
    if not text:
        return text
    text = text.strip()
    # Check city aliases first (e.g., "la" → "Los Angeles")
    if text.lower() in _CITY_ALIASES:
        return _CITY_ALIASES[text.lower()]
    # Check if it's a state abbreviation (1-2 uppercase letters)
    upper = text.upper().strip().rstrip(".")
    if upper in _US_STATES:
        return _US_STATES[upper]
    # Title-case each word, but keep state abbreviations uppercase
    parts = text.split(",")
    normalized = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i > 0 and len(part) <= 2 and part.upper() in _US_STATES:
            normalized.append(part.upper())
        else:
            normalized.append(part.title())
    return ", ".join(normalized)


_CITY_TO_STATE = {
    # Alabama
    "birmingham": "AL", "montgomery": "AL", "huntsville": "AL", "mobile": "AL",
    # Alaska
    "anchorage": "AK", "fairbanks": "AK", "juneau": "AK",
    # Arizona
    "phoenix": "AZ", "tucson": "AZ", "scottsdale": "AZ", "mesa": "AZ", "tempe": "AZ", "chandler": "AZ", "gilbert": "AZ",
    # Arkansas
    "little rock": "AR", "fayetteville": "AR", "fort smith": "AR", "bentonville": "AR",
    # California
    "los angeles": "CA", "la": "CA", "san francisco": "CA", "sf": "CA", "san diego": "CA",
    "san jose": "CA", "sacramento": "CA", "irvine": "CA", "oakland": "CA", "long beach": "CA",
    "fresno": "CA", "anaheim": "CA", "santa monica": "CA", "pasadena": "CA", "riverside": "CA",
    "bakersfield": "CA", "glendale": "CA", "burbank": "CA",
    # Colorado
    "denver": "CO", "colorado springs": "CO", "aurora": "CO", "boulder": "CO", "fort collins": "CO",
    # Connecticut
    "hartford": "CT", "new haven": "CT", "stamford": "CT", "bridgeport": "CT", "norwalk": "CT",
    # Delaware
    "wilmington": "DE", "dover": "DE", "newark": "DE",
    # DC
    "washington dc": "DC", "washington d.c.": "DC", "dc": "DC",
    # Florida
    "miami": "FL", "tampa": "FL", "orlando": "FL", "jacksonville": "FL",
    "fort lauderdale": "FL", "st. petersburg": "FL", "st petersburg": "FL",
    "west palm beach": "FL", "naples": "FL", "sarasota": "FL", "tallahassee": "FL",
    # Georgia
    "atlanta": "GA", "savannah": "GA", "augusta": "GA", "athens": "GA", "macon": "GA",
    # Hawaii
    "honolulu": "HI", "hilo": "HI", "kailua": "HI",
    # Idaho
    "boise": "ID", "meridian": "ID", "nampa": "ID", "idaho falls": "ID", "pocatello": "ID",
    # Illinois
    "chicago": "IL", "aurora": "IL", "naperville": "IL", "rockford": "IL", "joliet": "IL", "schaumburg": "IL",
    # Indiana
    "indianapolis": "IN", "fort wayne": "IN", "evansville": "IN", "south bend": "IN", "carmel": "IN",
    # Iowa
    "des moines": "IA", "cedar rapids": "IA", "davenport": "IA", "iowa city": "IA",
    # Kansas
    "wichita": "KS", "overland park": "KS", "kansas city": "KS", "topeka": "KS",
    # Kentucky
    "louisville": "KY", "lexington": "KY", "bowling green": "KY",
    # Louisiana
    "new orleans": "LA", "baton rouge": "LA", "shreveport": "LA", "lafayette": "LA",
    # Maine
    "portland": "ME", "bangor": "ME", "lewiston": "ME",
    # Maryland
    "baltimore": "MD", "annapolis": "MD", "rockville": "MD", "silver spring": "MD", "bethesda": "MD", "columbia": "MD",
    # Massachusetts
    "boston": "MA", "cambridge": "MA", "worcester": "MA", "springfield": "MA", "lowell": "MA",
    # Michigan
    "detroit": "MI", "grand rapids": "MI", "ann arbor": "MI", "lansing": "MI", "flint": "MI",
    # Minnesota
    "minneapolis": "MN", "st. paul": "MN", "st paul": "MN", "rochester": "MN", "duluth": "MN", "bloomington": "MN",
    # Mississippi
    "jackson": "MS", "gulfport": "MS", "biloxi": "MS",
    # Missouri
    "kansas city": "MO", "st. louis": "MO", "st louis": "MO", "springfield": "MO", "columbia": "MO",
    # Montana
    "billings": "MT", "missoula": "MT", "great falls": "MT", "bozeman": "MT",
    # Nebraska
    "omaha": "NE", "lincoln": "NE", "bellevue": "NE",
    # Nevada
    "las vegas": "NV", "reno": "NV", "henderson": "NV", "sparks": "NV",
    # New Hampshire
    "manchester": "NH", "nashua": "NH", "concord": "NH",
    # New Jersey
    "newark": "NJ", "jersey city": "NJ", "hoboken": "NJ", "trenton": "NJ", "atlantic city": "NJ", "princeton": "NJ",
    # New Mexico
    "albuquerque": "NM", "santa fe": "NM", "las cruces": "NM",
    # New York
    "new york": "NY", "new york city": "NY", "nyc": "NY", "manhattan": "NY",
    "brooklyn": "NY", "queens": "NY", "bronx": "NY", "buffalo": "NY",
    "white plains": "NY", "yonkers": "NY", "albany": "NY", "rochester": "NY", "syracuse": "NY",
    # North Carolina
    "charlotte": "NC", "raleigh": "NC", "durham": "NC", "greensboro": "NC",
    "winston-salem": "NC", "fayetteville": "NC", "research triangle": "NC", "asheville": "NC",
    # North Dakota
    "fargo": "ND", "bismarck": "ND", "grand forks": "ND",
    # Ohio
    "columbus": "OH", "cleveland": "OH", "cincinnati": "OH", "dayton": "OH", "toledo": "OH", "akron": "OH",
    # Oklahoma
    "oklahoma city": "OK", "tulsa": "OK", "norman": "OK",
    # Oregon
    "portland": "OR", "salem": "OR", "eugene": "OR", "bend": "OR", "beaverton": "OR",
    # Pennsylvania
    "philadelphia": "PA", "pittsburgh": "PA", "allentown": "PA", "harrisburg": "PA", "scranton": "PA",
    # Rhode Island
    "providence": "RI", "warwick": "RI", "newport": "RI", "cranston": "RI",
    # South Carolina
    "charleston": "SC", "columbia": "SC", "greenville": "SC", "myrtle beach": "SC",
    # South Dakota
    "sioux falls": "SD", "rapid city": "SD",
    # Tennessee
    "nashville": "TN", "memphis": "TN", "knoxville": "TN", "chattanooga": "TN", "murfreesboro": "TN",
    # Texas
    "houston": "TX", "dallas": "TX", "austin": "TX", "san antonio": "TX",
    "fort worth": "TX", "el paso": "TX", "plano": "TX", "arlington": "TX", "frisco": "TX",
    # Utah
    "salt lake city": "UT", "provo": "UT", "ogden": "UT", "st. george": "UT", "st george": "UT",
    # Vermont
    "burlington": "VT", "montpelier": "VT",
    # Virginia
    "richmond": "VA", "virginia beach": "VA", "arlington": "VA", "norfolk": "VA", "alexandria": "VA", "roanoke": "VA",
    # Washington
    "seattle": "WA", "tacoma": "WA", "spokane": "WA", "bellevue": "WA", "redmond": "WA", "vancouver": "WA",
    # West Virginia
    "charleston": "WV", "huntington": "WV", "morgantown": "WV",
    # Wisconsin
    "milwaukee": "WI", "madison": "WI", "green bay": "WI",
    # Wyoming
    "cheyenne": "WY", "casper": "WY", "laramie": "WY",
}


def _get_location_scope(location_str: str) -> dict:
    """Parse a location string into city/state components."""
    if not location_str:
        return {"city": None, "state": None, "scope": "national"}

    loc = location_str.strip()

    # "City, ST" or "City, State Name" pattern
    if "," in loc:
        parts = [p.strip() for p in loc.split(",", 1)]
        city = parts[0]
        st_part = parts[1].strip().rstrip(".")
        # Check abbreviation
        if st_part.upper() in _US_STATES:
            return {"city": city, "state": _US_STATES[st_part.upper()], "state_abbr": st_part.upper(), "scope": "city"}
        # Check full state name
        if st_part.lower() in _STATE_NAME_TO_ABBR:
            abbr = _STATE_NAME_TO_ABBR[st_part.lower()]
            return {"city": city, "state": _US_STATES[abbr], "state_abbr": abbr, "scope": "city"}
        # Unknown state portion
        return {"city": city, "state": st_part.title(), "state_abbr": None, "scope": "city"}

    # Check if it's a full state name
    if loc.lower() in _STATE_NAME_TO_ABBR:
        abbr = _STATE_NAME_TO_ABBR[loc.lower()]
        return {"city": None, "state": _US_STATES[abbr], "state_abbr": abbr, "scope": "state"}

    # Check if it's a state abbreviation
    if loc.upper() in _US_STATES:
        return {"city": None, "state": _US_STATES[loc.upper()], "state_abbr": loc.upper(), "scope": "state"}

    # City-to-state lookup for bare city names (try exact, then progressively shorter)
    loc_lower = loc.lower()
    abbr = _CITY_TO_STATE.get(loc_lower)
    if abbr:
        return {"city": loc, "state": _US_STATES[abbr], "state_abbr": abbr, "scope": "city"}

    # Unknown city — no state resolved
    return {"city": loc, "state": None, "state_abbr": None, "scope": "city"}


# Synonyms that map to canonical property types (longest first to avoid partial matches)
_PT_SYNONYMS = {
    "car wash": "Retail", "gas station": "Retail", "convenience store": "Retail",
    "shopping center": "Retail", "strip mall": "Retail",
    "data center": "Data Center", "data centres": "Data Center",
    "self-storage": "Self-Storage", "self storage": "Self-Storage", "storage unit": "Self-Storage",
    "warehouse": "Industrial", "logistics": "Industrial", "distribution": "Industrial",
    "fulfillment": "Industrial", "manufacturing": "Industrial", "factory": "Industrial",
    "storage": "Industrial", "flex space": "Industrial",
    "apartment": "Multifamily", "apartments": "Multifamily", "residential": "Multifamily",
    "condo": "Multifamily", "condos": "Multifamily", "townhome": "Multifamily",
    "restaurant": "Retail", "shop": "Retail", "shopping": "Retail", "mall": "Retail",
    "store": "Retail", "grocery": "Retail", "pharmacy": "Retail",
    "medical": "Healthcare", "hospital": "Healthcare", "clinic": "Healthcare",
    "urgent care": "Healthcare", "dental": "Healthcare",
    "hotel": "Hospitality", "motel": "Hospitality", "resort": "Hospitality",
    "senior living": "Healthcare", "assisted living": "Healthcare", "nursing": "Healthcare",
    "parking": "Other", "parking garage": "Other", "parking lot": "Other",
    "mixed-use": "Mixed-Use", "mixed use": "Mixed-Use", "live-work": "Mixed-Use",
}

# Region keywords → list of state abbreviations
_REGION_STATES = {
    "west coast":     ["CA", "WA", "OR"],
    "east coast":     ["NY", "FL", "MA", "NC", "VA", "NJ", "CT", "MD"],
    "midwest":        ["IL", "OH", "IN", "MI", "MN", "WI", "IA", "MO"],
    "south":          ["TX", "FL", "GA", "NC", "TN", "SC", "AL", "LA", "MS"],
    "southeast":      ["FL", "GA", "NC", "SC", "TN", "AL", "VA"],
    "southwest":      ["AZ", "NV", "NM", "CO"],
    "mountain west":  ["CO", "UT", "ID", "MT", "WY"],
    "northeast":      ["NY", "MA", "CT", "NJ", "PA", "NH", "VT", "ME", "RI"],
    "pacific northwest": ["WA", "OR"],
    "sun belt":       ["TX", "FL", "AZ", "GA", "NC", "TN", "NV", "SC"],
    "great plains":   ["ND", "SD", "NE", "KS", "OK"],
}


_FILLER_WORDS = {"potential", "places", "properties", "property", "investment",
                  "investments", "opportunities", "opportunity", "spaces", "space",
                  "buildings", "building", "for", "a", "an", "the", "some", "best",
                  "top", "good", "great", "cheap", "cheapest", "expensive", "near", "around"}


import re as _re

def _is_advisor_query(text: str) -> bool:
    """Return True if the text looks like a full investment query (not just property + location)."""
    t = text.lower()
    has_budget   = bool(_re.search(r'\$[\d,]+|\b\d+\s*(m|million|b|billion|k|thousand)\b', t))
    has_sqft     = bool(_re.search(r'[\d,]+\s*(sq\s?ft|sqft|sf|square\s?feet)', t))
    has_timeline = bool(_re.search(r'\d+[\s-]*(year|yr)', t))
    has_action   = any(w in t for w in [
        'build', 'invest', 'develop', 'construction', 'acquire', 'purchase',
        'looking to', 'want to', 'interested in', 'planning to', 'advise',
        'recommend', 'where should', 'best market', 'best city',
    ])
    # Trigger advisor if: (has financials + action) OR (budget + sqft) OR (3+ signals)
    signals = sum([has_budget, has_sqft, has_timeline, has_action])
    return (has_budget and has_action) or (has_budget and has_sqft) or signals >= 3


def _parse_intent(raw: str) -> dict:
    """Parse free-text input into property_type and location."""
    raw_lower = raw.lower().strip()

    # Detect property type — check multi-word synonyms first (longest match wins)
    prop_type = None
    for syn, canonical in _PT_SYNONYMS.items():
        if syn in raw_lower:
            prop_type = canonical
            break
    if not prop_type:
        for pt in ["industrial", "multifamily", "office", "retail", "data center", "healthcare", "hospitality"]:
            if pt in raw_lower:
                prop_type = pt.title()
                break

    # Detect region keywords
    region = None
    region_states = None
    for rname, rstates in _REGION_STATES.items():
        if rname in raw_lower:
            region = rname.title()
            region_states = rstates
            break

    # Extract location — everything after "in " or "on the "
    location = None
    for prep in [" in the ", " in ", " on the ", " on "]:
        if prep in raw_lower:
            location = raw[raw_lower.index(prep) + len(prep):].strip().rstrip(".")
            break

    # Handle "City, State Name" (e.g., "raleigh, north carolina")
    if not location and "," in raw:
        parts = [p.strip() for p in raw.rsplit(",", 1)]
        after_comma = parts[1].lower().rstrip(".")
        # Check if after comma is a state abbreviation (2 letters)
        if len(after_comma) <= 3 and after_comma.upper().rstrip(".") in _US_STATES:
            location = raw.strip()
        # Check if after comma is a full state name
        elif after_comma in _STATE_NAME_TO_ABBR:
            abbr = _STATE_NAME_TO_ABBR[after_comma]
            # Reconstruct as "City, ST" for downstream parsing
            city_part = parts[0]
            # Strip filler/property words from city part
            city_words = [w for w in city_part.split() if w.lower() not in _FILLER_WORDS
                          and w.lower() not in _PT_SYNONYMS
                          and w.lower() not in ["industrial", "multifamily", "office", "retail",
                                                 "data", "center", "healthcare", "hospitality"]]
            if city_words:
                location = " ".join(city_words) + ", " + abbr
            else:
                location = _US_STATES[abbr]

    # Strip filler words from location
    if location:
        loc_words = location.split()
        cleaned = [w for w in loc_words if w.lower() not in _FILLER_WORDS]
        if cleaned:
            location = " ".join(cleaned)

    # If we detected a region, don't treat it as a city/state location
    if region and not location:
        location = region
    elif region and location and location.lower() == region.lower():
        pass  # already set

    # Normalize
    location = _normalize_input(location) if location else None
    loc_scope = _get_location_scope(location)

    print(f"[Intent Parser] Parsed: type={prop_type}, location={location}, "
          f"city={loc_scope.get('city')}, state={loc_scope.get('state')}, "
          f"state_abbr={loc_scope.get('state_abbr')}")

    return {
        "property_type": prop_type,
        "location": location,
        "city": loc_scope.get("city"),
        "state": loc_scope.get("state"),
        "state_abbr": loc_scope.get("state_abbr"),
        "region": region,
        "region_states": region_states,
        "raw_input": raw,
    }


def _complete_onboarding(property_type=None, location=None, raw_input="", **kwargs):
    loc_scope = _get_location_scope(location)
    st.session_state.user_intent = {
        "property_type": property_type,
        "location": location,
        "city": kwargs.get("city") or loc_scope.get("city"),
        "state": kwargs.get("state") or loc_scope.get("state"),
        "state_abbr": kwargs.get("state_abbr") or loc_scope.get("state_abbr"),
        "region": kwargs.get("region"),
        "region_states": kwargs.get("region_states"),
        "raw_input": raw_input,
    }
    st.session_state.onboarding_complete = True
    if raw_input and raw_input.strip():
        if raw_input not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, raw_input)
            st.session_state.recent_searches = st.session_state.recent_searches[:6]


# ═══════════════════════════════════════════════════════════════════════════════
#  WELCOME / ONBOARDING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.onboarding_complete:

    # ── Query-param navigation (property card / example link clicks) ──────────
    # NOTE: set session state BEFORE clearing params to avoid double-rerun bug
    try:
        _qp_select = st.query_params.get("select")
        _qp_q      = st.query_params.get("q")
        _qp_clear  = st.query_params.get("clear_recent")
    except Exception:
        _qp_select = _qp_q = _qp_clear = None

    if _qp_select:
        _sel = _qp_select if isinstance(_qp_select, str) else _qp_select[0]
        if _sel == "Exploring":
            _complete_onboarding()
        else:
            _complete_onboarding(property_type=_sel)
        st.rerun()

    if _qp_q:
        _q = (_qp_q if isinstance(_qp_q, str) else _qp_q[0]).replace("+", " ")
        if _is_advisor_query(_q):
            _complete_onboarding(raw_input=_q)
            st.session_state.adv_home_prompt   = _q
            st.session_state.adv_auto_generate = True
            st.session_state.adv_navigate      = True
        else:
            _complete_onboarding(**_parse_intent(_q))
        st.rerun()

    if _qp_clear:
        st.session_state.recent_searches = []
        try: st.query_params.clear()
        except Exception: pass
        st.rerun()

    # ── Live ticker data ──────────────────────────────────────────────────────
    _tick_rates = read_cache("rates") or {}
    _rd = _tick_rates.get("data", _tick_rates)
    try: _tsy = f"{float(_rd.get('DGS10') or _rd.get('ten_year_yield') or _rd.get('treasury_10yr') or 4.5):.2f}%"
    except Exception: _tsy = "4.50%"

    _tick_cap = read_cache("cap_rate") or {}
    _cd = _tick_cap.get("data", _tick_cap)
    try: _cap_str = f"{float(_cd.get('national_avg_cap_rate', _cd.get('cap_rate', 5.6))):.2f}%"
    except Exception: _cap_str = "5.60%"

    _tick_ms  = read_cache("market_score") or {}
    _top_mkt  = "Austin, TX"
    try:
        _sc = _tick_ms.get("scores") or (_tick_ms.get("data") or {}).get("scores") or []
        if _sc: _top_mkt = _sc[0].get("market", "Austin, TX")
    except Exception: pass

    # ── SVG icons for property cards ──────────────────────────────────────────
    _PROP_ICONS = {
        "Industrial":  '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="14" rx="1"/><path d="M2 11h20M7 7V4M12 7V4M17 7V4"/><rect x="9" y="14" width="6" height="7" rx="0.5"/></svg>',
        "Multifamily": '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="3" y="2" width="18" height="20" rx="1"/><path d="M3 9h18M3 15h18M9 2v20M15 2v20"/></svg>',
        "Office":      '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="1"/><path d="M3 8h18M8 3v18M8 12h8M8 16h5"/></svg>',
        "Retail":      '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M3 9h18l-2 12H5L3 9z"/><path d="M3 9 5.5 3h13L21 9"/><path d="M9 21v-7h6v7"/></svg>',
        "Healthcare":  '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="15" rx="1"/><path d="M12 11v6M9 14h6"/><path d="M8 7V5a1 1 0 011-1h6a1 1 0 011 1v2"/></svg>',
        "Exploring":   '<svg width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20M2 12h20"/><path d="M4.9 7h14.2M4.9 17h14.2"/></svg>',
    }
    _prop_cards_html = "".join([
        f'<div class="prop-card" onclick="window.location.href=\'?select={name}\'">'
        f'{icon}<div class="prop-card-lbl">{name.upper()}</div></div>'
        for name, icon in _PROP_ICONS.items()
    ])

    # ── Recent searches HTML ──────────────────────────────────────────────────
    _recent_html = ""
    if st.session_state.recent_searches:
        _rs_items = "".join([
            f'<span class="rs-item" onclick="window.location.href=\'?q={_r.replace(" ","+")}\'">'
            f'<svg width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="opacity:.45;flex-shrink:0"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>'
            f'&nbsp;{_r}&nbsp;<span style="opacity:.4;font-size:.6rem;">&#8599;</span></span>'
            for _r in st.session_state.recent_searches[:4]
        ])
        _recent_html = f"""
        <div class="cre-wrap" style="padding-bottom:52px;border-top:1px solid rgba(200,160,64,.08);padding-top:28px;margin-top:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
            <span style="color:#3a2e1a;font-size:.65rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;">Recent Searches</span>
            <a href="?clear_recent=1" style="color:#3a2e1a;font-size:.72rem;text-decoration:none;">Clear all</a>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">{_rs_items}</div>
        </div>"""

    # ── Full-page HTML + CSS ──────────────────────────────────────────────────
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

      html, body, [class*="css"],
      [data-testid="stAppViewContainer"],
      [data-testid="stApp"],
      section[data-testid="stMain"] {{
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        background: #0d0b04 !important;
        color: #c8b890 !important;
      }}
      .main .block-container {{
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
      }}
      header[data-testid="stHeader"],
      [data-testid="stDecoration"],
      footer, #MainMenu {{ display: none !important; }}

      /* ── Navbar ─────────────────────────────────────────────────────────── */
      .cre-nav {{
        background: rgba(13,11,4,.97);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(200,160,64,.18);
        padding: 0 32px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .nav-logo {{
        width:28px; height:28px; border-radius:6px;
        background:{GOLD}; display:inline-flex;
        align-items:center; justify-content:center;
        font-size:.7rem; font-weight:800; color:#0d0b04;
        margin-right:10px; vertical-align:middle;
        font-family:'JetBrains Mono',monospace;
      }}
      .nav-brand {{ color:#e8e4d8; font-size:.93rem; font-weight:600; vertical-align:middle; }}
      .nav-sep   {{ color:rgba(200,160,64,.25); margin:0 10px; vertical-align:middle; }}
      .nav-school {{ color:#4a3820; font-size:.67rem; font-weight:500; letter-spacing:1.5px; vertical-align:middle; }}
      .nav-links {{ display:flex; align-items:center; gap:28px; }}
      .nav-link  {{ color:#6a5228; font-size:.8rem; }}
      .nav-cta {{
        background:transparent; border:1.5px solid {GOLD}; color:{GOLD};
        padding:6px 16px; border-radius:6px; font-size:.78rem; font-weight:600;
        cursor:default; font-family:'DM Sans',sans-serif; margin-left:24px;
      }}

      /* ── Ticker ─────────────────────────────────────────────────────────── */
      .cre-ticker {{
        background:#090700;
        border-bottom:1px solid rgba(200,160,64,.1);
        height:38px; display:flex; align-items:center; overflow:hidden;
      }}
      .t-item {{
        display:flex; align-items:center; gap:7px;
        padding:0 22px;
        border-right:1px solid rgba(200,160,64,.08);
        height:100%; white-space:nowrap;
      }}
      .t-lbl {{ color:#3a3020; font-size:.62rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; }}
      .t-val {{ color:#d8d0b8; font-family:'JetBrains Mono',monospace; font-size:.8rem; font-weight:600; }}
      .t-up  {{ color:#4caf50; font-size:.64rem; }}
      .t-dn  {{ color:#ef5350; font-size:.64rem; }}
      .t-badge {{
        background:rgba(76,175,80,.15); border:1px solid rgba(76,175,80,.3);
        color:#4caf50; font-size:.6rem; padding:1px 8px; border-radius:10px; font-weight:600;
      }}

      /* ── Hero ───────────────────────────────────────────────────────────── */
      .cre-hero {{ text-align:center; padding:68px 20px 40px; }}
      .hero-eyebrow {{
        display:flex; align-items:center; justify-content:center;
        gap:14px; margin-bottom:26px;
      }}
      .ey-line   {{ flex:0 0 64px; height:1px; background:linear-gradient(90deg,transparent,rgba(200,160,64,.4)); }}
      .ey-line-r {{ background:linear-gradient(270deg,transparent,rgba(200,160,64,.4)); }}
      .ey-text   {{ color:{GOLD}; font-size:.65rem; font-weight:500; letter-spacing:3.5px; }}
      .hero-title {{
        font-size:3.0rem; font-weight:700; color:{GOLD};
        line-height:1.1; margin-bottom:18px; letter-spacing:-.5px;
      }}
      .hero-sub {{
        font-size:1.02rem; color:#7a6840; line-height:1.65;
        max-width:580px; margin:0 auto 36px;
      }}

      /* ── Feature chips ──────────────────────────────────────────────────── */
      .f-chips {{
        display:flex; flex-wrap:wrap; justify-content:center;
        gap:8px; max-width:820px; margin:0 auto 52px;
      }}
      .f-chip {{
        background:transparent; border:1px solid rgba(200,160,64,.22);
        color:#7a6840; font-size:.72rem; padding:6px 14px; border-radius:4px;
        display:inline-flex; align-items:center; gap:7px;
      }}
      .f-check {{ color:{GOLD}; font-weight:700; }}

      /* ── Search input override ──────────────────────────────────────────── */
      [data-testid="stForm"] {{
        border:none !important; padding:0 !important; background:transparent !important;
      }}
      [data-testid="stForm"] > div {{ max-width:780px; margin:0 auto !important; }}
      [data-testid="stHorizontalBlock"] {{ gap:0 !important; }}

      [data-testid="stTextInput"] input {{
        background: rgba(13,11,4,.92)
          url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' fill='none' stroke='%234a3820' stroke-width='2' viewBox='0 0 24 24'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E")
          no-repeat left 18px center !important;
        border:1.5px solid rgba(200,160,64,.32) !important;
        border-right:none !important;
        border-radius:8px 0 0 8px !important;
        color:#e0d8c0 !important;
        font-size:.97rem !important;
        padding:18px 20px 18px 52px !important;
        height:58px !important;
        font-family:'DM Sans',sans-serif !important;
        caret-color:{GOLD};
      }}
      [data-testid="stTextInput"] input:focus {{
        border-color:rgba(200,160,64,.55) !important;
        box-shadow:0 0 0 3px rgba(200,160,64,.07) !important;
        outline:none !important;
      }}
      [data-testid="stTextInput"] input::placeholder {{ color:#3a2e1a !important; }}
      [data-testid="stTextInput"] > div {{ border:none !important; background:transparent !important; }}
      [data-testid="InputInstructions"] {{ display:none !important; }}

      [data-testid="stFormSubmitButton"] > button {{
        background:{GOLD} !important; color:#0d0b04 !important;
        border:none !important;
        border-radius:0 8px 8px 0 !important;
        padding:0 26px !important; height:58px !important;
        font-weight:700 !important; font-size:.85rem !important;
        font-family:'DM Sans',sans-serif !important;
        width:100% !important; letter-spacing:.3px;
      }}
      [data-testid="stFormSubmitButton"] > button:hover {{
        background:#e8c060 !important;
      }}

      /* ── Search examples ────────────────────────────────────────────────── */
      .s-examples {{
        text-align:center; color:#3a2e1a; font-size:.77rem; margin-bottom:52px;
      }}
      .s-ex {{
        color:{GOLD}; cursor:pointer;
        text-decoration:underline; text-decoration-color:rgba(200,160,64,.3);
      }}

      /* ── Property type section ──────────────────────────────────────────── */
      .cre-wrap {{ max-width:1160px; margin:0 auto; padding:0 48px; }}
      .prop-hdr {{
        text-align:center; color:#3a2e1a; font-size:.65rem; font-weight:600;
        letter-spacing:3px; text-transform:uppercase; margin-bottom:18px;
      }}
      .prop-grid {{
        display:grid; grid-template-columns:repeat(6,1fr);
        gap:12px; margin-bottom:52px;
      }}
      .prop-card {{
        background:rgba(255,255,255,.018);
        border:1px solid rgba(200,160,64,.15);
        border-radius:10px; padding:24px 12px 18px;
        text-align:center; cursor:pointer;
        transition:all .2s; color:#5a4820;
      }}
      .prop-card:hover {{
        background:rgba(200,160,64,.06);
        border-color:rgba(200,160,64,.38); color:{GOLD};
      }}
      .prop-card svg {{ display:block; margin:0 auto 12px; color:inherit; }}
      .prop-card-lbl {{ font-size:.6rem; font-weight:600; letter-spacing:2.5px; text-transform:uppercase; }}

      /* ── Recent searches ────────────────────────────────────────────────── */
      .rs-item {{
        display:inline-flex; align-items:center; gap:6px;
        background:rgba(255,255,255,.018);
        border:1px solid rgba(200,160,64,.1);
        border-radius:20px; color:#5a4820;
        font-size:.72rem; padding:5px 14px;
        cursor:pointer; white-space:nowrap;
        transition:border-color .2s;
      }}
      .rs-item:hover {{ border-color:rgba(200,160,64,.3); color:{GOLD}; }}

    </style>

    <!-- NAVBAR -->
    <div class="cre-nav">
      <div>
        <span class="nav-logo">&#9650;</span>
        <span class="nav-brand">CRE Intelligence Platform</span>
        <span class="nav-sep">|</span>
        <span class="nav-school">PURDUE &middot; DANIELS MSF</span>
      </div>
      <div style="display:flex;align-items:center;">
        <div class="nav-links">
          <span class="nav-link">Markets</span>
          <span class="nav-link">Watchlist</span>
          <span class="nav-link">Reports</span>
        </div>
        <button class="nav-cta">New Analysis</button>
      </div>
    </div>

    <!-- TICKER BAR -->
    <div class="cre-ticker">
      <div class="t-item">
        <span class="t-lbl">Ind. Cap Rate</span>
        <span class="t-val">{_cap_str}</span>
        <span class="t-dn">&#9660; 20bps</span>
      </div>
      <div class="t-item">
        <span class="t-lbl">Rent Growth</span>
        <span class="t-val">+8.0%</span>
        <span class="t-up">&#9650; YoY</span>
      </div>
      <div class="t-item">
        <span class="t-lbl">Nat. Vacancy</span>
        <span class="t-val">4.5%</span>
        <span class="t-up">&#9650; 60bps</span>
      </div>
      <div class="t-item">
        <span class="t-lbl">Top Market</span>
        <span class="t-val">{_top_mkt}</span>
        <span class="t-badge">High</span>
      </div>
      <div class="t-item">
        <span class="t-lbl">10Y Treasury</span>
        <span class="t-val">{_tsy}</span>
      </div>
      <div class="t-item">
        <span class="t-lbl">DSCR Min</span>
        <span class="t-val">1.25x</span>
      </div>
    </div>

    <!-- HERO -->
    <div class="cre-hero">
      <div class="hero-eyebrow">
        <div class="ey-line"></div>
        <span class="ey-text">AI-POWERED &middot; INSTITUTIONAL GRADE &middot; REAL-TIME</span>
        <div class="ey-line ey-line-r"></div>
      </div>
      <div class="hero-title">CRE Intelligence Platform</div>
      <div class="hero-sub">
        Market analysis, 10-year P&amp;L projections, debt structuring, and tax optimization &mdash; delivered in seconds.
      </div>
      <div class="f-chips">
        <span class="f-chip"><span class="f-check">&#10003;</span> Market Scoring</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> 10-Year P&amp;L Pro Forma</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> Financing &amp; DSCR</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> Depreciation Tax Shield</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> Opportunity Zone Benefits</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> Climate &amp; Risk Analysis</span>
        <span class="f-chip"><span class="f-check">&#10003;</span> AI Investment Rationale</span>
      </div>
    </div>

    """, unsafe_allow_html=True)

    # ── Search bar (form keeps input+button connected; Enter or click submits) ─
    with st.form("home_search_form", clear_on_submit=False):
        _sc, _bc = st.columns([7, 1.4])
        with _sc:
            user_input = st.text_input(
                "Search",
                placeholder="e.g., Industrial warehouse in Austin, TX",
                label_visibility="collapsed",
                key="home_search",
            )
        with _bc:
            submitted = st.form_submit_button("Analyze")
        if submitted and user_input:
            if _is_advisor_query(user_input):
                _complete_onboarding(raw_input=user_input)
                st.session_state.adv_home_prompt   = user_input
                st.session_state.adv_auto_generate = True
                st.session_state.adv_navigate      = True
            else:
                _complete_onboarding(**_parse_intent(user_input))
            st.rerun()

    # ── Example queries + property cards + recent searches ───────────────────
    st.markdown(f"""
    <div class="s-examples">
      Try:&nbsp;
      <span class="s-ex" onclick="window.location.href='?q=Multifamily+in+Nashville'">&ldquo;Multifamily in Nashville&rdquo;</span>
      &nbsp;&middot;&nbsp;
      <span class="s-ex" onclick="window.location.href='?q=Office+cap+rates+Chicago'">&ldquo;Office cap rates Chicago&rdquo;</span>
      &nbsp;&middot;&nbsp;
      <span class="s-ex" onclick="window.location.href='?q=Best+Sunbelt+markets+2026'">&ldquo;Best Sunbelt markets 2026&rdquo;</span>
    </div>

    <div class="cre-wrap">
      <div class="prop-hdr">OR SELECT A PROPERTY TYPE</div>
      <div class="prop-grid">{_prop_cards_html}</div>
    </div>
    {_recent_html}
    """, unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD (only shown after onboarding)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stApp"], section[data-testid="stMain"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0d0b04 !important;
    color: #c8b890 !important;
  }}

  .main .block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 1400px;
  }}

  /* ── Text ── */
  p, li, span, label, div {{ color: #c8b890; }}
  .stMarkdown p {{ color: #c8b890; }}
  h1, h2, h3, h4, h5, h6 {{ color: #d4a843 !important; }}
  .stCaption, [data-testid="stCaptionContainer"] p {{ color: #5a4820 !important; }}

  /* ── Metrics ── */
  [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: {GOLD} !important;
  }}
  [data-testid="stMetricLabel"] {{ color: #c8b890 !important; }}

  /* ── Data tables ── */
  [data-testid="stDataFrame"] {{ background: #171309; border: 1px solid #2a2208; border-radius: 6px; }}
  .stDataFrame th {{ background: #171309 !important; color: {GOLD} !important; font-size: 0.65rem !important; font-weight: 600 !important; text-transform: uppercase !important; }}
  .stDataFrame td {{ background: #171309 !important; color: #c8b890 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important; }}

  /* ── Alerts ── */
  [data-testid="stInfo"] {{ background: #131008; border-color: #2a2208; color: #c8b890; }}
  [data-testid="stWarning"] {{ background: #1a1208; border-color: #3a2e10; color: #c8b890; }}
  [data-testid="stSuccess"] {{ background: #0d2a12; border-color: #2a2208; color: #c8b890; }}

  /* ── Agent card ── */
  .agent-card {{
    background: #171309;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 12px 0;
    border: 1px solid #2a2208;
    border-left: 3px solid {GOLD};
  }}
  .agent-card .agent-label {{
    color: {GOLD};
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }}
  .agent-card .agent-text {{
    color: #c8b890;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }}

  /* ── Metric / stat cards ── */
  .metric-card {{
    background: #171309;
    border: 1px solid #2a2208;
    border-top: 2px solid {GOLD};
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    transition: box-shadow 0.2s;
    position: relative;
    overflow: hidden;
  }}
  .metric-card:hover {{ box-shadow: 0 2px 16px rgba(212,168,67,0.12); border-color: {GOLD}; }}
  .metric-card .label {{ font-size: 0.68rem; color: #6a5228; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
  .metric-card .value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 500; color: {GOLD}; margin: 4px 0; letter-spacing: -0.5px; }}
  .metric-card .sub   {{ font-size: 0.72rem; color: #6a5228; }}

  /* ── Section header ── */
  .section-header {{
    background: #171309;
    color: {GOLD};
    padding: 9px 16px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 22px 0 12px 0;
    border: 1px solid #2a2208;
    border-left: 3px solid {GOLD};
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  /* ── Listing card ── */
  .listing-card {{
    background: #171309;
    border: 1px solid #2a2208;
    border-left: 3px solid {GOLD};
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
  }}
  .listing-card .l-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: {GOLD}; }}
  .listing-card .l-address {{ font-size: 0.9rem; color: #c8b890; margin: 2px 0; }}
  .listing-card .l-detail {{ font-size: 0.8rem; color: #8a7040; }}
  .listing-card .l-tag {{
    display: inline-block;
    background: #1e1a08;
    color: {GOLD};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.65rem;
    font-weight: 600;
    margin-right: 4px;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}

  .status-ok    {{ color: #4a9e58; font-weight: 700; }}
  .status-error {{ color: #9e4a4a; font-weight: 700; }}
  .status-run   {{ color: {GOLD}; font-weight: 700; }}
  .status-idle  {{ color: #4a3e18; }}

  /* ── Tabs ── */
  div[data-testid="stTabs"] {{ background: #0d0b04 !important; }}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ border-bottom: 1px solid #2a2208 !important; background: #0d0b04 !important; }}
  div[data-testid="stTabs"] button[role="tab"] {{ color: #4a3e18 !important; font-weight: 500 !important; font-size: 0.82rem !important; }}
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color: {GOLD} !important; border-bottom-color: {GOLD} !important; font-weight: 600 !important; }}

  /* ── Buttons ── */
  .stButton > button {{
    background: transparent !important;
    color: {GOLD} !important;
    border: 1px solid #3a2e10 !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important;
    transition: all 0.2s !important;
  }}
  .stButton > button:hover {{ background: {GOLD} !important; color: #0d0b04 !important; }}

  /* ── Form controls ── */
  [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background: #1e1a0a !important;
    border-color: #3a2e10 !important;
    color: #8a7040 !important;
  }}
  .stTextInput input {{
    background: #1e1a0a !important;
    color: #8a7040 !important;
    border-color: #3a2e10 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
  }}

  /* ── Expander ── */
  [data-testid="stExpander"] {{
    background: #131008 !important;
    border: 1px solid #221e0a !important;
    border-radius: 6px !important;
  }}
  [data-testid="stExpander"] summary {{
    color: #6a5228 !important;
    font-size: 0.82rem !important;
  }}

  /* ── Plotly dark mode ── */
  .js-plotly-plot text {{ fill: #8a7040 !important; }}
  .js-plotly-plot .gtitle {{ fill: {GOLD} !important; }}
  .js-plotly-plot .xtick text, .js-plotly-plot .ytick text {{ fill: #8a7040 !important; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{ background: #0d0b04 !important; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: #0d0b04; }}
  ::-webkit-scrollbar-thumb {{ background: #3a2e10; border-radius: 2px; }}

  /* ── Footer accent line ── */
  body::after {{
    content: '';
    position: fixed;
    bottom: 0; left: 0; width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(212,168,67,0.12) 8%, rgba(212,168,67,0.55) 25%, #d4a843 50%, rgba(212,168,67,0.55) 75%, rgba(212,168,67,0.12) 92%, transparent 100%);
    z-index: 9998; pointer-events: none;
  }}
  .main .block-container {{ padding-bottom: 40px; }}

  /* Keep Streamlit's own topbar hidden so our header shows cleanly */
  [data-testid="stHeader"] {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header banner ────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:14px 24px; background:#1a1208;
            border-radius:10px 10px 0 0; border-bottom:1px solid #3a2e10;
            margin-bottom:0;">
  <!-- Left: Brand -->
  <div style="display:flex; align-items:center; gap:12px;">
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <rect width="32" height="32" rx="6" fill="#2a2008"/>
      <rect x="6" y="18" width="4" height="8" rx="1" fill="#d4a843"/>
      <rect x="12" y="12" width="4" height="14" rx="1" fill="#e8c060"/>
      <rect x="18" y="6" width="4" height="20" rx="1" fill="#d4a843"/>
      <rect x="24" y="14" width="2" height="12" rx="1" fill="#a07830"/>
      <path d="M6 18 L14 12 L20 6 L26 14" stroke="#f0d080" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    </svg>
    <div>
      <div style="font-size:16px; font-weight:500; color:#d4a843; letter-spacing:0.02em; line-height:1.2;">CRE Intelligence Platform</div>
      <div style="font-size:10px; color:#8a7040; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px;">AI-Powered Commercial Real Estate Intelligence</div>
    </div>
  </div>
  <!-- Divider -->
  <div style="width:1px; height:36px; background:#3a2e10; flex-shrink:0;"></div>
  <!-- Right: Purdue -->
  <div style="text-align:right;">
    <div style="font-size:14px; font-weight:500; color:#d4a843;">Purdue University</div>
    <div style="font-size:11px; color:#8a7040; margin-top:2px;">Daniels School of Business</div>
    <div style="font-size:10px; color:#5a4820; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">MSF Program</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Persistent Chat Bar ─────────────────────────────────────────────────────
_cur_intent = st.session_state.user_intent
_cur_pt = _cur_intent.get("property_type")
_cur_loc = _cur_intent.get("location")
if _cur_pt and _cur_loc:
    _focus_label = f"{_cur_pt} in {_cur_loc}"
elif _cur_pt:
    _focus_label = _cur_pt
elif _cur_loc:
    _focus_label = f"All types in {_cur_loc}"
else:
    _focus_label = "All markets"

_bar_left, _bar_right = st.columns([5, 1])
with _bar_left:
    _new_query = st.text_input(
        "Update focus",
        placeholder=f"Currently analyzing: {_focus_label} — type a new query to change",
        label_visibility="collapsed",
        key="chat_bar_input",
    )
    if _new_query:
        if _is_advisor_query(_new_query):
            st.session_state.adv_home_prompt   = _new_query
            st.session_state.adv_auto_generate = True
            st.session_state.adv_navigate      = True
            st.rerun()
        else:
            _new_intent = _parse_intent(_new_query)
            _complete_onboarding(**_new_intent)
            st.rerun()
with _bar_right:
    if st.button("Reset", use_container_width=True, key="reset_focus"):
        st.session_state.onboarding_complete = False
        st.rerun()

if _cur_pt or _cur_loc:
    st.markdown(
        f'<div style="background:#1e1a0a;color:{GOLD};padding:8px 16px;border-radius:6px;'
        f'border:1px solid #3a2e10;border-left:3px solid {GOLD};'
        f'font-size:0.82rem;margin-bottom:12px;letter-spacing:0.04em;text-transform:uppercase;">'
        f'Currently analyzing: <b>{_focus_label}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Sticky header injection ───────────────────────────────────────────────────
components.html("""
<script>
(function() {
  function applySticky() {
    var doc = window.parent.document;
    var blockContainer = doc.querySelector('.main .block-container');
    if (!blockContainer) return false;
    var vertBlock = blockContainer.querySelector('[data-testid="stVerticalBlock"]');
    if (!vertBlock) return false;
    var children = Array.from(vertBlock.children);
    if (children.length < 2) return false;

    // Measure combined height of header (child 0) + chat bar columns (child 1)
    var headerEl  = children[0];
    var chatBarEl = children[1];
    var totalH = headerEl.offsetHeight + chatBarEl.offsetHeight;
    if (totalH === 0) return false;

    // Make them sticky
    [headerEl, chatBarEl].forEach(function(el) {
      el.style.position  = 'sticky';
      el.style.top       = '0';
      el.style.zIndex    = '9999';
      el.style.background = '#0d0b04';
    });

    // Ensure no ancestor blocks sticky with overflow:hidden
    var el = headerEl.parentElement;
    while (el && el !== doc.body) {
      var ov = window.parent.getComputedStyle(el).overflow;
      if (ov === 'hidden') el.style.overflow = 'visible';
      el = el.parentElement;
    }
    return true;
  }

  var attempts = 0;
  var iv = setInterval(function() {
    if (applySticky() || ++attempts > 30) clearInterval(iv);
  }, 200);

  // Reapply on Streamlit reruns via MutationObserver
  var doc = window.parent.document;
  var observer = new MutationObserver(function() { applySticky(); });
  observer.observe(doc.body, { childList: true, subtree: true });
})();
</script>
""", height=0)


# ── Auto-navigate to Investment Advisor tab when routed from chat bar ─────────
if st.session_state.get("adv_navigate"):
    st.session_state.adv_navigate = False
    components.html("""
<script>
(function() {
  function clickAdvisorTab() {
    var doc = window.parent.document;
    var tabs = doc.querySelectorAll('button[role="tab"]');
    for (var i = 0; i < tabs.length; i++) {
      if (tabs[i].innerText.trim() === 'Investment Advisor') {
        tabs[i].click();
        return true;
      }
    }
    return false;
  }
  var attempts = 0;
  var iv = setInterval(function() {
    if (clickAdvisorTab() || ++attempts > 30) clearInterval(iv);
  }, 150);
})();
</script>
""", height=0)


# ── Helpers ──────────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _focus_parts() -> tuple:
    """Return (property_type, location, label) from session state."""
    i = st.session_state.user_intent
    pt, loc = i.get("property_type"), i.get("location")
    parts = [p for p in [pt, loc] if p]
    label = " in ".join(parts) if parts else "All markets"
    return pt, loc, label


def _personalized_title(tab_name: str) -> str:
    """Generate a personalized tab question based on user intent."""
    pt, loc, label = _focus_parts()
    if not pt and not loc:
        # Generic fallback titles
        return {
            "migration": "Where are people and companies moving in the US?",
            "pricing": "Where are the highest profit margins in commercial real estate today?",
            "predictions": "Which companies have announced new plants, factories, and facilities?",
            "buildings": "Cheapest commercial buildings to purchase in the top migration states",
            "announcements": "Where are companies building? Live facility & investment announcements across the US",
            "rates": "How do current interest rates affect CRE cap rates, valuations, and REIT debt risk?",
            "energy": "How are energy and material costs affecting CRE construction?",
            "sustainability": "Is green capital flowing into real estate?",
        }.get(tab_name, "")

    pt_str = pt or "commercial real estate"
    loc_str = loc or "the US"

    return {
        "migration": f"Is {loc_str} attracting population and business growth for {pt_str} investment?",
        "pricing": f"What are the profit margins for {pt_str} in {loc_str}?",
        "predictions": f"Which companies are building {pt_str} facilities in or near {loc_str}?",
        "buildings": f"Cheapest {pt_str} properties in {loc_str}",
        "announcements": f"{pt_str} facility announcements in the {loc_str} region",
        "rates": f"How do current interest rates affect {pt_str} cap rates and valuations?",
        "energy": f"How are energy costs affecting {pt_str} construction in {loc_str}?",
        "sustainability": f"ESG and sustainability trends for {pt_str}",
    }.get(tab_name, "")


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_insight(tab_name: str, property_type: str, location: str, data_summary: str) -> str:
    """Generate a personalized 2-3 sentence insight using Groq."""
    import os
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=key)
        pt_str = property_type or "commercial real estate"
        loc_str = location or "the US"
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a CRE investment analyst. Give exactly 2-3 sentences of actionable insight. Be specific with numbers when possible. No headers or bullet points."},
                {"role": "user", "content": f"Tab: {tab_name}. The investor is focused on {pt_str} in {loc_str}.\n\nRelevant data:\n{data_summary[:800]}\n\nGive a brief, personalized insight for this investor."},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def _show_tab_header(tab_name: str, description: str, agent_key: str, data_summary: str = ""):
    """Show personalized title, insight, and agent timestamp at the top of each tab."""
    title = _personalized_title(tab_name)
    st.markdown(f"#### {title}")
    st.markdown(description)
    pt, loc, label = _focus_parts()
    if pt or loc:
        insight = _generate_insight(tab_name, pt, loc, data_summary)
        if insight:
            st.markdown(
                f'<div style="background:#171309;border-left:3px solid {GOLD};padding:10px 16px;'
                f'border-radius:4px;margin:8px 0 12px 0;font-size:0.88rem;color:#e8e9ed;border:1px solid #2a2208;">'
                f'<b style="color:{GOLD};">Insight for {label}:</b> {insight}</div>',
                unsafe_allow_html=True,
            )
    agent_last_updated(agent_key)

def _intent_matches_state(loc_str: str, state_name: str, state_abbr: str) -> bool:
    """Check if user's location intent matches a state."""
    if not loc_str:
        return False
    loc_lower = loc_str.lower().strip()
    sn_lower = state_name.lower()
    sa_lower = state_abbr.lower()
    # Exact or near-exact matches only — avoid substring false positives
    if loc_lower == sn_lower or loc_lower == sa_lower:
        return True
    # "Indiana" in "warehouse in Indiana" — full state name in location string
    if sn_lower in loc_lower:
        return True
    # Location is the full state name
    if loc_lower in sn_lower and len(loc_lower) > 3:
        return True
    # Check if abbreviation appears as a whole word (e.g., "IN" not as part of "Indiana")
    import re
    if re.search(r'\b' + re.escape(sa_lower) + r'\b', loc_lower):
        return True
    return False

def _intent_matches_metro(loc_str: str, metro: str) -> bool:
    """Check if user's location intent matches a metro area."""
    if not loc_str:
        return False
    loc_lower = loc_str.lower()
    metro_lower = metro.lower()
    # Check city or state portion
    for part in loc_lower.replace(",", " ").split():
        if len(part) > 2 and part in metro_lower:
            return True
    return loc_lower in metro_lower or metro_lower.startswith(loc_lower.split(",")[0].strip())

def _intent_matches_property_type(user_pt: str, data_pt: str) -> bool:
    """Check if user's property_type intent matches a data row's property type."""
    if not user_pt:
        return False
    return user_pt.lower() in data_pt.lower()

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def _active_pt() -> str | None:
    """Return the active property type focus (e.g. 'Multifamily') or None."""
    return st.session_state.user_intent.get("property_type")


# Maps user_intent property_type → column/key names used in each tab
_PT_VAC_COL  = {"Industrial": "Industrial", "Multifamily": "Multifamily",
                "Office": "Office", "Retail": "Retail"}
_PT_CAP_COL  = {"Industrial": "Industrial", "Multifamily": "Multifamily",
                "Office": "Office", "Retail": "Retail"}
_PT_RG_LBL   = {"Multifamily": "Multifamily", "Industrial": "Industrial PSF",
                "Office": "Office PSF", "Retail": "Retail PSF"}
_PT_RG_KEY   = {"Multifamily": "multifamily", "Industrial": "industrial_psf",
                "Office": "office_psf", "Retail": "retail_psf"}


def _pt_focus_banner(pt: str | None):
    """Show a subtle banner when a property type focus is active."""
    if pt:
        st.markdown(
            f'<div style="background:#1a1500;border:1px solid #3a3010;border-radius:6px;'
            f'padding:7px 14px;margin-bottom:10px;font-size:0.82rem;color:#c8a040;">'
            f'Focus: <b>{pt}</b> &nbsp;·&nbsp; '
            f'<span style="color:#6a5228;">Relevant columns highlighted — update focus via the chat bar below</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def stale_banner(cache_key: str):
    c = read_cache(cache_key)
    if c["data"] is None:
        st.warning(f" Agent is fetching {cache_key} data for the first time — please wait ~30 seconds and refresh.")
        return False
    age = cache_age_label(cache_key)
    if c.get("stale"):
        st.warning(f" Data is stale (last updated {age}). Agent may be restarting.")
    else:
        st.caption(f" Last updated: {age} · Auto-refreshes in background")
    return True

def agent_last_updated(agent_name: str):
    st.caption(f"Last updated: {cache_age_label(agent_name)}")


def gauge_card(title: str, label: str, score: int, summary: str,
               agent_num: str, age_label: str,
               confidence: str = "High",
               low_good: bool = False,
               scale_labels: tuple = ("0", "25", "50", "75", "100")) -> str:
    """
    Returns HTML for a segmented gauge card matching the Inflation Regime design.
    low_good=True  → low score = green (e.g. tight credit is bad, loose is good reversed)
    low_good=False → high score = green (e.g. strong economy, loose credit)
    """
    total = 20
    filled = round(score / 100 * total)

    def _color(i):
        pct = score
        if low_good:
            # Low = good (green), high = bad (red)  — used for Rate Environment (high rates = bearish)
            if pct <= 35:   active = "#2e7d32"
            elif pct <= 60: active = "#d4a843"
            else:           active = "#b71c1c"
        else:
            if pct >= 65:   active = "#2e7d32"
            elif pct >= 40: active = "#d4a843"
            else:           active = "#b71c1c"
        return active if i < filled else "#2a2208"

    blocks = "".join(
        f"<div style='width:26px;height:26px;background:{_color(i)};"
        f"border-radius:3px;margin:0 3px;flex-shrink:0;'></div>"
        for i in range(total)
    )

    # Label color based on the filled color logic
    if low_good:
        lc = "#66bb6a" if score <= 35 else ("#d4a843" if score <= 60 else "#ef5350")
    else:
        lc = "#66bb6a" if score >= 65 else ("#d4a843" if score >= 40 else "#ef5350")

    s0, s25, s50, s75, s100 = scale_labels

    return f"""
    <div style="background:linear-gradient(135deg,#1a1208 0%,#141410 100%);
                border:1px solid #3a3a2a;border-radius:10px;
                padding:28px 36px 22px 36px;margin-bottom:24px;">
      <div style="text-align:center;letter-spacing:0.18em;font-size:0.78rem;
                  color:#a09070;margin-bottom:6px;">{title}</div>
      <div style="text-align:center;font-size:2.4rem;font-weight:800;
                  color:{lc};letter-spacing:0.06em;margin-bottom:20px;">{label}</div>
      <div style="text-align:right;font-size:0.8rem;color:#a09070;
                  letter-spacing:0.08em;margin-bottom:8px;">
        SCORE: <span style="color:#e8dfc4;font-weight:700;">{score} / 100</span>
      </div>
      <div style="display:flex;align-items:center;justify-content:center;margin-bottom:6px;">
        {blocks}
      </div>
      <div style="position:relative;height:16px;margin:0 6px 2px 6px;">
        <div style="position:absolute;left:calc({score}% - 7px);top:0;
                    width:0;height:0;
                    border-left:7px solid transparent;
                    border-right:7px solid transparent;
                    border-top:12px solid #e8dfc4;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:0.74rem;color:#666;margin:0 4px 16px 4px;">
        <span>{s0}</span><span>{s25}</span><span>{s50}</span><span>{s75}</span><span>{s100}</span>
      </div>
      <div style="font-size:0.88rem;color:#c8bfa8;line-height:1.65;margin-bottom:14px;">
        {summary}
      </div>
      <div style="padding-top:10px;border-top:1px solid #2a2208;
                  font-size:0.72rem;color:#666;letter-spacing:0.04em;">
        {agent_num} &nbsp;|&nbsp; Confidence: <span style="color:#d4a843;">{confidence}</span>
        &nbsp;|&nbsp; Last Updated: {age_label}
      </div>
    </div>"""



# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
main_tab_re, main_tab_advisor, main_tab_energy, main_tab_macro, main_tab_about = st.tabs(["Real Estate", "Investment Advisor", "Energy", "Macro Environment", "About"])

with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab_vacancy, tab_land, tab_caprate, tab_rent, tab_oz, tab_score, tab_climate = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "Vacancy Monitor",
        "Land & Development",
        "Cap Rate Monitor",
        "Rent Growth",
        "Opportunity Zones",
        "Market Score",
        "Climate Risk",
    ])


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 1 — POPULATION & MIGRATION
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab1:
        _show_tab_header(
            "migration",
            "Agent 1 tracks **domestic population migration** and **corporate headquarters relocations** "
            "to surface the highest-demand markets for CRE investment. Updates every 6 hours.",
            "migration",
        )

        cache = read_cache("migration")
        if not stale_banner("migration") or cache["data"] is None:
            st.stop()

        data     = cache["data"]
        mig_df   = pd.DataFrame(data["migration"])
        metros_df = pd.DataFrame(data["metros"])

        # ── Intent-based highlighting ──────────────────────────────────────────
        _mig_intent = st.session_state.user_intent
        _mig_state = _mig_intent.get("state")
        _mig_abbr = _mig_intent.get("state_abbr")
        _mig_city = _mig_intent.get("city")
        _mig_region = _mig_intent.get("region")
        _mig_region_states = _mig_intent.get("region_states")
        _mig_pt = _mig_intent.get("property_type")

        # Case A: Exact state or city specified → match by state_abbr
        if _mig_abbr:
            mig_df["_match"] = mig_df["state_abbr"] == _mig_abbr
            mig_df = mig_df.sort_values(["_match", "composite_score"], ascending=[False, False]).reset_index(drop=True)
            matched = mig_df[mig_df["_match"]]
            if not matched.empty:
                s = matched.iloc[0]
                _focus_label = f"{_mig_city}, {s['state_name']}" if _mig_city else s["state_name"]
                st.success(
                    f"Your focus area: **{_focus_label}** — "
                    f"Composite Score: {s['composite_score']}, Pop Growth: {s['pop_growth_pct']:+.1f}%, "
                    f"Business Score: {s['biz_score']}"
                )
            mig_df = mig_df.drop(columns=["_match"])
            # Sort metros
            if _mig_city:
                metros_df["_match"] = metros_df["Metro"].apply(lambda m: _intent_matches_metro(_mig_city, m))
                metros_df = metros_df.sort_values(["_match"], ascending=False).reset_index(drop=True)
                metros_df = metros_df.drop(columns=["_match"])

        # Case B: Region specified → find top migration scorer in that region
        elif _mig_region and _mig_region_states:
            _region_df = mig_df[mig_df["state_abbr"].isin(_mig_region_states)]
            if not _region_df.empty:
                _top_region = _region_df.sort_values("composite_score", ascending=False).iloc[0]
                st.success(
                    f"Your focus area: **{_top_region['state_name']}** (top {_mig_region} market) — "
                    f"Composite Score: {_top_region['composite_score']}, Pop Growth: {_top_region['pop_growth_pct']:+.1f}%, "
                    f"Business Score: {_top_region['biz_score']}"
                )
            # Sort region states to top
            mig_df["_match"] = mig_df["state_abbr"].isin(_mig_region_states)
            mig_df = mig_df.sort_values(["_match", "composite_score"], ascending=[False, False]).reset_index(drop=True)
            mig_df = mig_df.drop(columns=["_match"])

        # Case C: Property type only, no location → recommend #1 migration destination
        elif _mig_pt and not _mig_state and not _mig_city and not _mig_region:
            _top = mig_df.iloc[0]
            st.info(
                f"You're looking for **{_mig_pt}** properties. "
                f"Recommended market: **{_top['state_name']}** — #1 migration destination "
                f"(Composite: {_top['composite_score']}, Pop Growth: {_top['pop_growth_pct']:+.1f}%)"
            )

        # ── KPI strip (mockup stat cards) ─────────────────────────────────────
        top1      = mig_df.iloc[0]
        top_gain  = mig_df[mig_df["pop_growth_pct"] > 0].shape[0]
        top_loss  = mig_df[mig_df["pop_growth_pct"] < 0].shape[0]
        avg_grow  = mig_df["pop_growth_pct"].mean()

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _avg_bg  = "#0d2a12" if avg_grow > 0 else "#2a0d0d"
        _avg_fc  = "#4a9e58" if avg_grow > 0 else "#9e4a4a"
        _avg_lbl = "▲ Above avg" if avg_grow > 0 else "▼ Below avg"

        _sc1.markdown(f"""
        <div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#d4a843;"></div>
          <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">#1 DESTINATION</div>
          <div style="font-size:36px;font-weight:500;color:#d4a843;line-height:1;margin-bottom:4px;letter-spacing:-0.5px;">{top1["state_abbr"]}</div>
          <div style="font-size:13px;color:#8a7040;margin-bottom:10px;">{top1["state_name"]}</div>
          <div style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:4px 10px;border-radius:5px;background:#0d2a12;color:#4a9e58;">▲ Net inflow</div>
        </div>""", unsafe_allow_html=True)

        _sc2.markdown(f"""
        <div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#d4a843;opacity:0.35;"></div>
          <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">STATES GROWING</div>
          <div style="font-size:36px;font-weight:500;color:#d4a843;line-height:1;margin-bottom:4px;letter-spacing:-0.5px;">{top_gain}</div>
          <div style="font-size:13px;color:#8a7040;margin-bottom:10px;">Positive net migration</div>
          <div style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:4px 10px;border-radius:5px;background:#0d2a12;color:#4a9e58;">▲ Net inflow states</div>
        </div>""", unsafe_allow_html=True)

        _sc3.markdown(f"""
        <div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#d4a843;opacity:0.35;"></div>
          <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">STATES SHRINKING</div>
          <div style="font-size:36px;font-weight:500;color:#d4a843;line-height:1;margin-bottom:4px;letter-spacing:-0.5px;">{top_loss}</div>
          <div style="font-size:13px;color:#8a7040;margin-bottom:10px;">Negative net migration</div>
          <div style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:4px 10px;border-radius:5px;background:#2a0d0d;color:#9e4a4a;">▼ Outflow states</div>
        </div>""", unsafe_allow_html=True)

        _sc4.markdown(f"""
        <div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:#d4a843;opacity:0.35;"></div>
          <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">AVG POP GROWTH</div>
          <div style="font-size:36px;font-weight:500;color:#d4a843;line-height:1;margin-bottom:4px;letter-spacing:-0.5px;">{avg_grow:.2f}<span style="font-size:18px;">%</span></div>
          <div style="font-size:13px;color:#8a7040;margin-bottom:10px;">All states YoY</div>
          <div style="display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:4px 10px;border-radius:5px;background:{_avg_bg};color:{_avg_fc};">{_avg_lbl}</div>
        </div>""", unsafe_allow_html=True)

        # ── Two-panel: Top Destination States + Corporate HQ Relocations ──────
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        _pan_l, _pan_r = st.columns(2)

        with _pan_l:
            _top5 = mig_df.head(5)
            _max_net = max((_top5["pop_growth_pct"] * _top5["population"] / 100 / 1000).abs().max(), 1)
            _bar_rows = ""
            for _, _row in _top5.iterrows():
                _net_k = int(_row["pop_growth_pct"] * _row["population"] / 100 / 1000)
                _bw = max(int(abs(_net_k) / _max_net * 100), 5)
                _bar_rows += f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                  <span style="font-size:12px;color:#8a7040;width:28px;flex-shrink:0;">{_row["state_abbr"]}</span>
                  <div style="flex:1;height:6px;background:#1e1a08;border-radius:3px;overflow:hidden;">
                    <div style="width:{_bw}%;height:100%;border-radius:3px;background:linear-gradient(90deg,#d4a843,#e8c060);"></div>
                  </div>
                  <span style="font-size:12px;color:#d4a843;width:56px;text-align:right;">+{_net_k:,}K</span></div>"""
            st.markdown(f"""<div style="background:#131008;border:1px solid #221e0a;border-radius:8px;padding:16px 18px;">
              <div style="font-size:11px;color:#6a5228;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:14px;">Top Destination States</div>
              {_bar_rows}</div>""", unsafe_allow_html=True)

        with _pan_r:
            _inflows = [
                ("TX", "Tesla", "Austin, TX", "+12,000 jobs"),
                ("FL", "Citadel / Goldman Sachs", "Miami, FL", "+8,200 jobs"),
                ("NC", "Apple / Google", "Raleigh, NC", "+5,800 jobs"),
                ("AZ", "TSMC / Intel", "Phoenix, AZ", "+6,400 jobs"),
                ("TN", "Oracle / Ford BEV", "Nashville, TN", "+4,500 jobs"),
            ]
            _outflows = [
                ("CA", "Exodus", "San Francisco, CA", "−31K jobs"),
                ("NY", "Outflow", "New York, NY", "−18K jobs"),
                ("IL", "Outflow", "Chicago, IL", "−9K jobs"),
            ]
            _flow_html = ""
            for _, _co, _city, _jobs in _inflows:
                _flow_html += f"""<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #1a1608;">
                  <div style="width:22px;height:22px;border-radius:50%;background:#0d2a12;color:#4a9e58;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;">→</div>
                  <span style="font-size:12px;color:#8a7040;flex:1;">{_co} · {_city}</span>
                  <span style="font-size:12px;color:#4a9e58;font-weight:600;">{_jobs}</span></div>"""
            for _, _co, _city, _jobs in _outflows:
                _flow_html += f"""<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #1a1608;">
                  <div style="width:22px;height:22px;border-radius:50%;background:#2a0d0d;color:#9e4a4a;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;">←</div>
                  <span style="font-size:12px;color:#8a7040;flex:1;">{_co} · {_city}</span>
                  <span style="font-size:12px;color:#9e4a4a;font-weight:600;">{_jobs}</span></div>"""
            st.markdown(f"""<div style="background:#131008;border:1px solid #221e0a;border-radius:8px;padding:16px 18px;">
              <div style="font-size:11px;color:#6a5228;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px;">Corporate HQ Relocations · Recent</div>
              {_flow_html}</div>""", unsafe_allow_html=True)

        # ── MAP ────────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        _map_intent = st.session_state.user_intent
        _map_city = _map_intent.get("city")
        _map_state = _map_intent.get("state")
        _map_abbr = _map_intent.get("state_abbr")

        # Auto-select map level based on user intent
        _default_level = 0  # National
        if _map_city:
            _default_level = 2  # Metro
        elif _map_abbr:
            _default_level = 1  # State

        _map_level = st.radio(
            "Map Resolution",
            ["National", "State (Counties)", "Metro (Neighborhoods)"],
            index=_default_level,
            horizontal=True,
            key="migration_map_level",
        )

        _show_national_map = (_map_level == "National")
        _show_county_map = (_map_level == "State (Counties)")
        _show_metro_map = (_map_level == "Metro (Neighborhoods)")

        # ── COUNTY-LEVEL MAP ──────────────────────────────────────────────────
        if _show_county_map:
            from src.county_migration import get_county_data, COUNTY_GEOJSON_URL

            @st.cache_data(ttl=86400, show_spinner="Loading county boundaries...")
            def _load_county_geojson():
                import urllib.request, json as _json, ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(COUNTY_GEOJSON_URL, context=ctx) as resp:
                    return _json.loads(resp.read().decode())

            _sel_state_abbr = _map_abbr
            _all_abbrs = sorted(mig_df["state_abbr"].tolist())
            _default_idx = _all_abbrs.index(_sel_state_abbr) if _sel_state_abbr and _sel_state_abbr in _all_abbrs else 0
            if _sel_state_abbr and st.session_state.get("county_state_sel") != _sel_state_abbr:
                st.session_state["county_state_sel"] = _sel_state_abbr
            _sel_state_abbr = st.selectbox("Select State", _all_abbrs, index=_default_idx,
                                           format_func=lambda a: f"{a} — {_US_STATES.get(a, a)}", key="county_state_sel")

            county_df = get_county_data(_sel_state_abbr)
            if county_df.empty:
                st.info(f"No county data available for {_sel_state_abbr}.")
            else:
                counties_geojson = _load_county_geojson()
                section(f"County Migration — {_US_STATES.get(_sel_state_abbr, _sel_state_abbr)}")

                fig_county = go.Figure(go.Choroplethmapbox(
                    geojson=counties_geojson,
                    locations=county_df["fips"],
                    z=county_df["migration_score"],
                    featureidkey="id",
                    colorscale=[
                        [0.0, "#7f0000"], [0.25, "#c62828"], [0.45, "#d4c5a9"],
                        [0.55, "#a5d6a7"], [0.75, "#2e7d32"], [1.0, "#1b5e20"],
                    ],
                    zmin=0, zmax=100,
                    marker_line_width=0.5, marker_line_color="#333",
                    marker_opacity=0.85,
                    colorbar=dict(title=dict(text="Score", font=dict(size=10, color="#e8e9ed")),
                                  tickfont=dict(size=9, color="#e8e9ed"), thickness=12, len=0.6,
                                  bgcolor="#171309", bordercolor="#2a2208"),
                    text=county_df.apply(
                        lambda r: f"<b>{r['name']}</b><br>"
                                  f"Score: {r['migration_score']}<br>"
                                  f"Pop Growth: {r['pop_growth_pct']:+.1f}%<br>"
                                  f"Pop: {r['population']:,}<br>"
                                  f"{r['top_driver']}", axis=1),
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig_county.update_geos(
                    fitbounds="locations", visible=False,
                    bgcolor="#111111",
                    landcolor="#1a1a1a",
                    lakecolor="#0a0a0a",
                )
                fig_county.update_layout(
                    paper_bgcolor="#111111",
                    plot_bgcolor="#111111",
                    geo=dict(bgcolor="#111111"),
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=520,
                    font=dict(family="DM Sans", color="#e8e9ed"),
                )
                st.plotly_chart(fig_county, use_container_width=True, config={"displayModeBar": False})

                # County table
                section(f"County Rankings — {_US_STATES.get(_sel_state_abbr, _sel_state_abbr)}")
                _cdisp = county_df.sort_values("migration_score", ascending=False)[
                    ["name", "population", "pop_growth_pct", "migration_score", "top_driver"]
                ].copy()
                _cdisp.columns = ["County", "Population", "Pop Growth %", "Migration Score", "Key Driver"]
                _cdisp["Population"] = _cdisp["Population"].apply(lambda x: f"{x:,}")
                _cdisp["Pop Growth %"] = _cdisp["Pop Growth %"].apply(lambda x: f"{x:+.1f}%")
                st.dataframe(_cdisp, use_container_width=True, hide_index=True)

        # ── METRO / NEIGHBORHOOD MAP ──────────────────────────────────────────
        elif _show_metro_map:
            from src.zip_migration import (
                get_zip_data, get_available_metros, get_metro_center,
                generate_simple_metro_data, _STATE_CENTERS, _CITY_COORDS_FALLBACK,
            )

            _available = get_available_metros()
            _default_metro = None
            _use_fallback = False

            if _map_city:
                # Try matching to a detailed metro
                for m in _available:
                    if _map_city.lower() in m.lower() or m.lower() in _map_city.lower():
                        _default_metro = m
                        break

            if _default_metro:
                # Has detailed metro data — use selectbox
                if st.session_state.get("metro_sel") != _default_metro:
                    st.session_state["metro_sel"] = _default_metro
                _metro_idx = _available.index(_default_metro)
                _sel_metro = st.selectbox("Select Metro", _available, index=_metro_idx, key="metro_sel")
                zip_df = get_zip_data(_sel_metro)
                center = get_metro_center(_sel_metro)
            else:
                # No detailed data — use fallback generation
                _use_fallback = True
                _fb_city = _map_city
                _fb_abbr = _map_abbr
                _fb_center = None

                # Try city coords fallback
                if _fb_city and _fb_city.lower() in _CITY_COORDS_FALLBACK:
                    lat, lon, st_code = _CITY_COORDS_FALLBACK[_fb_city.lower()]
                    _fb_center = (lat, lon)
                    _fb_abbr = _fb_abbr or st_code
                # Try state centers
                elif _fb_abbr and _fb_abbr in _STATE_CENTERS:
                    _fb_city_name, lat, lon = _STATE_CENTERS[_fb_abbr]
                    _fb_center = (lat, lon)
                    _fb_city = _fb_city or _fb_city_name

                if _fb_center and _fb_city and _fb_abbr:
                    # Also show selectbox with detailed metros as option
                    _all_options = [f"{_fb_city} (generated)"] + _available
                    _sel = st.selectbox("Select Metro", _all_options, index=0, key="metro_sel_fb")
                    if _sel != _all_options[0]:
                        # User switched to a detailed metro
                        zip_df = get_zip_data(_sel)
                        center = get_metro_center(_sel)
                        _use_fallback = False
                        _sel_metro = _sel
                    else:
                        zip_df = generate_simple_metro_data(_fb_city, _fb_abbr, _fb_center[0], _fb_center[1])
                        center = _fb_center
                        _sel_metro = _fb_city
                else:
                    # Total fallback — show selectbox of available metros
                    _sel_metro = st.selectbox("Select Metro", _available, index=0, key="metro_sel")
                    zip_df = get_zip_data(_sel_metro)
                    center = get_metro_center(_sel_metro)

            print(f"[Migration Map] city={_map_city}, state={_map_abbr}, metro={_default_metro or 'fallback'}, fallback={_use_fallback}")

            if zip_df.empty or not center:
                st.info(f"No neighborhood data available.")
            else:
                _section_label = f"Neighborhood Migration — {_sel_metro}"
                if _use_fallback:
                    _section_label = f"Area Migration — {_sel_metro} (estimated)"
                section(_section_label)

                # Color by score
                _type_shapes = {"Urban Core": "circle", "Suburban": "diamond", "Exurban": "square"}
                fig_metro = go.Figure(go.Scattermapbox(
                    lat=zip_df["lat"], lon=zip_df["lon"],
                    mode="markers+text",
                    marker=dict(
                        size=zip_df["migration_score"].clip(10, 95) / 5,
                        color=zip_df["migration_score"],
                        colorscale=[
                            [0.0, "#7f0000"], [0.3, "#c62828"], [0.5, "#d4c5a9"],
                            [0.7, "#2e7d32"], [1.0, "#1b5e20"],
                        ],
                        cmin=0, cmax=100,
                        showscale=True,
                        colorbar=dict(title=dict(text="Score", font=dict(color="#e8e9ed")),
                                      thickness=10, len=0.5,
                                      tickfont=dict(color="#e8e9ed")),
                    ),
                    text=zip_df["name"],
                    textposition="top center",
                    textfont=dict(size=9, color="#e8e9ed"),
                    customdata=zip_df[["name", "migration_score", "pop_growth_pct", "median_rent_growth_pct", "neighborhood_type"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Score: %{customdata[1]}<br>"
                        "Pop Growth: %{customdata[2]:+.1f}%<br>"
                        "Rent Growth: %{customdata[3]:+.1f}%<br>"
                        "Type: %{customdata[4]}<extra></extra>"
                    ),
                ))
                _map_zoom = 9.5 if _use_fallback else 10.5
                fig_metro.update_layout(
                    mapbox=dict(
                        style="carto-darkmatter",
                        center=dict(lat=center[0], lon=center[1]),
                        zoom=_map_zoom,
                    ),
                    paper_bgcolor="#0d0b04",
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=520,
                    font=dict(family="DM Sans", color="#e8e9ed"),
                )
                st.plotly_chart(fig_metro, use_container_width=True, config={"displayModeBar": False})

                # Neighborhood table
                section(f"Neighborhood Rankings — {_sel_metro}")
                _zdisp = zip_df.sort_values("migration_score", ascending=False)[
                    ["name", "neighborhood_type", "migration_score", "pop_growth_pct", "median_rent_growth_pct"]
                ].copy()
                _zdisp.columns = ["Neighborhood", "Type", "Migration Score", "Pop Growth %", "Rent Growth %"]
                _zdisp["Pop Growth %"] = _zdisp["Pop Growth %"].apply(lambda x: f"{x:+.1f}%")
                _zdisp["Rent Growth %"] = _zdisp["Rent Growth %"].apply(lambda x: f"{x:+.1f}%")
                st.dataframe(_zdisp, use_container_width=True, hide_index=True)

        # ── NATIONAL MAP (default) ────────────────────────────────────────────
        else:
            pass  # fall through to existing national map code below

        if _show_national_map:
            # City coordinate lookup for zoom
            _CITY_COORDS = {
                "los angeles": (34.05, -118.24), "new york": (40.71, -74.01), "chicago": (41.88, -87.63),
                "houston": (29.76, -95.37), "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
                "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16), "dallas": (32.78, -96.80),
                "austin": (30.27, -97.74), "san francisco": (37.77, -122.42), "seattle": (47.61, -122.33),
                "denver": (39.74, -104.99), "nashville": (36.16, -86.78), "miami": (25.76, -80.19),
                "atlanta": (33.75, -84.39), "charlotte": (35.23, -80.84), "raleigh": (35.78, -78.64),
                "orlando": (28.54, -81.38), "tampa": (27.95, -82.46), "salt lake city": (40.76, -111.89),
                "las vegas": (36.17, -115.14), "portland": (45.52, -122.68), "boise": (43.62, -116.21),
                "minneapolis": (44.98, -93.27), "indianapolis": (39.77, -86.16), "columbus": (39.96, -82.99),
                "detroit": (42.33, -83.05), "boston": (42.36, -71.06), "kansas city": (39.10, -94.58),
                "richmond": (37.54, -77.44), "pittsburgh": (40.44, -79.99), "sacramento": (38.58, -121.49),
            }
            # State center coordinates for state-level zoom
            _STATE_COORDS = {
                "AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.3, -111.7), "AR": (34.8, -92.2),
                "CA": (37.2, -119.5), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
                "FL": (28.6, -82.5), "GA": (33.0, -83.5), "HI": (20.8, -156.3), "ID": (44.4, -114.6),
                "IL": (40.0, -89.2), "IN": (39.8, -86.3), "IA": (42.0, -93.5), "KS": (38.5, -98.3),
                "KY": (37.8, -85.3), "LA": (31.0, -92.0), "ME": (45.3, -69.2), "MD": (39.0, -76.7),
                "MA": (42.2, -71.5), "MI": (44.3, -85.5), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
                "MO": (38.4, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8), "NV": (39.3, -116.6),
                "NH": (43.7, -71.6), "NJ": (40.1, -74.7), "NM": (34.5, -106.0), "NY": (42.9, -75.5),
                "NC": (35.5, -79.8), "ND": (47.5, -100.5), "OH": (40.4, -82.8), "OK": (35.5, -97.5),
                "OR": (44.0, -120.5), "PA": (41.0, -77.5), "RI": (41.7, -71.5), "SC": (34.0, -81.0),
                "SD": (44.4, -100.2), "TN": (35.9, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
                "VT": (44.0, -72.7), "VA": (37.5, -78.9), "WA": (47.4, -120.7), "WV": (38.6, -80.6),
                "WI": (44.6, -89.7), "WY": (43.0, -107.5), "DC": (38.9, -77.0),
            }

            # Determine map mode
            _zoom_coords = None
            _zoom_scale = None
            _map_title_suffix = "Where America is Moving"
            _focus_abbr = _map_abbr  # state to highlight

            coords = _CITY_COORDS.get(_map_city.lower()) if _map_city else None
            if coords:
                _zoom_coords = coords
                _zoom_scale = 3.5
                _map_title_suffix = f"Focused on {_map_city}" + (f", {_map_state}" if _map_state else "")
            # Try to resolve state from city if not already known
            if not _focus_abbr and _map_state:
                _focus_abbr = _STATE_NAME_TO_ABBR.get(_map_state.lower(), "").upper() or None
            elif _map_state and _map_abbr:
                coords = _STATE_COORDS.get(_map_abbr)
                if coords:
                    _zoom_coords = coords
                    _zoom_scale = 2.5
                _map_title_suffix = f"Focused on {_map_state}"

            section(f" US Population Growth Map — {_map_title_suffix}")

            map_col, legend_col = st.columns([3, 1])
            with map_col:
                # Build marker sizes: highlight the focus state
                _marker_line_widths = []
                _marker_line_colors = []
                for abbr in mig_df["state_abbr"]:
                    if _focus_abbr and abbr == _focus_abbr:
                        _marker_line_widths.append(3)
                        _marker_line_colors.append(GOLD)
                    else:
                        _marker_line_widths.append(0.5)
                        _marker_line_colors.append("#999")

                def _classify(score):
                    if score >= 70:   return "High Growth"
                    elif score >= 55: return "Growing"
                    elif score >= 45: return "Stable"
                    elif score >= 25: return "Declining"
                    else:             return "High Outflow"

                fig_map = go.Figure(go.Choropleth(
                    locations=mig_df["state_abbr"],
                    z=mig_df["composite_score"],
                    locationmode="USA-states",
                    colorscale=[
                        [0.0,  "#7f0000"],
                        [0.25, "#c62828"],
                        [0.45, "#e57373"],
                        [0.55, "#d4c5a9"],
                        [0.70, "#81c784"],
                        [0.85, "#2e7d32"],
                        [1.0,  "#1b5e20"],
                    ],
                    zmin=0, zmax=100,
                    marker=dict(line=dict(width=_marker_line_widths, color=_marker_line_colors)),
                    colorbar=dict(
                        title=dict(text="Migration<br>Score", font=dict(size=11, color="#c8b890")),
                        tickfont=dict(size=10, color="#c8b890"),
                        thickness=14, len=0.65,
                        bgcolor="#1a1208",
                        bordercolor="#3a3a2a",
                        borderwidth=1,
                    ),
                    text=mig_df.apply(
                        lambda r: (
                            f"<b>{r['state_name']}</b><br>"
                            f"Migration Score: {r['composite_score']:.0f}<br>"
                            f"Classification: {_classify(r['composite_score'])}<br>"
                            f"Key Drivers: {r['growth_drivers']}"
                        ),
                        axis=1
                    ),
                    hovertemplate="%{text}<extra></extra>",
                ))

                # Add city marker if zoomed to city
                if _map_city and _zoom_coords:
                    fig_map.add_trace(go.Scattergeo(
                        lat=[_zoom_coords[0]], lon=[_zoom_coords[1]],
                        mode="markers+text",
                        marker=dict(size=14, color=GOLD, line=dict(width=2, color=BLACK), symbol="star"),
                        text=[_map_city],
                        textposition="top center",
                        textfont=dict(size=12, color=BLACK, family="Source Sans Pro"),
                        showlegend=False,
                        hovertemplate=f"<b>{_map_city}</b><extra></extra>",
                    ))


                # Set map center and zoom
                _geo_base = dict(
                    scope="usa", showlakes=True, lakecolor="#1a2535",
                    bgcolor="#0d0b04", showland=True, landcolor="#1e2018",
                    showframe=False, showcoastlines=False,
                    subunitcolor="#3a3a2a", subunitwidth=0.5,
                )
                if _zoom_coords and _zoom_scale:
                    fig_map.update_layout(
                        geo=dict(**_geo_base,
                                 projection_scale=_zoom_scale,
                                 center=dict(lat=_zoom_coords[0], lon=_zoom_coords[1])),
                    )
                else:
                    fig_map.update_layout(
                        geo=dict(**_geo_base,
                                 projection_scale=1, center=dict(lat=38, lon=-96)),
                    )

                fig_map.update_layout(
                    paper_bgcolor="#1a1208",
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=460,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    dragmode=False,
                )
                st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
                st.caption(
                    "Darker green indicates states with the strongest combined population inflow and business migration. "
                    "These markets historically see the earliest and sharpest increases in CRE demand — "
                    "particularly for multifamily, industrial, and mixed-use properties."
                )

            with legend_col:
                st.markdown(
                    "<div style='margin-top:24px;padding:14px 12px;background:#1a1208;"
                    "border:1px solid #3a3a2a;border-radius:8px;'>"
                    "<div style='font-size:0.78rem;font-weight:700;color:#d4a843;"
                    "letter-spacing:0.08em;margin-bottom:10px;'>MIGRATION SCORE</div>",
                    unsafe_allow_html=True
                )
                for color, label, desc in [
                    ("#1b5e20", "High Growth", "70 – 100"),
                    ("#2e7d32", "Growing",     "55 – 70"),
                    ("#d4c5a9", "Stable",      "45 – 55"),
                    ("#c62828", "Declining",   "25 – 45"),
                    ("#7f0000", "High Outflow","< 25"),
                ]:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin:5px 0;font-size:0.80rem;'>"
                        f"<div style='width:14px;height:14px;background:{color};border-radius:2px;"
                        f"margin-right:9px;flex-shrink:0;border:1px solid #555;'></div>"
                        f"<span style='color:#e8dfc4;font-weight:600;'>{label}</span>"
                        f"<span style='color:#888;font-size:0.74rem;margin-left:6px;'>{desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(
                    "<div style='margin-top:10px;padding-top:8px;border-top:1px solid #3a3a2a;"
                    "font-size:0.72rem;color:#888;line-height:1.5;'>"
                    "60% Population Growth<br>+ 40% Business Migration Index</div></div>",
                    unsafe_allow_html=True
                )

        # ── Top 10 States ──────────────────────────────────────────────────────
        section(" Top 10 States for CRE Investment (Migration Score)")
        top10 = mig_df.head(10)
        fig_bar = go.Figure(go.Bar(
            x=top10["composite_score"], y=top10["state_abbr"],
            orientation="h",
            marker=dict(color=top10["composite_score"],
                        colorscale=[[0, GOLD_DARK], [1, GOLD]], showscale=False),
            text=top10["composite_score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            textfont=dict(color="#c8b890", size=12),
            customdata=top10[["state_name", "pop_growth_pct", "key_companies"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pop Growth: %{customdata[1]:+.2f}%<br>"
                "Key Companies: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
            xaxis=dict(showgrid=True, gridcolor="#2a2208", range=[0, 110],
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c8b890", size=12)),
            margin=dict(t=20, b=20, l=60, r=60),
            height=320, font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Rankings combine population growth rate (60% weight) and business migration index (40% weight). "
            "States at the top represent the most favorable macro conditions for new CRE investment — "
            "higher scores correlate with rising rental demand, tighter vacancy, and upward rent pressure."
        )

        # ── Metro Table ────────────────────────────────────────────────────────
        _metro_city = st.session_state.user_intent.get("city")
        _metro_title = " Top Metro Areas — Population, Jobs & CRE Demand"
        if _metro_city:
            _metro_title = f" Metro Area Focus: {_metro_city} & Comparisons"
            # Show matched metro callout
            _matched_metro = metros_df[metros_df["Metro"].str.lower().str.contains(_metro_city.lower())]
            if not _matched_metro.empty:
                _mm = _matched_metro.iloc[0]
                st.markdown(
                    f'<div style="background:{BLACK};color:{GOLD};padding:12px 18px;border-radius:6px;margin-bottom:12px;">'
                    f'<span style="font-size:1.1rem;font-weight:700;">{_mm["Metro"]}</span><br>'
                    f'<span style="color:#eee;font-size:0.9rem;">'
                    f'Pop Growth: {_mm["Pop Growth %"]:+.1f}% · Job Growth: {_mm["Job Growth %"]:+.1f}% · '
                    f'Corp HQ Moves: {_mm["Corp HQ Moves"]:+d} · CRE Demand: {_mm["CRE Demand"]}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No metro data for {_metro_city}. Showing all metros for comparison.")

        section(_metro_title)

        # ── Sort control ───────────────────────────────────────────────────────
        _sort_opts = ["CRE Demand", "Pop Growth", "Job Growth", "HQ Moves"]
        _sort_by = st.radio(
            "Sort by", _sort_opts, horizontal=True,
            key="metro_sort_key", label_visibility="collapsed"
        )

        metros_disp = metros_df.copy()
        _demand_order = {"Very High": 5, "High": 4, "Moderate": 3, "Weak": 2, "Declining": 1}
        if _sort_by == "CRE Demand":
            metros_disp["_s"] = metros_disp["CRE Demand"].map(_demand_order).fillna(0)
            metros_disp = metros_disp.sort_values("_s", ascending=False).drop(columns=["_s"])
        elif _sort_by == "Pop Growth":
            metros_disp = metros_disp.sort_values("Pop Growth %", ascending=False)
        elif _sort_by == "Job Growth":
            metros_disp = metros_disp.sort_values("Job Growth %", ascending=False)
        else:
            metros_disp = metros_disp.sort_values("Corp HQ Moves", ascending=False)
        metros_disp = metros_disp.reset_index(drop=True)

        _max_pop = max(metros_disp["Pop Growth %"].max(), 0.01)
        _max_job = max(metros_disp["Job Growth %"].max(), 0.01)
        _max_hq  = max(int(metros_disp["Corp HQ Moves"].max()), 1)

        def _demand_badge_style(v):
            if v in ("Very High", "High"):
                return "background:#0d2a12;color:#4a9e58"
            elif v == "Moderate":
                return "background:#2a1a04;color:#a07830"
            return "background:#2a0d0d;color:#9e4a4a"

        def _hq_dots(n, mx):
            filled = round(min(n, mx) / mx * 5) if mx > 0 else 0
            return "".join(
                f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                f'background:{"#4a9e58" if i < filled else "#2a2208"};margin-right:2px;"></span>'
                for i in range(5)
            )

        _rows_html = ""
        for _ri, _rr in metros_disp.iterrows():
            _pop = _rr["Pop Growth %"]
            _job = _rr["Job Growth %"]
            _hq  = int(_rr["Corp HQ Moves"])
            _dem = _rr["CRE Demand"]
            _pbw = max(0, _pop / _max_pop * 100)
            _jbw = max(0, _job / _max_job * 100)
            _highlight = "background:#1e1a08;" if _metro_city and _metro_city.lower() in _rr["Metro"].lower() else ""
            _rows_html += f"""
<div style="display:flex;align-items:center;padding:10px 16px;border-bottom:1px solid #1e1a08;gap:8px;{_highlight}">
  <div style="width:24px;font-size:12px;color:#4a3e18;flex-shrink:0;text-align:center;">{_ri+1}</div>
  <div style="flex:2;font-size:14px;font-weight:500;color:#d4a843;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_rr["Metro"]}</div>
  <div style="flex:1.5;display:flex;flex-direction:column;gap:3px;">
    <span style="font-size:12px;color:#c8b890;">{_pop:+.1f}%</span>
    <div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">
      <div style="height:100%;width:{_pbw:.0f}%;background:linear-gradient(90deg,#2a7a38,#4a9e58);border-radius:2px;"></div>
    </div>
  </div>
  <div style="flex:1.5;display:flex;flex-direction:column;gap:3px;">
    <span style="font-size:12px;color:#c8b890;">{_job:+.1f}%</span>
    <div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">
      <div style="height:100%;width:{_jbw:.0f}%;background:linear-gradient(90deg,#2a7a38,#4a9e58);border-radius:2px;"></div>
    </div>
  </div>
  <div style="flex:1.5;display:flex;align-items:center;gap:5px;">
    <span style="font-size:12px;color:#c8b890;min-width:28px;">{_hq:+d}</span>
    {_hq_dots(_hq, _max_hq)}
  </div>
  <div style="flex:1;text-align:right;">
    <span style="font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;{_demand_badge_style(_dem)}">{_dem}</span>
  </div>
</div>"""

        st.markdown(f"""
<div style="background:#131008;border:1px solid #221e0a;border-radius:10px;overflow:hidden;margin-top:4px;">
  <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #1e1a08;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:3px;height:20px;background:#d4a843;border-radius:2px;flex-shrink:0;"></div>
      <span style="font-size:11px;font-weight:600;color:#d4a843;letter-spacing:0.08em;text-transform:uppercase;">Top Metro Areas — Population, Jobs &amp; CRE Demand</span>
    </div>
    <span style="font-size:11px;color:#6a5228;">{len(metros_disp)} markets tracked</span>
  </div>
  <div style="display:flex;padding:8px 16px;border-bottom:1px solid #1e1a08;gap:8px;">
    <div style="width:24px;"></div>
    <div style="flex:2;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">METRO</div>
    <div style="flex:1.5;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">POP GROWTH</div>
    <div style="flex:1.5;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">JOB GROWTH</div>
    <div style="flex:1.5;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">CORP HQ MOVES</div>
    <div style="flex:1;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;text-align:right;">CRE DEMAND</div>
  </div>
  {_rows_html}
</div>""", unsafe_allow_html=True)

        # ── Bubble chart ───────────────────────────────────────────────────────
        section(" Business Migration vs. Population Growth")
        biz_data = mig_df[mig_df["biz_score"] != 50].copy()
        fig_bubble = px.scatter(
            biz_data, x="pop_growth_pct", y="biz_score",
            size="population", color="composite_score",
            text="state_abbr",
            color_continuous_scale=[[0, "#b71c1c"], [0.5, GOLD], [1, "#1b5e20"]],
            labels={"pop_growth_pct": "Population Growth % (YoY)",
                    "biz_score": "Business Migration Score",
                    "composite_score": "Composite"},
            hover_data={"state_name": True, "key_companies": True, "growth_drivers": True},
            size_max=70,
        )
        fig_bubble.update_traces(textposition="middle center", textfont=dict(size=9, color="#c8b890"))
        fig_bubble.update_layout(
            plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
            xaxis=dict(showgrid=True, gridcolor="#2a2208", zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            yaxis=dict(showgrid=True, gridcolor="#2a2208",
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            coloraxis_showscale=False, margin=dict(t=20, b=40),
            height=380, font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption(
            "Each bubble is a state. Bubble size reflects the composite migration score. "
            "The ideal CRE investment target sits in the upper-right — high population growth AND strong business migration. "
            "States in the lower-left are losing both residents and corporate presence."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Migration Score** = 60% Population Growth + 40% Business Migration Index

- **Population Growth (%)**: Year-over-year change in state population from the US Census Bureau Population Estimates Program (2023).
- **Business Migration Index (0-100)**: Composite of corporate relocation announcements, new facility investments, and state economic incentive competitiveness.
- **Composite Score (0-100)**: Weighted blend of both signals — higher scores indicate states attracting both residents and employers.
- **Metro Rankings**: Top metros within each state ranked by job growth, rent growth, and migration inflows.

**Data Source:** US Census Bureau Population Estimates API (2023), Bureau of Labor Statistics (BLS), corporate announcement filings.

**Update Frequency:** Every 6 hours via Agent 1.
""")


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 2 — CRE PRICING & PROFIT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab2:
        _show_tab_header(
            "pricing",
            "Agent 2 pulls **live REIT pricing**, estimates **cap rates** and **NOI margins** by property type, "
            "and ranks market × property type combinations. Updates every hour.",
            "pricing",
        )

        from src.cre_pricing import CAP_RATE_BENCHMARKS, REIT_UNIVERSE
        import random as _rnd

        cache_p = read_cache("pricing")
        if not stale_banner("pricing") or cache_p["data"] is None:
            st.stop()

        pdata   = cache_p["data"]
        reit_df = pd.DataFrame(pdata["reits"])

        # ── Prior-year benchmarks for YoY delta badges ──────────────────────────
        _PT_PRIOR = {
            "Industrial / Logistics":      {"cap_rate": 0.048, "noi_margin": 0.702, "vacancy": 0.051, "rent_growth": 0.105},
            "Multifamily / Residential":   {"cap_rate": 0.048, "noi_margin": 0.598, "vacancy": 0.050, "rent_growth": 0.035},
            "Retail":                      {"cap_rate": 0.072, "noi_margin": 0.560, "vacancy": 0.082, "rent_growth": 0.008},
            "Office":                      {"cap_rate": 0.076, "noi_margin": 0.520, "vacancy": 0.162, "rent_growth": -0.020},
            "Healthcare / Medical Office": {"cap_rate": 0.062, "noi_margin": 0.625, "vacancy": 0.058, "rent_growth": 0.032},
            "Self-Storage":                {"cap_rate": 0.058, "noi_margin": 0.675, "vacancy": 0.092, "rent_growth": 0.024},
            "Data Centers":                {"cap_rate": 0.055, "noi_margin": 0.525, "vacancy": 0.028, "rent_growth": 0.098},
        }

        # Tab label → CAP_RATE_BENCHMARKS key
        _PT_MAP = {
            "Industrial / Logistics": "Industrial / Logistics",
            "Multifamily":            "Multifamily / Residential",
            "Retail":                 "Retail",
            "Office":                 "Office",
            "Healthcare":             "Healthcare / Medical Office",
            "Self-Storage":           "Self-Storage",
            "Data Centers":           "Data Centers",
        }

        _pricing_tabs = st.tabs(list(_PT_MAP.keys()))

        # Pre-compute 24-month trend data (seeded — same values every render)
        _rnd.seed(7)
        _trend_months = pd.date_range(end=datetime.today(), periods=24, freq="MS")
        _ind_trend = [0.062 + (0.056 - 0.062) * i / 23 + _rnd.gauss(0, 0.0008) for i in range(24)]
        _off_trend = [0.074 + (0.085 - 0.074) * i / 23 + _rnd.gauss(0, 0.0012) for i in range(24)]
        _mf_trend  = [0.050 + (0.052 - 0.050) * i / 23 + _rnd.gauss(0, 0.0006) for i in range(24)]

        for _ptab, (_pt_label, _pt_key) in zip(_pricing_tabs, _PT_MAP.items()):
            with _ptab:
                bench      = CAP_RATE_BENCHMARKS.get(_pt_key, {})
                prior      = _PT_PRIOR.get(_pt_key, bench)
                _sub_reit  = reit_df[reit_df["Property Type"] == _pt_key].copy().reset_index(drop=True)

                _cap   = bench.get("cap_rate",    0.056)
                _noi   = bench.get("noi_margin",  0.72)
                _rent  = bench.get("rent_growth", 0.08)
                _vac   = bench.get("vacancy",     0.045)
                _pcap  = prior.get("cap_rate",    _cap)
                _pnoi  = prior.get("noi_margin",  _noi)
                _prent = prior.get("rent_growth", _rent)
                _pvac  = prior.get("vacancy",     _vac)

                _cap_d  = (_cap  - _pcap)  * 10000
                _noi_d  = (_noi  - _pnoi)  * 10000
                _rent_d = (_rent - _prent) * 10000
                _vac_d  = (_vac  - _pvac)  * 10000

                def _badge(delta, label, good_if_positive=True):
                    good  = (delta > 0) if good_if_positive else (delta < 0)
                    clr   = "#4a9e58" if good else "#ef5350"
                    bg    = "#0d2a12" if good else "#2a0d0d"
                    arrow = "▲" if delta > 0 else "▼"
                    s     = "+" if delta > 0 else ""
                    return (f'<span style="display:inline-flex;align-items:center;gap:3px;font-size:11px;'
                            f'padding:3px 9px;border-radius:4px;background:{bg};color:{clr};margin-top:6px;">'
                            f'{arrow} {s}{delta:.0f}bps {label}</span>')

                # ── 4 KPI cards ────────────────────────────────────────────────
                _max_rent = max(b["rent_growth"] for b in CAP_RATE_BENCHMARKS.values())
                _rent_sub = "Sector high" if abs(_rent - _max_rent) < 0.001 else "Market consensus"
                _vac_sub  = "Tightening YoY" if _vac < _pvac else "Rising YoY"
                _cap_sub  = "Market benchmark"
                _noi_sub  = "After opex, before CapEx"

                _KPI_EXPLAIN = {
                    "AVG CAP RATE":   ("NOI ÷ Property Value",       "A lower cap rate = higher asset value. Compressed cap rates signal strong investor demand. Rises when interest rates climb or asset values fall."),
                    "NOI MARGIN":     ("NOI ÷ Gross Revenue",        "Measures operating efficiency after expenses (insurance, maintenance, mgmt fees) but before CapEx and debt service. Higher = more cash flow per dollar of revenue."),
                    "RENT GROWTH YOY":("(Rent Now − Rent Yr Ago) ÷ Rent Yr Ago", "Positive rent growth compounds NOI over time. Driven by supply/demand imbalance, lease rollovers at market rates, and inflation passthroughs."),
                    "AVG VACANCY":    ("Unleased SF ÷ Total Rentable SF", "Lower vacancy tightens rent pricing power. Rising vacancy signals oversupply or demand softness — a leading indicator of cap rate expansion."),
                }

                _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                for _col, _lbl, _val_str, _sub, _badge_html, _border in [
                    (_kc1, "AVG CAP RATE",     f"{_cap*100:.2f}%",    _cap_sub,  _badge(_cap_d,  "YoY", False), "#d4a843"),
                    (_kc2, "NOI MARGIN",        f"{_noi*100:.1f}%",    _noi_sub,  _badge(_noi_d,  "YoY", True),  "#4a9e58"),
                    (_kc3, "RENT GROWTH YOY",   f"{_rent*100:+.1f}%", _rent_sub, _badge(_rent_d, "YoY", True),  "#4a9e58" if _rent > 0 else "#ef5350"),
                    (_kc4, "AVG VACANCY",       f"{_vac*100:.1f}%",    _vac_sub,  _badge(_vac_d,  "vs prior", False), "#d4a843" if _vac < 0.10 else "#ef5350"),
                ]:
                    _exp_formula, _exp_desc = _KPI_EXPLAIN[_lbl]
                    _col.markdown(f"""
<div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;min-height:130px;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{_border};border-radius:10px 10px 0 0;"></div>
  <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">{_lbl}</div>
  <div style="font-size:34px;font-weight:500;color:{_border};line-height:1;margin-bottom:4px;">{_val_str}</div>
  <div style="font-size:11px;color:#6a5228;margin-bottom:2px;">{_sub}</div>
  {_badge_html}
</div>
<div style="margin-top:6px;padding:10px 12px;background:#0f0c05;border:1px solid #1e1a08;border-radius:8px;">
  <div style="font-size:10px;color:#d4a843;letter-spacing:0.05em;margin-bottom:4px;font-family:monospace;">{_exp_formula}</div>
  <div style="font-size:11px;color:#5a4820;line-height:1.5;">{_exp_desc}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                # ── Row 2: trend chart (left) + scatter (right) ────────────────
                _cl, _cr = st.columns([6, 4])

                with _cl:
                    _fig_trend = go.Figure()
                    _fig_trend.add_trace(go.Scatter(
                        x=list(_trend_months), y=[v * 100 for v in _ind_trend],
                        name="Industrial", mode="lines",
                        line=dict(color="#d4a843", width=2),
                    ))
                    _fig_trend.add_trace(go.Scatter(
                        x=list(_trend_months), y=[v * 100 for v in _off_trend],
                        name="Office", mode="lines",
                        line=dict(color="#ef5350", width=2, dash="dot"),
                    ))
                    _fig_trend.add_trace(go.Scatter(
                        x=list(_trend_months), y=[v * 100 for v in _mf_trend],
                        name="Multifamily", mode="lines",
                        line=dict(color="#42a5f5", width=2, dash="dot"),
                    ))
                    _fig_trend.update_layout(
                        paper_bgcolor="#171309", plot_bgcolor="#171309",
                        annotations=[
                            dict(text="CAP RATE TREND — INDUSTRIAL VS. OFFICE VS. MULTIFAMILY",
                                 xref="paper", yref="paper", x=0, y=1.12, showarrow=False,
                                 font=dict(size=10, color="#6a5228"), xanchor="left"),
                            dict(text="24-MONTH TRAILING",
                                 xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                                 font=dict(size=10, color="#4a3e18"), xanchor="right"),
                        ],
                        xaxis=dict(showgrid=False, tickfont=dict(color="#6a5228", size=10),
                                   tickformat="%b", dtick="M2", tickcolor="#2a2208"),
                        yaxis=dict(showgrid=True, gridcolor="#1e1a08",
                                   tickfont=dict(color="#6a5228", size=10),
                                   ticksuffix="%", tickformat=".1f"),
                        legend=dict(orientation="h", y=-0.22, x=0,
                                    font=dict(color="#8a7040", size=11),
                                    bgcolor="rgba(0,0,0,0)"),
                        margin=dict(t=50, b=60, l=50, r=20),
                        height=330,
                        font=dict(family="Source Sans Pro"),
                    )
                    st.plotly_chart(_fig_trend, use_container_width=True, key=f"trend_{_pt_label}")

                with _cr:
                    if not _sub_reit.empty:
                        _tickers    = _sub_reit["Ticker"].tolist()
                        _caps_sc    = (_sub_reit["Cap Rate"] * 100).tolist()
                        _nois_sc    = (_sub_reit["NOI Margin"] * 100).tolist()
                        _max_noi_sc = max(_nois_sc)
                        _sc_colors  = ["#4a9e58" if n >= _max_noi_sc * 0.97 else "#d4a843" for n in _nois_sc]
                        # Alternate label positions to avoid overlap
                        _label_pos  = ["top right", "top left", "bottom right", "bottom left",
                                        "top center", "bottom center"]

                        _fig_sc = go.Figure()
                        for _i, (_tk, _cr_v, _nm_v, _clr) in enumerate(
                                zip(_tickers, _caps_sc, _nois_sc, _sc_colors)):
                            _fig_sc.add_trace(go.Scatter(
                                x=[_cr_v], y=[_nm_v],
                                mode="markers+text", text=[_tk],
                                textposition=_label_pos[_i % len(_label_pos)],
                                textfont=dict(size=10, color="#c8b890"),
                                marker=dict(size=16, color=_clr, opacity=0.88,
                                            line=dict(width=1.5, color="#0d0b04")),
                                showlegend=False,
                                hovertemplate=(f"<b>{_tk}</b><br>Cap Rate: {_cr_v:.2f}%"
                                               f"<br>NOI Margin: {_nm_v:.1f}%<extra></extra>"),
                            ))
                        _fig_sc.update_layout(
                            paper_bgcolor="#171309", plot_bgcolor="#171309",
                            annotations=[dict(
                                text="CAP RATE VS. NOI MARGIN BY TICKER",
                                xref="paper", yref="paper", x=0, y=1.14,
                                showarrow=False, font=dict(size=10, color="#6a5228"), xanchor="left",
                            )],
                            xaxis=dict(showgrid=True, gridcolor="#1e1a08",
                                       tickfont=dict(color="#6a5228", size=10),
                                       ticksuffix="%", tickformat=".1f",
                                       title=dict(text="Cap Rate", font=dict(color="#6a5228", size=10))),
                            yaxis=dict(showgrid=True, gridcolor="#1e1a08",
                                       tickfont=dict(color="#6a5228", size=10),
                                       ticksuffix="%", tickformat=".0f",
                                       title=dict(text="NOI Margin", font=dict(color="#6a5228", size=10))),
                            margin=dict(t=55, b=50, l=60, r=20),
                            height=330,
                        )
                        st.plotly_chart(_fig_sc, use_container_width=True, key=f"scatter_{_pt_label}")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Company table ───────────────────────────────────────────────
                if not _sub_reit.empty:
                    _max_noi_tbl = _sub_reit["NOI Margin"].max()
                    _rows_tbl = ""
                    for _, _r in _sub_reit.iterrows():
                        _px   = f"${_r['Price']:.2f}"  if pd.notna(_r.get("Price"))       else "N/A"
                        _dr   = float(_r.get("Daily Return", 0) or 0)
                        _dy   = float(_r.get("Div Yield",    0) or 0)
                        _cr_v = float(_r.get("Cap Rate",     0) or 0)
                        _nm_v = float(_r.get("NOI Margin",   0) or 0)
                        _vr_v = float(_r.get("Vacancy Rate", 0) or 0)
                        _rg_v = float(_r.get("Rent Growth",  0) or 0)
                        _foc  = str(_r.get("Market Focus", ""))

                        _dr_color = "#4a9e58" if _dr >= 0 else "#ef5350"
                        _nm_color = "#4a9e58" if _nm_v >= _max_noi_tbl * 0.95 else "#d4a843" if _nm_v >= _max_noi_tbl * 0.80 else "#c8b890"
                        _vr_color = "#4a9e58" if _vr_v < 0.04 else "#d4a843" if _vr_v < 0.09 else "#ef5350"
                        _rg_color = "#4a9e58" if _rg_v > 0.05 else "#d4a843" if _rg_v > 0 else "#ef5350"
                        _bar_w    = max(4, int(_nm_v / max(_max_noi_tbl, 0.01) * 64))
                        _dot_n    = 3 if _nm_v >= _max_noi_tbl * 0.95 else 2 if _nm_v >= _max_noi_tbl * 0.80 else 1
                        _dots     = "".join(
                            f'<span style="color:#d4a843;font-size:9px;">●</span>' for _ in range(_dot_n)
                        ) + "".join(
                            f'<span style="color:#2a2208;font-size:9px;">●</span>' for _ in range(3 - _dot_n)
                        )

                        _rows_tbl += f"""
<div style="display:flex;align-items:center;padding:11px 14px;border-bottom:1px solid #1a1608;gap:0;">
  <div style="width:72px;font-size:14px;font-weight:700;color:#d4a843;">{_r['Ticker']}</div>
  <div style="flex:2.2;font-size:13px;color:#c8b890;">{_r['Company']}</div>
  <div style="flex:1.2;">
    <span style="font-size:11px;padding:2px 8px;border-radius:4px;background:#1e1a08;color:#8a7040;white-space:nowrap;">{_foc}</span>
  </div>
  <div style="width:72px;font-size:13px;color:#c8b890;">{_px}</div>
  <div style="width:80px;font-size:13px;font-weight:500;color:{_dr_color};">{_dr*100:+.1f}%</div>
  <div style="width:76px;font-size:13px;color:#c8b890;">{_dy*100:.1f}%</div>
  <div style="width:74px;font-size:13px;color:#d4a843;">{_cr_v*100:.1f}%</div>
  <div style="width:100px;">
    <div style="font-size:13px;color:{_nm_color};margin-bottom:3px;">{_nm_v*100:.1f}%</div>
    <div style="height:3px;background:#1e1a08;border-radius:2px;width:64px;">
      <div style="height:100%;width:{_bar_w}px;background:#4a9e58;border-radius:2px;"></div>
    </div>
  </div>
  <div style="width:68px;font-size:13px;color:{_vr_color};">{_vr_v*100:.1f}%</div>
  <div style="width:52px;font-size:13px;color:{_rg_color};">{_rg_v*100:+.1f}%</div>
  <div style="width:44px;text-align:right;">{_dots}</div>
</div>"""

                    st.markdown(f"""
<div style="background:#131008;border:1px solid #221e0a;border-radius:8px;overflow:hidden;margin-top:4px;">
  <div style="display:flex;padding:7px 14px;border-bottom:1px solid #2a2208;background:#0f0c05;gap:0;">
    <div style="width:72px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">TICKER</div>
    <div style="flex:2.2;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">COMPANY</div>
    <div style="flex:1.2;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">FOCUS</div>
    <div style="width:72px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">PRICE</div>
    <div style="width:80px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">DAILY RTN</div>
    <div style="width:76px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">DIV YIELD</div>
    <div style="width:74px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">CAP RATE</div>
    <div style="width:100px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">NOI MARGIN</div>
    <div style="width:68px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">VACANCY</div>
    <div style="width:52px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;">RENT G.</div>
    <div style="width:44px;font-size:9px;color:#4a3e18;letter-spacing:0.1em;text-transform:uppercase;text-align:right;">RATING</div>
  </div>
  {_rows_tbl}
</div>""", unsafe_allow_html=True)

        with st.expander("How These Metrics Are Calculated"):
            st.markdown("""
#### KPI Cards
| Metric | Formula | Source |
|--------|---------|--------|
| **Avg Cap Rate** | Net Operating Income ÷ Property Value | CBRE / Green Street sector benchmarks |
| **NOI Margin** | Net Operating Income ÷ Gross Revenue (after opex, before CapEx) | REIT 10-K filings, SNL Real Estate |
| **Rent Growth YoY** | (Current Avg Rent − Prior Year Avg Rent) ÷ Prior Year Avg Rent | CoStar / CBRE market reports |
| **Avg Vacancy** | Unleased SF ÷ Total Rentable SF | JLL / CBRE quarterly vacancy surveys |

**YoY delta badges** show the change in basis points (bps) vs. the prior year. Green = improving (NOI up, vacancy down), Red = deteriorating.

#### Cap Rate Trend Chart
24-month trailing cap rate history for Industrial, Office, and Multifamily — generated from known sector benchmarks interpolated monthly. Industrial cap rates have **compressed** as e-commerce demand keeps vacancy near historic lows. Office cap rates have **expanded** as WFH and maturity defaults push values down.

#### Cap Rate vs. NOI Margin Scatter
Each circle is a REIT ticker. **Upper-left = best in class** — low cap rate (high asset values) and high NOI margin (efficient operations). REXR (Rexford Industrial) leads in the Industrial sector due to SoCal infill scarcity. Lower-right tickers face cap rate pressure and thinner margins.

#### Company Table
- **DAILY RTN**: 1-day price return from Yahoo Finance (live)
- **DIV YIELD**: Annual dividend ÷ current price
- **CAP RATE**: Per-REIT estimate from Green Street / SNL filings
- **NOI MARGIN spark bar**: Width scaled to best-in-class NOI for that property type
- **RATING (●●●)**: 3 dots = top-quartile NOI, 2 dots = mid, 1 dot = below median

**Data Sources:** yfinance (live prices), CBRE/JLL/Green Street/CoStar 2024-2025 benchmarks, REIT 10-K filings.

**Update Frequency:** Every hour via Agent 2.
""")




    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CONFIRMED FACILITY ANNOUNCEMENTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab3:
        _show_tab_header(
            "predictions",
            "Agent 3 scans live news feeds and uses AI to extract **confirmed** corporate facility announcements — "
            "new manufacturing plants, data centers, warehouses, training centers, and headquarters. Updates every 24 hours.",
            "predictions",
        )

        cache3 = read_cache("predictions")
        if not stale_banner("predictions") or cache3["data"] is None:
            st.stop()

        pdata3 = cache3["data"]

        # ── Confirmed Announcements ────────────────────────────────────────────
        section("Confirmed Plant & Facility Announcements")
        confirmed = pdata3.get("confirmed_announcements", [])

        # Filter out error placeholders
        confirmed = [a for a in confirmed if a.get("company") and a.get("company") != "Error"]

        # Sort by relevance to user intent
        _user_pt3 = st.session_state.user_intent.get("property_type")
        _user_loc3 = st.session_state.user_intent.get("location")
        # Map property_type to facility types
        _PT_TO_FACILITY = {
            "industrial": ["manufacturing plant", "warehouse / distribution", "battery plant", "semiconductor fab"],
            "office": ["headquarters", "research & development"],
            "data center": ["data center"],
            "retail": ["other"],
        }
        _matching_types = _PT_TO_FACILITY.get((_user_pt3 or "").lower(), [])
        if _user_pt3 or _user_loc3:
            def _ann_score(a):
                score = 0
                if _matching_types and a.get("type", "").lower() in _matching_types:
                    score += 2
                if _user_loc3 and _user_loc3.lower() in (a.get("location", "") or "").lower():
                    score += 1
                return score
            confirmed = sorted(confirmed, key=_ann_score, reverse=True)

        if not confirmed:
            st.info(
                "No confirmed facility announcements found in the current news cycle. "
                "Agent 3 refreshes every 24 hours — check back after the next run or ensure GROQ_API_KEY is set in .env."
            )
        else:
            # ── Summary metrics ────────────────────────────────────────────────
            type_counts = {}
            for a in confirmed:
                t = a.get("type", "Other")
                type_counts[t] = type_counts.get(t, 0) + 1

            m_cols = st.columns(min(4, len(type_counts) + 1))
            m_cols[0].metric("Total Announcements", len(confirmed))
            for i, (t, cnt) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])[:3], 1):
                m_cols[i].metric(t, cnt)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Type badge colors ──────────────────────────────────────────────
            TYPE_COLORS = {
                "Manufacturing Plant":    ("#1b5e20", "#e8f5e9"),
                "Warehouse / Distribution": ("#0d47a1", "#e3f2fd"),
                "Data Center":            ("#4a148c", "#f3e5f5"),
                "Semiconductor Fab":      ("#b71c1c", "#ffebee"),
                "Battery Plant":          ("#e65100", "#fff3e0"),
                "Headquarters":           ("#263238", "#eceff1"),
                "Training Center":        ("#004d40", "#e0f2f1"),
                "Research & Development": ("#880e4f", "#fce4ec"),
                "Other":                  ("#5d4037", "#efebe9"),
            }

            # ── Announcement cards ─────────────────────────────────────────────
            for ann in confirmed:
                co      = ann.get("company", "Unknown")
                ticker  = ann.get("ticker", "")
                atype   = ann.get("type", "Other")
                loc     = ann.get("location", "")
                invest  = ann.get("investment", "")
                jobs    = ann.get("jobs", "")
                detail  = ann.get("detail", "")
                impact  = ann.get("cre_impact", "")
                source  = ann.get("source", "")

                badge_fg, badge_bg = TYPE_COLORS.get(atype, ("#5d4037", "#efebe9"))
                ticker_str = f" &nbsp;·&nbsp; <span style='color:#888;font-size:0.8rem;'>{ticker}</span>" if ticker else ""
                invest_str = f"<b>Investment:</b> {invest}" if invest else ""
                jobs_str   = f"<b>Jobs:</b> {jobs}" if jobs else ""
                meta_parts = [x for x in [invest_str, jobs_str] if x]
                meta_line  = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_parts) if meta_parts else ""

                st.markdown(f"""
                <div style="background:#1a1208;border:1px solid #a07830;border-radius:8px;
                            padding:16px 20px;margin-bottom:12px;border-left:4px solid {badge_fg};">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="background:{badge_bg};color:{badge_fg};font-size:0.72rem;
                                 font-weight:700;padding:2px 8px;border-radius:4px;
                                 text-transform:uppercase;letter-spacing:0.5px;">{atype}</span>
                    <span style="font-size:1rem;font-weight:700;color:#e8dfc4;">{co}{ticker_str}</span>
                    <span style="font-size:0.85rem;color:#a09880;margin-left:auto;">{loc}</span>
                  </div>
                  <div style="font-size:0.9rem;color:#e8dfc4;margin-bottom:6px;">{detail}</div>
                  {"<div style='font-size:0.82rem;color:#555;margin-bottom:6px;'>" + meta_line + "</div>" if meta_line else ""}
                  {"<div style='font-size:0.82rem;color:#1b5e20;'><b>CRE Opportunity:</b> " + impact + "</div>" if impact else ""}
                  <div style="font-size:0.72rem;color:#6a6050;margin-top:6px;">Source: {source}</div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "Announcements extracted from live RSS feeds: Reuters, Manufacturing.net, IndustryWeek, "
                "PR Newswire, Business Wire, Dept. of Energy, Commerce Dept., EDA, and industry publications. "
                "Only confirmed/announced projects are shown — not speculation."
            )

        # ── Top 5 States context ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section("Top 5 States Attracting Corporate Investment")
        top5 = pdata3.get("top5_states", [])
        if top5:
            top5_df = pd.DataFrame(top5)
            cols_show = [c for c in ["state_name","state_abbr","pop_growth_pct","biz_score","key_companies","growth_drivers"] if c in top5_df.columns]
            top5_df = top5_df[cols_show].copy()
            if "pop_growth_pct" in top5_df.columns:
                top5_df["pop_growth_pct"] = top5_df["pop_growth_pct"].apply(lambda x: f"{x:+.2f}%")
            rename = {"state_name": "State", "state_abbr": "Abbr", "pop_growth_pct": "Pop Growth",
                      "biz_score": "Business Score", "key_companies": "Recent Corporate Moves",
                      "growth_drivers": "Growth Drivers"}
            top5_df.rename(columns=rename, inplace=True)
            st.dataframe(top5_df, use_container_width=True, hide_index=True)
            st.caption(
                "These states rank highest on combined population growth and business migration scores. "
                "Cross-reference with announcements above to identify where CRE demand is building fastest."
            )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 4 — CHEAPEST BUILDINGS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab4:
        _show_tab_header(
            "buildings",
            "Agent 3 sources the lowest-price commercial listings in the states with the highest "
            "migration and business growth scores. **Agent 21 (RentCast)** overlays live property data "
            "when API key is configured (free tier: 50 calls/month). Updates every 24 hours.",
            "predictions",
        )

        cache4 = read_cache("predictions")
        if not stale_banner("predictions") or cache4["data"] is None:
            st.stop()

        pdata4   = cache4["data"]
        listings = pdata4.get("listings", {})
        top3_abbr = pdata4.get("top3_abbr", [])

        # ── Overlay RentCast live data if available ──────────────────────────
        _rc_cache = read_cache("rentcast")
        _rc_data = _rc_cache.get("data") or {}
        _rc_listings = _rc_data.get("listings", {})
        _rc_sources = _rc_data.get("source_states", {})
        _rc_live_count = _rc_data.get("live_listing_count", 0)
        _rc_remaining = _rc_data.get("api_calls_remaining", 0)
        _rc_has_key = _rc_data.get("has_api_key", False)

        # Merge RentCast listings into the main listings dict (RentCast takes priority)
        for abbr, rc_list in _rc_listings.items():
            if rc_list:
                listings[abbr] = rc_list
                if abbr not in top3_abbr:
                    top3_abbr.append(abbr)

        # Show RentCast API status strip
        if _rc_has_key:
            _rc_used = _rc_data.get("api_calls_used", 0)
            _live_states = [k for k, v in _rc_sources.items() if v == "live"]
            if _rc_live_count > 0:
                st.markdown(
                    f'<div style="background:#1b5e20;color:#fff;padding:6px 14px;border-radius:6px;'
                    f'font-size:0.82rem;margin-bottom:12px;display:inline-block;">'
                    f'<b>LIVE DATA</b> &nbsp; {_rc_live_count} listings from RentCast API '
                    f'&middot; {_rc_used}/50 calls used this month '
                    f'&middot; Live states: {", ".join(_live_states)}'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.caption(f"RentCast API configured — {_rc_remaining} calls remaining this month. "
                           f"Showing cached/mock data until next agent run.")

        # On-demand: generate listings for user's target state if not already cached
        _user_abbr_b = st.session_state.user_intent.get("state_abbr")
        if _user_abbr_b and _user_abbr_b not in listings:
            try:
                listings[_user_abbr_b] = get_cheapest_buildings(_user_abbr_b, n=10)
                if _user_abbr_b not in top3_abbr:
                    top3_abbr = [_user_abbr_b] + list(top3_abbr)
            except Exception:
                pass

        if not listings:
            st.info("Listings will appear after the first scheduled agent run (every 24 hours).")
        else:

            _intent4 = st.session_state.user_intent
            _user_loc4 = _intent4.get("location")
            _user_city4 = _intent4.get("city")
            _user_state4 = _intent4.get("state")
            _user_abbr4 = _intent4.get("state_abbr")
            _user_pt4 = _intent4.get("property_type")

            # Flatten all listings into one list for filtering
            _all_listings = []
            for abbr, lst in listings.items():
                for l in lst:
                    if isinstance(l, dict):
                        l["_source_abbr"] = abbr
                        _all_listings.append(l)

            # Filter by city if specified
            _filtered = _all_listings
            _filter_note = ""
            if _user_city4:
                _city_match = [l for l in _all_listings if _user_city4.lower() in (l.get("city", "") or "").lower()]
                if _city_match:
                    _filtered = _city_match
                else:
                    # Try state-level match
                    _state_match = []
                    if _user_abbr4:
                        _state_match = [l for l in _all_listings if (l.get("state", "") or "").upper() == _user_abbr4]
                    elif _user_state4:
                        _state_match = [l for l in _all_listings
                                        if _user_state4.lower() in (l.get("state", "") or "").lower()
                                        or _intent_matches_state(_user_state4, "", l.get("_source_abbr", ""))]
                    if _state_match:
                        _filtered = _state_match
                        _nearby = set(l.get("city", "Unknown") for l in _state_match[:5])
                        _filter_note = f"No listings found in {_user_city4}. Showing nearby: {', '.join(_nearby)}"
                    else:
                        _filter_note = f"No listings found in {_user_city4} or surrounding area. Showing top migration states instead."
                        _filtered = _all_listings
            elif _user_abbr4 or _user_state4:
                _state_match = []
                if _user_abbr4:
                    _state_match = [l for l in _all_listings if l.get("_source_abbr") == _user_abbr4]
                if not _state_match and _user_state4:
                    _state_match = [l for l in _all_listings
                                    if _intent_matches_state(_user_state4, "", l.get("_source_abbr", ""))]
                if _state_match:
                    _filtered = _state_match
                else:
                    _filter_note = f"No listings found in {_user_state4 or _user_abbr4}. Showing top migration states instead."

            # Filter by property type if specified
            if _user_pt4:
                _pt_match = [l for l in _filtered if _user_pt4.lower() in (l.get("property_type", "") or "").lower()]
                if _pt_match:
                    _filtered = _pt_match
                elif _filtered != _all_listings:
                    _filter_note += f" No exact {_user_pt4} listings — showing all property types."

            if _filter_note:
                st.info(_filter_note)

            # Group filtered listings by state for display
            _display_groups = {}
            for l in _filtered:
                key = l.get("_source_abbr", "??")
                _display_groups.setdefault(key, []).append(l)

            # Sort so user's focus state appears first
            _sorted_abbr4 = list(_display_groups.keys())
            if _user_abbr4 and _user_abbr4 in _sorted_abbr4:
                _sorted_abbr4.remove(_user_abbr4)
                _sorted_abbr4.insert(0, _user_abbr4)

            # Build section title
            _pt_label = f"{_user_pt4} " if _user_pt4 else "Commercial "
            _loc_label = _user_loc4 or "Top Migration States"

            for abbr in _sorted_abbr4:
                group = _display_groups[abbr]

                # Resolve state name
                mig_cache = read_cache("migration")
                state_name = abbr
                if mig_cache["data"]:
                    mig_df2 = pd.DataFrame(mig_cache["data"]["migration"])
                    row = mig_df2[mig_df2["state_abbr"] == abbr]
                    if not row.empty:
                        state_name = row.iloc[0]["state_name"]

                section(f" Cheapest {_pt_label}Properties — {state_name} ({abbr})")
                st.caption(f"Showing {len(group)} listings sorted by asking price")

                for listing in group:
                    tax          = estimate_property_tax(listing)
                    price_fmt    = f"${listing.get('price', 0):,}"
                    sqft_fmt     = f"{listing.get('sqft', 0):,} sqft"
                    ppsf_fmt     = f"${listing.get('price_per_sqft', 0):.0f}/sqft"
                    cap_fmt      = f"{listing.get('cap_rate', 0):.2f}% cap rate"
                    noi_fmt      = f"${listing.get('noi_annual', 0):,}/yr NOI"
                    dom_fmt      = f"{listing.get('days_on_market', 0)}d on market"
                    built_fmt    = f"Built {listing.get('year_built', 'N/A')}"
                    pt_fmt       = listing.get("property_type", "")
                    addr_fmt     = f"{listing.get('address', '')}, {listing.get('city', '')}, {listing.get('state', '')}"
                    highlights   = listing.get("highlights", "")
                    tax_fmt      = f"${tax['annual_tax']:,}/yr est. tax ({tax['tax_rate_pct']}% rate · ${tax['tax_per_sqft']}/sqft)"
                    tax_noi_note = f" · {tax['tax_as_pct_noi']}% of NOI" if tax['tax_as_pct_noi'] else ""

                    # Highlight if city matches
                    _is_city_match = _user_city4 and _user_city4.lower() in (listing.get("city", "") or "").lower()
                    _is_live = listing.get("_source") == "rentcast"
                    _border_clr = "#1b5e20" if _is_city_match or _is_live else GOLD
                    _live_badge = ('<span style="background:#1b5e20;color:#fff;padding:2px 8px;'
                                   'border-radius:4px;font-size:0.7rem;font-weight:700;'
                                   'margin-left:8px;vertical-align:middle;">LIVE DATA</span>') if _is_live else ""

                    st.markdown(f"""
                    <div class="listing-card" style="border-left-color:{_border_clr};">
                      <div class="l-price">{price_fmt}{_live_badge}</div>
                      <div class="l-address">{addr_fmt}</div>
                      <div style="margin:4px 0;">
                        <span class="l-tag">{pt_fmt}</span>
                        <span class="l-tag">{cap_fmt}</span>
                        <span class="l-tag">{dom_fmt}</span>
                      </div>
                      <div class="l-detail">{sqft_fmt} · {ppsf_fmt} · {noi_fmt} · {built_fmt}</div>
                      <div class="l-detail" style="color:#c8a96e;margin-top:4px;">
                        Property Tax: {tax_fmt}{tax_noi_note}
                      </div>
                      {"<div class='l-detail' style='color:#555;margin-top:4px;font-style:italic;'> " + highlights + "</div>" if highlights else ""}
                    </div>
                    """, unsafe_allow_html=True)

            # ── Why these markets? ────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Why These Markets? Investment Thesis")

            mig_cache2 = read_cache("migration")
            if mig_cache2["data"]:
                mig_df3 = pd.DataFrame(mig_cache2["data"]["migration"])
                top3_rows = mig_df3[mig_df3["state_abbr"].isin(top3_abbr)].head(3)
                for _, row in top3_rows.iterrows():
                    with st.expander(f" {row['state_name']} ({row['state_abbr']}) — Why Invest Here?"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Population Growth", f"{row['pop_growth_pct']:+.2f}%", "YoY")
                        col_b.metric("Business Score", str(row['biz_score']), "Migration index")
                        col_c.metric("Composite Score", str(row['composite_score']), "0–100 scale")
                        st.markdown(f"**Key Companies:** {row['key_companies']}")
                        st.markdown(f"**Growth Drivers:** {row['growth_drivers']}")


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 5 — SYSTEM MONITOR / DEBUGGER
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab5:
        _show_tab_header(
            "announcements",
            "Agent 5 monitors **news wires, government press releases, and industry publications** "
            "every 4 hours — surfacing companies that have announced new manufacturing plants, "
            "training centers, data centers, warehouses, and other large facilities.",
            "news",
        )

        cache_news = read_cache("news")
        if not stale_banner("news") or cache_news["data"] is None:
            st.stop()

        ndata = cache_news["data"]

        # ── KPI strip ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Articles Found", str(ndata.get("article_count", 0)), "Facility announcements"), unsafe_allow_html=True)
        c2.markdown(metric_card("Verified / High", str(ndata.get("verified_count", 0)), "Independent sources only"), unsafe_allow_html=True)
        c3.markdown(metric_card("Sources Checked", str(ndata.get("sources_checked", 0)), "Incl. Bloomberg + Reuters"), unsafe_allow_html=True)
        fetched = ndata.get("fetched_at", "")
        fetched_label = datetime.fromisoformat(fetched).strftime("%b %d, %Y %I:%M %p") if fetched else "N/A"
        c4.markdown(metric_card("Last Scan", fetched_label, "Updates every 4 hours"), unsafe_allow_html=True)

        # ── Credibility breakdown bar ──────────────────────────────────────
        cred_bk = ndata.get("credibility_breakdown", {})
        tier_bk = ndata.get("tier_breakdown", {})
        if cred_bk:
            _total_art = max(sum(cred_bk.values()), 1)
            _cred_colors = {"VERIFIED": "#4caf50", "HIGH": "#8bc34a", "MODERATE": "#ff9800", "LOW": "#f44336"}
            _bars_parts = []
            _label_parts = []
            for lbl, cnt in cred_bk.items():
                if cnt > 0:
                    clr = _cred_colors.get(lbl, "#555")
                    _bars_parts.append(
                        f'<div title="{lbl}: {cnt}" style="flex:{cnt};background:{clr};'
                        f'height:8px;border-radius:2px;"></div>'
                    )
                    _label_parts.append(
                        f'<span style="color:{clr};font-size:11px;">&#9632; {lbl} {cnt}</span>'
                    )
            _bars   = "".join(_bars_parts)
            _labels = " &nbsp;".join(_label_parts)
            st.markdown(f"""
<div style="background:#16140a;border:1px solid #2a2208;border-radius:6px;padding:10px 16px;margin-bottom:16px;">
  <div style="font-size:11px;color:#6a5228;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">
    Source Credibility Breakdown
  </div>
  <div style="display:flex;gap:3px;margin-bottom:8px;">{_bars}</div>
  <div>{_labels}</div>
</div>""", unsafe_allow_html=True)

        # ── AI Summary ────────────────────────────────────────────────────────
        section(" Agent 5 — AI Investment Brief: Facility Announcements")
        summary = ndata.get("summary", "")
        if summary:
            st.markdown(f"""
<div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 20px;border-left:3px solid #d4a843;margin-bottom:8px;">
  <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
    Agent 5 · Industry Announcements · {datetime.today().strftime('%b %d, %Y')}
  </div>
  <div style="font-size:14px;color:#c8b890;line-height:1.7;">{summary}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("News summary is generated every 4 hours. Ensure GROQ_API_KEY is set in .env.")

        # ── Raw Article Feed ─────────────────────────────────────────────────
        raw = ndata.get("raw_articles", [])
        if raw:
            section(f" Raw Announcement Feed ({len(raw)} articles)")

            _fcol1, _fcol2 = st.columns(2)
            feed_type_filter = _fcol1.selectbox(
                "Filter by source type",
                options=["All", "news", "industry", "press", "government"],
                key="news_filter",
            )
            cred_filter = _fcol2.selectbox(
                "Minimum credibility",
                options=["All", "MODERATE", "HIGH", "VERIFIED"],
                key="news_cred_filter",
            )

            _cred_min_map = {"All": 0, "MODERATE": 45, "HIGH": 65, "VERIFIED": 80}
            _cred_min_score = _cred_min_map.get(cred_filter, 0)

            _cred_badge_style = {
                "VERIFIED": "background:#0d1a0d;color:#4caf50;border:1px solid #2d5a2d;",
                "HIGH":     "background:#121a0d;color:#8bc34a;border:1px solid #3a5a1a;",
                "MODERATE": "background:#1a1200;color:#ff9800;border:1px solid #5a4000;",
                "LOW":      "background:#1a0d0d;color:#f44336;border:1px solid #5a1a1a;",
            }
            _tier_badge_style = {
                1: "background:#0d1a0d;color:#4caf50;",
                2: "background:#0d1a2a;color:#4a8abf;",
                3: "background:#0d2a12;color:#4a9e58;",
                4: "background:#2a1a0d;color:#bf8a4a;",
            }
            _tier_labels = {1: "Independent News", 2: "Government", 3: "Trade Press", 4: "Press Release"}
            _left_colors = {"VERIFIED": "#4caf50", "HIGH": "#8bc34a", "MODERATE": "#ff9800", "LOW": "#f44336"}

            def _render_article(art):
                tier       = art.get("tier", 4)
                link       = art.get("link", "#")
                title      = art.get("title", "No title")
                desc       = art.get("description", "")[:300]
                src        = art.get("source", "")
                cred_lbl   = art.get("credibility_label", "LOW")
                cred_score = art.get("credibility_score", 0)
                confirms   = art.get("confirming_sources", [])
                age_days   = art.get("age_days")

                _cred_sty  = _cred_badge_style.get(cred_lbl, _cred_badge_style["LOW"])
                _tier_sty  = _tier_badge_style.get(tier, "background:#2a2208;color:#d4a843;")
                _tier_lbl  = _tier_labels.get(tier, "Unknown")
                _age_lbl   = f"{age_days}d ago" if age_days is not None else ""
                _left_clr  = _left_colors.get(cred_lbl, "#d4a843")

                # Per-tier card styling
                if cred_lbl == "VERIFIED":
                    _card_bg      = "#0e1a0a"
                    _card_border  = f"2px solid #4caf50"
                    _card_shadow  = "box-shadow:0 0 12px rgba(76,175,80,0.18);"
                    _card_left    = "5px solid #4caf50"
                    _verified_banner = (
                        '<div style="font-size:10px;font-weight:700;letter-spacing:0.12em;'
                        'color:#4caf50;margin-bottom:8px;">★ VERIFIED SOURCE</div>'
                    )
                elif cred_lbl == "HIGH":
                    _card_bg      = "#0e140a"
                    _card_border  = "1px solid #3a5a1a"
                    _card_shadow  = ""
                    _card_left    = "4px solid #8bc34a"
                    _verified_banner = ""
                elif cred_lbl == "MODERATE":
                    _card_bg      = "#171309"
                    _card_border  = "1px solid #2a2208"
                    _card_shadow  = ""
                    _card_left    = "3px solid #ff9800"
                    _verified_banner = ""
                else:  # LOW
                    _card_bg      = "#141210"
                    _card_border  = "1px solid #1e1a14"
                    _card_shadow  = ""
                    _card_left    = "2px solid #f44336"
                    _verified_banner = ""

                _title_clr  = "#d4a843" if cred_lbl != "VERIFIED" else "#7ecb80"
                _title_html = (
                    f'<a href="{link}" target="_blank" style="color:{_title_clr};font-weight:600;'
                    f'font-size:15px;text-decoration:none;line-height:1.4;">{title}</a>'
                    if link and link != "#"
                    else f'<span style="color:{_title_clr};font-weight:600;font-size:15px;">{title}</span>'
                )
                _cred_badge = (
                    f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;'
                    f'{_cred_sty}letter-spacing:0.08em;font-weight:700;">'
                    f'{cred_lbl} {cred_score}</span>'
                )
                _tier_badge = (
                    f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;'
                    f'{_tier_sty}letter-spacing:0.06em;">{_tier_lbl}</span>'
                )
                _confirm_badge = (
                    f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;'
                    f'background:#0a1a0a;color:#66bb6a;border:1px solid #1a4a1a;" '
                    f'title="Also reported by: {", ".join(confirms)}">'
                    f'&#10003; {len(confirms)} confirming source{"s" if len(confirms)!=1 else ""}'
                    f'</span>'
                ) if confirms else ""
                _src_label  = f'<span style="font-size:11px;color:#6a5228;">{src}</span>' if src else ""
                _date_label = f'<span style="font-size:11px;color:#4a3e18;">{_age_lbl}</span>' if _age_lbl else ""
                _desc_clr   = "#8a7040" if cred_lbl != "LOW" else "#5a5040"

                st.markdown(f"""
<div style="background:{_card_bg};border:{_card_border};border-radius:10px;
            padding:16px 18px;margin:8px 0;border-left:{_card_left};{_card_shadow}">
  {_verified_banner}
  <div style="margin-bottom:10px;">{_title_html}</div>
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
    {_cred_badge}
    {_tier_badge}
    {_confirm_badge}
    {_src_label}
    {_date_label}
  </div>
  <div style="font-size:13px;color:{_desc_clr};line-height:1.6;">{desc}</div>
</div>""", unsafe_allow_html=True)

            shown       = 0
            low_articles = []

            for art in raw:
                if feed_type_filter != "All" and art.get("feed_type") != feed_type_filter:
                    continue
                if art.get("credibility_score", 0) < _cred_min_score:
                    continue

                cred_lbl = art.get("credibility_label", "LOW")

                if cred_lbl == "LOW" and cred_filter == "All":
                    # Collect LOW articles to render collapsed
                    low_articles.append(art)
                else:
                    _render_article(art)
                    shown += 1

            # Render LOW articles collapsed
            if low_articles:
                with st.expander(f"Low credibility articles ({len(low_articles)}) — unverified / press releases"):
                    for art in low_articles:
                        _render_article(art)
                shown += len(low_articles)

            if shown == 0:
                st.info("No articles match the current filters.")

        st.caption(
            "Sources: Bloomberg, Reuters, AP News, Manufacturing.net, IndustryWeek, "
            "PR Newswire, Business Wire, US Dept of Energy, US Dept of Commerce, "
            "EDA, Expansion Solutions, Site Selection Magazine."
        )

        with st.expander("How Credibility Is Scored"):
            st.markdown("""
**Source Tiers (base score):**
- **Tier 1 — Independent News** (85 base): Bloomberg, Reuters, AP News — independently reported journalism
- **Tier 2 — Government** (80 base): DOE, Commerce Dept, EDA — official announcements
- **Tier 3 — Trade Press** (65 base): Manufacturing.net, IndustryWeek, Site Selection — credible industry reporting
- **Tier 4 — Press Releases** (30 base): PR Newswire, Business Wire — company self-published, unverified

**Credibility Bonuses:**
- Dollar amount disclosed: +10 pts · Jobs count: +5 pts · Location named: +5 pts · Timeline given: +5 pts
- Confirmed by 1 other independent source: +10 pts · Confirmed by 2+: +20 pts

**Age Penalty:** >7 days: −10 pts · >14 days: −20 pts · >30 days: filtered out entirely

**Labels:** VERIFIED (80+) · HIGH (65+) · MODERATE (45+) · LOW (<45)

The Groq AI brief only uses MODERATE+ articles — press releases not confirmed by independent sources are excluded.

**Update Frequency:** Company Predictions via Agent 3 (every 24h), Industry Announcements via Agent 5 (every 4h).
""")


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 7 — VACANCY RATE MONITOR
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_vacancy:
        vac_cache = read_cache("vacancy")
        vac_data  = vac_cache["data"]

        if not vac_data:
            # Use module-level data directly if cache not yet populated
            from src.vacancy_agent import run_vacancy_agent as _run_vac
            vac_data = _run_vac()

        national  = vac_data.get("national", NATIONAL_VACANCY)
        mkt_rows  = vac_data.get("market_rows", [])
        as_of     = vac_data.get("data_as_of", "Q1 2025")

        st.markdown(
            f"Commercial vacancy rates by property type and market. "
            f"Data as of **{as_of}**. Sources: CBRE, JLL, CoStar market reports."
        )

        # ── National Snapshot ────────────────────────────────────────────────
        section(" National Vacancy by Property Type")
        nat_cols = st.columns(len(national))
        for col, (ptype, info) in zip(nat_cols, national.items()):
            trend   = info["trend"]
            arrow   = TREND_ARROW[trend]
            color   = TREND_COLOR[trend]
            delta   = info["rate"] - info["prior_year"]
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{ptype}</div>
              <div class="value">{info['rate']}%</div>
              <div class="sub" style="color:{color};">{arrow} {trend.title()} ({delta:+.1f}pp YoY)</div>
              <div style="font-size:0.72rem;color:#888;margin-top:6px;">{info['note']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Office vacancy is at historic highs driven by remote/hybrid work. "
            "Industrial remains tight despite new supply. Retail is recovering in "
            "experience-oriented and grocery-anchored formats. Multifamily softening "
            "in Sunbelt markets from record 2024-2025 deliveries."
        )

        # ── Focus banner ─────────────────────────────────────────────────────
        _vac_pt = _active_pt()
        _pt_focus_banner(_vac_pt)
        _vac_focus_col = _PT_VAC_COL.get(_vac_pt) if _vac_pt else None

        # ── Market Heatmap ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Vacancy Rate by Market and Property Type")

        if mkt_rows:
            vac_df = pd.DataFrame(mkt_rows)
            pivot  = vac_df.pivot_table(
                index="market", columns="property_type", values="vacancy_rate"
            ).round(1)

            # Order columns: Office, Industrial, Retail, Multifamily (whichever exist)
            _vac_col_order = ["Office", "Industrial", "Retail", "Multifamily", "Mixed-Use"]
            _vac_cols = [c for c in _vac_col_order if c in pivot.columns] + \
                        [c for c in pivot.columns if c not in _vac_col_order]

            def _vac_cell_color(v: float) -> str:
                """Low vacancy = teal/green (tight). High vacancy = orange/red (soft)."""
                _stops = [
                    ( 2.0, (26,  122, 106)),   # deep teal  — very tight
                    ( 5.0, (45,  150,  64)),   # green
                    ( 8.0, (106, 176,  24)),   # lime-green
                    (12.0, (168, 162,  24)),   # yellow-lime
                    (16.0, (212, 148,  28)),   # gold
                    (20.0, (212, 100,  28)),   # orange
                    (26.0, (200,  50,  30)),   # red-orange
                ]
                if v <= _stops[0][0]:
                    return "rgb(%d,%d,%d)" % _stops[0][1]
                if v >= _stops[-1][0]:
                    return "rgb(%d,%d,%d)" % _stops[-1][1]
                for _i in range(len(_stops) - 1):
                    _lv, _lc = _stops[_i]
                    _hv, _hc = _stops[_i + 1]
                    if _lv <= v <= _hv:
                        _t = (v - _lv) / (_hv - _lv)
                        return "rgb(%d,%d,%d)" % (
                            int(_lc[0] + _t * (_hc[0] - _lc[0])),
                            int(_lc[1] + _t * (_hc[1] - _lc[1])),
                            int(_lc[2] + _t * (_hc[2] - _lc[2])),
                        )
                return "rgb(150,150,150)"

            # Header row — dim non-focus columns when focus is active
            _vhdr_cells = ""
            for c in _vac_cols:
                _is_focus_col = (not _vac_focus_col) or (c == _vac_focus_col)
                _hdr_opacity = "1" if _is_focus_col else "0.25"
                _hdr_clr     = "#c8a040" if _is_focus_col else "#5a5030"
                _hdr_fw      = "700" if _is_focus_col else "400"
                _vhdr_cells += (
                    f'<th style="padding:10px 8px 14px;color:{_hdr_clr};font-size:0.78rem;'
                    f'font-weight:{_hdr_fw};letter-spacing:0.08em;text-align:center;'
                    f'border-bottom:1px solid #2a2410;opacity:{_hdr_opacity};">{c.upper()}</th>'
                )
            _vhdr = (
                f'<tr><th style="padding:10px 8px 14px;border-bottom:1px solid #2a2410;"></th>'
                f'{_vhdr_cells}</tr>'
            )

            # Data rows — dim non-focus columns
            _vrows_html = ""
            import numpy as _np
            for _mkt in pivot.index:
                _vcells = (
                    f'<td style="padding:7px 12px 7px 0;text-align:right;color:#c8b890;'
                    f'font-size:0.88rem;white-space:nowrap;border-bottom:1px solid #1e1c0e;">'
                    f'{_mkt}</td>'
                )
                for _col in _vac_cols:
                    _is_focus_col = (not _vac_focus_col) or (_col == _vac_focus_col)
                    _cell_opacity = "1" if _is_focus_col else "0.22"
                    _val = pivot.loc[_mkt, _col] if _col in pivot.columns else None
                    if _val is not None and not (_np.isnan(_val) if hasattr(_val, '__float__') else False) and float(_val) > 0:
                        _bg = _vac_cell_color(float(_val))
                        _vcells += (
                            f'<td style="padding:7px 6px;border-bottom:1px solid #1e1c0e;opacity:{_cell_opacity};">'
                            f'<div style="background:{_bg};border-radius:7px;padding:9px 0;'
                            f'text-align:center;color:#fff;font-size:0.92rem;font-weight:600;'
                            f'letter-spacing:0.04em;min-width:80px;">{float(_val):.1f}%</div></td>'
                        )
                    else:
                        _vcells += (
                            f'<td style="padding:7px 6px;border-bottom:1px solid #1e1c0e;opacity:{_cell_opacity};">'
                            f'<div style="text-align:center;color:#4a4530;font-size:0.92rem;">—</div></td>'
                        )
                _vrows_html += f"<tr>{_vcells}</tr>"

            _vac_hm_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:20px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">VACANCY RATE HEATMAP</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">All markets × property types &mdash; CBRE / JLL / CoStar Q1 2025</div>
  </div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead>{_vhdr}</thead>
      <tbody>{_vrows_html}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    Teal/green = tight market (low vacancy, strong demand) &nbsp;·&nbsp; Gold/orange/red = soft market (excess supply)
    &nbsp;·&nbsp; Source: CBRE / JLL / CoStar Q1 2025 &nbsp;·&nbsp; Not financial advice.
  </div>
</div>
"""
            st.markdown(_vac_hm_html, unsafe_allow_html=True)

        # ── Market Detail Table ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Market Detail — Vacancy vs. National Average")

        if mkt_rows:
            detail_df = pd.DataFrame(mkt_rows)

            # Build header
            _md_headers = ["MARKET", "PROPERTY TYPE", "VACANCY %", "TREND", "VS. NATIONAL"]
            _md_hcells = "".join(
                f'<th style="padding:12px 16px 14px;color:#c8a040;font-size:0.78rem;'
                f'font-weight:700;letter-spacing:0.09em;text-align:{"left" if i < 2 else "right"};'
                f'border-bottom:1px solid #2a2410;">{h}</th>'
                for i, h in enumerate(_md_headers)
            )

            # Build rows — dim non-focus property type rows when focus is active
            _md_rows_html = ""
            for _, _row in detail_df.iterrows():
                _trend_str  = f"{TREND_ARROW[_row['trend']]} {_row['trend'].title()}"
                _trend_c    = TREND_COLOR[_row["trend"]]
                _vs         = float(_row.get("vs_national", 0) or 0)
                _vs_str     = f"{_vs:+.1f}pp"
                _vs_c       = "#66bb6a" if _vs < -2 else ("#ef5350" if _vs > 2 else "#c8a040")
                _vac_val    = float(_row["vacancy_rate"])
                _is_focus_row = (not _vac_focus_col) or (_row["property_type"] == _vac_focus_col)
                _row_opacity  = "1" if _is_focus_row else "0.25"
                _row_bg       = "#1a1500" if _is_focus_row and _vac_focus_col else "transparent"
                _row_style    = f"border-bottom:1px solid #1e1c0e;background:{_row_bg};opacity:{_row_opacity};"
                _td           = f'style="padding:12px 16px;{_row_style}'
                _md_rows_html += (
                    f'<tr>'
                    f'<td {_td}text-align:left;color:#c8b890;font-size:0.9rem;white-space:nowrap;">{_row["market"]}</td>'
                    f'<td {_td}text-align:left;color:#a09070;font-size:0.87rem;">{_row["property_type"]}</td>'
                    f'<td {_td}text-align:right;color:#c8a040;font-size:0.92rem;font-weight:600;letter-spacing:0.05em;">{_vac_val:.1f}%</td>'
                    f'<td {_td}text-align:right;color:{_trend_c};font-size:0.87rem;">{_trend_str}</td>'
                    f'<td {_td}text-align:right;color:{_vs_c};font-size:0.87rem;font-weight:600;">{_vs_str}</td>'
                    f'</tr>'
                )

            _md_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:20px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">MARKET DETAIL — VACANCY VS. NATIONAL AVERAGE</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Vacancy rate &amp; trend by market and property type &mdash; CBRE / JLL / CoStar Q1 2025</div>
  </div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead><tr>{_md_hcells}</tr></thead>
      <tbody>{_md_rows_html}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    vs. National = pp difference from property-type national average &nbsp;·&nbsp;
    <span style="color:#66bb6a;">Green</span> = tighter than avg &nbsp;·&nbsp;
    <span style="color:#ef5350;">Red</span> = looser than avg &nbsp;·&nbsp;
    Not financial advice.
  </div>
</div>
"""
            st.markdown(_md_html, unsafe_allow_html=True)

        # ── Absorption Rate ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Net Absorption — Q1 2025 (thousands sq ft)")
        st.markdown(
            "Net absorption = space leased minus space vacated. "
            "Positive = more demand than supply returned. "
            "Negative = tenants leaving faster than new leases are signed."
        )

        # National absorption summary cards
        nat_abs = vac_data.get("national_absorption") or {}
        if not nat_abs:
            st.info("Absorption data loading — refresh in ~30 seconds.")
        abs_cols = st.columns(max(len(nat_abs), 1))
        for col, (ptype, info) in zip(abs_cols, nat_abs.items()):
            net   = info["net_msf"]
            prior = info["prior_quarter"]
            trend = info["trend"]
            color = "#66bb6a" if net > 0 else "#ef5350"
            trend_color = {"improving": "#66bb6a", "slowing": "#d4a843", "worsening": "#ef5350", "stable": "#d4a843"}.get(trend, "#888")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{ptype}</div>
              <div class="value" style="color:{color};">{net:+.1f}M sf</div>
              <div class="sub">Prior Qtr: {prior:+.1f}M sf</div>
              <div style="font-size:0.72rem;color:{trend_color};margin-top:4px;">{trend.title()}</div>
            </div>""", unsafe_allow_html=True)

        st.caption("National net absorption in millions of square feet. Q1 2025. Sources: CBRE, JLL, CoStar.")

        # Market-level absorption bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        abs_rows = vac_data.get("absorption_rows", [])
        if abs_rows:
            abs_df = pd.DataFrame(abs_rows)

            _pt_options = ["Industrial", "Office", "Retail", "Multifamily"]
            _sel_pt = st.selectbox("Property Type", _pt_options, key="abs_pt_sel")
            _abs_filtered = abs_df[abs_df["property_type"] == _sel_pt].sort_values(
                "net_absorption_ksf", ascending=True
            )

            bar_colors = [
                "#66bb6a" if v > 0 else "#ef5350"
                for v in _abs_filtered["net_absorption_ksf"]
            ]
            fig_abs = go.Figure(go.Bar(
                x=_abs_filtered["net_absorption_ksf"],
                y=_abs_filtered["market"],
                orientation="h",
                marker=dict(color=bar_colors),
                text=_abs_filtered["net_absorption_ksf"].apply(
                    lambda v: f"{v:+,} ksf"
                ),
                textposition="outside",
                textfont=dict(color="#c8b890", size=10),
                hovertemplate="<b>%{y}</b><br>Net Absorption: %{x:,} ksf<extra></extra>",
            ))
            fig_abs.add_vline(x=0, line_width=1, line_color="#888")
            fig_abs.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                xaxis=dict(
                    title="Net Absorption (thousands sq ft)",
                    gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                    title_font=dict(color="#c8b890"),
                ),
                yaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=20, b=40, l=180, r=80),
                height=520,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_abs, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Green bars = net positive absorption (demand outpacing supply returns). "
                "Red bars = net negative absorption (more space returned than leased). "
                "Industrial dominates positive absorption in Sun Belt markets. "
                "Office shows persistent negative absorption in most major metros."
            )

        # ── Land Availability by Market ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Developable Land Availability — Q1 2025")
        st.markdown(
            "Entitled or shovel-ready vacant land acres actively available for development "
            "across top CRE markets. Larger supply + lower entitlement timeline = faster "
            "path to breaking ground. Sources: CoStar Land, state planning databases."
        )

        _land_avail = vac_data.get("land_availability") or LAND_AVAILABILITY
        if _land_avail:
            _la_df = pd.DataFrame([
                {
                    "Market": mkt,
                    "Industrial (ac)": info["industrial_ac"],
                    "Mixed-Use (ac)":  info["mixed_use_ac"],
                    "Residential (ac)": info["residential_ac"],
                    "Total (ac)": info["industrial_ac"] + info["mixed_use_ac"] + info["residential_ac"],
                    "Avg $/Acre": info["avg_ppa"],
                    "Entitlement (mo)": info["entitlement_mo"],
                    "Pipeline": TREND_ARROW.get(info["pipeline_trend"], "→") + " " + info["pipeline_trend"].title(),
                }
                for mkt, info in _land_avail.items()
            ]).sort_values("Total (ac)", ascending=False)

            # Bar chart: total available acres by market
            _la_colors = [
                TREND_COLOR.get(
                    list(_land_avail.values())[i]["pipeline_trend"], "#d4a843"
                )
                for i in range(len(_la_df))
            ]
            fig_land = go.Figure(go.Bar(
                y=_la_df["Market"],
                x=_la_df["Total (ac)"],
                orientation="h",
                marker=dict(color=_la_colors, opacity=0.85),
                customdata=_la_df[["Industrial (ac)", "Mixed-Use (ac)", "Residential (ac)", "Avg $/Acre", "Entitlement (mo)"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Industrial: %{customdata[0]:,} ac<br>"
                    "Mixed-Use: %{customdata[1]:,} ac<br>"
                    "Residential: %{customdata[2]:,} ac<br>"
                    "Total: %{x:,} ac<br>"
                    "Avg $/acre: $%{customdata[3]:,}<br>"
                    "Entitlement: ~%{customdata[4]} months<extra></extra>"
                ),
                text=_la_df["Total (ac)"].apply(lambda v: f"{v:,} ac"),
                textposition="outside",
                textfont=dict(color="#c8b890", size=10),
            ))
            fig_land.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                xaxis=dict(
                    title="Total Developable Acres",
                    gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                    title_font=dict(color="#c8b890"),
                ),
                yaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=20, b=40, l=180, r=100),
                height=520,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_land, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Bar color = pipeline trend: green = more entitlements coming online (rising supply), "
                "gold = stable, red = fewer new entitlements (constrained market). "
                "Sun Belt markets (Dallas, Houston, Phoenix) lead in available developable land."
            )

            # Detail table
            st.markdown("<br>", unsafe_allow_html=True)

            _LND_C = {"Industrial": "#2bbfb0", "Mixed-Use": "#a09040", "Residential": "#a07830"}

            # Legend dots
            _legend_html = " &nbsp;&nbsp; ".join(
                f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82rem;color:#c8b890;">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{c};"></span>{lbl}</span>'
                for lbl, c in _LND_C.items()
            )

            # Header
            _lnd_hdrs = ["MARKET", "INDUSTRIAL", "MIXED-USE", "RESIDENTIAL", "TOTAL (AC)", "MIX"]
            _lnd_aligns = ["left", "right", "right", "right", "right", "left"]
            _lnd_hcells = "".join(
                f'<th style="padding:12px 14px 14px;color:#c8a040;font-size:0.78rem;font-weight:700;'
                f'letter-spacing:0.09em;text-align:{_lnd_aligns[i]};border-bottom:1px solid #2a2410;">{h}</th>'
                for i, h in enumerate(_lnd_hdrs)
            )

            # Rows
            _lnd_rows_html = ""
            for _, _lr in _la_df.iterrows():
                _ind = int(_lr["Industrial (ac)"])
                _mix = int(_lr["Mixed-Use (ac)"])
                _res = int(_lr["Residential (ac)"])
                _tot = _ind + _mix + _res or 1
                _i_pct = _ind / _tot * 100
                _m_pct = _mix / _tot * 100
                _r_pct = _res / _tot * 100
                _bar = (
                    f'<div style="display:flex;height:10px;border-radius:4px;overflow:hidden;min-width:80px;">'
                    f'<div style="width:{_i_pct:.1f}%;background:{_LND_C["Industrial"]};"></div>'
                    f'<div style="width:{_m_pct:.1f}%;background:{_LND_C["Mixed-Use"]};margin:0 1px;"></div>'
                    f'<div style="width:{_r_pct:.1f}%;background:{_LND_C["Residential"]};"></div>'
                    f'</div>'
                )
                _sep = "border-bottom:1px solid #1e1c0e;"
                _lnd_rows_html += (
                    f'<tr>'
                    f'<td style="padding:13px 14px;{_sep}color:#c8b890;font-size:0.9rem;white-space:nowrap;">{_lr["Market"]}</td>'
                    f'<td style="padding:13px 14px;{_sep}text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_ind:,}</td>'
                    f'<td style="padding:13px 14px;{_sep}text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_mix:,}</td>'
                    f'<td style="padding:13px 14px;{_sep}text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_res:,}</td>'
                    f'<td style="padding:13px 14px;{_sep}text-align:right;color:#c8a040;font-size:0.9rem;font-weight:700;letter-spacing:0.04em;">{_tot:,}</td>'
                    f'<td style="padding:13px 14px;{_sep}min-width:100px;">{_bar}</td>'
                    f'</tr>'
                )

            _lnd_table_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:18px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">LAND MARKET DETAIL TABLE</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Developable acreage, pricing &amp; pipeline activity &mdash; CoStar / CBRE Q1 2025</div>
  </div>
  <div style="margin-bottom:16px;">{_legend_html}</div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead><tr>{_lnd_hcells}</tr></thead>
      <tbody>{_lnd_rows_html}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    Acreage = entitled or shovel-ready developable land actively available &nbsp;·&nbsp;
    MIX bar shows Industrial / Mixed-Use / Residential proportion &nbsp;·&nbsp;
    Source: CoStar Land / CBRE Q1 2025 &nbsp;·&nbsp; Not financial advice.
  </div>
</div>
"""
            st.markdown(_lnd_table_html, unsafe_allow_html=True)
            st.caption(
                "Entitlement timeline = estimated months from land purchase to permitted/shovel-ready status. "
                "Markets like New York and Los Angeles require 3-5 years; Sun Belt typically 12-20 months."
            )

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — LAND & DEVELOPMENT SITES
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_land:
        st.markdown("#### Land & Development Sites")
        st.markdown(
            "Browse empty land parcels across top CRE markets. Agent 13 tracks entitlement "
            "pipeline activity, notable transactions, price trends, and demand signals — "
            "calibrated to CoStar Land / LoopNet Q1 2025 benchmarks. Updates every 12 hours."
        )
        agent_last_updated("land_market")

        # Load land market agent cache
        _lm_cache = read_cache("land_market")
        _lm_data  = _lm_cache.get("data") or {}
        _lm_intel = _lm_data.get("market_intelligence", {})
        _lm_signals = _lm_data.get("demand_signals", {})
        _lm_txns    = _lm_data.get("notable_transactions", [])
        _lm_ents    = _lm_data.get("new_entitlements", [])
        _lm_prices  = _lm_data.get("price_trends", {})

        # ── AI Market Intelligence Brief ──────────────────────────────────
        if _lm_intel.get("summary"):
            section(" Agent 13 — Land Market Intelligence")
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-label">Agent 13 &nbsp;·&nbsp; Land Market Agent &nbsp;·&nbsp; {datetime.today().strftime('%b %d, %Y')}</div>
              <div class="agent-text">{_lm_intel['summary']}</div>
              {"<div style='margin-top:12px;padding:8px 12px;background:#1e1a0a;border-radius:6px;border-left:3px solid #66bb6a;'><span style='color:#66bb6a;font-weight:700;'>Best Opportunity:</span> <span style='color:#c8bfa8;'>" + _lm_intel.get('top_opportunity','') + "</span></div>" if _lm_intel.get('top_opportunity') else ""}
              {"<div style='margin-top:8px;padding:8px 12px;background:#1e1a0a;border-radius:6px;border-left:3px solid #ef5350;'><span style='color:#ef5350;font-weight:700;'>Key Risk:</span> <span style='color:#c8bfa8;'>" + _lm_intel.get('top_risk','') + "</span></div>" if _lm_intel.get('top_risk') else ""}
            </div>""", unsafe_allow_html=True)

        # ── Demand Signals ────────────────────────────────────────────────
        if _lm_signals:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Land Demand by Use Type")
            _sig_cols = st.columns(len(_lm_signals))
            for col, (use, sig) in zip(_sig_cols, _lm_signals.items()):
                _sc = "#66bb6a" if sig["score"] >= 70 else ("#d4a843" if sig["score"] >= 45 else "#ef5350")
                _arr = TREND_ARROW.get(sig["trend"], "→")
                _short = use.replace("_land_demand", "").replace("_", " ").title()
                col.markdown(f"""
                <div class="metric-card">
                  <div class="label">{_short}</div>
                  <div class="value" style="color:{_sc};">{sig['score']}</div>
                  <div class="sub" style="color:{_sc};">{_arr} {sig['trend'].title()}</div>
                  <div style="font-size:0.72rem;color:#7a7a6a;margin-top:6px;line-height:1.4;">{sig['note']}</div>
                </div>""", unsafe_allow_html=True)
            st.caption("Demand score 0-100: higher = stronger institutional land buying activity for that use type.")

        # ── Price Trends ──────────────────────────────────────────────────
        if _lm_prices:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Land Price Trends — $/Acre YoY")
            _pt_df = pd.DataFrame([
                {
                    "Market":          mkt,
                    "Current $/Acre":  pt["current_ppa"],
                    "Prior Year":      pt["prior_year_ppa"],
                    "YoY %":           pt["yoy_pct"],
                    "Trend":           TREND_ARROW.get(pt["trend"], "→") + " " + pt["trend"].title(),
                }
                for mkt, pt in _lm_prices.items()
            ]).sort_values("YoY %", ascending=False)

            fig_pt = go.Figure()
            _pt_colors = [
                "#66bb6a" if v > 5 else ("#ef5350" if v < 0 else "#d4a843")
                for v in _pt_df["YoY %"]
            ]
            fig_pt.add_trace(go.Bar(
                y=_pt_df["Market"],
                x=_pt_df["YoY %"],
                orientation="h",
                marker=dict(color=_pt_colors, opacity=0.85),
                text=_pt_df["YoY %"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
                textfont=dict(color="#c8b890", size=10),
                customdata=_pt_df[["Current $/Acre", "Prior Year"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "YoY Change: %{x:+.1f}%<br>"
                    "Current: $%{customdata[0]:,}/ac<br>"
                    "Prior Year: $%{customdata[1]:,}/ac<extra></extra>"
                ),
            ))
            fig_pt.add_vline(x=0, line_width=1, line_color="#888")
            fig_pt.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                xaxis=dict(title="YoY % Change", gridcolor="#2a2208",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                yaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=20, b=40, l=180, r=80),
                height=480,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_pt, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Green = land prices rising more than 5% YoY (strong demand). "
                "Gold = moderate appreciation. Red = prices flat or declining (soft market)."
            )

        # ── Recent Entitlement Filings ────────────────────────────────────
        if _lm_ents:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Recent Entitlement Filings — Q1 2025")
            st.markdown(
                "New entitlement applications submitted to planning departments across top markets. "
                "Industrial and mixed-use dominate the pipeline — a leading indicator of where "
                "developers expect demand to materialize 12-24 months from now."
            )
            _ent_df = pd.DataFrame(_lm_ents).rename(columns={
                "market": "Market", "acres": "Acres",
                "zoning": "Zoning", "applicant_type": "Applicant",
                "est_months_to_shovel": "Est. Months to Shovel-Ready",
                "filed": "Filed",
            })
            st.dataframe(_ent_df, use_container_width=True, hide_index=True)
            st.caption(
                "Months to shovel-ready = estimated time from filing to all permits in hand. "
                "Corp. User = company buying land for its own facility. REIT = speculative development."
            )

        # ── Notable Transactions ──────────────────────────────────────────
        if _lm_txns:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Notable Land Transactions — Q3 2024 – Q1 2025")
            for txn in _lm_txns:
                _total = txn["acres"] * txn["price_per_acre"]
                st.markdown(f"""
                <div class="listing-card" style="border-left-color:#42a5f5;">
                  <div style="display:flex;justify-content:space-between;">
                    <div>
                      <div class="l-price">{txn['buyer']} — {txn['market']}</div>
                      <div class="l-address">{txn['use']} &nbsp;·&nbsp; {txn['acres']:,} acres &nbsp;·&nbsp; {txn['quarter']}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:1.4rem;font-weight:700;color:#42a5f5;">${_total:,.0f}</div>
                      <div style="font-size:0.8rem;color:#a09070;">${txn['price_per_acre']:,}/acre</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        _land_col_filter, _land_col_info = st.columns([2, 3])

        with _land_col_filter:
            _land_states_sorted = sorted(LAND_PRICE_PER_ACRE.keys())
            _sel_land_state = st.selectbox("State", _land_states_sorted,
                                           index=_land_states_sorted.index("TX") if "TX" in _land_states_sorted else 0,
                                           key="land_state_sel")
            _land_zoning_opts = ["All"] + ZONING_TYPES
            _sel_land_zoning  = st.selectbox("Zoning Filter", _land_zoning_opts, key="land_zoning_sel")
            _land_ent_opts    = ["All"] + ENTITLEMENT_STATUS
            _sel_land_ent     = st.selectbox("Entitlement Status", _land_ent_opts, key="land_ent_sel")

        parcels = get_land_parcels(_sel_land_state, n=10)

        # Apply filters
        if _sel_land_zoning != "All":
            parcels = [p for p in parcels if p["zoning"] == _sel_land_zoning] or parcels
        if _sel_land_ent != "All":
            parcels = [p for p in parcels if p["entitlement_status"] == _sel_land_ent] or parcels

        with _land_col_info:
            if parcels:
                _avg_ppa   = sum(p["price_per_acre"] for p in parcels) / len(parcels)
                _avg_score = sum(p["dev_potential_score"] for p in parcels) / len(parcels)
                _shovel    = sum(1 for p in parcels if p["entitlement_status"] == "Shovel-Ready")
                _avg_ac    = sum(p["acres"] for p in parcels) / len(parcels)
                _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                _kc1.markdown(metric_card("Avg Price/Acre", f"${_avg_ppa:,.0f}", _sel_land_state), unsafe_allow_html=True)
                _kc2.markdown(metric_card("Avg Dev Score",  f"{_avg_score:.0f}/100", "Entitlement-based"), unsafe_allow_html=True)
                _kc3.markdown(metric_card("Shovel-Ready",   str(_shovel), f"of {len(parcels)} shown"), unsafe_allow_html=True)
                _kc4.markdown(metric_card("Avg Acreage",    f"{_avg_ac:.1f} ac", "Per listing"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section(f" Available Parcels — {_sel_land_state}")
        st.caption(f"Showing {len(parcels)} listings sorted by asking price")

        _ent_color = {
            "Raw / Unentitled": "#ef5350",
            "Entitled":         "#d4a843",
            "Permitted":        "#42a5f5",
            "Shovel-Ready":     "#66bb6a",
        }
        _util_icon = {"At Site": "✔", "Available": "~", "Stubbed": "~", "Not Available": "✗"}

        for p in parcels:
            _ec   = _ent_color.get(p["entitlement_status"], "#888")
            _ui   = _util_icon.get(p["utilities"], "?")
            _score_bar_w = p["dev_potential_score"]
            _score_color = "#66bb6a" if _score_bar_w >= 75 else ("#d4a843" if _score_bar_w >= 50 else "#ef5350")
            st.markdown(f"""
            <div class="listing-card" style="border-left-color:{_ec};">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div class="l-price">${p['price']:,}
                    <span style="font-size:0.9rem;font-weight:400;color:#a09070;"> — {p['acres']} acres @ ${p['price_per_acre']:,}/ac</span>
                  </div>
                  <div class="l-address">{p['address']}, {p['city']}, {p['state']}</div>
                </div>
                <div style="text-align:right;min-width:120px;">
                  <div style="font-size:0.72rem;color:#a09070;letter-spacing:0.08em;">DEV POTENTIAL</div>
                  <div style="font-size:1.4rem;font-weight:800;color:{_score_color};">{p['dev_potential_score']}</div>
                  <div style="width:100%;height:4px;background:#2a2208;border-radius:2px;margin-top:2px;">
                    <div style="width:{_score_bar_w}%;height:4px;background:{_score_color};border-radius:2px;"></div>
                  </div>
                </div>
              </div>
              <div style="margin:6px 0;">
                <span class="l-tag" style="border-color:{_ec};color:{_ec};">{p['entitlement_status']}</span>
                <span class="l-tag">{p['zoning']}</span>
                <span class="l-tag">{p['days_on_market']}d on market</span>
              </div>
              <div class="l-detail">
                Road frontage: {p['road_frontage_ft']} ft &nbsp;·&nbsp;
                Utilities: {_ui} {p['utilities']} &nbsp;·&nbsp;
                {p['acres']} acres
              </div>
              {"<div class='l-detail' style='color:#555;margin-top:4px;font-style:italic;'>" + p['highlights'] + "</div>" if p['highlights'] else ""}
            </div>""", unsafe_allow_html=True)

        # ── Development potential guide ───────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Entitlement Status Guide")
        st.markdown(
            "Understanding the entitlement pipeline helps investors gauge timeline risk and "
            "development cost. Shovel-ready sites command a premium but eliminate the longest "
            "regulatory risk periods."
        )
        _guide_cols = st.columns(4)
        _guide = [
            ("Raw / Unentitled", "#ef5350", "0–30", "No permits filed. Longest timeline. "
             "Lowest acquisition cost. Highest regulatory risk."),
            ("Entitled",         "#d4a843", "31–65", "Zoning approved, environmental cleared. "
             "Permits not yet issued. Moderate cost and risk."),
            ("Permitted",        "#42a5f5", "66–82", "Building permits issued or near-approval. "
             "Engineering & utility plans typically complete."),
            ("Shovel-Ready",     "#66bb6a", "83–100", "All permits in hand. Utilities stubbed. "
             "Construction can start immediately. Highest premium."),
        ]
        for col, (label, color, score_rng, desc) in zip(_guide_cols, _guide):
            col.markdown(f"""
            <div class="metric-card" style="border-left:3px solid {color};">
              <div class="label" style="color:{color};">{label}</div>
              <div class="value" style="font-size:1.1rem;color:{color};">Score {score_rng}</div>
              <div style="font-size:0.78rem;color:#9a9080;margin-top:6px;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        # ── Market context ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Land Market Context — {_sel_land_state}".format(_sel_land_state=_sel_land_state))
        _land_mkt_match = {k: v for k, v in LAND_AVAILABILITY.items()
                           if k.endswith(f", {_sel_land_state}")}
        if _land_mkt_match:
            _lm_cols = st.columns(min(len(_land_mkt_match), 3))
            for col, (mkt, li) in zip(_lm_cols, _land_mkt_match.items()):
                _total = li["industrial_ac"] + li["mixed_use_ac"] + li["residential_ac"]
                _pt    = TREND_ARROW.get(li["pipeline_trend"], "→")
                col.markdown(f"""
                <div class="metric-card">
                  <div class="label">{mkt.split(",")[0]}</div>
                  <div class="value" style="font-size:1.1rem;">{_total:,} ac</div>
                  <div class="sub">Avg ${li['avg_ppa']:,}/acre</div>
                  <div style="font-size:0.78rem;color:#9a9080;margin-top:4px;">
                    Industrial: {li['industrial_ac']:,} ac<br>
                    Mixed-Use: {li['mixed_use_ac']:,} ac<br>
                    Entitlement: ~{li['entitlement_mo']} mo<br>
                    Pipeline: {_pt} {li['pipeline_trend'].title()}
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption(
                f"Detailed market-level land data is available for: "
                + ", ".join(set(k.split(", ")[1] for k in LAND_AVAILABILITY))
            )

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — MARKET OPPORTUNITY SCORE
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_score:
        st.markdown("#### Which CRE markets offer the best composite opportunity today?")
        st.markdown(
            "Agent 18 synthesizes migration, vacancy, rent growth, cap rate attractiveness, "
            "land availability, and macro conditions into a single 0–100 composite score "
            "per market. Updates every 6 hours."
        )
        agent_last_updated("market_score")

        _ms_cache = read_cache("market_score")
        _ms_data  = _ms_cache.get("data") or {}

        if not _ms_data:
            st.info(" Market Score agent is running its first computation — refresh in ~30 seconds.")
        else:
            _ms_rankings  = _ms_data.get("rankings", [])
            _ms_top3      = _ms_data.get("top3_markets", [])
            _ms_avoid     = _ms_data.get("avoid_markets", [])
            _ms_avg       = _ms_data.get("avg_score", 0)
            _ms_fw        = _ms_data.get("factor_weights", {})

            # ── Score explainer ───────────────────────────────────────────────
            with st.expander("How the composite score is calculated"):
                st.markdown("""
**The composite score (0–100)** is a weighted average of 7 signals pulled from all other agents. Higher = better investment environment.

| Factor | Weight | What it captures |
|--------|--------|-----------------|
| Migration / Population | 25% | Net domestic migration, population growth, business relocations |
| Labor (via migration proxy) | 15% | Employment base and tenant demand potential |
| Vacancy & Absorption | 20% | Current vacancy rate vs. national average; positive net absorption |
| Rent Growth Momentum | 15% | YoY rent growth weighted: industrial 40%, multifamily 35%, retail 15%, office 10% |
| Cap Rate Attractiveness | 10% | Treasury spread — wider spread = higher score |
| Land Availability | 10% | Developable acres + entitlement timeline (faster = better) |
| Macro Environment | 5% | Interest rate environment + credit conditions signal |

**Climate Risk Adjustment** (applied after weighting): Markets with a state climate risk score ≥ 60 receive a composite penalty of `min(10, (score − 60) × 0.20)` points. A Severe-risk market (score 85) loses up to 5 pts from its composite. This is shown as a red annotation on the score.

**Grade scale:** A ≥ 80 · B+ ≥ 70 · B ≥ 60 · C+ ≥ 50 · C ≥ 40 · D < 40
                """)

            # ── Summary cards ─────────────────────────────────────────────────
            section(" Composite Market Rankings")
            _ms_c1, _ms_c2, _ms_c3 = st.columns(3)
            _ms_c1.markdown(metric_card("Avg Market Score", f"{_ms_avg}/100", "19 tracked markets"), unsafe_allow_html=True)
            _ms_c2.markdown(metric_card("Top Markets", " · ".join(_ms_top3[:2]) if _ms_top3 else "—", "Highest composite scores"), unsafe_allow_html=True)
            _ms_c3.markdown(metric_card("Avoid", " · ".join(_ms_avoid[:2]) if _ms_avoid else "—", "Lowest composite scores"), unsafe_allow_html=True)

            # ── Rankings bar chart ────────────────────────────────────────────
            if _ms_rankings:
                _ms_df = pd.DataFrame(_ms_rankings)
                _ms_df["color"] = _ms_df["composite"].apply(
                    lambda s: "#66bb6a" if s >= 70 else ("#d4a843" if s >= 50 else "#ef5350")
                )
                fig_ms = go.Figure(go.Bar(
                    x=_ms_df["composite"],
                    y=_ms_df["market"],
                    orientation="h",
                    marker_color=_ms_df["color"].tolist(),
                    text=[f"{r['grade']}  {r['composite']}" for _, r in _ms_df.iterrows()],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Score: %{x}<extra></extra>",
                ))
                fig_ms.update_layout(
                    xaxis=dict(range=[0, 105], title="Composite Score (0–100)",
                               gridcolor="#2a2208", color="#8a7040"),
                    yaxis=dict(categoryorder="total ascending", color="#8a7040"),
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    margin=dict(l=160, r=80, t=20, b=40), height=580,
                )
                st.plotly_chart(fig_ms, use_container_width=True)

            # ── Factor weight legend ──────────────────────────────────────────
            section(" Factor Weights")
            _ms_fw_labels = {
                "migration": "Migration / Population (+ Labor proxy)",
                "vacancy":   "Vacancy & Absorption",
                "rent":      "Rent Growth Momentum",
                "cap_rate":  "Cap Rate Attractiveness",
                "land":      "Land Availability",
                "macro":     "Macro Environment",
            }
            _ms_fw_cols = st.columns(len(_ms_fw_labels))
            for _ms_col, (_ms_k, _ms_lbl) in zip(_ms_fw_cols, _ms_fw_labels.items()):
                _ms_pct = int(_ms_fw.get(_ms_k, 0) * 100)
                _ms_extra = " (+15% labor)" if _ms_k == "migration" else ""
                _ms_col.markdown(metric_card(_ms_lbl, f"{_ms_pct}%{_ms_extra}", "weight"), unsafe_allow_html=True)

            # ── Top 10 factor breakdown table ─────────────────────────────────
            _FACTOR_COLS = [
                ("Migration", "migration", "#2bbfb0"),
                ("Vacancy",   "vacancy",   "#c8a040"),
                ("Rent",      "rent",      "#c8a040"),
                ("Cap Rate",  "cap_rate",  "#c8a040"),
                ("Land",      "land",      "#c8a040"),
                ("Macro",     "macro",     "#c8a040"),
            ]

            # Header
            _mfb_fcols_hdr = "".join(
                f'<th style="padding:10px 10px 14px;color:#c8a040;font-size:0.75rem;font-weight:700;'
                f'letter-spacing:0.09em;text-align:center;border-bottom:1px solid #2a2410;">{lbl.upper()}</th>'
                for lbl, _, _ in _FACTOR_COLS
            )
            _mfb_hdr = f"""
<tr>
  <th style="padding:10px 10px 14px;color:#c8a040;font-size:0.75rem;font-weight:700;letter-spacing:0.09em;text-align:center;border-bottom:1px solid #2a2410;width:30px;">#</th>
  <th style="padding:10px 14px 14px;color:#c8a040;font-size:0.75rem;font-weight:700;letter-spacing:0.09em;text-align:left;border-bottom:1px solid #2a2410;">MARKET</th>
  <th style="padding:10px 14px 14px;border-bottom:1px solid #2a2410;width:100px;"></th>
  <th style="padding:10px 10px 14px;color:#c8a040;font-size:0.75rem;font-weight:700;letter-spacing:0.09em;text-align:right;border-bottom:1px solid #2a2410;">SCORE</th>
  <th style="padding:10px 10px 14px;color:#c8a040;font-size:0.75rem;font-weight:700;letter-spacing:0.09em;text-align:center;border-bottom:1px solid #2a2410;">GRADE</th>
  {_mfb_fcols_hdr}
  <th style="padding:10px 10px 14px;color:#ef5350;font-size:0.75rem;font-weight:700;letter-spacing:0.09em;text-align:center;border-bottom:1px solid #2a2410;">CLIMATE ADJ.</th>
</tr>"""

            # Grade badge colors
            def _grade_badge(g):
                _gc = {"A": "#2bbfb0", "B+": "#c8a040", "B": "#7a6830", "C+": "#6a5828", "C": "#4a3820", "D": "#3a2010"}.get(g, "#3a3020")
                _tc = "#fff" if g in ("A",) else "#c8b870"
                return (
                    f'<span style="display:inline-block;background:{_gc};color:{_tc};'
                    f'font-size:0.78rem;font-weight:700;padding:3px 10px;border-radius:5px;'
                    f'letter-spacing:0.04em;">{g}</span>'
                )

            # Rows
            _mfb_rows = ""
            for _ri, _rr in enumerate(_ms_rankings[:10]):
                _f       = _rr["factors"]
                _sc      = float(_rr["composite"])
                _penalty = float(_rr.get("climate_penalty", 0) or 0)
                _raw_sc  = float(_rr.get("raw_composite", _sc) or _sc)
                _sep     = "border-bottom:1px solid #1e1c0e;"

                # Mini score bar (80px track) — uses raw score for bar length
                _bar_fill = min(_raw_sc / 100 * 80, 80)
                _score_bar = (
                    f'<div style="width:80px;height:6px;background:#2a2410;border-radius:3px;overflow:hidden;">'
                    f'<div style="width:{_bar_fill:.1f}px;height:6px;background:#c8a040;border-radius:3px;"></div>'
                    f'</div>'
                )

                # Composite score cell — annotate penalty if present
                if _penalty > 0:
                    _score_cell = (
                        f'<div style="font-size:1.05rem;font-weight:700;color:#c8a040;">{_sc:.1f}</div>'
                        f'<div style="font-size:0.72rem;color:#ef5350;margin-top:2px;">−{_penalty:.1f} climate</div>'
                    )
                else:
                    _score_cell = f'<div style="font-size:1.05rem;font-weight:700;color:#c8a040;">{_sc:.1f}</div>'

                # Factor cells: value + mini underbar
                _fcells = ""
                for _, _fkey, _fcolor in _FACTOR_COLS:
                    _fv = float(_f.get(_fkey, 0) or 0)
                    _fw = min(_fv / 100 * 44, 44)
                    _fcells += (
                        f'<td style="padding:12px 10px;{_sep}text-align:center;">'
                        f'<div style="font-size:1rem;font-weight:700;color:{_fcolor};letter-spacing:0.02em;">{round(_fv)}</div>'
                        f'<div style="margin:4px auto 0;width:44px;height:3px;background:#2a2410;border-radius:2px;">'
                        f'<div style="width:{_fw:.1f}px;height:3px;background:{_fcolor};border-radius:2px;"></div></div>'
                        f'</td>'
                    )

                # Climate adjustment cell
                if _penalty > 0:
                    _clim_cell = (
                        f'<td style="padding:12px 10px;{_sep}text-align:center;">'
                        f'<span style="color:#ef5350;font-size:0.88rem;font-weight:700;">−{_penalty:.1f}</span>'
                        f'</td>'
                    )
                else:
                    _clim_cell = (
                        f'<td style="padding:12px 10px;{_sep}text-align:center;">'
                        f'<span style="color:#4a4530;font-size:0.88rem;">—</span>'
                        f'</td>'
                    )

                _rank_c = "#c8a040" if _ri < 3 else "#7a7050"
                _mfb_rows += (
                    f'<tr>'
                    f'<td style="padding:12px 10px;{_sep}text-align:center;color:{_rank_c};font-size:0.88rem;">{_rr["rank"]}</td>'
                    f'<td style="padding:12px 14px;{_sep}color:#c8b870;font-size:0.95rem;font-weight:600;white-space:nowrap;">{_rr["market"]}</td>'
                    f'<td style="padding:12px 14px;{_sep}vertical-align:middle;">{_score_bar}</td>'
                    f'<td style="padding:12px 10px;{_sep}text-align:right;">{_score_cell}</td>'
                    f'<td style="padding:12px 10px;{_sep}text-align:center;">{_grade_badge(_rr["grade"])}</td>'
                    f'{_fcells}'
                    f'{_clim_cell}'
                    f'</tr>'
                )

            _mfb_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:16px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">MARKET FACTOR BREAKDOWN &mdash; TOP 10</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Composite investment score by market &mdash; CoStar / CBRE Q1 2025</div>
  </div>
  <div style="font-size:0.78rem;color:#7a7050;font-style:italic;margin-bottom:18px;">
    Factor scores out of 100. Migration is the primary differentiator across markets.
    <span style="color:#ef5350;">Climate Adj.</span> = points deducted for high physical climate risk (state score ≥ 60).
  </div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead>{_mfb_hdr}</thead>
      <tbody>{_mfb_rows}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    Scores are composite index values. Not financial advice.
  </div>
</div>
"""
            st.markdown(_mfb_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — CAP RATE MONITOR
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_caprate:
        st.markdown("#### Are cap rates offering attractive spreads over the risk-free rate?")
        st.markdown(
            "Agent 14 pulls live commercial mortgage rates and 10-year treasury yields from FRED, "
            "then computes cap rate spreads by property type — signaling where valuations are "
            "attractive, fair, or compressed. Updates every 6 hours."
        )
        agent_last_updated("cap_rate")

        _cap_cache = read_cache("cap_rate")
        _cap_data  = _cap_cache.get("data") or {}

        if not _cap_data:
            st.info(" Cap Rate agent is fetching data — refresh in ~30 seconds.")
        else:
            _cap_national = _cap_data.get("national", {})
            _cap_mktcaps  = _cap_data.get("market_cap_rates", {})
            _cap_t10y     = _cap_data.get("treasury_10y")
            _cap_cmr      = _cap_data.get("commercial_mortgage_rate")
            _cap_spreads  = _cap_data.get("spreads", {})
            _cap_err      = _cap_data.get("error")

            _cap_ta = {"rising": "↑", "falling": "↓", "stable": "→"}
            _cap_tc = {"rising": "#ef5350", "falling": "#66bb6a", "stable": "#d4a843"}

            # ── Cap rate explainer ────────────────────────────────────────────
            with st.expander("How to read cap rates & spread signals"):
                st.markdown("""
**Cap Rate (Capitalization Rate)** = Net Operating Income ÷ Property Value. A higher cap rate means a cheaper price relative to income (better yield for buyers). A lower cap rate means a premium price (compressed yield).

**Treasury Spread** = Cap Rate − 10-Year Treasury Yield. This measures how much extra return CRE offers over the "risk-free" rate.

| Signal | Spread | What it means |
|--------|--------|---------------|
| **ATTRACTIVE** | > 2.5pp | Wide spread → buyers earn a meaningful premium over treasuries. Good time to acquire. |
| **FAIR** | 1.5–2.5pp | Normal historical range. Neither particularly cheap nor expensive. |
| **COMPRESSED** | < 1.5pp | Thin spread → CRE priced near treasury yields. Limited margin of safety; higher downside risk if rates rise. |

**Why it matters:** When the Fed raises rates, treasury yields rise. If cap rates don't rise in tandem, spreads compress and CRE valuations come under pressure.
                """)

            # ── Rate context ──────────────────────────────────────────────────
            _cap_rc1, _cap_rc2, _cap_rc3 = st.columns(3)
            _t10y_str = f"{_cap_t10y:.2f}%" if _cap_t10y else "Set FRED_API_KEY"
            _cmr_str  = f"{_cap_cmr:.2f}%" if _cap_cmr else "Set FRED_API_KEY"
            _cap_rc1.markdown(metric_card("10Y Treasury", _t10y_str, "DGS10 · FRED live"), unsafe_allow_html=True)
            _cap_rc2.markdown(metric_card("Comm. Mortgage Rate", _cmr_str, "RIFLPBCIANM · FRED live"), unsafe_allow_html=True)
            _cap_rc3.markdown(metric_card("Benchmarks", _cap_data.get("data_as_of", "Q1 2025"), "CoStar / CBRE"), unsafe_allow_html=True)
            if _cap_err and not _cap_t10y:
                st.info(f" {_cap_err}")

            # ── Focus banner ──────────────────────────────────────────────────
            _cap_pt = _active_pt()
            _pt_focus_banner(_cap_pt)
            _cap_focus_col = _PT_CAP_COL.get(_cap_pt) if _cap_pt else None

            # ── National cap rates ────────────────────────────────────────────
            section(" National Cap Rates by Property Type")
            _cap_ncols = st.columns(len(_cap_national))
            for _cap_col, (_ptype, _pd) in zip(_cap_ncols, _cap_national.items()):
                _sp_info    = _cap_spreads.get(_ptype, {})
                _sig        = _sp_info.get("signal", "")
                _sig_c      = {"attractive": "#66bb6a", "fair": "#d4a843", "compressed": "#ef5350"}.get(_sig, "#a09880")
                _is_focus   = (not _cap_focus_col) or (_ptype == _cap_focus_col)
                _card_html  = metric_card(_ptype, f"{_pd['rate']}%", f"{_cap_ta.get(_pd['trend'],'')} vs {_pd['prior_year']}% prior year")
                if _cap_focus_col and _is_focus:
                    _card_html = f'<div style="outline:2px solid #c8a040;border-radius:10px;">{_card_html}</div>'
                elif _cap_focus_col and not _is_focus:
                    _card_html = f'<div style="opacity:0.25;">{_card_html}</div>'
                _cap_col.markdown(_card_html, unsafe_allow_html=True)
                if _sig:
                    _cap_col.markdown(
                        f"<div style='text-align:center;font-size:0.77rem;color:{_sig_c};"
                        f"margin-top:-6px;margin-bottom:8px;font-weight:700;"
                        f"opacity:{"1" if _is_focus else "0.25"};'>{_sig.upper()}</div>",
                        unsafe_allow_html=True,
                    )

            # ── Spread table ──────────────────────────────────────────────────
            if _cap_spreads:
                section(" Treasury Spread Analysis")
                _sp_rows = []
                for _pt, _sp in _cap_spreads.items():
                    _sp_rows.append({
                        "Property Type": _pt,
                        "Cap Rate":      f"{_sp['cap_rate']:.1f}%",
                        "10Y Spread":    f"{_sp['treasury_spread']:+.2f}pp",
                        "Signal":        _sp["signal"].upper(),
                    })
                st.dataframe(pd.DataFrame(_sp_rows).set_index("Property Type"), use_container_width=True, height=220)
                st.caption("Spread > 2.5pp = Attractive · 1.5–2.5pp = Fair · < 1.5pp = Compressed vs. current 10Y Treasury.")

            # ── Market cap rate heatmap ───────────────────────────────────────
            section(" Market Cap Rate Heatmap (All Markets × Property Types)")
            if _cap_mktcaps:
                _hm_ptypes = ["Office", "Industrial", "Retail", "Multifamily"]

                def _cap_cell_color(v: float) -> str:
                    """Teal (low) → green → lime → gold → orange (high)."""
                    _stops = [
                        (3.5, (26,  122, 106)),
                        (4.5, (29,  140,  94)),
                        (5.0, (45,  150,  64)),
                        (5.5, (74,  160,  32)),
                        (6.0, (120, 168,  24)),
                        (6.5, (168, 168,  24)),
                        (7.0, (212, 160,  32)),
                        (7.5, (212, 120,  32)),
                        (8.5, (212,  72,  32)),
                    ]
                    if v <= _stops[0][0]:
                        return "rgb(%d,%d,%d)" % _stops[0][1]
                    if v >= _stops[-1][0]:
                        return "rgb(%d,%d,%d)" % _stops[-1][1]
                    for _i in range(len(_stops) - 1):
                        _lv, _lc = _stops[_i]
                        _hv, _hc = _stops[_i + 1]
                        if _lv <= v <= _hv:
                            _t = (v - _lv) / (_hv - _lv)
                            return "rgb(%d,%d,%d)" % (
                                int(_lc[0] + _t * (_hc[0] - _lc[0])),
                                int(_lc[1] + _t * (_hc[1] - _lc[1])),
                                int(_lc[2] + _t * (_hc[2] - _lc[2])),
                            )
                    return "rgb(150,150,150)"

                # Build header row — dim non-focus columns
                _hm_header_cells = ""
                for p in _hm_ptypes:
                    _is_fc = (not _cap_focus_col) or (p == _cap_focus_col)
                    _hm_header_cells += (
                        f'<th style="padding:10px 8px 14px;color:{"#c8a040" if _is_fc else "#5a5030"};'
                        f'font-size:0.78rem;font-weight:{"700" if _is_fc else "400"};'
                        f'letter-spacing:0.08em;text-align:center;opacity:{"1" if _is_fc else "0.3"};'
                        f'border-bottom:1px solid #2a2410;">{p.upper()}</th>'
                    )
                _hm_header = (
                    f'<tr><th style="padding:10px 8px 14px;border-bottom:1px solid #2a2410;"></th>'
                    f'{_hm_header_cells}</tr>'
                )

                # Build data rows — dim non-focus columns
                _hm_rows_html = ""
                for _mkt, _mdata in _cap_mktcaps.items():
                    _cells = f'<td style="padding:7px 12px 7px 0;text-align:right;color:#c8b890;font-size:0.88rem;white-space:nowrap;border-bottom:1px solid #1e1c0e;">{_mkt}</td>'
                    for _pt in _hm_ptypes:
                        _is_fc  = (not _cap_focus_col) or (_pt == _cap_focus_col)
                        _col_op = "1" if _is_fc else "0.22"
                        _val = _mdata.get(_pt)
                        if _val and _val > 0:
                            _bg  = _cap_cell_color(float(_val))
                            _cells += (
                                f'<td style="padding:7px 6px;border-bottom:1px solid #1e1c0e;opacity:{_col_op};">'
                                f'<div style="background:{_bg};border-radius:7px;padding:9px 0;'
                                f'text-align:center;color:#fff;font-size:0.92rem;font-weight:600;'
                                f'letter-spacing:0.04em;min-width:80px;">{_val:.1f}%</div></td>'
                            )
                        else:
                            _cells += (
                                f'<td style="padding:7px 6px;border-bottom:1px solid #1e1c0e;opacity:{_col_op};">'
                                f'<div style="text-align:center;color:#4a4530;font-size:0.92rem;">—</div></td>'
                            )
                    _hm_rows_html += f"<tr>{_cells}</tr>"

                _hm_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:20px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">MARKET CAP RATE HEATMAP</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">All markets × property types &mdash; CoStar / CBRE Q1 2025</div>
  </div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead>{_hm_header}</thead>
      <tbody>{_hm_rows_html}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    Teal/green = compressed yield (premium pricing) &nbsp;·&nbsp; Gold/orange = higher yield (cheaper entry)
    &nbsp;·&nbsp; Source: CoStar / CBRE Q1 2025 &nbsp;·&nbsp; Not financial advice.
  </div>
</div>
"""
                st.markdown(_hm_html, unsafe_allow_html=True)

            # ── Property-type analyst notes ───────────────────────────────────
            section(" Analyst Notes")
            for _ptype, _pd in _cap_national.items():
                _t_c      = _cap_tc.get(_pd["trend"], "#d4a843")
                _t_a      = _cap_ta.get(_pd["trend"], "")
                _is_focus = (not _cap_focus_col) or (_ptype == _cap_focus_col)
                _note_op  = "1" if _is_focus else "0.28"
                st.markdown(
                    f"<div style='background:#171309;border-left:3px solid {_t_c};"
                    f"padding:8px 16px;border-radius:4px;margin-bottom:8px;opacity:{_note_op};'>"
                    f"<b style='color:#e8dfc4;'>{_ptype}</b> "
                    f"<span style='color:{_t_c};font-size:0.85rem;'>{_t_a} {_pd['trend']} — {_pd['rate']}% cap rate</span>"
                    f" &nbsp;·&nbsp; <span style='color:#a09880;font-size:0.83rem;'>{_pd['note']}</span></div>",
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — RENT GROWTH TRACKER
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_rent:
        st.markdown("#### Where is rent growth accelerating — and where is it declining?")
        st.markdown(
            "Agent 15 tracks YoY rent growth across multifamily, industrial, office, and retail "
            "at national and market levels. Combines FRED CPI shelter indices with "
            "Zillow / CoStar Q1 2025 benchmarks. Updates every 6 hours."
        )
        agent_last_updated("rent_growth")

        _rg_cache = read_cache("rent_growth")
        _rg_data  = _rg_cache.get("data") or {}

        if not _rg_data:
            st.info(" Rent Growth agent is fetching data — refresh in ~30 seconds.")
        else:
            _rg_national = _rg_data.get("national", {})
            _rg_market   = _rg_data.get("market_rent_growth", {})
            _rg_top_mf   = _rg_data.get("top_markets_multifamily", [])
            _rg_top_ind  = _rg_data.get("top_markets_industrial", [])
            _rg_fred     = _rg_data.get("fred_rent_series", {})
            _rg_err      = _rg_data.get("error")

            _rg_ta = {"rising": "↑", "falling": "↓", "stable": "→"}
            _rg_tc = {"rising": "#66bb6a", "falling": "#ef5350", "stable": "#d4a843"}

            if _rg_err and not _rg_fred.get("cpi_rent"):
                st.info(f" {_rg_err}")

            with st.expander("How to read rent growth indicators"):
                st.markdown("""
**YoY Rent Growth %** measures how much asking or effective rents have changed versus the same period last year. Positive = landlords gaining pricing power; negative = tenants have leverage.

| Property Type | Healthy Range | Notes |
|--------------|--------------|-------|
| **Multifamily** | +2% to +5% | Above 5% = supply shortage; below 0% = oversupply (common in Sunbelt 2024-25) |
| **Industrial** | +5% to +12% | Driven by e-commerce and nearshoring; still elevated but normalizing from 2022 peaks |
| **Retail** | +1% to +3% | Limited new supply supporting positive growth; grocery-anchored outperforms |
| **Office** | Negative | Structural headwind from remote work; effective rents negative after concessions |

**PSF** = Per Square Foot (annual). Used for industrial and office leases.
**FRED CPI Rent** = Bureau of Labor Statistics measure of residential rent inflation — a leading indicator for multifamily rent trends.
                """)

            # ── Focus banner ──────────────────────────────────────────────────
            _rg_pt_focus  = _active_pt()
            _pt_focus_banner(_rg_pt_focus)
            _rg_focus_lbl = _PT_RG_LBL.get(_rg_pt_focus) if _rg_pt_focus else None
            _rg_focus_key = _PT_RG_KEY.get(_rg_pt_focus) if _rg_pt_focus else None

            # ── National overview ─────────────────────────────────────────────
            section(" National Rent Growth by Property Type (YoY %)")
            _rg_ncols = st.columns(len(_rg_national))
            for _rg_col, (_rg_pt, _rg_d) in zip(_rg_ncols, _rg_national.items()):
                _rg_yoy    = _rg_d["yoy_pct"]
                _rg_color  = "#66bb6a" if _rg_yoy > 1 else ("#ef5350" if _rg_yoy < 0 else "#d4a843")
                _is_focus  = (not _rg_pt_focus) or (_rg_pt == _rg_pt_focus)
                _card_html = metric_card(
                    _rg_pt,
                    f"<span style='color:{_rg_color}'>{_rg_yoy:+.1f}%</span>",
                    f"{_rg_ta.get(_rg_d['trend'],'')} vs {_rg_d['prior_year']:+.1f}% prior year",
                )
                if _rg_pt_focus and _is_focus:
                    _card_html = f'<div style="outline:2px solid #c8a040;border-radius:10px;">{_card_html}</div>'
                elif _rg_pt_focus and not _is_focus:
                    _card_html = f'<div style="opacity:0.25;">{_card_html}</div>'
                _rg_col.markdown(_card_html, unsafe_allow_html=True)

            # ── Top 5 — Multifamily ───────────────────────────────────────────
            section(" Top 5 Markets — Multifamily Rent Growth (YoY %)")
            if _rg_top_mf:
                _rg_mf_cols = st.columns(5)
                for _rg_c, _rg_e in zip(_rg_mf_cols, _rg_top_mf):
                    _rg_y = _rg_e["yoy_pct"]
                    _rg_c_ = "#66bb6a" if _rg_y > 1 else ("#ef5350" if _rg_y < 0 else "#d4a843")
                    _rg_c.markdown(metric_card(_rg_e["market"], f"<span style='color:{_rg_c_}'>{_rg_y:+.1f}%</span>", "multifamily YoY"), unsafe_allow_html=True)

            # ── Top 5 — Industrial ────────────────────────────────────────────
            section(" Top 5 Markets — Industrial Rent Growth (PSF YoY %)")
            if _rg_top_ind:
                _rg_ind_cols = st.columns(5)
                for _rg_c, _rg_e in zip(_rg_ind_cols, _rg_top_ind):
                    _rg_y = _rg_e["yoy_pct"]
                    _rg_c_ = "#66bb6a" if _rg_y > 3 else ("#d4a843" if _rg_y > 0 else "#ef5350")
                    _rg_c.markdown(metric_card(_rg_e["market"], f"<span style='color:{_rg_c_}'>{_rg_y:+.1f}%</span>", "industrial PSF YoY"), unsafe_allow_html=True)

            # ── All-market heatmap ────────────────────────────────────────────
            section(" Rent Growth Heatmap — All Markets")
            if _rg_market:
                _rg_mkts     = list(_rg_market.keys())
                _rg_ptypes   = ["multifamily", "industrial_psf", "office_psf", "retail_psf"]
                _rg_lbls     = ["Multifamily", "Industrial PSF", "Office PSF", "Retail PSF"]
                # Reorder: put focus type first when active
                if _rg_focus_key and _rg_focus_key in _rg_ptypes:
                    _fi = _rg_ptypes.index(_rg_focus_key)
                    _rg_ptypes = [_rg_ptypes[_fi]] + [p for i, p in enumerate(_rg_ptypes) if i != _fi]
                    _rg_lbls   = [_rg_lbls[_fi]]   + [l for i, l in enumerate(_rg_lbls)   if i != _fi]
                _rg_matrix = [[_rg_market[m].get(p, 0) for p in _rg_ptypes] for m in _rg_mkts]
                fig_rg = go.Figure(go.Heatmap(
                    z=_rg_matrix, x=_rg_lbls, y=_rg_mkts,
                    colorscale=[[0, "#7f1d1d"], [0.5, "#1a1208"], [1, "#1b4332"]],
                    zmid=0,
                    text=[[f"{v:+.1f}%" for v in row] for row in _rg_matrix],
                    texttemplate="%{text}",
                    showscale=True,
                    hovertemplate="<b>%{y}</b> · %{x}<br>YoY: %{z:+.1f}%<extra></extra>",
                ))
                fig_rg.update_layout(
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    margin=dict(l=160, r=40, t=20, b=40), height=580,
                    xaxis=dict(color="#8a7040"), yaxis=dict(color="#8a7040"),
                )
                st.plotly_chart(fig_rg, use_container_width=True)

            # ── FRED CPI shelter series ───────────────────────────────────────
            _rg_cpi = _rg_fred.get("cpi_rent", [])
            _rg_oer = _rg_fred.get("oer", [])
            if _rg_cpi or _rg_oer:
                section(" FRED CPI Shelter Indices (Live)")
                fig_cpi = go.Figure()
                if _rg_cpi:
                    fig_cpi.add_trace(go.Scatter(
                        x=[o["date"] for o in _rg_cpi], y=[o["value"] for o in _rg_cpi],
                        name="CPI Rent (CUSR0000SEHA)", line=dict(color=GOLD, width=2),
                    ))
                if _rg_oer:
                    fig_cpi.add_trace(go.Scatter(
                        x=[o["date"] for o in _rg_oer], y=[o["value"] for o in _rg_oer],
                        name="Owners' Equiv. Rent (CUSR0000SEHA2)", line=dict(color="#66bb6a", width=2),
                    ))
                fig_cpi.update_layout(
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    xaxis=dict(gridcolor="#2a2208", color="#8a7040"),
                    yaxis=dict(gridcolor="#2a2208", color="#8a7040", title="Index Level"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8b890")),
                    margin=dict(t=20, b=40), height=280,
                )
                st.plotly_chart(fig_cpi, use_container_width=True)
                st.caption("U.S. Bureau of Labor Statistics via FRED. Live when FRED_API_KEY is set.")

            # ── Analyst notes ─────────────────────────────────────────────────
            section(" Analyst Notes")
            for _rg_pt, _rg_d in _rg_national.items():
                _rg_t_c   = _rg_tc.get(_rg_d["trend"], "#d4a843")
                _rg_t_a   = _rg_ta.get(_rg_d["trend"], "")
                _is_focus = (not _rg_pt_focus) or (_rg_pt == _rg_pt_focus)
                _note_op  = "1" if _is_focus else "0.28"
                st.markdown(
                    f"<div style='background:#171309;border-left:3px solid {_rg_t_c};"
                    f"padding:8px 16px;border-radius:4px;margin-bottom:8px;opacity:{_note_op};'>"
                    f"<b style='color:#e8dfc4;'>{_rg_pt}</b> "
                    f"<span style='color:{_rg_t_c};font-size:0.85rem;'>{_rg_t_a} {_rg_d['trend']}</span>"
                    f" &nbsp;·&nbsp; <span style='color:#a09880;font-size:0.83rem;'>{_rg_d['note']}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Benchmarks: Zillow Research / CoStar Q1 2025. Not financial advice.")

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — OPPORTUNITY ZONES & TAX INCENTIVES
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_oz:
        st.markdown("#### Where can Opportunity Zone tax benefits enhance CRE returns?")
        st.markdown(
            "Agent 16 maps federal Opportunity Zone designations and state-level CRE tax incentive "
            "programs to market opportunities. Covers 15 major OZ metros and 15 state programs. "
            "Source: IRS Rev. Rul. 2018-29, HUD OZ designations, state economic development agencies."
        )
        agent_last_updated("opportunity_zone")

        _oz_cache = read_cache("opportunity_zone")
        _oz_data  = _oz_cache.get("data") or {}

        if not _oz_data:
            st.info(" Opportunity Zone agent is loading — refresh in ~30 seconds.")
        else:
            _oz_markets   = _oz_data.get("oz_markets", {})
            _oz_state_inc = _oz_data.get("state_incentives", {})
            _oz_fed_ben   = _oz_data.get("federal_benefits", [])
            _oz_ranked    = _oz_data.get("top_markets_by_score", [])

            with st.expander("How Opportunity Zones work"):
                st.markdown("""
**Opportunity Zones (OZs)** are census tracts designated by the IRS where investors can defer and reduce capital gains taxes by investing through a **Qualified Opportunity Fund (QOF)**.

**The three key benefits:**
1. **Deferral** — Reinvest capital gains into a QOF and defer the original tax until Dec 31, 2026 (or earlier sale)
2. **Step-Up** — Partial basis increase if held 5+ years (primarily applies to investments made before end of 2021)
3. **Permanent Exclusion** — Hold for 10+ years and pay **zero tax** on appreciation inside the QOF — the primary economic driver

**CRE application:** Buy or develop property inside an OZ through a QOF. The 10-year hold threshold aligns well with a value-add or development hold period. Industrial, multifamily, and mixed-use assets in high-score OZ markets offer the best risk-adjusted returns.

**Important:** Must invest via a QOF structure; the 90% asset test applies; consult a qualified tax attorney before proceeding.
                """)

            # ── Federal OZ benefit cards ───────────────────────────────────────
            section(" Federal Opportunity Zone Tax Benefits")
            _oz_fed_cols = st.columns(len(_oz_fed_ben))
            for _oz_fc, _oz_b in zip(_oz_fed_cols, _oz_fed_ben):
                _oz_fc.markdown(metric_card(_oz_b["benefit"], "", _oz_b["detail"]), unsafe_allow_html=True)

            # ── Top OZ markets bar chart ──────────────────────────────────────
            section(" OZ Markets Ranked by Opportunity Score")
            if _oz_ranked:
                _oz_names   = [r[0] for r in _oz_ranked[:12]]
                _oz_scores  = [r[1] for r in _oz_ranked[:12]]
                _oz_colors  = ["#66bb6a" if s >= 80 else ("#d4a843" if s >= 70 else "#ef5350") for s in _oz_scores]
                fig_oz = go.Figure(go.Bar(
                    x=_oz_scores, y=_oz_names,
                    orientation="h",
                    marker_color=_oz_colors,
                    text=[str(s) for s in _oz_scores],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>OZ Score: %{x}<extra></extra>",
                ))
                fig_oz.update_layout(
                    xaxis=dict(range=[0, 105], title="Opportunity Score", gridcolor="#2a2208", color="#8a7040"),
                    yaxis=dict(categoryorder="total ascending", color="#8a7040"),
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    margin=dict(l=180, r=60, t=20, b=40), height=420,
                )
                st.plotly_chart(fig_oz, use_container_width=True)

            # ── OZ market detail cards ────────────────────────────────────────
            section(" OZ Market Details")
            _oz_sorted = sorted(_oz_markets.items(), key=lambda x: x[1]["opportunity_score"], reverse=True)
            for _oz_i in range(0, min(len(_oz_sorted), 12), 3):
                _oz_row = st.columns(3)
                for _oz_j, _oz_col in enumerate(_oz_row):
                    if _oz_i + _oz_j >= len(_oz_sorted):
                        break
                    _oz_mkt, _oz_info = _oz_sorted[_oz_i + _oz_j]
                    _oz_sc   = _oz_info["opportunity_score"]
                    _oz_sc_c = "#66bb6a" if _oz_sc >= 80 else ("#d4a843" if _oz_sc >= 70 else "#ef5350")
                    _oz_zones = " · ".join(_oz_info["key_zones"][:2])
                    _oz_types = ", ".join(_oz_info["cre_types"])
                    _oz_col.markdown(f"""
                    <div style="background:#171309;border:1px solid #2a2208;border-top:2px solid {_oz_sc_c};
                                border-radius:8px;padding:16px 20px;margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="color:#e8dfc4;font-weight:700;font-size:0.9rem;">{_oz_mkt}</span>
                        <span style="color:{_oz_sc_c};font-weight:800;font-size:1.1rem;">{_oz_sc}</span>
                      </div>
                      <div style="color:#a09880;font-size:0.77rem;margin-bottom:4px;">{_oz_info['tract_count']} OZ census tracts</div>
                      <div style="color:#d4a843;font-size:0.77rem;margin-bottom:4px;">Zones: {_oz_zones}</div>
                      <div style="color:#7a9870;font-size:0.75rem;">{_oz_types}</div>
                    </div>""", unsafe_allow_html=True)

            # ── State incentive programs table ────────────────────────────────
            section(" State CRE Tax Incentive Programs")
            if _oz_state_inc:
                # CRE type chip styles
                _TI_CHIP = {
                    "Industrial":   ("border:1px solid #2bbfb0;background:#0d2420;color:#2bbfb0;"),
                    "Office":       ("border:1px solid #6080c0;background:#0d1828;color:#8aabdf;"),
                    "Multifamily":  ("border:1px solid #9070c0;background:#1a0d28;color:#b090d8;"),
                    "Mixed-Use":    ("border:1px solid #c8a040;background:#1a1408;color:#c8a040;"),
                    "Retail":       ("border:1px solid #60a850;background:#0d1a08;color:#80c868;"),
                    "Data Center":  ("border:1px solid #7090b8;background:#0d1218;color:#90b0d8;"),
                }
                _chip_css = "display:inline-block;padding:2px 9px;border-radius:5px;font-size:0.75rem;font-weight:600;margin:2px 3px 2px 0;white-space:nowrap;"

                # Collect all unique types for filter
                _all_types = sorted({t for v in _oz_state_inc.values() for t in v["cre_types"]})
                _filter_opts = ["All Types"] + _all_types

                # Filter widget
                st.markdown(
                    "<style>.stRadio > div{display:flex;flex-wrap:wrap;gap:8px;}"
                    ".stRadio label{background:#1e1c10;border:1px solid #3a3420;border-radius:8px;"
                    "padding:7px 18px;color:#c8a040;font-size:0.8rem;font-weight:700;"
                    "letter-spacing:0.07em;cursor:pointer;margin:0!important;}"
                    ".stRadio label:has(input:checked){background:#2a2410;border-color:#c8a040;}"
                    "</style>",
                    unsafe_allow_html=True,
                )
                _ti_filter = st.radio(
                    "Filter by property type",
                    _filter_opts,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="ti_filter",
                )

                # Filter data
                _ti_filtered = {
                    k: v for k, v in _oz_state_inc.items()
                    if _ti_filter == "All Types" or _ti_filter in v["cre_types"]
                }
                _ti_count = len(_ti_filtered)

                # Header
                _ti_sep = "border-bottom:1px solid #2a2410;"
                _ti_hcells = "".join(
                    f'<th style="padding:11px 14px 13px;color:#c8a040;font-size:0.76rem;font-weight:700;'
                    f'letter-spacing:0.09em;text-align:{al};{_ti_sep}">{h}</th>'
                    for h, al in [("STATE","center"),("PROGRAM","left"),("BENEFIT","left"),("CRE TYPES","left"),("CAP","left")]
                )

                # Rows
                _ti_rows_html = ""
                for _ti_abbr, _ti_si in _ti_filtered.items():
                    _ti_fallback = _TI_CHIP["Mixed-Use"]
                    _ti_type_chips = "".join(
                        f'<span style="{_chip_css}{_TI_CHIP.get(t, _ti_fallback)}">{t}</span>'
                        for t in _ti_si["cre_types"]
                    )
                    _row_sep = "border-bottom:1px solid #1e1c0e;"
                    _ti_rows_html += (
                        f'<tr>'
                        f'<td style="padding:14px 10px;{_row_sep}text-align:center;vertical-align:top;">'
                        f'<span style="display:inline-block;background:#2a2410;border:1px solid #3a3020;'
                        f'color:#c8a040;font-size:0.8rem;font-weight:700;padding:5px 10px;border-radius:5px;'
                        f'letter-spacing:0.06em;">{_ti_abbr}</span></td>'
                        f'<td style="padding:14px 14px;{_row_sep}vertical-align:top;color:#c8b870;'
                        f'font-size:0.92rem;font-weight:600;min-width:180px;max-width:220px;">{_ti_si["program"]}</td>'
                        f'<td style="padding:14px 14px;{_row_sep}vertical-align:top;color:#9a9070;'
                        f'font-size:0.86rem;min-width:180px;max-width:240px;line-height:1.5;">{_ti_si["benefit"]}</td>'
                        f'<td style="padding:14px 14px;{_row_sep}vertical-align:top;min-width:160px;">{_ti_type_chips}</td>'
                        f'<td style="padding:14px 14px;{_row_sep}vertical-align:top;color:#c8a040;'
                        f'font-size:0.86rem;font-family:monospace;white-space:nowrap;min-width:140px;">{_ti_si["cap"]}</td>'
                        f'</tr>'
                    )

                _ti_html = f"""
<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:20px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">STATE CRE TAX INCENTIVE PROGRAMS</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Federal &amp; state-level incentives by property type &mdash; IRS / State Policy 2024&ndash;25</div>
  </div>
  <div style="font-size:0.8rem;color:#7a7050;margin-bottom:16px;">Showing {_ti_count} program{"s" if _ti_count != 1 else ""}</div>
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead><tr>{_ti_hcells}</tr></thead>
      <tbody>{_ti_rows_html}</tbody>
    </table>
  </div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">
    Source: IRS, HUD Opportunity Zone designations, state economic development agencies.
    Not financial or legal advice. Consult a tax advisor.
  </div>
</div>
"""
                st.markdown(_ti_html, unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 8 — CLIMATE RISK
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_climate:
        from src.climate_risk_agent import (
            WEIGHTS as CR_WEIGHTS,
            WEIGHT_LABELS as CR_WEIGHT_LABELS,
            PROPERTY_RISK_CONTEXT,
            score_label as cr_score_label,
            label_color as cr_label_color,
        )

        _show_tab_header(
            "climate_risk",
            "Agent 14 tracks **climate hazard exposure** across US CRE markets — scoring flood, "
            "wildfire, extreme heat, hurricane/wind, and sea level rise risk at the state and metro level. "
            "Data from OpenFEMA, NIFC, and NOAA. Updates every 24 hours.",
            "climate_risk",
        )

        cr_cache = read_cache("climate_risk")
        cr_data  = cr_cache.get("data")

        if not cr_data:
            st.info(" Climate Risk agent is running for the first time — fetching FEMA and NIFC data. This may take 60–90 seconds. Refresh when ready.")
            st.stop()

        state_scores = cr_data.get("states", {})
        metro_scores = cr_data.get("metros", [])
        top_risk     = cr_data.get("top_risk_states", [])
        low_risk     = cr_data.get("lowest_risk_states", [])
        sources      = cr_data.get("data_sources", [])
        ts           = cr_data.get("timestamp", "")
        if ts:
            st.caption(f"Last updated: {ts[:16].replace('T', ' ')} UTC · Sources: {', '.join(sources[:2])}")

        # ── Personalized insight ─────────────────────────────────────────────
        pt, loc, focus_label = _focus_parts()
        if pt or loc:
            risk_context = PROPERTY_RISK_CONTEXT.get(pt, "all major climate hazards") if pt else "all major climate hazards"
            loc_state = st.session_state.user_intent.get("state")
            loc_data  = state_scores.get(loc_state, {}) if loc_state else {}
            focus_metro = next((m for m in metro_scores if loc and loc.lower() in m["metro"].lower()), None)
            if focus_metro:
                loc_score = focus_metro["composite_score"]
                loc_label = focus_metro["label"]
                loc_display = focus_metro["metro"]
            elif loc_data:
                loc_score = loc_data.get("composite_score", "N/A")
                loc_label = loc_data.get("label", "")
                loc_display = loc_state
            else:
                loc_score = None
                loc_label = ""
                loc_display = loc or "US"

            if loc_score is not None:
                color = cr_label_color(loc_label)
                st.markdown(
                    f'<div style="background:#1a1a12;border-left:3px solid {color};padding:10px 16px;'
                    f'border-radius:4px;margin-bottom:16px;">'
                    f'<span style="color:{color};font-weight:600;">{loc_display} — {loc_label} ({loc_score}/100)</span>'
                    f'<br><span style="font-size:0.85rem;color:#c8bfa0;">'
                    f'{pt or "CRE"} in this market is exposed to {risk_context}.</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Score legend ─────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Score Legend & Factor Weights")
        leg_cols = st.columns(4)
        for col, (label_name, color, rng) in zip(leg_cols, [
            ("Low",      "#4caf50", "0–25"),
            ("Moderate", "#ff9800", "26–50"),
            ("High",     "#f44336", "51–75"),
            ("Severe",   "#7b1fa2", "76–100"),
        ]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label_name}</div>'
                f'<div class="value" style="color:{color};font-size:1.4rem;">{rng}</div></div>',
                unsafe_allow_html=True,
            )

        wt_cols = st.columns(5)
        for col, (factor, weight) in zip(wt_cols, CR_WEIGHTS.items()):
            col.markdown(
                f'<div class="metric-card"><div class="label">{CR_WEIGHT_LABELS[factor]}</div>'
                f'<div class="value" style="font-size:1.3rem;">{int(weight*100)}%</div></div>',
                unsafe_allow_html=True,
            )

        # ── US Choropleth — State-level composite scores ──────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" US Climate Risk Map — State Composite Score (0–100)")

        if state_scores:
            map_states  = list(state_scores.keys())
            map_scores  = [state_scores[s]["composite_score"] for s in map_states]
            map_labels  = [state_scores[s]["label"] for s in map_states]
            map_hover   = [
                f"{s}: {state_scores[s]['composite_score']:.1f} ({state_scores[s]['label']})<br>"
                f"Flood: {state_scores[s]['factors']['flood']:.0f} | "
                f"Fire: {state_scores[s]['factors']['wildfire']:.0f} | "
                f"Heat: {state_scores[s]['factors']['heat']:.0f} | "
                f"Wind: {state_scores[s]['factors']['wind']:.0f} | "
                f"SLR: {state_scores[s]['factors']['sea_level']:.0f}"
                for s in map_states
            ]

            fig_map = go.Figure(go.Choropleth(
                locations=map_states,
                z=map_scores,
                locationmode="USA-states",
                colorscale=[
                    [0.0,  "#1b5e20"],
                    [0.25, "#66bb6a"],
                    [0.50, "#ff9800"],
                    [0.75, "#f44336"],
                    [1.0,  "#4a148c"],
                ],
                zmin=0, zmax=100,
                colorbar=dict(
                    title=dict(text="Risk Score", font=dict(color="#e8dfc4", size=11)),
                    tickfont=dict(color="#e8dfc4", size=10),
                    thickness=14,
                    bgcolor="#16160f",
                    bordercolor="#3a3a2a",
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0 Low", "25", "50", "75", "100 Severe"],
                ),
                hovertext=map_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            ))
            fig_map.update_layout(
                geo=dict(
                    scope="usa",
                    bgcolor="#0f0f0c",
                    showlakes=False,
                    lakecolor="#0f0f0c",
                    landcolor="#1a1a12",
                ),
                paper_bgcolor="#16160f",
                margin=dict(t=10, b=0, l=0, r=0),
                height=420,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Green = Low risk (0–25). Orange = Moderate (26–50). Red = High (51–75). Purple = Severe (76–100). "
                "Composite score weighted: Flood 25% | Wildfire 20% | Heat 20% | Wind 20% | Sea Level Rise 15%."
            )

        # ── Metro Risk Heatmap — factors × metros ─────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Metro Climate Risk Heatmap — Factor Breakdown")

        if metro_scores:
            top_metros = metro_scores[:20]  # top 20 by composite score
            hm_metros  = [m["metro"] for m in top_metros]
            hm_factors = list(CR_WEIGHTS.keys())
            hm_labels  = [CR_WEIGHT_LABELS[f] for f in hm_factors]
            hm_z       = [[m["factors"][f] for f in hm_factors] for m in top_metros]

            fig_hm = go.Figure(go.Heatmap(
                z=hm_z,
                x=hm_labels,
                y=hm_metros,
                colorscale=[
                    [0.0,  "#1b5e20"],
                    [0.3,  "#66bb6a"],
                    [0.55, "#fff9c4"],
                    [0.75, "#f44336"],
                    [1.0,  "#4a148c"],
                ],
                zmin=0, zmax=100,
                text=[[f"{v:.0f}" for v in row] for row in hm_z],
                texttemplate="%{text}",
                textfont=dict(size=11, color="#0f0f0c"),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}/100<extra></extra>",
                colorbar=dict(
                    title=dict(text="Score", font=dict(color="#e8dfc4", size=11)),
                    tickfont=dict(color="#e8dfc4", size=10),
                    thickness=14,
                    bgcolor="#16160f",
                    bordercolor="#3a3a2a",
                ),
            ))
            fig_hm.update_layout(
                plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                margin=dict(t=10, b=20, l=160, r=20),
                height=600,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
                xaxis=dict(side="top", tickfont=dict(color="#e8dfc4", size=12)),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=11)),
            )
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
            st.caption("Top 20 highest-risk metros. Each cell = factor score 0–100. Darker purple = greater hazard exposure.")

        # ── Top Risk / Safest Markets side by side ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col_risk, col_safe = st.columns(2)

        with col_risk:
            section(" 10 Highest-Risk States")
            if top_risk:
                tr_df = pd.DataFrame(top_risk)
                fig_tr = go.Figure(go.Bar(
                    x=tr_df["score"],
                    y=tr_df["state"],
                    orientation="h",
                    marker=dict(
                        color=tr_df["score"],
                        colorscale=[
                            [0.0, "#ff9800"], [0.5, "#f44336"], [1.0, "#4a148c"]
                        ],
                        cmin=40, cmax=100,
                    ),
                    text=[f"{s['score']:.0f} — {s['label']}" for s in top_risk],
                    textposition="inside",
                    textfont=dict(color="#fff", size=11),
                    hovertemplate="<b>%{y}</b>: %{x:.1f}/100<extra></extra>",
                ))
                fig_tr.update_layout(
                    plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                    margin=dict(t=10, b=10, l=40, r=10),
                    height=340,
                    xaxis=dict(range=[0, 100], tickfont=dict(color="#e8dfc4", size=10), gridcolor="#2a2a1a"),
                    yaxis=dict(tickfont=dict(color="#e8dfc4", size=11), autorange="reversed"),
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
                )
                st.plotly_chart(fig_tr, use_container_width=True, config={"displayModeBar": False})

        with col_safe:
            section(" 10 Lowest-Risk States")
            if low_risk:
                lr_df = pd.DataFrame(low_risk)
                fig_lr = go.Figure(go.Bar(
                    x=lr_df["score"],
                    y=lr_df["state"],
                    orientation="h",
                    marker=dict(
                        color=lr_df["score"],
                        colorscale=[[0.0, "#1b5e20"], [0.5, "#66bb6a"], [1.0, "#fff9c4"]],
                        cmin=0, cmax=40,
                    ),
                    text=[f"{s['score']:.0f} — {s['label']}" for s in low_risk],
                    textposition="inside",
                    textfont=dict(color="#0f0f0c", size=11),
                    hovertemplate="<b>%{y}</b>: %{x:.1f}/100<extra></extra>",
                ))
                fig_lr.update_layout(
                    plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                    margin=dict(t=10, b=10, l=40, r=10),
                    height=340,
                    xaxis=dict(range=[0, 100], tickfont=dict(color="#e8dfc4", size=10), gridcolor="#2a2a1a"),
                    yaxis=dict(tickfont=dict(color="#e8dfc4", size=11), autorange="reversed"),
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
                )
                st.plotly_chart(fig_lr, use_container_width=True, config={"displayModeBar": False})

        # ── Metro Detail Table ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" All Metro Markets — Full Risk Breakdown")

        if metro_scores:
            tbl_rows = []
            for m in metro_scores:
                f = m["factors"]
                tbl_rows.append({
                    "Metro":          m["metro"],
                    "State":          m["state"],
                    "Score":          m["composite_score"],
                    "Risk Level":     m["label"],
                    "Flood":          f["flood"],
                    "Wildfire":       f["wildfire"],
                    "Heat":           f["heat"],
                    "Wind":           f["wind"],
                    "Sea Level Rise": f["sea_level"],
                })
            tbl_df = pd.DataFrame(tbl_rows)

            def _style_score(val):
                if val >= 76:   return "color:#ce93d8;font-weight:600"
                if val >= 51:   return "color:#ef5350;font-weight:600"
                if val >= 26:   return "color:#ffa726;font-weight:600"
                return "color:#66bb6a;font-weight:600"

            st.dataframe(
                tbl_df.style.map(_style_score, subset=["Score", "Flood", "Wildfire", "Heat", "Wind", "Sea Level Rise"]),
                use_container_width=True,
                hide_index=True,
                height=500,
            )
            st.caption("Scores 0–100. Green = Low | Orange = Moderate | Red = High | Purple = Severe.")

        # ── Trend Chart — disaster events per year for focused market ─────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Disaster Event Trend — Annual FEMA Declarations")

        trend_data = None
        trend_title = "US Overall"

        if pt or loc:
            # Try to find focused metro or state trend
            if loc:
                focus_m = next((m for m in metro_scores if loc.lower() in m["metro"].lower()), None)
                if focus_m:
                    trend_data  = focus_m["trend"]
                    trend_title = focus_m["metro"]
                elif loc_state and loc_state in state_scores:
                    trend_data  = state_scores[loc_state]["trend"]
                    trend_title = loc_state

        if not trend_data and state_scores:
            # Default: show the highest-risk state trend
            top_state = top_risk[0]["state"] if top_risk else list(state_scores.keys())[0]
            trend_data  = state_scores[top_state]["trend"]
            trend_title = f"{top_state} (highest risk)"

        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            fig_trend = go.Figure(go.Bar(
                x=trend_df["year"],
                y=trend_df["events"],
                marker=dict(
                    color=trend_df["events"],
                    colorscale=[[0.0, "#388e3c"], [0.5, "#f57c00"], [1.0, "#c62828"]],
                ),
                hovertemplate="<b>%{x}</b>: %{y} FEMA disaster declarations<extra></extra>",
            ))
            fig_trend.update_layout(
                title=dict(
                    text=f"FEMA Disaster Declarations — {trend_title}",
                    font=dict(color="#e8dfc4", size=13),
                ),
                plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                margin=dict(t=40, b=30, l=50, r=20),
                height=280,
                xaxis=dict(tickfont=dict(color="#e8dfc4", size=11), gridcolor="#2a2a1a"),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=11), gridcolor="#2a2a1a", title="Declarations"),
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
            )
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Annual count of FEMA disaster declarations (flood, hurricane, wildfire, severe storm) for the selected market. "
                "Rising bars indicate worsening climate event frequency."
            )


with main_tab_energy:
    tab_energy, tab_esg = st.tabs([
        "Energy & Construction Costs",
        "Sustainability",
    ])


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — ENERGY & CONSTRUCTION COSTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_energy:
        st.markdown("#### How are energy and material costs affecting CRE construction?")
        st.markdown(
            "Agent 6 tracks **oil, natural gas, copper, and steel** prices to derive a "
            "**Construction Cost Signal** that indicates whether building costs are rising or easing. Updates every 6 hours."
        )
        agent_last_updated("energy")

        cache_e = read_cache("energy_data")
        if cache_e["data"] is None:
            st.warning(" Energy agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_e = cache_age_label("energy_data")
        st.caption(f" Last updated: {age_e} · Auto-refreshes in background")

        edata = cache_e["data"]
        commodities = edata.get("commodities", [])
        cost_signal = edata.get("construction_cost_signal", "UNKNOWN")
        avg_momentum = edata.get("avg_momentum_pct", 0)

        # ── KPI strip ──────────────────────────────────────────────────────────
        signal_color = {"HIGH": "HIGH", "MODERATE": "MOD", "LOW": "LOW"}.get(cost_signal, "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Construction Cost Signal", f"{signal_color} {cost_signal}",
                                 "Based on commodity momentum"), unsafe_allow_html=True)
        c2.markdown(metric_card("Avg Momentum", f"{avg_momentum:+.1f}%",
                                 "vs 60-day SMA"), unsafe_allow_html=True)
        c3.markdown(metric_card("Commodities Tracked", str(len(commodities)),
                                 "Oil, Gas, Copper, Steel, Energy"), unsafe_allow_html=True)
        c4.markdown(metric_card("Trading Days", str(edata.get("trading_days_analysed", 0)),
                                 "6-month lookback"), unsafe_allow_html=True)

        # ── Commodity Price Table ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Commodity Prices vs 60-Day Moving Average")

        if commodities:
            comm_df = pd.DataFrame(commodities)

            # Bar chart — % above/below SMA
            colors_comm = [GOLD if p >= 0 else "#c62828" for p in comm_df["pct_above_sma"]]
            fig_comm = go.Figure(go.Bar(
                x=comm_df["label"], y=comm_df["pct_above_sma"],
                marker_color=colors_comm,
                text=comm_df["pct_above_sma"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                customdata=comm_df[["latest_price", "sma_60"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "SMA-60: $%{customdata[1]:.2f}<br>"
                    "vs SMA: %{y:+.1f}%<extra></extra>"
                ),
            ))
            fig_comm.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                yaxis_title="% Above/Below SMA-60",
                yaxis=dict(gridcolor="#2a2208", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_comm, use_container_width=True)
            st.caption(
                "Each bar shows the commodity's latest price versus its 60-day moving average. "
                "Bars above the baseline indicate costs are elevated — signaling higher construction costs for developers. "
                "Copper and steel are the most direct inputs for building; oil and natural gas drive operating expenses and transport."
            )

            # Detail table
            section(" Commodity Detail")
            disp_comm = comm_df[["label", "latest_price", "sma_60", "pct_above_sma"]].copy()
            disp_comm.columns = ["Commodity", "Latest Price", "SMA-60", "% vs SMA"]
            disp_comm["Latest Price"] = disp_comm["Latest Price"].apply(lambda x: f"${x:.2f}")
            disp_comm["SMA-60"] = disp_comm["SMA-60"].apply(lambda x: f"${x:.2f}")
            disp_comm["% vs SMA"] = disp_comm["% vs SMA"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(disp_comm, use_container_width=True, hide_index=True)

        st.caption(
            "Data sourced from Yahoo Finance (USO, UNG, XLE, CPER, SLX). "
            "Construction Cost Signal: HIGH (>+5%), MODERATE (±5%), LOW (<−5%) based on avg momentum vs 60-day SMA."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Construction Cost Signal** = Average % deviation of key commodities from their 60-day Simple Moving Average (SMA).

- **HIGH** = Average deviation > +10% above SMA (commodity prices surging — expect higher construction bids)
- **MODERATE** = Average deviation within +/-10% of SMA (costs stable)
- **LOW** = Average deviation > 10% below SMA (commodity prices falling — favorable for new development)

**Commodities Tracked:**
- **Crude Oil (USO):** Drives transportation and equipment fuel costs
- **Natural Gas (UNG):** Heating/cooling during construction, operating cost baseline
- **Copper (CPER):** Wiring, plumbing, HVAC — direct construction input
- **Steel (SLX):** Structural framing, rebar — largest material cost component

**PnL Impact:** A 20% rise in steel and copper prices can add 5-8% to total hard construction costs on a typical CRE project.

**Data Source:** Yahoo Finance commodity ETFs, updated every 6 hours via Agent 7.
""")


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — SUSTAINABILITY & ESG
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_esg:
        st.markdown("#### Is green capital flowing into real estate?")
        st.markdown(
            "Agent 7 monitors **clean-energy ETFs** and **green REITs** to gauge ESG momentum "
            "relative to the broad market (SPY). Updates every 6 hours."
        )
        agent_last_updated("sustainability")

        cache_s = read_cache("sustainability_data")
        if cache_s["data"] is None:
            st.warning(" Sustainability agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_s = cache_age_label("sustainability_data")
        st.caption(f" Last updated: {age_s} · Auto-refreshes in background")

        sdata = cache_s["data"]
        clean_energy = sdata.get("clean_energy", [])
        green_reits = sdata.get("green_reits", [])
        esg_signal = sdata.get("esg_momentum_signal", "UNKNOWN")
        bench_ret = sdata.get("benchmark_return_pct") or 0
        avg_clean_ret = sdata.get("avg_clean_energy_return_pct") or 0

        # ── KPI strip ──────────────────────────────────────────────────────────
        esg_icon = ""
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("ESG Momentum Signal", f"{esg_icon} {esg_signal}",
                                 "Clean energy vs SPY"), unsafe_allow_html=True)
        c2.markdown(metric_card("Clean Energy Avg Return", f"{avg_clean_ret:+.1f}%",
                                 "6-month trailing"), unsafe_allow_html=True)
        c3.markdown(metric_card("SPY Benchmark Return", f"{bench_ret:+.1f}%",
                                 "6-month trailing"), unsafe_allow_html=True)
        spread = (avg_clean_ret or 0) - (bench_ret or 0)
        c4.markdown(metric_card("Spread vs Market", f"{spread:+.1f} pp",
                                 "Positive = outperforming"), unsafe_allow_html=True)

        # ── Clean Energy ETFs ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Clean Energy ETF Performance (ICLN, TAN, QCLN)")

        if clean_energy:
            ce_df = pd.DataFrame(clean_energy)
            fig_ce = go.Figure()
            colors_ce = [GOLD if r >= 0 else "#c62828" for r in ce_df["period_return_pct"]]
            fig_ce.add_trace(go.Bar(
                x=ce_df["label"], y=ce_df["period_return_pct"],
                marker_color=colors_ce,
                text=ce_df["period_return_pct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                name="6mo Return",
                customdata=ce_df[["latest_price", "sma_60", "pct_above_sma"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "6mo Return: %{y:+.1f}%<br>"
                    "vs SMA-60: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ))
            # Add SPY benchmark line
            fig_ce.add_hline(y=bench_ret, line_dash="dash", line_color=BLACK,
                             annotation_text=f"SPY: {bench_ret:+.1f}%",
                             annotation_position="top right")
            fig_ce.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#2a2208", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_ce, use_container_width=True)
            st.caption(
                "Tracks ICLN (global clean energy), TAN (solar), and QCLN (clean tech) ETFs. "
                "Sustained outperformance signals growing institutional capital flows into green energy — "
                "a leading indicator of demand for solar-ready industrial space, EV charging infrastructure, and LEED-certified buildings."
            )

        # ── Green REITs ────────────────────────────────────────────────────────
        section(" Green REIT Performance (PLD, EQIX, ARE)")

        if green_reits:
            gr_df = pd.DataFrame(green_reits)
            colors_gr = [GOLD if r >= 0 else "#c62828" for r in gr_df["period_return_pct"]]
            fig_gr = go.Figure(go.Bar(
                x=gr_df["label"], y=gr_df["period_return_pct"],
                marker_color=colors_gr,
                text=gr_df["period_return_pct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                customdata=gr_df[["latest_price", "sma_60", "pct_above_sma"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "6mo Return: %{y:+.1f}%<br>"
                    "vs SMA-60: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ))
            fig_gr.add_hline(y=bench_ret, line_dash="dash", line_color=BLACK,
                             annotation_text=f"SPY: {bench_ret:+.1f}%",
                             annotation_position="top right")
            fig_gr.update_layout(
                plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#2a2208", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_gr, use_container_width=True)
            st.caption(
                "Prologis (PLD) is the world's largest industrial REIT with a strong LEED portfolio. "
                "Equinix (EQIX) powers its data centers with 90%+ renewable energy. "
                "Alexandria (ARE) focuses on carbon-neutral life science campuses. "
                "Outperformance vs. SPY indicates ESG-focused capital allocators are actively rotating into these names."
            )

        # ── Combined Detail Table ──────────────────────────────────────────────
        section(" Full Detail — Clean Energy & Green REITs")
        all_esg = clean_energy + green_reits
        if all_esg:
            esg_df = pd.DataFrame(all_esg)
            disp_esg = esg_df[["label", "latest_price", "period_return_pct", "sma_60", "pct_above_sma"]].copy()
            disp_esg.columns = ["Security", "Price", "6mo Return", "SMA-60", "% vs SMA"]
            disp_esg["Price"] = disp_esg["Price"].apply(lambda x: f"${x:.2f}")
            disp_esg["6mo Return"] = disp_esg["6mo Return"].apply(lambda x: f"{x:+.1f}%")
            disp_esg["SMA-60"] = disp_esg["SMA-60"].apply(lambda x: f"${x:.2f}")
            disp_esg["% vs SMA"] = disp_esg["% vs SMA"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(disp_esg, use_container_width=True, hide_index=True)

        st.caption(
            "Data sourced from Yahoo Finance. ESG Momentum Signal: STRONG (clean energy outperforms SPY by >2pp), "
            "NEUTRAL (±2pp), WEAK (underperforms by >2pp). Green REITs: Prologis (LEED), Equinix (renewables), Alexandria (carbon-neutral)."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**ESG Momentum Signal** = Relative performance of clean energy ETFs vs. S&P 500 (SPY) over 6 months.

- **STRONG** = Clean energy basket outperforms SPY by > 2 percentage points
- **NEUTRAL** = Performance within +/- 2 percentage points of SPY
- **WEAK** = Clean energy basket underperforms SPY by > 2 percentage points

**Clean Energy Basket:** ICLN (global clean energy), TAN (solar), QCLN (clean tech) — equal-weighted average return.

**Green REITs:**
- **Prologis (PLD):** World's largest industrial REIT, extensive LEED-certified portfolio
- **Equinix (EQIX):** Data center REIT, 90%+ renewable energy usage
- **Alexandria (ARE):** Life science REIT, carbon-neutral campus commitments

**Why It Matters:** Sustained ESG outperformance signals institutional capital rotation into green assets — a leading indicator for demand in solar-ready industrial, EV infrastructure, and LEED-certified buildings.

**Data Source:** Yahoo Finance, updated every 6 hours via Agent 8.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TAB — MACRO ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
with main_tab_macro:
    tab_rates, tab_labor, tab_gdp, tab_inflation, tab_credit, tab_distressed = st.tabs([
        "Rate Environment",
        "Labor Market & Tenant Demand",
        "GDP & Economic Growth",
        "Inflation",
        "Credit & Capital Markets",
        "CMBS & Distressed",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — RATE ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_rates:
        _show_tab_header(
            "rates",
            "Agent 6 pulls **live rate data from FRED**, classifies the rate environment, "
            "computes dynamic cap rate adjustments by property type, and scores REIT refinancing risk. "
            "Updates every hour. Requires `FRED_API_KEY` in `.env`.",
            "rates",
        )

        cache_r = read_cache("rates")
        rdata   = cache_r.get("data") or {}

        if not rdata or rdata.get("error"):
            err = rdata.get("error") if rdata else None
            if err:
                st.warning(f" {err}")
            else:
                st.info("Rate data is fetched every hour. Check that `FRED_API_KEY` is set in `.env`.")
            st.stop()

        rates       = rdata.get("rates", {})
        env         = rdata.get("environment", {})
        cap_adj     = rdata.get("cap_rate_adjustments", [])
        debt_risk   = rdata.get("reit_debt_risk", [])
        yc          = rdata.get("yield_curve", {})
        current_10y = rdata.get("current_10y") or 0.0
        baseline    = rdata.get("baseline_10y") or 4.0
        cached_at   = rdata.get("cached_at", "")

        # ── Signal Banner ───────────────────────────────────────────────────────
        signal    = env.get("signal", "CAUTIOUS")
        _re_score = env.get("score", 50)
        _re_sum   = env.get("summary", "")
        _re_conf  = env.get("confidence", "High")
        st.markdown(gauge_card(
            title       = "RATE ENVIRONMENT",
            label       = signal,
            score       = _re_score,
            summary     = _re_sum,
            agent_num   = "A6  Agent 6",
            age_label   = cache_age_label("rates"),
            confidence  = _re_conf,
            low_good    = True,
            scale_labels= ("BEARISH", "25", "CAUTIOUS", "75", "BULLISH"),
        ), unsafe_allow_html=True)
        with st.expander("How to read this indicator"):
            st.markdown("""
**What it measures:** The overall interest rate climate for CRE borrowing and valuation — combining the 10-year Treasury yield, Fed Funds rate, yield curve shape, and mortgage spreads.

| Signal | Score | What it means for CRE |
|--------|-------|----------------------|
| **BULLISH** | 75–100 | Low/falling rates → lower cap rate requirements, rising valuations, cheap debt. Best time to acquire or refinance. |
| **CAUTIOUS** | 25–74 | Rates are elevated or uncertain. Underwrite conservatively; favor shorter loan terms. |
| **BEARISH** | 0–24 | High rates compress valuations, choke deal flow, and increase refinancing risk. Defensive positioning recommended. |

**Key inputs:** 10Y Treasury (DGS10), Fed Funds Rate, 2Y Treasury, yield curve spread, IG corporate spread — all pulled live from FRED.
            """)

        # ── Key Rate Cards ──────────────────────────────────────────────────────
        section(" Current Rates")
        display_rates = [
            "10Y Treasury", "2Y Treasury", "Fed Funds Rate",
            "SOFR", "30Y Mortgage", "Prime Rate", "IG Corp Spread",
        ]
        card_cols = st.columns(len(display_rates))
        for col, name in zip(card_cols, display_rates):
            r = rates.get(name)
            if not r:
                col.markdown(metric_card(name, "N/A", "No data"), unsafe_allow_html=True)
                continue
            curr  = r["current"]
            d1w   = r.get("delta_1w")
            unit  = r["unit"]
            val_s = f"{curr:.2f}{unit}" if unit == "%" else f"{curr:.0f}{unit}"
            delta_s = ""
            if d1w is not None:
                arrow = "▲" if d1w > 0 else ("▼" if d1w < 0 else "→")
                color = "#b71c1c" if d1w > 0 else ("#1b5e20" if d1w < 0 else "#555")
                delta_s = f"<span style='color:{color};font-size:0.78rem;'>{arrow} {abs(d1w):.2f}{unit} 1W</span>"
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{name}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_s}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Yield Curve + Rate Comparison Table ─────────────────────────────────
        col_yc, col_tbl = st.columns([3, 2])
        with col_yc:
            section(" Yield Curve (Current Shape)")
            if yc:
                tenor_order = ["3M", "2Y", "5Y", "10Y", "30Y"]
                tenors  = [t for t in tenor_order if t in yc]
                values  = [yc[t] for t in tenors]
                inverted = yc.get("2Y", 0) > yc.get("10Y", 0)
                line_clr = "#b71c1c" if inverted else "#1b5e20"
                fig_yc = go.Figure()
                fig_yc.add_trace(go.Scatter(
                    x=tenors, y=values,
                    mode="lines+markers+text",
                    line=dict(color=line_clr, width=3),
                    marker=dict(size=10, color=line_clr),
                    text=[f"{v:.2f}%" for v in values],
                    textposition="top center",
                    textfont=dict(size=11, color="#c8b890"),
                    hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
                ))
                fig_yc.add_hline(y=values[0] if values else 0, line_dash="dot",
                                  line_color="#aaa", line_width=1)
                fig_yc.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Yield (%)", ticksuffix="%",
                               gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                               title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    margin=dict(t=30, b=30, l=60, r=20), height=300,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    annotations=[dict(
                        text=" INVERTED" if inverted else "Normal slope",
                        x=0.5, y=1.08, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12,
                        color="#b71c1c" if inverted else "#1b5e20", family="Source Sans Pro"),
                    )],
                )
                st.plotly_chart(fig_yc, use_container_width=True)
                st.caption(
                    "A normal (upward-sloping) yield curve signals healthy economic expectations. "
                    "An inverted curve — where short-term rates exceed long-term — historically precedes recessions "
                    "and tightens CRE lending conditions as banks compress their net interest margins."
                )
            else:
                st.info("Yield curve data unavailable.")

        with col_tbl:
            section(" Rate Change Summary")
            tbl_rows = []
            for name in display_rates:
                r = rates.get(name)
                if not r:
                    continue
                unit = r["unit"]
                def _fmt_rate(v):
                    if v is None: return "—"
                    return f"{v:+.2f}{unit}" if unit == "%" else f"{v:+.0f}{unit}"
                tbl_rows.append({
                    "Rate":   name,
                    "Now":    f"{r['current']:.2f}{unit}" if unit == "%" else f"{r['current']:.0f}{unit}",
                    "1W Δ":   _fmt_rate(r.get("delta_1w")),
                    "1M Δ":   _fmt_rate(r.get("delta_1m")),
                    "1Y Δ":   _fmt_rate(r.get("delta_1y")),
                })
            if tbl_rows:
                tbl_df = pd.DataFrame(tbl_rows)
                st.dataframe(tbl_df, use_container_width=True, hide_index=True, height=280)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Rate Trends (12 months) ──────────────────────────────────────────────
        section(" Rate Trends — Past 12 Months")
        trend_series = ["10Y Treasury", "2Y Treasury", "Fed Funds Rate", "SOFR"]
        trend_colors = ["#1565c0", "#e65100", "#1b5e20", "#6a1b9a"]
        fig_tr = go.Figure()
        for sname, clr in zip(trend_series, trend_colors):
            r = rates.get(sname)
            if not r or not r.get("series"):
                continue
            series_pts = r["series"]
            cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            pts = [o for o in series_pts if o["date"] >= cutoff]
            if not pts:
                continue
            dates  = [o["date"] for o in pts]
            values = [o["value"] for o in pts]
            fig_tr.add_trace(go.Scatter(
                x=dates, y=values, name=sname,
                mode="lines", line=dict(color=clr, width=2),
                hovertemplate=f"{sname}: %{{y:.3f}}%<br>%{{x}}<extra></extra>",
            ))
        t10_s = rates.get("10Y Treasury", {}).get("series", [])
        t2_s  = rates.get("2Y Treasury",  {}).get("series", [])
        if t10_s and t2_s:
            t10_d = {o["date"]: o["value"] for o in t10_s}
            t2_d  = {o["date"]: o["value"] for o in t2_s}
            cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            inv_dates = sorted(d for d in t10_d if d >= cutoff and d in t2_d and t2_d[d] > t10_d[d])
            if inv_dates:
                fig_tr.add_vrect(
                    x0=inv_dates[0], x1=inv_dates[-1],
                    fillcolor="rgba(183,28,28,0.07)", line_width=0,
                    annotation_text="Inverted", annotation_position="top left",
                    annotation_font=dict(color="#b71c1c", size=10),
                )
        fig_tr.update_layout(
            paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
            yaxis=dict(title="Rate (%)", ticksuffix="%", gridcolor="#2a2208",
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            xaxis=dict(tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#c8b890", size=11)),
            margin=dict(t=40, b=40), height=380,
            font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_tr, use_container_width=True)
        st.caption(
            "Tracks the 10-year Treasury, 2-year Treasury, and Fed Funds rate over the past 12 months. "
            "The spread between the 10Y and 2Y (the yield curve slope) is a key leading indicator — "
            "a narrowing or inverted spread signals reduced appetite for long-duration CRE debt."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cap Rate Adjustment Impact ───────────────────────────────────────────
        section(f" Cap Rate Adjustment — Current 10Y ({current_10y:.2f}%) vs. {baseline:.1f}% Baseline")
        if cap_adj:
            adj_df = pd.DataFrame(cap_adj)
            col_bars, col_impact = st.columns([3, 2])
            with col_bars:
                pt_labels = [p.split("/")[0].strip() for p in adj_df["Property Type"]]
                base_caps = adj_df["Baseline Cap Rate"].tolist()
                adj_caps  = adj_df["Adjusted Cap Rate"].tolist()
                adj_bps   = adj_df["Rate Adjustment bps"].tolist()
                bar_colors = ["#b71c1c" if v > 0 else "#1b5e20" for v in adj_bps]

                fig_cap = go.Figure()
                fig_cap.add_trace(go.Bar(
                    name="Baseline Cap Rate",
                    x=pt_labels, y=base_caps,
                    marker_color="#d4a843",
                    text=[f"{v:.2f}%" for v in base_caps],
                    textposition="inside", textfont=dict(color="#c8b890", size=10),
                ))
                fig_cap.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate",
                    x=pt_labels, y=adj_caps,
                    marker_color=bar_colors,
                    opacity=0.85,
                    text=[f"{v:.2f}%\n({'+' if d>0 else ''}{d:.0f}bps)" for v, d in zip(adj_caps, adj_bps)],
                    textposition="outside", textfont=dict(color="#c8b890", size=9),
                ))
                fig_cap.update_layout(
                    barmode="group",
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickangle=-20, tickfont=dict(color="#c8b890", size=9)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#c8b890")),
                    margin=dict(t=40, b=60), height=360,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_cap, use_container_width=True)
                st.caption(
                    f"Adjustment = (10Y Treasury − {baseline:.1f}% baseline) × property-type interest rate beta. "
                    "Red bars mean cap rates have expanded from the baseline — asset values have declined for the same NOI. "
                    "Green bars mean cap rate compression — favorable for existing owners but tougher for new buyers on yield."
                )

            with col_impact:
                st.markdown("**Profit Margin Impact by Property Type**")
                disp_adj = adj_df[["Property Type", "Baseline Cap Rate", "Adjusted Cap Rate",
                                    "Rate Adjustment bps", "Static Margin %", "Adj Margin %", "Margin Delta bps"]].copy()
                disp_adj["Property Type"] = disp_adj["Property Type"].str.split("/").str[0].str.strip()

                def _colour_delta(val):
                    try:
                        v = float(val)
                        if v < 0: return "color: #b71c1c"
                        if v > 0: return "color: #1b5e20"
                    except: pass
                    return ""

                # Highlight user's focus property type row
                _user_pt_rate = st.session_state.user_intent.get("property_type")
                def _highlight_focus_row(row):
                    if _user_pt_rate and _intent_matches_property_type(_user_pt_rate, row["Property Type"]):
                        return ["background-color: #f5f0e6; font-weight: 700"] * len(row)
                    return [""] * len(row)

                styled = disp_adj.style.applymap(_colour_delta, subset=["Margin Delta bps"]).apply(_highlight_focus_row, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True, height=290)
                delta_10y = (current_10y or 0) - baseline
                direction = "above" if delta_10y > 0 else "below"
                st.caption(
                    f"10Y is {abs(delta_10y):.2f}% {direction} the {baseline:.1f}% baseline. "
                    "Higher cap rates → lower property multiples → compressed effective margins."
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── REIT Refinancing Risk ────────────────────────────────────────────────
        section(" REIT Near-Term Refinancing Risk")
        if debt_risk:
            debt_df = pd.DataFrame(debt_risk)
            risk_colors = {"High": "#b71c1c", "Medium": "#e65100", "Low": "#1b5e20"}

            col_chart, col_tbl2 = st.columns([2, 3])
            with col_chart:
                sorted_df = debt_df.sort_values("Risk %", ascending=True).tail(20)
                bar_clrs  = [risk_colors.get(r, "#888") for r in sorted_df["Risk Level"]]
                fig_debt  = go.Figure(go.Bar(
                    x=sorted_df["Risk %"],
                    y=sorted_df["Ticker"],
                    orientation="h",
                    marker_color=bar_clrs,
                    text=sorted_df["Risk %"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    customdata=sorted_df[["Name", "Near-Term Debt $B", "Total Debt $B"]].values,
                    hovertemplate=(
                        "<b>%{y} — %{customdata[0]}</b><br>"
                        "Near-term: $%{customdata[1]:.2f}B<br>"
                        "Total debt: $%{customdata[2]:.2f}B<br>"
                        "Risk: %{x:.1f}%<extra></extra>"
                    ),
                ))
                fig_debt.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    xaxis=dict(title="Near-Term Debt / Total Debt (%)", ticksuffix="%",
                               gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                               title_font=dict(color="#c8b890")),
                    yaxis=dict(tickfont=dict(color="#c8b890", size=9)),
                    margin=dict(t=20, b=40, l=60, r=60), height=460,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                fig_debt.add_vline(x=25, line_dash="dash", line_color="#b71c1c",
                                    annotation_text="High risk threshold",
                                    annotation_font=dict(color="#b71c1c", size=9))
                fig_debt.add_vline(x=10, line_dash="dot", line_color="#e65100",
                                    annotation_text="Med threshold",
                                    annotation_font=dict(color="#e65100", size=9))
                st.plotly_chart(fig_debt, use_container_width=True)
                st.caption(
                    "Estimates the share of each REIT's debt maturing within 12 months relative to its market cap. "
                    "REITs above the medium threshold face refinancing pressure — in a high-rate environment, "
                    "rolling debt at elevated rates compresses FFO and can force asset sales or equity dilution."
                )

            with col_tbl2:
                def _risk_style(val):
                    return {"High": "color:#b71c1c;font-weight:700",
                            "Medium": "color:#e65100;font-weight:600",
                            "Low": "color:#1b5e20"}.get(val, "")
                styled_debt = debt_df.style.applymap(_risk_style, subset=["Risk Level"])
                st.dataframe(styled_debt, use_container_width=True, hide_index=True, height=460)

            high_risk = debt_df[debt_df["Risk Level"] == "High"]
            if not high_risk.empty:
                names = ", ".join(high_risk["Ticker"].tolist())
                st.warning(
                    f" **High refinancing risk:** {names} — these REITs have ≥25% of debt maturing "
                    f"within 12 months. Elevated 10Y rates ({current_10y:.2f}%) increase rollover costs."
                )
            else:
                st.success(" No REITs with high near-term refinancing risk detected.")

            st.caption(
                "Near-term debt = current portion of long-term debt from latest quarterly balance sheet (yfinance). "
                "Risk % = near-term / total debt. High ≥ 25%, Medium 10–25%, Low < 10%."
            )
        else:
            st.info("REIT debt data not yet available. The agent will populate this on its next run.")

        st.markdown("<br>", unsafe_allow_html=True)
        if cap_adj and current_10y:
            delta = current_10y - baseline
            direction_word = "above" if delta > 0 else "below"
            impact_word    = "expanding cap rates" if delta > 0 else "compressing cap rates"
            st.info(
                f" **Impact on Pricing & Profit tab:** With 10Y at **{current_10y:.2f}%** "
                f"({abs(delta):.2f}% {direction_word} the {baseline:.1f}% static benchmark), "
                f"rates are currently **{impact_word}** across all property types. "
                f"The adjustments above are applied in the rate-adjusted view on the Pricing & Profit tab."
            )

        st.caption(
            "Data: Federal Reserve Bank of St. Louis (FRED). "
            "Cap rate adjustment model: adjusted = benchmark + (10Y − baseline) × sector beta. "
            "This is research, not financial advice."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Cap Rate Adjustment Model:**

Cap Rate (adjusted) = Benchmark Cap Rate + (Current 10Y Treasury - 3.5% historical average) x Sector Beta

- **Sector Beta** reflects each property type's sensitivity to rate changes: Office (1.0x) and Retail (0.9x) are most sensitive; Industrial (0.6x) and Multifamily (0.7x) are more resilient.
- A **100bp rate increase** typically expands cap rates **50-75bp** across most property types.
- **PnL Impact:** On a $10M property, a 50bp cap rate expansion = ~$800K loss in implied asset value.

**Yield Curve Shape:**
- **Normal (10Y > 2Y):** Healthy economy, favorable for long-term CRE financing
- **Inverted (2Y > 10Y):** Recession signal, short-term borrowing costs exceed long-term — stress on floating-rate CRE debt
- **Flat:** Transition period, uncertainty in rate direction

**REIT Refinancing Risk:** Scored 0-100 based on debt maturity profile, floating-rate debt exposure, and current spread vs. origination spread.

**Data Source:** Federal Reserve Bank of St. Louis (FRED) — 40+ series including DFF, DGS2, DGS10, DGS30, T10Y2Y, T10Y3M. Updated every hour via Agent 6.
""")


    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — LABOR MARKET & TENANT DEMAND
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_labor:
        st.markdown("#### Where is job growth strongest — and which property types benefit?")
        st.markdown(
            "Agent 9 pulls **BLS payroll data**, **FRED labor series** (unemployment, job openings, quits), "
            "and **sector ETF momentum** to map employment trends to CRE tenant demand by property type. "
            "Updates every 6 hours."
        )
        agent_last_updated("labor_market")

        cache_lm = read_cache("labor_market")
        ldata    = cache_lm.get("data") or {}

        if not ldata:
            st.info(" Labor market agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()

        fred_labor   = ldata.get("fred_labor", {})
        bls_sectors  = ldata.get("bls_sectors", [])
        sector_etfs  = ldata.get("sector_etfs", [])
        metro_unemp  = ldata.get("metro_unemployment", [])
        demand_sig   = ldata.get("demand_signal", {})

        # ── Demand Signal Banner ────────────────────────────────────────────────
        sig_label = demand_sig.get("label", "UNKNOWN")
        sig_score = demand_sig.get("score", 50)
        _td_sum   = "Synthesized from nonfarm payrolls, job openings, unemployment trend, and sector ETF momentum. STRONG >= 65 · MODERATE 41-64 · SOFT <= 40"
        st.markdown(gauge_card(
            title       = "TENANT DEMAND",
            label       = sig_label,
            score       = sig_score,
            summary     = _td_sum,
            agent_num   = "A9  Agent 9",
            age_label   = cache_age_label("labor_market"),
            scale_labels= ("SOFT", "25", "MODERATE", "75", "STRONG"),
        ), unsafe_allow_html=True)
        with st.expander("How to read this indicator"):
            st.markdown("""
**What it measures:** The strength of employment-driven demand for commercial space — the primary driver of office, industrial, and retail lease-up.

| Signal | Score | What it means for CRE |
|--------|-------|----------------------|
| **STRONG** | 65–100 | Low unemployment, rising payrolls, high job openings → businesses expanding → tenants leasing more space. Landlords hold pricing power. |
| **MODERATE** | 41–64 | Mixed signals. Demand is present but softening. Selective acquisitions in supply-constrained markets remain viable. |
| **SOFT** | 0–40 | Rising unemployment or falling payrolls → tenants downsizing, sublease supply rising, lease concessions increasing. |

**Key inputs:** Nonfarm Payrolls, Unemployment Rate (U-3), JOLTS Job Openings, Quits Rate, Labor Force Participation, Avg Hourly Earnings — all from FRED.
            """)

        # ── KPI Strip — National Labor ──────────────────────────────────────────
        section(" National Labor Market Snapshot")
        kpi_keys = [
            ("Unemployment Rate",         "%",  "Civilian U-3"),
            ("Nonfarm Payrolls",           "K",  "Total employed"),
            ("Job Openings (JOLTS)",       "K",  "Open positions"),
            ("Quits Rate",                 "%",  "Voluntary separations"),
            ("Labor Force Participation",  "%",  "Ages 16+"),
            ("Avg Hourly Earnings",        "$",  "Private sector"),
        ]
        kpi_cols = st.columns(len(kpi_keys))
        for col, (key, unit, sub) in zip(kpi_cols, kpi_keys):
            r = fred_labor.get(key, {})
            cur = r.get("current")
            d1m = r.get("delta_1m")
            if cur is None:
                col.markdown(metric_card(key, "N/A", sub), unsafe_allow_html=True)
                continue
            if unit == "K":
                val_s = f"{cur/1000:.1f}M" if cur >= 1000 else f"{cur:.0f}K"
            elif unit == "$":
                val_s = f"${cur:.2f}"
            else:
                val_s = f"{cur:.1f}%"
            delta_html = ""
            if d1m is not None:
                arrow = "▲" if d1m > 0 else ("▼" if d1m < 0 else "→")
                # For unemployment, down is good; for everything else, up is good
                good_up = key != "Unemployment Rate"
                good    = (d1m > 0) == good_up
                clr     = "#1b5e20" if good else "#b71c1c"
                if unit == "K":
                    d_s = f"{d1m/1000:+.1f}M" if abs(d1m) >= 1000 else f"{d1m:+.0f}K"
                elif unit == "$":
                    d_s = f"{d1m:+.2f}"
                else:
                    d_s = f"{d1m:+.2f}%"
                delta_html = f"<span style='color:{clr};font-size:0.78rem;'>{arrow} {d_s} 1M</span>"
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{key}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_html or sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BLS Sector Payrolls ─────────────────────────────────────────────────
        section(" Employment by Sector — CRE Demand Drivers")
        if bls_sectors:
            sec_df = pd.DataFrame(bls_sectors)
            # Filter out "Total Private" for the chart (keep it in table)
            chart_df = sec_df[sec_df["label"] != "Total Private"].copy()
            bar_clrs = [GOLD if v > 0 else "#c62828" for v in chart_df["mom_pct"]]

            fig_sec = go.Figure(go.Bar(
                x=chart_df["mom_pct"],
                y=chart_df["label"],
                orientation="h",
                marker_color=bar_clrs,
                text=chart_df["mom_pct"].apply(lambda v: f"{v:+.2f}%"),
                textposition="outside",
                customdata=chart_df[["employment_k", "property_type", "period"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Employment: %{customdata[0]:,.0f}K<br>"
                    "MoM Change: %{x:+.2f}%<br>"
                    "CRE Type: %{customdata[1]}<br>"
                    "Period: %{customdata[2]}<extra></extra>"
                ),
            ))
            fig_sec.add_vline(x=0, line_color="#333", line_width=1)
            fig_sec.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                xaxis=dict(title="Month-over-Month Change (%)", ticksuffix="%",
                           gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                           title_font=dict(color="#c8b890")),
                yaxis=dict(tickfont=dict(color="#c8b890", size=10)),
                margin=dict(t=20, b=40, l=220, r=80), height=400,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

            # Detail table
            disp_sec = sec_df[["label", "employment_k", "mom_pct", "property_type", "period"]].copy()
            disp_sec.columns = ["Sector", "Employment (K)", "MoM %", "CRE Demand Driver", "Period"]
            disp_sec["Employment (K)"] = disp_sec["Employment (K)"].apply(lambda x: f"{x:,.0f}K")
            disp_sec["MoM %"] = disp_sec["MoM %"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(disp_sec, use_container_width=True, hide_index=True)
            st.caption(
                "BLS Supersector payrolls (monthly). Positive MoM = expanding employment = rising tenant demand. "
                "Professional & Business Services and Financial Activities drive Office demand; "
                "Manufacturing and Trade/Transport drive Industrial demand."
            )
        else:
            st.info("BLS sector data not yet available.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sector ETF Tenant Demand Signals ────────────────────────────────────
        section(" Sector ETF Momentum → Tenant Demand by Property Type")
        if sector_etfs:
            etf_df = pd.DataFrame(sector_etfs)
            sig_colors = {"EXPANDING": "#1b5e20", "FLAT": "#e65100", "CONTRACTING": "#b71c1c"}
            bar_clrs_etf = [sig_colors.get(s, "#888") for s in etf_df["signal"]]

            fig_etf = go.Figure(go.Bar(
                x=etf_df["label"],
                y=etf_df["return_6mo"],
                marker_color=bar_clrs_etf,
                text=etf_df["return_6mo"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
                customdata=etf_df[["property_type", "latest_price", "pct_vs_sma", "signal"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "6mo Return: %{y:+.1f}%<br>"
                    "Price: $%{customdata[1]:.2f}<br>"
                    "vs SMA-60: %{customdata[2]:+.1f}%<br>"
                    "Signal: %{customdata[3]}<br>"
                    "CRE Type: %{customdata[0]}<extra></extra>"
                ),
            ))
            fig_etf.add_hline(y=0, line_color="#333", line_width=1)
            fig_etf.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                yaxis=dict(title="6-Month Return (%)", ticksuffix="%",
                           gridcolor="#2a2208", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickangle=-20, tickfont=dict(color="#c8b890", size=9)),
                margin=dict(t=30, b=80), height=360,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_etf, use_container_width=True)

            # Property type signal summary
            pt_summary = {}
            for row in sector_etfs:
                pt = row["property_type"]
                if pt not in pt_summary:
                    pt_summary[pt] = []
                pt_summary[pt].append(row["return_6mo"])
            pt_rows = [{"CRE Property Type": pt,
                        "Avg Sector Return": f"{sum(v)/len(v):+.1f}%",
                        "Demand Signal": "EXPANDING" if sum(v)/len(v) > 2 else ("CONTRACTING" if sum(v)/len(v) < -2 else "FLAT")}
                       for pt, v in sorted(pt_summary.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)]
            st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Rising sector ETFs signal expanding corporate employment → more office/industrial/retail leasing. "
                "EXPANDING > +2% return, CONTRACTING < -2%, FLAT in between. "
                "Data: Yahoo Finance (6-month trailing). Rate-limited — refreshes on scheduled runs."
            )
        else:
            st.info("Sector ETF data temporarily unavailable (Yahoo Finance rate limit). Will populate on next scheduled run.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metro Market Unemployment ────────────────────────────────────────────
        section(" State Unemployment — Top CRE Destination Markets")
        if metro_unemp:
            mu_df = pd.DataFrame(metro_unemp)
            tight_clr  = "#1b5e20"
            bal_clr    = GOLD
            loose_clr  = "#b71c1c"
            bar_clrs_mu = [
                tight_clr if r == "TIGHT" else (loose_clr if r == "LOOSE" else bal_clr)
                for r in mu_df["signal"]
            ]

            fig_mu = go.Figure(go.Bar(
                x=mu_df["unemp_rate"],
                y=mu_df["market"],
                orientation="h",
                marker_color=bar_clrs_mu,
                text=mu_df["unemp_rate"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                customdata=mu_df[["delta_1m", "signal", "period"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Unemployment: %{x:.1f}%<br>"
                    "MoM Δ: %{customdata[0]:+.1f}pp<br>"
                    "Labor Market: %{customdata[1]}<br>"
                    "Period: %{customdata[2]}<extra></extra>"
                ),
            ))
            fig_mu.add_vline(x=4.0, line_dash="dash", line_color=tight_clr,
                              annotation_text="Tight (<4%)",
                              annotation_font=dict(color=tight_clr, size=9))
            fig_mu.add_vline(x=6.0, line_dash="dash", line_color=loose_clr,
                              annotation_text="Loose (>6%)",
                              annotation_font=dict(color=loose_clr, size=9))
            fig_mu.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                xaxis=dict(title="Unemployment Rate (%)", ticksuffix="%",
                           gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                           title_font=dict(color="#c8b890")),
                yaxis=dict(tickfont=dict(color="#c8b890", size=9)),
                margin=dict(t=20, b=40, l=260, r=80), height=360,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_mu, use_container_width=True)

            # Table
            mu_disp = mu_df[["market", "unemp_rate", "delta_1m", "signal", "period"]].copy()
            mu_disp.columns = ["State / Key Metros", "Unemployment %", "MoM Δ (pp)", "Labor Market", "Period"]
            mu_disp["Unemployment %"] = mu_disp["Unemployment %"].apply(lambda x: f"{x:.1f}%")
            mu_disp["MoM Δ (pp)"] = mu_disp["MoM Δ (pp)"].apply(lambda x: f"{x:+.1f}")

            def _tight_style(val):
                return {"TIGHT": "color:#1b5e20;font-weight:700",
                        "BALANCED": "color:#e65100",
                        "LOOSE": "color:#b71c1c;font-weight:700"}.get(val, "")
            styled_mu = mu_disp.style.applymap(_tight_style, subset=["Labor Market"])
            st.dataframe(styled_mu, use_container_width=True, hide_index=True)
            st.caption(
                "TIGHT labor markets (<4% unemployment) indicate strong local economies — higher occupier demand "
                "and rent growth potential. LOOSE (>6%) may signal weaker absorption. "
                "Data: FRED state unemployment rates (BLS LAUS). Updated monthly."
            )
        else:
            st.info("Metro unemployment data not yet available.")

        st.caption(
            "Data: Bureau of Labor Statistics (BLS), Federal Reserve (FRED), Yahoo Finance. "
            "Tenant Demand Signal score: 0–100 based on payroll growth, job openings, unemployment trend, and sector momentum. "
            "This is research, not investment advice."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Tenant Demand Signal (0-100):**

Composite score based on four equally weighted components:
1. **Nonfarm Payroll Growth (25%):** Monthly job additions — strong payroll growth = more tenants needing space
2. **JOLTS Job Openings (25%):** Forward-looking indicator of hiring intent — high openings signal future lease demand
3. **Unemployment Trend (25%):** Direction matters more than level — declining unemployment = tightening labor market
4. **Sector ETF Momentum (25%):** 6-month return of sector ETFs (XLI, XLK, XLF, etc.) mapped to property types

**Property Type Mapping:**
- Manufacturing/Logistics payrolls -> Industrial demand
- Professional/Business services -> Office demand
- Leisure/Hospitality -> Retail/Hospitality demand
- Education/Healthcare -> Medical office demand

**Labor Market Classification:**
- **TIGHT** (< 4% unemployment): Strong occupier demand, rent growth potential
- **BALANCED** (4-6%): Stable absorption
- **LOOSE** (> 6%): Weaker absorption, potential vacancy risk

**Data Source:** BLS Public API (supersector payrolls), FRED (UNRATE, JTSJOL, PAYEMS), Yahoo Finance (sector ETFs). Updated every 6 hours via Agent 9.
""")

    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — GDP & ECONOMIC GROWTH
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_gdp:
        st.markdown("#### What economic cycle phase are we in — and what does it mean for CRE?")
        st.markdown(
            "Agent 10 tracks **real GDP growth, industrial production, retail sales, consumer sentiment, "
            "and the leading economic index** to classify the economic cycle and map it to CRE property type outlook. "
            "Updates every 6 hours."
        )
        agent_last_updated("gdp_data")

        cache_gdp = read_cache("gdp_data")
        gdata = cache_gdp.get("data") or {}
        if not gdata:
            st.info(" GDP agent is fetching data — please refresh in ~30 seconds.")
            st.stop()

        g_series = gdata.get("series", {})
        g_cycle  = gdata.get("cycle", {})

        # ── Cycle Signal Banner ─────────────────────────────────────────────────
        cycle_label = g_cycle.get("label", "UNKNOWN")
        cycle_score = g_cycle.get("score", 50)
        _ec_sum     = g_cycle.get("cre_implication", "") or g_cycle.get("summary", "")
        st.markdown(gauge_card(
            title       = "ECONOMIC CYCLE",
            label       = cycle_label,
            score       = cycle_score,
            summary     = _ec_sum,
            agent_num   = "A10  Agent 10",
            age_label   = cache_age_label("gdp_data"),
            scale_labels= ("CONTRACTION", "25", "SLOWDOWN", "75", "EXPANSION"),
        ), unsafe_allow_html=True)
        with st.expander("How to read this indicator"):
            st.markdown("""
**What it measures:** Where the US economy sits in its business cycle — which directly drives CRE occupancy, rent growth, and transaction volume.

| Signal | Score | What it means for CRE |
|--------|-------|----------------------|
| **EXPANSION** | 65–100 | GDP growing, consumer spending up, businesses investing → rising occupancy across all property types, rent growth likely. Peak valuations possible. |
| **SLOWDOWN** | 25–74 | Growth decelerating. Industrial and logistics hold up; office and retail face headwinds. Reduce leverage exposure. |
| **CONTRACTION** | 0–24 | Negative GDP, rising layoffs → vacancy climbing, cap rates rising, values declining. Distressed opportunities may emerge. |

**Key inputs:** Real GDP Growth Rate, Industrial Production Index, Chicago Fed National Activity Index (CFNAI), Retail Sales, Real PCE, Consumer Sentiment — from FRED.
            """)

        # ── KPI Strip ──────────────────────────────────────────────────────────
        section(" Key Economic Indicators")
        kpi_defs = [
            ("Real GDP Growth Rate",        "%",   "Annualized",          False),
            ("Industrial Production Index", "idx", "Level",               False),
            ("Retail Sales",                "$M",  "Monthly",             False),
            ("Consumer Sentiment",          "idx", "U of Michigan",       False),
            ("Chicago Fed Activity Index",  "idx", "Chicago Fed (CFNAI)", False),
            ("Real PCE",                    "$B",  "Personal consumption", False),
        ]
        kpi_cols = st.columns(len(kpi_defs))
        for col, (key, unit, sub, _) in zip(kpi_cols, kpi_defs):
            r = g_series.get(key, {})
            cur = r.get("current")
            d1y = r.get("delta_1y")
            if cur is None:
                col.markdown(metric_card(key.split(" (")[0], "N/A", sub), unsafe_allow_html=True)
                continue
            if unit == "$M":
                val_s = f"${cur/1000:.1f}B"
            elif unit == "$B":
                val_s = f"${cur:,.0f}B"
            elif unit == "%":
                val_s = f"{cur:.1f}%"
            else:
                val_s = f"{cur:.1f}"
            delta_html = ""
            if d1y is not None:
                arrow = "▲" if d1y > 0 else "▼"
                good  = d1y > 0
                clr   = "#1b5e20" if good else "#b71c1c"
                if unit == "$M":
                    d_s = f"{d1y/1000:+.1f}B"
                elif unit == "$B":
                    d_s = f"{d1y:+,.0f}B"
                elif unit == "%":
                    d_s = f"{d1y:+.1f}pp"
                else:
                    d_s = f"{d1y:+.1f}"
                delta_html = f"<span style='color:{clr};font-size:0.78rem;'>{arrow} {d_s} 1Y</span>"
            short_key = key.split(" (")[0].replace(" Index", "").replace(" Rate", "")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{short_key}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_html or sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GDP Growth + IPI Trend ──────────────────────────────────────────────
        col_gdp, col_ipi = st.columns(2)

        with col_gdp:
            section(" Real GDP Growth Rate (Quarterly)")
            gdp_s = g_series.get("Real GDP Growth Rate", {}).get("series", [])
            if gdp_s:
                dates  = [o["date"] for o in gdp_s]
                values = [o["value"] for o in gdp_s]
                bar_clrs = [GOLD if v >= 0 else "#c62828" for v in values]
                fig_gdp = go.Figure(go.Bar(
                    x=dates, y=values, marker_color=bar_clrs,
                    text=[f"{v:+.1f}%" for v in values], textposition="outside",
                    hovertemplate="Q: %{x}<br>Growth: %{y:+.2f}%<extra></extra>",
                ))
                fig_gdp.add_hline(y=0, line_color="#333", line_width=1)
                fig_gdp.add_hline(y=2, line_dash="dot", line_color="#1b5e20",
                                   annotation_text="2% trend", annotation_position="top right",
                                   annotation_font=dict(color="#1b5e20", size=9))
                fig_gdp.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Annualized %", ticksuffix="%", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890")),
                    margin=dict(t=30, b=40), height=320,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_gdp, use_container_width=True)
                st.caption("Quarterly real GDP growth (annualized). Consecutive negative quarters = recession definition.")
            else:
                st.info("GDP growth series not yet available.")

        with col_ipi:
            section(" Industrial Production Index")
            ipi_s = g_series.get("Industrial Production Index", {}).get("series", [])
            if ipi_s:
                dates  = [o["date"] for o in ipi_s]
                values = [o["value"] for o in ipi_s]
                fig_ipi = go.Figure(go.Scatter(
                    x=dates, y=values, mode="lines",
                    line=dict(color=GOLD, width=2.5),
                    fill="tozeroy", fillcolor="rgba(207,185,145,0.15)",
                    hovertemplate="Date: %{x}<br>IPI: %{y:.1f}<extra></extra>",
                ))
                fig_ipi.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Index (2017=100)", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890")),
                    margin=dict(t=30, b=40), height=320,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_ipi, use_container_width=True)
                st.caption("Industrial Production Index (2017=100). Drives demand for industrial/logistics CRE.")
            else:
                st.info("Industrial production series not yet available.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Consumer Sentiment + Leading Index ─────────────────────────────────
        col_cs, col_lei = st.columns(2)

        with col_cs:
            section(" Consumer Sentiment")
            cs_s = g_series.get("Consumer Sentiment", {}).get("series", [])
            if cs_s:
                dates  = [o["date"] for o in cs_s]
                values = [o["value"] for o in cs_s]
                fig_cs = go.Figure(go.Scatter(
                    x=dates, y=values, mode="lines+markers",
                    line=dict(color="#1565c0", width=2),
                    marker=dict(size=4, color="#1565c0"),
                    hovertemplate="Date: %{x}<br>Sentiment: %{y:.1f}<extra></extra>",
                ))
                fig_cs.add_hrect(y0=80, y1=max(values) + 5, fillcolor="rgba(27,94,32,0.05)",
                                  line_width=0, annotation_text="Strong",
                                  annotation_font=dict(color="#1b5e20", size=9))
                fig_cs.add_hrect(y0=0, y1=65, fillcolor="rgba(183,28,28,0.05)",
                                  line_width=0, annotation_text="Weak",
                                  annotation_font=dict(color="#b71c1c", size=9))
                fig_cs.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Index", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890")),
                    margin=dict(t=30, b=40), height=300,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_cs, use_container_width=True)
                st.caption("U of Michigan Consumer Sentiment. High sentiment → retail & hospitality CRE demand.")
            else:
                st.info("Consumer sentiment series not yet available.")

        with col_lei:
            section(" Chicago Fed National Activity Index (3-Month MA)")
            lei_s = g_series.get("Chicago Fed Activity Index", {}).get("series", [])
            if lei_s:
                dates  = [o["date"] for o in lei_s]
                values = [o["value"] for o in lei_s]
                bar_clrs_cfnai = [GOLD if v >= 0 else "#c62828" for v in values]
                fig_lei = go.Figure(go.Bar(
                    x=dates, y=values, marker_color=bar_clrs_cfnai,
                    hovertemplate="Date: %{x}<br>CFNAI-MA3: %{y:.2f}<extra></extra>",
                ))
                fig_lei.add_hline(y=0, line_color="#333", line_width=1.5)
                fig_lei.add_hline(y=-0.7, line_dash="dash", line_color="#b71c1c",
                                   annotation_text="Recession signal (<−0.7)",
                                   annotation_font=dict(color="#b71c1c", size=9))
                fig_lei.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="Index (0 = trend growth)", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890")),
                    margin=dict(t=30, b=40), height=300,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_lei, use_container_width=True)
                st.caption("Chicago Fed National Activity Index (3-month MA). 0 = trend growth. Below −0.7 historically signals recession onset.")
            else:
                st.info("Chicago Fed Activity Index not yet available.")

        # ── CRE Cycle Implications Table ───────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" CRE Property Type Outlook by Cycle Phase")
        outlook_data = {
            "Property Type":     ["Office", "Industrial / Logistics", "Multifamily", "Retail", "Healthcare / Life Sci", "Data Centers"],
            "Expansion":         ["", "", "", "", "", ""],
            "Slowdown":          ["", "", "", "", "", ""],
            "Contraction":       ["", "", "", "", "", ""],
            "Current Outlook":   [
                "" if cycle_label == "EXPANSION" else ("" if cycle_label == "SLOWDOWN" else ""),
                "" if cycle_label in ("EXPANSION", "SLOWDOWN") else "",
                "",
                "" if cycle_label == "EXPANSION" else ("" if cycle_label == "SLOWDOWN" else ""),
                "",
                "" if cycle_label in ("EXPANSION", "SLOWDOWN") else "",
            ],
        }
        outlook_df = pd.DataFrame(outlook_data)
        st.dataframe(outlook_df, use_container_width=True, hide_index=True)
        st.caption(
            " Favorable · Neutral · Cautious. Based on current economic cycle classification. "
            "Industrial and Healthcare tend to be more defensive; Office and Retail more cyclical."
        )
        st.caption("Data: Federal Reserve Bank of St. Louis (FRED). This is research, not financial advice.")

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Economic Cycle Classification:**

The cycle phase is determined by combining multiple indicators:
- **Real GDP Growth (GDPC1):** Quarter-over-quarter annualized rate
- **Industrial Production (INDPRO):** Monthly index of factory, mining, and utility output
- **Consumer Sentiment (UMCSENT):** University of Michigan survey — leading indicator of consumer spending
- **Chicago Fed National Activity Index (CFNAI):** 85-indicator composite of national economic activity

**Cycle Phases:**
- **EXPANSION:** GDP > 2%, Industrial Production rising, Sentiment > 80, CFNAI > 0
- **SLOWDOWN:** GDP 0-2%, mixed signals, Sentiment declining
- **CONTRACTION:** GDP < 0%, Industrial Production falling, Sentiment < 60, CFNAI < -0.7

**CRE Impact by Cycle Phase:**
- *Expansion:* All property types benefit — strongest for Office and Retail
- *Slowdown:* Industrial and Healthcare more defensive; Office and Retail vulnerable
- *Contraction:* Multifamily most resilient (people always need housing); Office and Retail face rising vacancies

**Data Source:** FRED (GDPC1, INDPRO, UMCSENT, CFNAI, RSXFS). Updated every 6 hours via Agent 10.
""")


    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — INFLATION
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_inflation:
        st.markdown("#### How is inflation affecting CRE valuations, rents, and construction costs?")
        st.markdown(
            "Agent 11 tracks **CPI, core inflation, shelter & rent inflation, PPI, and market-implied "
            "breakeven inflation** to assess real return erosion, rent growth, and replacement cost trends. "
            "Updates every 6 hours."
        )
        agent_last_updated("inflation_data")

        cache_inf = read_cache("inflation_data")
        idata = cache_inf.get("data") or {}
        if not idata:
            st.info(" Inflation agent is fetching data — please refresh in ~30 seconds.")
            st.stop()

        inf_series = idata.get("series", {})
        inf_signal = idata.get("signal", {})

        # ── Signal Banner — Segmented Gauge Card ────────────────────────────────
        inf_label    = inf_signal.get("label", "UNKNOWN")
        inf_score    = inf_signal.get("score", 50)
        inf_confidence = inf_signal.get("confidence", "High")
        _age_inf     = cache_age_label("inflation_data")

        st.markdown(gauge_card(
            title="INFLATION REGIME",
            label=inf_label,
            score=inf_score,
            summary=inf_signal.get("summary", ""),
            agent_num=11,
            age_label=_age_inf,
            confidence=inf_confidence,
            low_good=True,
            scale_labels=("COOLING", "25", "MODERATE", "75", "HOT"),
        ), unsafe_allow_html=True)
        with st.expander("How to read this indicator"):
            st.markdown("""
**What it measures:** The current inflation environment and its impact on CRE construction costs, cap rates, and real returns.

| Signal | Score | What it means for CRE |
|--------|-------|----------------------|
| **COOLING** | 0–35 | CPI falling toward 2% target. Fed likely cutting rates → cap rate compression, rising valuations, easier financing. |
| **MODERATE** | 36–74 | Inflation in the 2–4% range. Mixed — some replacement cost support for values, but rate uncertainty limits cap rate compression. |
| **HOT** | 75–100 | CPI well above target. Fed holds rates high → higher cap rates, value pressure on stabilized assets. Construction cost inflation eats development margins. |

**Key inputs:** CPI All Items, Core CPI (ex food & energy), CPI Shelter, CPI Rent, 5-Year Breakeven Inflation, 1-Year Inflation Expectations (U of Michigan) — from FRED.
            """)

        # ── KPI Strip ──────────────────────────────────────────────────────────
        section(" Inflation Dashboard")
        inf_kpis = [
            ("CPI All Items",           "YoY %", "Headline"),
            ("Core CPI",                "YoY %", "Ex food & energy"),
            ("CPI Shelter",             "YoY %", "Housing costs"),
            ("CPI Rent",                "YoY %", "Primary residence"),
            ("5Y Breakeven Inflation",  "%",      "Market expectation"),
            ("1Y Inflation Expectations","%",     "U of Michigan"),
        ]
        inf_cols = st.columns(len(inf_kpis))
        for col, (key, unit_label, sub) in zip(inf_cols, inf_kpis):
            r = inf_series.get(key, {})
            # Use YoY for CPI/PPI index series, current for breakeven/expectations
            val = r.get("yoy_pct") if r.get("yoy_pct") is not None else r.get("current")
            d1m = r.get("delta_1m")
            if val is None:
                col.markdown(metric_card(key.replace(" All Items", "").replace(" Inflation", ""), "N/A", sub), unsafe_allow_html=True)
                continue
            val_s = f"{val:.1f}%"
            delta_html = ""
            if d1m is not None:
                arrow = "▲" if d1m > 0 else "▼"
                # For inflation, up = bad (hot), down = good (cooling)
                clr = "#b71c1c" if d1m > 0 else "#1b5e20"
                delta_html = f"<span style='color:{clr};font-size:0.78rem;'>{arrow} {abs(d1m):.3f} 1M</span>"
            short = key.replace(" All Items", "").replace(" Inflation", "").replace(" Expectations", " Exp.")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{short}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_html or sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CPI Components Chart ────────────────────────────────────────────────
        col_cpi, col_ppi = st.columns(2)

        with col_cpi:
            section(" CPI Trends — Headline vs Core vs Shelter")
            cpi_keys   = ["CPI All Items", "Core CPI", "CPI Shelter", "CPI Rent"]
            cpi_colors = ["#1565c0", "#e65100", GOLD, "#6a1b9a"]
            fig_cpi = go.Figure()
            for ckey, clr in zip(cpi_keys, cpi_colors):
                s = inf_series.get(ckey, {}).get("series", [])
                if not s:
                    continue
                # Compute rolling YoY from index series
                if len(s) >= 13:
                    yoy_pts = [{"date": s[i]["date"],
                                "value": round((s[i]["value"] - s[i-12]["value"]) / s[i-12]["value"] * 100, 2)}
                               for i in range(12, len(s))]
                else:
                    yoy_pts = s
                dates  = [o["date"] for o in yoy_pts]
                values = [o["value"] for o in yoy_pts]
                fig_cpi.add_trace(go.Scatter(
                    x=dates, y=values, name=ckey,
                    mode="lines", line=dict(color=clr, width=2),
                    hovertemplate=f"{ckey}: %{{y:.2f}}% YoY<br>%{{x}}<extra></extra>",
                ))
            fig_cpi.add_hline(y=2, line_dash="dot", line_color="#1b5e20",
                               annotation_text="Fed 2% target",
                               annotation_font=dict(color="#1b5e20", size=9))
            fig_cpi.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                yaxis=dict(title="YoY %", ticksuffix="%", gridcolor="#2a2208",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#c8b890", size=10)),
                margin=dict(t=40, b=40), height=340,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_cpi, use_container_width=True)
            st.caption(
                "CPI Shelter and CPI Rent measure housing cost inflation — directly relevant to multifamily NOI growth. "
                "Core CPI (ex food & energy) is the Fed's primary policy target."
            )

        with col_ppi:
            section(" PPI — Producer & Construction Input Costs")
            ppi_keys   = ["PPI All Commodities", "PPI Manufacturing"]
            ppi_colors = ["#c62828", "#e65100"]
            fig_ppi = go.Figure()
            for pkey, clr in zip(ppi_keys, ppi_colors):
                s = inf_series.get(pkey, {}).get("series", [])
                if not s or len(s) < 13:
                    continue
                yoy_pts = [{"date": s[i]["date"],
                            "value": round((s[i]["value"] - s[i-12]["value"]) / s[i-12]["value"] * 100, 2)}
                           for i in range(12, len(s))]
                dates  = [o["date"] for o in yoy_pts]
                values = [o["value"] for o in yoy_pts]
                fig_ppi.add_trace(go.Scatter(
                    x=dates, y=values, name=pkey,
                    mode="lines", line=dict(color=clr, width=2),
                    hovertemplate=f"{pkey}: %{{y:.2f}}% YoY<br>%{{x}}<extra></extra>",
                ))
            fig_ppi.add_hline(y=0, line_color="#333", line_width=1)
            fig_ppi.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                yaxis=dict(title="YoY %", ticksuffix="%", gridcolor="#2a2208",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#c8b890", size=10)),
                margin=dict(t=40, b=40), height=340,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_ppi, use_container_width=True)
            st.caption(
                "Rising PPI increases replacement cost of new CRE — supporting values of existing assets. "
                "Elevated construction input costs also deter new supply, tightening vacancy."
            )

        # ── Breakeven Inflation Chart ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Market-Implied Inflation Expectations (Breakeven Rates)")
        be_keys   = ["5Y Breakeven Inflation", "10Y Breakeven Inflation"]
        be_colors = ["#1565c0", "#6a1b9a"]
        fig_be = go.Figure()
        for bkey, clr in zip(be_keys, be_colors):
            s = inf_series.get(bkey, {}).get("series", [])
            if not s:
                continue
            dates  = [o["date"] for o in s]
            values = [o["value"] for o in s]
            fig_be.add_trace(go.Scatter(
                x=dates, y=values, name=bkey,
                mode="lines", line=dict(color=clr, width=2),
                hovertemplate=f"{bkey}: %{{y:.2f}}%<br>%{{x}}<extra></extra>",
            ))
        fig_be.add_hline(y=2.0, line_dash="dot", line_color="#1b5e20",
                          annotation_text="Fed 2% target",
                          annotation_font=dict(color="#1b5e20", size=9))
        fig_be.add_hline(y=2.5, line_dash="dash", line_color="#b71c1c",
                          annotation_text="Concern threshold",
                          annotation_font=dict(color="#b71c1c", size=9))
        fig_be.update_layout(
            paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
            yaxis=dict(title="Implied Inflation (%)", ticksuffix="%", gridcolor="#2a2208",
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            xaxis=dict(tickfont=dict(color="#c8b890")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#c8b890", size=11)),
            margin=dict(t=40, b=40), height=320,
            font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_be, use_container_width=True)
        st.caption(
            "Breakeven inflation = nominal Treasury yield minus TIPS yield — the market's consensus inflation forecast. "
            "Elevated breakevens signal that the Fed is unlikely to cut rates soon, keeping cap rates elevated "
            "and compressing CRE asset values."
        )
        st.caption("Data: Federal Reserve Bank of St. Louis (FRED). This is research, not financial advice.")

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Inflation Regime Classification:**

- **HOT:** Headline CPI > 4% YoY or Core CPI > 3.5% YoY — erosion of real returns, Fed likely tightening
- **MODERATE:** Headline CPI 2-4% YoY — manageable inflation, favorable for CRE with rent escalators
- **COOLING:** Headline CPI < 2% YoY — disinflation, potential rate cuts ahead (positive for asset values)

**Key Series Tracked:**
- **CPI Headline (CPIAUCSL):** All items, urban consumers — broadest inflation measure
- **CPI Core (CPILFESL):** Excludes food & energy — underlying inflation trend
- **CPI Shelter (CUSR0000SAH1):** Largest CPI component (~35% weight) — directly reflects rent/housing costs
- **CPI Rent (CUSR0000SEHA):** Rent of primary residence — most direct CRE inflation indicator
- **PPI Construction (WPUIP2311001):** Producer prices for construction inputs — signals replacement cost pressure
- **5Y & 10Y Breakeven Inflation (T5YIE, T10YIE):** Market-implied inflation expectations from TIPS spreads

**Why It Matters for CRE:**
- Rising shelter/rent CPI validates rent growth assumptions in underwriting
- Elevated PPI signals higher replacement costs — supporting existing asset values
- Breakeven inflation above 2.5% suggests the Fed won't cut rates soon — keeping cap rates elevated

**Data Source:** FRED (BLS CPI/PPI series, Treasury breakevens). Updated every 6 hours via Agent 11.
""")


    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — CREDIT & CAPITAL MARKETS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_credit:
        st.markdown("#### Is capital available for CRE — and at what cost?")
        st.markdown(
            "Agent 12 monitors **corporate credit spreads, bank lending standards, VIX volatility, "
            "and the BAA–AAA spread** to gauge whether debt capital is flowing into or out of CRE. "
            "Updates every 6 hours."
        )
        agent_last_updated("credit_data")

        cache_cr = read_cache("credit_data")
        crdata = cache_cr.get("data") or {}
        if not crdata:
            st.info(" Credit agent is fetching data — please refresh in ~30 seconds.")
            st.stop()

        cr_series = crdata.get("series", {})
        cr_signal = crdata.get("signal", {})

        # ── Signal Banner — Segmented Gauge Card ────────────────────────────────
        cr_label      = cr_signal.get("label", "UNKNOWN")
        cr_score      = cr_signal.get("score", 50)
        cr_confidence = cr_signal.get("confidence", "High")
        _age_cr       = cache_age_label("credit_data")

        st.markdown(gauge_card(
            title="CREDIT CONDITIONS",
            label=cr_label,
            score=cr_score,
            summary=cr_signal.get("summary", ""),
            agent_num=12,
            age_label=_age_cr,
            confidence=cr_confidence,
            low_good=False,
            scale_labels=("TIGHT", "25", "NEUTRAL", "75", "LOOSE"),
        ), unsafe_allow_html=True)
        with st.expander("How to read this indicator"):
            st.markdown("""
**What it measures:** Whether debt capital is flowing freely into CRE or being restricted — tracking corporate credit spreads, bank lending standards, and market volatility.

| Signal | Score | What it means for CRE |
|--------|-------|----------------------|
| **LOOSE** | 65–100 | Spreads narrow, banks easing standards, VIX low → debt is cheap and available. Strong deal volume, high leverage possible. |
| **NEUTRAL** | 25–74 | Normal credit access. Standard underwriting prevails. Selective lenders; moderate leverage recommended. |
| **TIGHT** | 0–24 | Spreads wide, banks tightening, VIX elevated → lenders pulling back. Higher equity requirements, fewer loans closing. Distressed deals may emerge. |

**Key inputs:** IG & HY Corporate Spreads, BBB Spread (CRE proxy), BAA–AAA Quality Spread, VIX Volatility Index, Fed Senior Loan Officer Survey (CRE tightening %) — from FRED.
            """)

        # ── KPI Strip ──────────────────────────────────────────────────────────
        section(" Credit Market Snapshot")
        cr_kpis = [
            ("IG Corporate Spread",  "bps", "Investment grade"),
            ("HY Corporate Spread",  "bps", "High yield"),
            ("BBB Corporate Spread", "bps", "BBB-rated (CRE proxy)"),
            ("BAA-AAA Spread",       "%",   "Credit quality gap"),
            ("VIX",                  "pts", "Market fear gauge"),
            ("CRE Loan Tightening",  "%",   "Net % banks tightening"),
        ]
        cr_cols = st.columns(len(cr_kpis))
        for col, (key, unit, sub) in zip(cr_cols, cr_kpis):
            r = cr_series.get(key, {})
            cur = r.get("current")
            d1m = r.get("delta_1m")
            if cur is None:
                col.markdown(metric_card(key, "N/A", sub), unsafe_allow_html=True)
                continue
            val_s = f"{cur:.0f}{unit}" if unit == "bps" else (f"{cur:.1f}" if unit == "pts" else f"{cur:.2f}%")
            delta_html = ""
            if d1m is not None:
                arrow = "▲" if d1m > 0 else "▼"
                # Wide spreads / high VIX = bad; tightening lending = bad
                clr = "#b71c1c" if d1m > 0 else "#1b5e20"
                d_s = f"{d1m:+.0f}{unit}" if unit == "bps" else f"{d1m:+.2f}"
                delta_html = f"<span style='color:{clr};font-size:0.78rem;'>{arrow} {d_s} 1M</span>"
            short = key.replace(" Corporate", "").replace(" Spread", " Sprd")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{short}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_html or sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Spread History Charts ───────────────────────────────────────────────
        col_spreads, col_vix = st.columns([3, 2])

        with col_spreads:
            section(" Credit Spread History — IG, HY & BBB")
            fig_sp = go.Figure()
            spread_keys   = ["IG Corporate Spread", "HY Corporate Spread", "BBB Corporate Spread"]
            spread_colors = ["#1565c0", "#c62828", GOLD]
            spread_axis   = [1, 2, 1]  # HY on secondary axis
            for skey, clr, ax in zip(spread_keys, spread_colors, spread_axis):
                s = cr_series.get(skey, {}).get("series", [])
                if not s:
                    continue
                dates  = [o["date"] for o in s]
                values = [o["value"] for o in s]
                fig_sp.add_trace(go.Scatter(
                    x=dates, y=values, name=skey,
                    mode="lines", line=dict(color=clr, width=2),
                    yaxis=f"y{ax}" if ax == 2 else "y",
                    hovertemplate=f"{skey}: %{{y:.0f}}bps<br>%{{x}}<extra></extra>",
                ))
            fig_sp.update_layout(
                paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                yaxis=dict(title="IG / BBB Spread (bps)", gridcolor="#2a2208",
                           tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                yaxis2=dict(title="HY Spread (bps)", overlaying="y", side="right",
                            tickfont=dict(color="#c62828"), title_font=dict(color="#c62828")),
                xaxis=dict(tickfont=dict(color="#c8b890")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#c8b890", size=10)),
                margin=dict(t=40, b=40), height=360,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "IG (Investment Grade) and BBB spreads track the cost of the debt that finances most CRE. "
                "HY spreads (right axis) are a leading risk-sentiment indicator — sharp spikes precede transaction freezes."
            )

        with col_vix:
            section(" VIX — Market Volatility")
            vix_s = cr_series.get("VIX", {}).get("series", [])
            if vix_s:
                dates  = [o["date"] for o in vix_s]
                values = [o["value"] for o in vix_s]
                bar_clrs_vix = ["#b71c1c" if v > 30 else (GOLD if v > 20 else "#1b5e20") for v in values]
                fig_vix = go.Figure(go.Bar(
                    x=dates, y=values, marker_color=bar_clrs_vix,
                    hovertemplate="Date: %{x}<br>VIX: %{y:.1f}<extra></extra>",
                ))
                fig_vix.add_hline(y=20, line_dash="dot", line_color=GOLD,
                                   annotation_text="Elevated (>20)",
                                   annotation_font=dict(color=GOLD_DARK, size=9))
                fig_vix.add_hline(y=30, line_dash="dash", line_color="#b71c1c",
                                   annotation_text="Stress (>30)",
                                   annotation_font=dict(color="#b71c1c", size=9))
                fig_vix.update_layout(
                    paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                    yaxis=dict(title="VIX Level", gridcolor="#2a2208",
                               tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
                    xaxis=dict(tickfont=dict(color="#c8b890")),
                    margin=dict(t=30, b=40), height=360,
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                )
                st.plotly_chart(fig_vix, use_container_width=True)
                st.caption("VIX > 20 = elevated uncertainty. VIX > 30 = stress — CRE deal pipelines freeze as buyers demand higher risk premiums.")
            else:
                st.info("VIX series not yet available.")

        # ── Lending Standards Chart ─────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Bank Lending Standards — C&I and CRE Loans (Net % Tightening)")
        fig_ls = go.Figure()
        ls_keys   = ["C&I Loan Tightening", "CRE Loan Tightening"]
        ls_colors = ["#1565c0", GOLD]
        for lkey, clr in zip(ls_keys, ls_colors):
            s = cr_series.get(lkey, {}).get("series", [])
            if not s:
                continue
            dates  = [o["date"] for o in s]
            values = [o["value"] for o in s]
            fig_ls.add_trace(go.Scatter(
                x=dates, y=values, name=lkey,
                mode="lines+markers", line=dict(color=clr, width=2),
                marker=dict(size=5),
                hovertemplate=f"{lkey}: %{{y:.1f}}%<br>%{{x}}<extra></extra>",
            ))
        fig_ls.add_hline(y=0, line_color="#333", line_width=1.5)
        fig_ls.add_hrect(y0=20, y1=100, fillcolor="rgba(183,28,28,0.05)",
                          line_width=0, annotation_text="Tightening territory",
                          annotation_font=dict(color="#b71c1c", size=9))
        fig_ls.add_hrect(y0=-100, y1=-10, fillcolor="rgba(27,94,32,0.05)",
                          line_width=0, annotation_text="Easing territory",
                          annotation_font=dict(color="#1b5e20", size=9))
        fig_ls.update_layout(
            paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
            yaxis=dict(title="Net % Tightening", ticksuffix="%", gridcolor="#2a2208",
                       zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            xaxis=dict(tickfont=dict(color="#c8b890")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#c8b890", size=11)),
            margin=dict(t=40, b=40), height=340,
            font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_ls, use_container_width=True)
        st.caption(
            "Fed Senior Loan Officer Survey (quarterly). Positive = banks tightening loan standards (less credit supply). "
            "Negative = easing (more credit supply). CRE loan tightening directly restricts acquisition and development financing."
        )
        st.caption("Data: Federal Reserve Bank of St. Louis (FRED). This is research, not financial advice.")

    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — CMBS & DISTRESSED ASSET MONITOR
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_distressed:
        st.markdown("#### Where is CRE distress concentrated — and where are the opportunities?")
        st.markdown(
            "Agent 17 tracks CMBS delinquency rates by property type, the known distressed "
            "asset pipeline, national distress signals, and live CRE loan conditions from FRED. "
            "Powered by Groq AI analysis when API key is configured. Updates every 6 hours."
        )
        agent_last_updated("distressed")

        _dst_cache = read_cache("distressed")
        _dst_data  = _dst_cache.get("data") or {}

        if not _dst_data:
            st.info(" Distressed Asset agent is fetching data — refresh in ~30 seconds.")
        else:
            _dst_dlq    = _dst_data.get("cmbs_delinquency", {})
            _dst_pipe   = _dst_data.get("distressed_pipeline", [])
            _dst_sigs   = _dst_data.get("distress_signals", {})
            _dst_intel  = _dst_data.get("market_intelligence", {})
            _dst_fred   = _dst_data.get("fred_cre_delinquency", [])
            _dst_bbb    = _dst_data.get("fred_bbb_spread", [])

            _dst_ta = {"rising": "↑", "falling": "↓", "stable": "→"}
            _dst_tc = {"rising": "#ef5350", "falling": "#66bb6a", "stable": "#d4a843"}
            _dst_status_c = {
                "REO":               "#ef5350",
                "Maturity Default":  "#ef5350",
                "Special Servicing": "#d4a843",
                "Watchlist":         "#42a5f5",
                "Modified":          "#66bb6a",
            }

            # ── AI Brief ──────────────────────────────────────────────────────
            if _dst_intel.get("summary"):
                section(" Agent 17 — CMBS & Distressed Intelligence")
                _dst_groq_lbl = "Groq AI" if _dst_intel.get("groq_used") else "Static Brief"
                st.markdown(f"""
                <div class="agent-card">
                  <div class="agent-label">Agent 17 &nbsp;·&nbsp; CMBS & Distressed Monitor &nbsp;·&nbsp; {_dst_groq_lbl}</div>
                  <div class="agent-text">{_dst_intel['summary']}</div>
                  {"<div style='margin-top:12px;padding:8px 12px;background:#1e1a0a;border-radius:6px;border-left:3px solid #66bb6a;'><span style='color:#66bb6a;font-weight:700;'>Best Opportunity:</span> <span style='color:#c8bfa8;'>" + _dst_intel.get('top_opportunity','') + "</span></div>" if _dst_intel.get('top_opportunity') else ""}
                  {"<div style='margin-top:8px;padding:8px 12px;background:#1e1a0a;border-radius:6px;border-left:3px solid #ef5350;'><span style='color:#ef5350;font-weight:700;'>Key Risk:</span> <span style='color:#c8bfa8;'>" + _dst_intel.get('key_risk','') + "</span></div>" if _dst_intel.get('key_risk') else ""}
                </div>""", unsafe_allow_html=True)

            # ── Distress explainer ────────────────────────────────────────────
            with st.expander("How to read CMBS delinquency & distress signals"):
                st.markdown("""
**CMBS (Commercial Mortgage-Backed Securities)** are bonds backed by commercial real estate loans. The delinquency rate measures what percentage of those loans are 30+ days past due.

| Rate | Severity | Context |
|------|----------|---------|
| < 2% | Low | Normal/healthy — minimal stress |
| 2–5% | Moderate | Elevated — watch for loan modifications and maturity extensions |
| 5–10% | High | Significant distress — expect special servicing, discounted sales |
| > 10% | Crisis | GFC-level stress (office/retail peaked ~10–12% in 2020) |

**Distress statuses:**
- **REO** (Real Estate Owned) — Lender has taken back the property; available at steep discount
- **Maturity Default** — Borrower couldn't refinance at loan maturity; often the first step toward REO
- **Special Servicing** — Loan transferred to a workout specialist; active negotiation underway
- **Watchlist** — Flagged for potential stress; borrower still current but at risk

**Opportunity:** Distressed assets often trade at 30–60% below peak value, creating entry points for repositioning or conversion plays.
                """)

            # ── National distress signals ──────────────────────────────────────
            section(" National Distress Signals")
            _dst_sig_cols = st.columns(min(len(_dst_sigs), 5))
            for _dst_sc, (_dst_sk, _dst_sv) in zip(_dst_sig_cols, _dst_sigs.items()):
                _dst_lbl   = _dst_sk.replace("_", " ").title()
                _dst_val   = f"${_dst_sv['amount_bn']}B" if "amount_bn" in _dst_sv else f"{_dst_sv.get('rate_pct','—')}%"
                _dst_trend = _dst_sv.get("trend", "stable")
                _dst_ta_   = _dst_ta.get(_dst_trend, "")
                _dst_tc_   = _dst_tc.get(_dst_trend, "#d4a843")
                _dst_note  = _dst_sv.get("note", "")[:70]
                _dst_sc.markdown(
                    f"<div class='metric-card' style='border-top:2px solid {_dst_tc_};'>"
                    f"<div class='label' style='font-size:0.72rem;'>{_dst_lbl}</div>"
                    f"<div class='value' style='font-size:1.4rem;color:{_dst_tc_};'>{_dst_val}</div>"
                    f"<div class='sub' style='color:{_dst_tc_};'>{_dst_ta_} {_dst_trend}</div>"
                    f"<div style='color:#6a6a5a;font-size:0.71rem;margin-top:4px;'>{_dst_note}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── CMBS delinquency table & bar chart ────────────────────────────
            section(" CMBS Delinquency Rates by Property Type")
            _dst_col_table, _dst_col_chart = st.columns([1, 1])
            with _dst_col_table:
                _dst_dlq_rows = []
                for _dst_pt, _dst_d in _dst_dlq.items():
                    _dst_dlq_rows.append({
                        "Property Type": _dst_pt,
                        "Rate":          f"{_dst_d['rate_pct']}%",
                        "Prior Year":    f"{_dst_d['prior_year']}%",
                        "YoY":           f"{_dst_d['rate_pct'] - _dst_d['prior_year']:+.1f}pp",
                        "Trend":         f"{_dst_ta.get(_dst_d['trend'],'')} {_dst_d['trend']}",
                    })
                if _dst_dlq_rows:
                    st.dataframe(pd.DataFrame(_dst_dlq_rows).set_index("Property Type"), use_container_width=True, height=260)
                for _dst_pt, _dst_d in _dst_dlq.items():
                    _dst_t_c = _dst_tc.get(_dst_d["trend"], "#d4a843")
                    _dst_t_a = _dst_ta.get(_dst_d["trend"], "")
                    st.markdown(
                        f"<div style='background:#171309;border-left:3px solid {_dst_t_c};"
                        f"padding:6px 14px;border-radius:4px;margin-bottom:6px;font-size:0.8rem;'>"
                        f"<b style='color:#e8dfc4;'>{_dst_pt}</b> "
                        f"<span style='color:{_dst_t_c};'>{_dst_t_a} {_dst_d['rate_pct']}%</span>"
                        f" &nbsp;·&nbsp; <span style='color:#a09880;'>{_dst_d['note']}</span></div>",
                        unsafe_allow_html=True,
                    )
            with _dst_col_chart:
                if _dst_dlq:
                    _dst_bar_types  = list(_dst_dlq.keys())
                    _dst_bar_rates  = [_dst_dlq[t]["rate_pct"] for t in _dst_bar_types]
                    _dst_bar_colors = ["#ef5350" if r > 6 else ("#d4a843" if r > 3 else "#66bb6a") for r in _dst_bar_rates]
                    fig_dlq = go.Figure(go.Bar(
                        x=_dst_bar_types, y=_dst_bar_rates,
                        marker_color=_dst_bar_colors,
                        text=[f"{r}%" for r in _dst_bar_rates],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Delinquency: %{y}%<extra></extra>",
                    ))
                    fig_dlq.update_layout(
                        xaxis=dict(color="#8a7040"),
                        yaxis=dict(title="Delinquency Rate (%)", gridcolor="#2a2208", color="#8a7040", range=[0, 12]),
                        plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                        font=dict(family="Source Sans Pro", color="#c8b890"),
                        margin=dict(t=20, b=40), height=340,
                    )
                    st.plotly_chart(fig_dlq, use_container_width=True)

            # ── Distressed pipeline ───────────────────────────────────────────
            section(" Known Distressed Asset Pipeline")
            if _dst_pipe:
                _dst_pipe_rows = []
                for _dst_a in _dst_pipe:
                    _dst_status = _dst_a.get("status", "")
                    _dst_s_c    = _dst_status_c.get(_dst_status, "#a09880")
                    _dst_pipe_rows.append({
                        "Asset":       _dst_a["asset"],
                        "Type":        _dst_a["type"],
                        "Loan":        f"${_dst_a['loan_amount'] / 1_000_000:.0f}M",
                        "Status":      _dst_status,
                        "Market":      _dst_a["market"],
                        "Opportunity": _dst_a["opportunity"],
                    })
                st.dataframe(pd.DataFrame(_dst_pipe_rows).set_index("Asset"), use_container_width=True, height=380)

            # ── FRED live credit indicators ───────────────────────────────────
            if _dst_fred or _dst_bbb:
                section(" FRED Live Credit Indicators")
                _dst_fc1, _dst_fc2 = st.columns(2)
                with _dst_fc1:
                    if _dst_fred:
                        fig_fd = go.Figure(go.Scatter(
                            x=[o["date"] for o in _dst_fred],
                            y=[o["value"] for o in _dst_fred],
                            name="CRE Delinquency Rate",
                            line=dict(color="#ef5350", width=2),
                            fill="tozeroy", fillcolor="rgba(239,83,80,0.08)",
                        ))
                        fig_fd.update_layout(
                            title="Bank CRE Loan Delinquency Rate (DRCRELEXBS)",
                            plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                            font=dict(family="Source Sans Pro", color="#c8b890", size=11),
                            xaxis=dict(gridcolor="#2a2208", color="#8a7040"),
                            yaxis=dict(gridcolor="#2a2208", color="#8a7040", title="Rate %"),
                            margin=dict(t=40, b=40), height=280,
                        )
                        st.plotly_chart(fig_fd, use_container_width=True)
                with _dst_fc2:
                    if _dst_bbb:
                        fig_bbb = go.Figure(go.Scatter(
                            x=[o["date"] for o in _dst_bbb],
                            y=[o["value"] for o in _dst_bbb],
                            name="BBB Corp Spread",
                            line=dict(color=GOLD, width=2),
                            fill="tozeroy", fillcolor="rgba(207,185,145,0.08)",
                        ))
                        fig_bbb.update_layout(
                            title="BBB Corporate Spread — CMBS Proxy (BAMLC0A4CBBB)",
                            plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                            font=dict(family="Source Sans Pro", color="#c8b890", size=11),
                            xaxis=dict(gridcolor="#2a2208", color="#8a7040"),
                            yaxis=dict(gridcolor="#2a2208", color="#8a7040", title="Spread (bps)"),
                            margin=dict(t=40, b=40), height=280,
                        )
                        st.plotly_chart(fig_bbb, use_container_width=True)

            st.caption(
                "Source: Trepp, MSCI Real Capital Analytics Q1 2025, FRED (DRCRELEXBS, BAMLC0A4CBBB). "
                "Not financial advice. Distressed assets listed are representative examples, not a complete universe."
            )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**Credit Conditions Classification:**

- **LOOSE:** IG spreads < 100bp, VIX < 15, bank lending easing — abundant capital, aggressive CRE lending
- **NEUTRAL:** IG spreads 100-200bp, VIX 15-25 — normal market conditions
- **TIGHT:** IG spreads > 200bp, VIX > 25, bank lending tightening — restricted capital, higher borrowing costs

**Key Indicators:**
- **Investment Grade (IG) Spread (BAMLC0A4CBBB):** BBB corporate bond yield minus Treasury — cost of corporate borrowing
- **High Yield (HY) Spread (BAMLH0A0HYM2):** Junk bond spread — risk appetite measure
- **VIX (VIXCLS):** CBOE Volatility Index — market uncertainty and risk aversion
- **Moody's BAA-AAA Spread:** Credit quality premium — widening signals deteriorating credit conditions
- **Fed Senior Loan Officer Survey:** Quarterly survey on bank lending standards for C&I and CRE loans

**PnL Impact:**
- A 100bp widening in IG spreads typically adds 75-100bp to CRE mortgage rates
- On a $10M property with 65% LTV, a 100bp rate increase adds ~$65K/year to debt service
- Tightening lending standards reduce available acquisition financing, cooling transaction volume and prices

**Data Source:** FRED (corporate spreads, VIX, lending surveys), updated every 6 hours via Agent 12.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TAB — INVESTMENT ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
with main_tab_advisor:
    from src.recommendation_engine import (
        build_recommendation, parse_prompt,
        ALL_MARKETS, _PROP_SYNONYMS,
    )

    st.markdown("""
<div style="background:linear-gradient(135deg,#1a1208 0%,#2a1e08 100%);
            border:1px solid #a07830; border-top:3px solid #d4a843;
            border-radius:10px; padding:28px 36px; margin-bottom:16px;">
  <div style="color:#d4a843;font-size:1.45rem;font-weight:700;letter-spacing:1px;">
    AI-Powered Commercial Real Estate Advisor
  </div>
  <div style="color:#a09880;font-size:0.92rem;margin-top:6px;max-width:780px;line-height:1.65;">
    Describe your investment goal in plain English. The platform parses your intent, scores
    every candidate market across 10+ live data signals, and generates a full institutional-grade
    investment brief &mdash; complete with a year-by-year P&amp;L pro forma, debt structure,
    depreciation tax shield, and Opportunity Zone analysis.
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;">
    <span style="background:#1e2e0a;border:1px solid #507028;color:#80a848;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; Market Scoring</span>
    <span style="background:#0a1e1e;border:1px solid #287068;color:#48a898;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; 10-Year P&amp;L Pro Forma</span>
    <span style="background:#1a1408;border:1px solid #705828;color:#a88048;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; Financing &amp; DSCR</span>
    <span style="background:#1a0a1e;border:1px solid #603880;color:#9868b8;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; Depreciation Tax Shield</span>
    <span style="background:#1e0a0a;border:1px solid #802828;color:#b85858;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; Opportunity Zone Benefits</span>
    <span style="background:#0a0e1e;border:1px solid #284870;color:#4878a8;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; Climate &amp; Risk Analysis</span>
    <span style="background:#1a1208;border:1px solid #705020;color:#a87838;font-size:0.74rem;
                 padding:4px 10px;border-radius:12px;letter-spacing:0.05em;">&#10003; AI Investment Rationale</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Session state for advisor ─────────────────────────────────────────────
    if "adv_result" not in st.session_state:
        st.session_state.adv_result = None
    if "adv_show_followup" not in st.session_state:
        st.session_state.adv_show_followup = False
    if "adv_parsed" not in st.session_state:
        st.session_state.adv_parsed = None

    # Pre-populate text area with home/chat bar query if routed here
    _home_prompt = st.session_state.get("adv_home_prompt")
    if _home_prompt and not st.session_state.get("adv_prompt_text"):
        st.session_state["adv_prompt_text"] = _home_prompt

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#16140a;border:1px solid #3a3020;border-radius:8px;
            padding:20px 24px 8px 24px;margin-bottom:8px;">
  <div style="color:#d4a843;font-weight:600;font-size:0.95rem;margin-bottom:10px;">
    Describe your investment
  </div>
""", unsafe_allow_html=True)

    prompt_input = st.text_area(
        label="prompt",
        label_visibility="collapsed",
        placeholder='e.g. "I want to build a 50,000 sq ft warehouse in southern Texas with a $8M budget over a 5-year hold"',
        height=90,
        key="adv_prompt_text",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    col_parse, col_generate, col_regen = st.columns([1.5, 1.8, 1], gap="small")

    with col_parse:
        do_parse = st.button("Analyze Prompt", key="adv_btn_parse", use_container_width=True)

    if do_parse and prompt_input.strip():
        try:
            parsed = parse_prompt(prompt_input.strip())
            st.session_state.adv_parsed = parsed
            st.session_state.adv_show_followup = bool(parsed["missing_fields"])
            st.session_state.adv_result = None
            if not parsed.get("missing_fields"):
                st.success("Prompt analyzed — all fields detected. Click **Generate Recommendation** to proceed.")
            else:
                st.info(f"Prompt analyzed — {len(parsed['missing_fields'])} field(s) need clarification below.")
        except Exception as _pe:
            st.error(f"Could not parse prompt: {_pe}")

    # ── Follow-up inputs for missing fields ───────────────────────────────────
    followup_values = {}
    if st.session_state.adv_parsed and st.session_state.adv_show_followup:
        missing = st.session_state.adv_parsed.get("missing_fields", [])
        if missing:
            st.markdown("""
<div style="background:#1a1208;border-left:3px solid #d4a843;border-radius:4px;
            padding:12px 18px;margin:12px 0 8px 0;color:#e8dfc4;font-size:0.9rem;">
  A few details were not found in your prompt. Fill in the fields below to improve accuracy
  (or leave blank to use platform defaults).
</div>
""", unsafe_allow_html=True)
            fu_cols = st.columns(min(len(missing), 4))
            for i, field in enumerate(missing):
                col = fu_cols[i % len(fu_cols)]
                if field == "property_type":
                    prop_options = sorted(set(_PROP_SYNONYMS.values()))
                    sel = col.selectbox("Property Type", [""] + prop_options, key="adv_fu_pt")
                    followup_values["property_type"] = sel or None
                elif field == "location":
                    followup_values["location_raw"] = col.text_input(
                        "Target Market / Region", key="adv_fu_loc",
                        placeholder="e.g. Texas, Southeast, Phoenix"
                    ) or None
                elif field == "budget":
                    raw_budget = col.text_input(
                        "Total Budget ($)", key="adv_fu_budget",
                        placeholder="e.g. 8M or 8000000"
                    )
                    if raw_budget:
                        try:
                            v = raw_budget.strip().lower().replace(",", "")
                            if v.endswith("m"):   v = float(v[:-1]) * 1_000_000
                            elif v.endswith("b"): v = float(v[:-1]) * 1_000_000_000
                            elif v.endswith("k"): v = float(v[:-1]) * 1_000
                            else:                 v = float(v)
                            followup_values["budget"] = v
                        except Exception:
                            followup_values["budget"] = None
                elif field == "sqft":
                    raw_sf = col.text_input(
                        "Square Footage", key="adv_fu_sqft",
                        placeholder="e.g. 50000"
                    )
                    if raw_sf:
                        try:
                            followup_values["sqft"] = float(raw_sf.replace(",", ""))
                        except Exception:
                            followup_values["sqft"] = None
                elif field == "timeline_years":
                    followup_values["timeline_years"] = col.number_input(
                        "Hold Period (years)", min_value=1, max_value=30,
                        value=5, key="adv_fu_timeline"
                    )

    # ── Generate / Regenerate buttons ─────────────────────────────────────────
    with col_generate:
        do_generate = st.button("Generate Recommendation", key="adv_btn_gen",
                                type="primary", use_container_width=True)

    with col_regen:
        do_regen = st.button("Regenerate", key="adv_btn_regen", use_container_width=True,
                             disabled=(st.session_state.adv_result is None))

    def _run_advisor(prompt_text, override_fields):
        parsed = parse_prompt(prompt_text)
        for k, v in override_fields.items():
            if v is not None:
                parsed[k] = v
        if not parsed.get("property_type"):  parsed["property_type"]  = "Industrial"
        if not parsed.get("location_raw"):   parsed["location_raw"]   = "sunbelt"
        if not parsed.get("budget"):         parsed["budget"]         = 10_000_000
        if not parsed.get("sqft"):           parsed["sqft"]           = 50_000
        if not parsed.get("timeline_years"): parsed["timeline_years"] = 5
        if not parsed.get("risk_tolerance"): parsed["risk_tolerance"] = "moderate"
        return build_recommendation(parsed)

    # Pick up auto-generate flag set by chat bar routing
    _auto_gen = st.session_state.get("adv_auto_generate", False)
    if _auto_gen:
        st.session_state.adv_auto_generate = False   # consume flag

    trigger_generate = do_generate or do_regen or _auto_gen

    _effective_prompt = prompt_input.strip() or (st.session_state.get("adv_home_prompt") or "").strip()

    if trigger_generate and _effective_prompt:
        # Clear the home prompt so it doesn't re-fire on next rerun
        st.session_state.adv_home_prompt = None
        with st.spinner("Analyzing markets and building your investment brief…"):
            try:
                result = _run_advisor(_effective_prompt, followup_values)
                st.session_state.adv_result = result
                st.session_state.adv_show_followup = False
            except Exception as _adv_err:
                st.error(f"Error generating recommendation: {_adv_err}")

    # ══════════════════════════════════════════════════════════════════════════
    #  REPORT OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    _adv_result = st.session_state.adv_result
    if _adv_result and "error" not in _adv_result:
        primary    = _adv_result["primary"]
        runners    = _adv_result["runners"]
        financials = _adv_result["financials"]
        weights    = _adv_result["weights"]
        narrative  = _adv_result["narrative"]
        params     = _adv_result["params"]
        all_scored = _adv_result.get("all_scored", [])

        # ── Parsed-params banner ─────────────────────────────────────────────
        _prop_type = params.get("property_type", "")
        _location  = params.get("location_raw", "")
        _timeline  = params.get("timeline_years", "N/A")
        _risk_tol  = (params.get("risk_tolerance") or "moderate").title()
        _budget_m  = f"${params['budget']/1_000_000:.1f}M" if params.get("budget") else "N/A"
        _sqft_k    = f"{params['sqft']/1000:.0f}K sq ft" if params.get("sqft") else "N/A"

        st.markdown(f"""
<div style="background:#16140a;border:1px solid #3a3020;border-radius:6px;
            padding:10px 18px;margin:8px 0 20px 0;font-size:0.88rem;color:#a09880;">
  <span style="color:#d4a843;font-weight:600;">Parsed: </span>
  {_prop_type} &nbsp;&middot;&nbsp; {_location or 'All Markets'} &nbsp;&middot;&nbsp;
  {_budget_m} budget &nbsp;&middot;&nbsp; {_sqft_k} &nbsp;&middot;&nbsp;
  {_timeline}-yr hold &nbsp;&middot;&nbsp; {_risk_tol} risk
  <span style="float:right;color:#5a5040;">Generated {_adv_result['generated_at'][:16].replace('T', ' ')} UTC</span>
</div>
""", unsafe_allow_html=True)

        # ── Summary metric cards ─────────────────────────────────────────────
        section(" Summary")
        _c1, _c2, _c3, _c4, _c5 = st.columns(5)

        def _adv_score_color(s):
            if s >= 75: return "#4caf50"
            if s >= 55: return "#ff9800"
            if s >= 35: return "#f44336"
            return "#9e9e9e"

        _opp = primary["opportunity_score"]
        _roi = financials["roi_pct"]
        _roi_color = "#4caf50" if _roi > 30 else ("#ff9800" if _roi > 10 else "#f44336")

        for _col, (_lbl, _val, _clr) in zip(
            [_c1, _c2, _c3, _c4, _c5],
            [
                ("Opportunity Score",  f"{_opp:.1f}/100",                             _adv_score_color(_opp)),
                ("Est. Total Cost",    f"${financials['total_cost']/1e6:.2f}M",         "#e8dfc4"),
                ("Estimated ROI",      f"{_roi}%",                                     _roi_color),
                ("Buildout Timeline",  f"{financials['buildout_months']} mo",           "#e8dfc4"),
                ("Est. Exit Value",    f"${financials['exit_value']/1e6:.2f}M",          "#d4a843"),
            ]
        ):
            _col.markdown(
                f'<div class="metric-card"><div class="label">{_lbl}</div>'
                f'<div class="value" style="color:{_clr};font-size:1.5rem;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Primary recommendation ────────────────────────────────────────────
        section(f" Primary Recommendation — {primary['market']}")
        _col_left, _col_right = st.columns([1.05, 0.95], gap="large")

        with _col_left:
            _bd         = primary.get("factor_scores", {})
            _fnames     = list(_bd.keys())
            _raw_s      = [_bd[f]["raw_score"] for f in _fnames]
            _wt_s       = [_bd[f]["weighted"]   for f in _fnames]
            _flabels    = [f.replace("_", " ").title() for f in _fnames]

            _fig_score = go.Figure()
            _fig_score.add_trace(go.Bar(
                x=_raw_s, y=_flabels, orientation="h",
                marker=dict(color=[_adv_score_color(s) for s in _raw_s]),
                text=[f"{s:.0f}" for s in _raw_s],
                textposition="auto",
                textfont=dict(color="#fff", size=11),
                name="Raw Score",
                hovertemplate="<b>%{y}</b><br>Raw: %{x:.1f}/100<extra></extra>",
            ))
            _fig_score.add_trace(go.Bar(
                x=_wt_s, y=_flabels, orientation="h",
                marker=dict(color="rgba(212,168,67,0.35)", line=dict(color="#d4a843", width=1)),
                text=[f"{v:.1f} wt" for v in _wt_s],
                textposition="auto",
                textfont=dict(color="#d4a843", size=10),
                name="Weighted Score",
                hovertemplate="<b>%{y}</b><br>Weighted: %{x:.1f}<extra></extra>",
            ))
            _fig_score.update_layout(
                barmode="overlay",
                plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                margin=dict(t=10, b=10, l=120, r=10),
                height=280,
                legend=dict(font=dict(color="#a09880", size=10), bgcolor="rgba(0,0,0,0)",
                            orientation="h", y=1.08, x=0),
                xaxis=dict(range=[0, 100], tickfont=dict(color="#e8dfc4", size=10), gridcolor="#2a2a1a"),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=11)),
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
            )
            st.plotly_chart(_fig_score, use_container_width=True, config={"displayModeBar": False})
            st.caption("Bar length = raw factor score (0–100). Gold overlay = weighted contribution to composite.")

        with _col_right:
            _cap   = primary.get("cap_rate", 0)
            _rg    = primary.get("rent_growth", 0)
            _clbl  = primary.get("climate_label", "N/A")
            _cscr  = primary.get("climate_score", 0)
            _gdp   = primary.get("gdp_cycle", "N/A")
            _cred  = primary.get("credit_signal", "N/A")
            _mig   = primary.get("mig_score", 50)
            _cc    = ("#4caf50" if _cscr < 25 else "#ff9800" if _cscr < 50
                      else "#f44336" if _cscr < 75 else "#9c27b0")
            _rg_c  = "#4caf50" if _rg > 3 else ("#ff9800" if _rg > 0 else "#f44336")

            # Climate score penalty (matches market_score_agent formula)
            _clim_penalty = round(min(10.0, (_cscr - 60) * 0.20), 1) if _cscr >= 60 else 0.0
            _clim_penalty_row = ""
            if _clim_penalty > 0:
                _clim_penalty_row = (
                    f'<tr><td style="color:#ef5350;padding:5px 0;font-size:0.82rem;">Score Adj. (climate)</td>'
                    f'<td style="color:#ef5350;font-weight:700;text-align:right;font-size:0.82rem;">−{_clim_penalty:.1f} pts</td></tr>'
                )

            st.markdown(f"""
<div style="background:#1a1208;border:1px solid #2a2010;border-radius:8px;padding:18px 20px;">
  <div style="color:#d4a843;font-weight:600;font-size:0.95rem;margin-bottom:12px;">Market Signals</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
    <tr><td style="color:#a09880;padding:5px 0;">Cap Rate</td>
        <td style="color:#e8dfc4;font-weight:600;text-align:right;">{_cap:.2f}%</td></tr>
    <tr><td style="color:#a09880;padding:5px 0;">Rent Growth YoY</td>
        <td style="color:{_rg_c};font-weight:600;text-align:right;">{_rg:+.1f}%</td></tr>
    <tr><td style="color:#a09880;padding:5px 0;">Climate Risk</td>
        <td style="color:{_cc};font-weight:600;text-align:right;">{_clbl} ({_cscr:.0f}/100)</td></tr>
    {_clim_penalty_row}
    <tr><td style="color:#a09880;padding:5px 0;">GDP Cycle</td>
        <td style="color:#e8dfc4;font-weight:600;text-align:right;">{_gdp.title()}</td></tr>
    <tr><td style="color:#a09880;padding:5px 0;">Credit Conditions</td>
        <td style="color:#e8dfc4;font-weight:600;text-align:right;">{_cred.title()}</td></tr>
    <tr><td style="color:#a09880;padding:5px 0;">Migration Score</td>
        <td style="color:#e8dfc4;font-weight:600;text-align:right;">{_mig:.0f}/100</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Financials breakdown ──────────────────────────────────────────────
        section(" Financial Estimates")

        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        for _fc, (_lbl, _val, _clr) in zip(
            [_fc1, _fc2, _fc3, _fc4],
            [
                ("Land Cost",          f"${financials['land_cost']/1e6:.2f}M",         "#e8dfc4"),
                ("Construction",       f"${financials['construction_cost']/1e6:.2f}M", "#e8dfc4"),
                ("Soft Costs",         f"${financials['soft_costs']/1e6:.2f}M",         "#e8dfc4"),
                ("Total Project Cost", f"${financials['total_cost']/1e6:.2f}M",         "#d4a843"),
            ]
        ):
            _fc.markdown(
                f'<div class="metric-card"><div class="label">{_lbl}</div>'
                f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
        _fc5, _fc6, _fc7, _fc8 = st.columns(4)
        _irr_c = "#4caf50" if financials["irr_est"] > 10 else "#ff9800"
        _pft_c = "#4caf50" if financials["total_profit"] > 0 else "#f44336"
        for _fc, (_lbl, _val, _clr) in zip(
            [_fc5, _fc6, _fc7, _fc8],
            [
                ("Annual NOI",     f"${financials['annual_noi']/1e3:.0f}K",     "#e8dfc4"),
                ("Cumulative NOI", f"${financials['total_noi']/1e6:.2f}M",       "#e8dfc4"),
                ("Estimated IRR",  f"{financials['irr_est']:.1f}%",              _irr_c),
                ("Total Profit",   f"${financials['total_profit']/1e6:.2f}M",    _pft_c),
            ]
        ):
            _fc.markdown(
                f'<div class="metric-card"><div class="label">{_lbl}</div>'
                f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        _esig = financials.get("energy_signal", "MODERATE")
        _emult = {"LOW": "0.88×", "MODERATE": "1.0×", "HIGH": "1.20×"}.get(_esig, "1.0×")
        st.caption(
            f"Construction cost signal: **{_esig}** (platform energy agent). "
            f"Cost multiplier: {_emult}. Buildout estimate: {financials['buildout_months']} months."
        )

        # ── Financing Structure ───────────────────────────────────────────────
        financing = res.get("financing", {})
        if financing:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Financing Structure")
            _fn1, _fn2, _fn3, _fn4, _fn5 = st.columns(5)
            _dscr_c = "#4caf50" if financing.get("dscr", 0) >= 1.25 else ("#ff9800" if financing.get("dscr", 0) >= 1.0 else "#f44336")
            _coc_c  = "#4caf50" if financing.get("cash_on_cash_pct", 0) >= 6 else "#ff9800"
            _irrl_c = "#4caf50" if financing.get("leveraged_irr_pct", 0) >= 15 else "#ff9800"
            for _fc, (_lbl, _val, _clr) in zip(
                [_fn1, _fn2, _fn3, _fn4, _fn5],
                [
                    ("LTV",               f"{financing['ltv_pct']:.0f}%",                         "#e8dfc4"),
                    ("Loan Amount",        f"${financing['loan_amount']/1e6:.2f}M",                "#e8dfc4"),
                    ("Equity Required",    f"${financing['equity_required']/1e6:.2f}M",            "#d4a843"),
                    ("Annual Debt Service",f"${financing['annual_debt_service']/1e3:.0f}K",        "#e8dfc4"),
                    ("DSCR",               f"{financing['dscr']:.2f}x",                            _dscr_c),
                ]
            ):
                _fc.markdown(
                    f'<div class="metric-card"><div class="label">{_lbl}</div>'
                    f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
            _fn6, _fn7, _fn8, _fn9 = st.columns(4)
            for _fc, (_lbl, _val, _clr) in zip(
                [_fn6, _fn7, _fn8, _fn9],
                [
                    ("Loan Rate",          f"{financing['loan_rate_pct']:.2f}% / {financing['amort_years']}yr", "#e8dfc4"),
                    ("Cash Flow After DS", f"${financing['cash_flow_after_ds']/1e3:.0f}K/yr",                  _coc_c),
                    ("Cash-on-Cash",       f"{financing['cash_on_cash_pct']:.1f}%",                            _coc_c),
                    ("Leveraged IRR",      f"{financing['leveraged_irr_pct']:.1f}%",                           _irrl_c),
                ]
            ):
                _fc.markdown(
                    f'<div class="metric-card"><div class="label">{_lbl}</div>'
                    f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── 10-Year P&L Pro Forma ─────────────────────────────────────────────
        proforma = res.get("proforma", [])
        if proforma:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" 10-Year P&L Pro Forma")
            _pf_hdr = ["Year","Gross Revenue","Vacancy Loss","EGI","Operating Exp.","NOI","Debt Service","Cash Flow","Cumulative CF"]
            _pf_keys = ["year","gross_revenue","vacancy_loss","egi","opex","noi","debt_service","cf_after_ds","cum_cf"]
            _pf_aligns = ["center"] + ["right"] * 8

            def _pf_fmt(k, v):
                if k == "year": return str(v)
                return f"${v/1e3:.0f}K" if abs(v) < 1e6 else f"${v/1e6:.2f}M"

            _pf_hcells = "".join(
                f'<th style="padding:8px 10px;color:#c8a040;font-size:0.72rem;font-weight:700;'
                f'letter-spacing:0.08em;text-align:{al};border-bottom:1px solid #2a2410;'
                f'white-space:nowrap;">{h}</th>'
                for h, al in zip(_pf_hdr, _pf_aligns)
            )
            _pf_rows_html = ""
            for _row in proforma:
                _noi_c  = "#4caf50" if _row["noi"] > 0 else "#f44336"
                _cf_c   = "#4caf50" if _row["cf_after_ds"] >= 0 else "#f44336"
                _cum_c  = "#4caf50" if _row["cum_cf"] >= 0 else "#f44336"
                _sep    = "border-bottom:1px solid #1a1808;"
                _cells  = ""
                for k, al in zip(_pf_keys, _pf_aligns):
                    v = _row[k]
                    fmt = _pf_fmt(k, v)
                    if k == "noi":
                        clr = _noi_c
                    elif k == "cf_after_ds":
                        clr = _cf_c
                    elif k == "cum_cf":
                        clr = _cum_c
                    elif k == "year":
                        clr = "#c8a040"
                    else:
                        clr = "#c8b890"
                    _cells += (f'<td style="padding:7px 10px;{_sep}text-align:{al};'
                               f'color:{clr};font-size:0.83rem;white-space:nowrap;">{fmt}</td>')
                _pf_rows_html += f"<tr>{_cells}</tr>"

            st.markdown(
                f'<div style="background:#13110a;border-radius:10px;padding:20px 24px;'
                f'margin-bottom:8px;overflow-x:auto;">'
                f'<table style="border-collapse:collapse;width:100%;font-family:\'Source Sans Pro\',sans-serif;">'
                f'<thead><tr>{_pf_hcells}</tr></thead>'
                f'<tbody>{_pf_rows_html}</tbody></table></div>',
                unsafe_allow_html=True,
            )
            st.caption("NOI = EGI − Operating Expenses. Cash Flow = NOI − Debt Service. "
                       "Year-1 shows lease-up vacancy premium. Rent grows at market rate; OpEx inflates 2.5%/yr.")

        # ── Tax & Depreciation Benefits ───────────────────────────────────────
        tax = res.get("tax_benefits", {})
        if tax:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Tax & Depreciation Benefits")

            _tx1, _tx2, _tx3, _tx4, _tx5 = st.columns(5)
            for _fc, (_lbl, _val, _clr) in zip(
                [_tx1, _tx2, _tx3, _tx4, _tx5],
                [
                    ("Building Value",        f"${tax['building_value']/1e6:.2f}M",      "#e8dfc4"),
                    ("Year-1 Depreciation",   f"${tax['yr1_depreciation']/1e3:.0f}K",    "#d4a843"),
                    ("Year-1 Tax Shield",     f"${tax['yr1_tax_shield']/1e3:.0f}K",      "#4caf50"),
                    ("Annual Tax Shield",     f"${tax['annual_tax_shield']/1e3:.0f}K/yr","#80c858"),
                    ("10-Yr Tax Savings",     f"${tax['cum10_tax_savings']/1e6:.2f}M",   "#4caf50"),
                ]
            ):
                _fc.markdown(
                    f'<div class="metric-card"><div class="label">{_lbl}</div>'
                    f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)

            # Depreciation breakdown table
            _depr_rows = [
                ("Personal Property (5-yr)", tax["personal_prop_value"], f"{tax['bonus_depr_pct']}% bonus + straight-line"),
                ("Land Improvements (15-yr)", tax["land_improv_value"],  f"{tax['bonus_depr_pct']}% bonus + straight-line"),
                (f"Structure ({tax['struct_depr_life']}-yr)",            tax["structure_value"],   "Straight-line"),
                ("Land (non-depreciable)",   tax["land_value"],          "N/A"),
            ]
            _depr_html = ""
            for _dn, _dv, _dm in _depr_rows:
                _bar_w = int(_dv / max(tax["building_value"] + tax["land_value"], 1) * 280)
                _depr_html += (
                    f'<tr>'
                    f'<td style="padding:8px 12px;color:#c8b890;font-size:0.84rem;border-bottom:1px solid #1e1c0e;">{_dn}</td>'
                    f'<td style="padding:8px 12px;text-align:right;color:#c8a040;font-size:0.84rem;border-bottom:1px solid #1e1c0e;">${_dv/1e6:.2f}M</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid #1e1c0e;">'
                    f'<div style="background:#2a2410;border-radius:3px;height:8px;width:300px;">'
                    f'<div style="background:#c8a040;border-radius:3px;height:8px;width:{_bar_w}px;"></div></div></td>'
                    f'<td style="padding:8px 12px;color:#7a7060;font-size:0.78rem;border-bottom:1px solid #1e1c0e;">{_dm}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="background:#13110a;border-radius:10px;padding:20px 24px;margin-bottom:8px;">'
                f'<table style="border-collapse:collapse;width:100%;">'
                f'<thead><tr>'
                f'<th style="padding:8px 12px;color:#c8a040;font-size:0.72rem;letter-spacing:0.08em;border-bottom:1px solid #2a2410;">COMPONENT</th>'
                f'<th style="padding:8px 12px;color:#c8a040;font-size:0.72rem;letter-spacing:0.08em;text-align:right;border-bottom:1px solid #2a2410;">VALUE</th>'
                f'<th style="padding:8px 12px;border-bottom:1px solid #2a2410;"></th>'
                f'<th style="padding:8px 12px;color:#c8a040;font-size:0.72rem;letter-spacing:0.08em;border-bottom:1px solid #2a2410;">METHOD</th>'
                f'</tr></thead><tbody>{_depr_html}</tbody></table></div>',
                unsafe_allow_html=True,
            )

            if tax.get("oz_eligible"):
                st.markdown(
                    f'<div style="background:#0a1e0a;border:1px solid #2a6030;border-left:4px solid #4caf50;'
                    f'border-radius:6px;padding:12px 18px;margin-top:4px;color:#80c858;font-size:0.88rem;">'
                    f'<strong>&#9733; Opportunity Zone</strong> &mdash; {tax["oz_note"]}</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Assumes {tax['tax_rate_pct']}% combined federal tax rate. "
                f"Cost-segregation allocation: 15% personal property (5-yr), 10% land improvements (15-yr), "
                f"75% structure ({tax['struct_depr_life']}-yr). "
                f"{tax['bonus_depr_pct']}% bonus depreciation applied (2025 schedule)."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Investment narrative ──────────────────────────────────────────────
        section(" Investment Rationale")
        for _para in narrative.strip().split("\n\n"):
            if _para.strip():
                st.markdown(
                    f'<div style="background:#16140a;border-left:3px solid #a07830;'
                    f'border-radius:4px;padding:14px 18px;margin-bottom:12px;'
                    f'color:#e8dfc4;font-size:0.93rem;line-height:1.65;">{_para.strip()}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Conditional: Climate risk detail if High/Severe ───────────────────
        if primary.get("climate_score", 0) >= 50:
            section(f" Climate Risk Alert — {primary['market']} ({primary['climate_label']})")
            _cf = primary.get("climate_factors", {})
            if _cf:
                _cf_cols = st.columns(len(_cf))
                _cf_display = {"flood": "Flood", "wildfire": "Wildfire",
                               "heat": "Extreme Heat", "wind": "Wind/Hurricane",
                               "sea_level": "Sea Level Rise"}
                for _cfc, (_fk, _fv) in zip(_cf_cols, _cf.items()):
                    _fc_color = ("#4caf50" if _fv < 25 else "#ff9800" if _fv < 50
                                 else "#f44336" if _fv < 75 else "#9c27b0")
                    _cfc.markdown(
                        f'<div class="metric-card"><div class="label">{_cf_display.get(_fk, _fk)}</div>'
                        f'<div class="value" style="color:{_fc_color};font-size:1.3rem;">{_fv:.0f}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.warning(
                f"{primary['market']} carries **{primary['climate_label'].upper()}** physical climate risk "
                f"(composite {primary['climate_score']:.0f}/100). "
                f"Factor higher insurance budgets, resilience design costs, and potential exit cap rate "
                f"expansion into your underwriting.",
                icon="⚠️",
            )
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Runner-up comparison ──────────────────────────────────────────────
        if runners:
            section(" Runner-Up Markets")
            _compare = [primary] + runners
            _cmp_rows = []
            for _i, _m in enumerate(_compare):
                _bd_m  = _m.get("factor_scores", {})
                _cscr2 = _m.get("climate_score", 0)
                _clim_pen2 = round(min(10.0, (_cscr2 - 60) * 0.20), 1) if _cscr2 >= 60 else 0.0
                _clim_str  = f"{_cscr2:.0f} ({_m.get('climate_label', 'N/A')})"
                if _clim_pen2 > 0:
                    _clim_str += f" −{_clim_pen2:.1f}pts"
                _cmp_rows.append({
                    "Rank":              "Primary" if _i == 0 else f"#{_i+1} Runner-Up",
                    "Market":            _m["market"],
                    "Opp. Score":        _m["opportunity_score"],
                    "Cap Rate":          f"{_m.get('cap_rate', 0):.2f}%",
                    "Rent Growth":       f"{_m.get('rent_growth', 0):+.1f}%",
                    "Climate Risk":      _clim_str,
                    "Mkt Fundamentals":  f"{_bd_m.get('market_fundamentals', {}).get('raw_score', 0):.0f}",
                    "Migration Score":   f"{_m.get('mig_score', 0):.0f}",
                })
            _cmp_df = pd.DataFrame(_cmp_rows)

            def _adv_style_rank(val):
                if val == "Primary": return "color:#d4a843;font-weight:700"
                return "color:#a09880"

            def _adv_style_climate(val):
                """Colour-code climate risk: green low, orange mid, red/purple high."""
                try:
                    score = float(str(val).split("(")[0].strip())
                except Exception:
                    return ""
                if score < 25:   return "color:#4caf50;font-weight:600"
                if score < 50:   return "color:#ff9800;font-weight:600"
                if score < 75:   return "color:#f44336;font-weight:600"
                return "color:#9c27b0;font-weight:700"

            st.dataframe(
                _cmp_df.style
                       .map(_adv_style_rank,    subset=["Rank"])
                       .map(_adv_style_climate, subset=["Climate Risk"]),
                use_container_width=True,
                hide_index=True,
            )

            # Composite score bar chart
            _cmp_names  = [_m["market"] for _m in _compare]
            _cmp_scores = [_m["opportunity_score"] for _m in _compare]
            _fig_cmp = go.Figure(go.Bar(
                x=_cmp_names,
                y=_cmp_scores,
                marker=dict(color=[_adv_score_color(s) for s in _cmp_scores]),
                text=[f"{s:.1f}" for s in _cmp_scores],
                textposition="outside",
                textfont=dict(color="#e8dfc4", size=12),
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}/100<extra></extra>",
            ))
            _fig_cmp.update_layout(
                plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                margin=dict(t=20, b=20, l=20, r=20),
                height=260,
                xaxis=dict(tickfont=dict(color="#e8dfc4", size=11), gridcolor="#2a2a1a"),
                yaxis=dict(range=[0, 110], tickfont=dict(color="#e8dfc4", size=10),
                           gridcolor="#2a2a1a", title="Opportunity Score"),
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
                showlegend=False,
            )
            st.plotly_chart(_fig_cmp, use_container_width=True, config={"displayModeBar": False})
            st.caption("Composite opportunity scores (0–100). Green ≥ 75, Orange 55–74, Red < 55.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── All scored markets (collapsed) ────────────────────────────────────
        if all_scored and len(all_scored) > 3:
            with st.expander(f"All {len(all_scored)} candidate markets scored"):
                _all_rows = []
                for _i, _m in enumerate(all_scored):
                    _all_rows.append({
                        "Rank":           _i + 1,
                        "Market":         _m["market"],
                        "Score":          _m["opportunity_score"],
                        "Cap Rate":       f"{_m.get('cap_rate', 0):.2f}%",
                        "Rent Growth":    f"{_m.get('rent_growth', 0):+.1f}%",
                        "Climate Score":  _m.get("climate_score", 0),
                        "Migration":      f"{_m.get('mig_score', 0):.0f}",
                    })
                _all_df = pd.DataFrame(_all_rows)
                def _adv_style_score(val):
                    if val >= 75: return "color:#4caf50;font-weight:600"
                    if val >= 55: return "color:#ff9800;font-weight:600"
                    if val >= 35: return "color:#f44336;font-weight:600"
                    return "color:#9e9e9e"
                st.dataframe(
                    _all_df.style.map(_adv_style_score, subset=["Score"]),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

        # ── Methodology & weights ─────────────────────────────────────────────
        with st.expander("Scoring Methodology & Factor Weights"):
            _meth   = weights.get("methodology", "")
            _wsrc   = "AI-determined (Groq)" if weights.get("source") == "groq" else "Platform default"
            st.markdown(
                f'<div style="background:#16140a;border-left:3px solid #d4a843;border-radius:4px;'
                f'padding:12px 16px;margin-bottom:16px;color:#e8dfc4;font-size:0.9rem;">'
                f'<b>Weight source:</b> {_wsrc}<br>'
                f'<b>Methodology:</b> {_meth}</div>',
                unsafe_allow_html=True,
            )
            _wt_data  = weights.get("weights", {})
            _rat_data = weights.get("rationales", {})
            if _wt_data:
                _wt_rows = []
                for _f, _w in sorted(_wt_data.items(), key=lambda x: -x[1]):
                    _bd_f = primary.get("factor_scores", {}).get(_f, {})
                    _wt_rows.append({
                        "Factor":         _f.replace("_", " ").title(),
                        "Weight":         f"{_w*100:.0f}%",
                        "Raw Score":      f"{_bd_f.get('raw_score', 0):.1f}/100",
                        "Contribution":   f"{_bd_f.get('weighted', 0):.1f} pts",
                        "Rationale":      _rat_data.get(_f, ""),
                    })
                st.dataframe(pd.DataFrame(_wt_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

    elif _adv_result and "error" in _adv_result:
        st.error(_adv_result["error"])

    else:
        # Idle state — example prompts
        st.markdown("""
<div style="background:#16140a;border:1px solid #2a2010;border-radius:8px;
            padding:24px 28px;margin-top:12px;">
  <div style="color:#d4a843;font-weight:600;margin-bottom:14px;">Example prompts to get started:</div>
  <ul style="color:#a09880;font-size:0.92rem;line-height:2.0;padding-left:20px;margin:0;">
    <li>"I want to build a 50,000 sq ft warehouse in southern Texas with an $8M budget over a 5-year hold."</li>
    <li>"Looking for multifamily development in the Southeast. $15M budget, 7-year hold, moderate risk."</li>
    <li>"Office development, 20,000 sq ft, $12M, in the Sunbelt &mdash; where should I build?"</li>
    <li>"Conservative industrial play in the Midwest, $5M, 3 years."</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TAB — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with main_tab_about:
    tab_about_team, tab_about_monitor = st.tabs(["Meet the Team", "System Monitor"])

    # ── Meet the Team ─────────────────────────────────────────────────────────
    with tab_about_team:
        import base64 as _b64

        # ── Team config — drop a photo file in app/assets/team/ named exactly
        #    as the "photo" field below (e.g. aayman.jpg) and it auto-appears.
        #    Supported formats: jpg, jpeg, png, webp
        _TEAM = [
            {
                "name":     "Aayman Afzal",
                "role":     "MSF Candidate",
                "linkedin": "https://www.linkedin.com/in/aayman-afzal",
                "photo":    "aayman.jpg",
            },
            {
                "name":     "Ajinkya Kodnikar",
                "role":     "MSF Candidate",
                "linkedin": "https://www.linkedin.com/in/ajinkyakodnikar",
                "photo":    "ajinkya.jpg",
            },
            {
                "name":     "Oyu Amar",
                "role":     "MSF Candidate",
                "linkedin": "https://www.linkedin.com/in/oyu-amar/",
                "photo":    "oyu.jpg",
            },
            {
                "name":     "Ricardo Ruiz",
                "role":     "MSF Candidate",
                "linkedin": "https://www.linkedin.com/in/ricardo-ruiz1",
                "photo":    "ricardo.jpg",
            },
        ]

        from pathlib import Path as _Path
        _ASSETS_DIR = _Path(__file__).parent / "assets" / "team"

        def _photo_html(filename: str) -> str:
            """Return an <img> tag if the photo file exists, else a fallback avatar."""
            for ext in [filename, filename.replace(".jpg", ".jpeg"),
                        filename.replace(".jpg", ".png"), filename.replace(".jpg", ".webp")]:
                p = _ASSETS_DIR / ext
                if p.exists():
                    mime = "image/jpeg" if ext.endswith((".jpg", ".jpeg")) else (
                           "image/png" if ext.endswith(".png") else "image/webp")
                    data = _b64.b64encode(p.read_bytes()).decode()
                    return (
                        f'<img src="data:{mime};base64,{data}" '
                        f'style="width:96px;height:96px;border-radius:50%;'
                        f'object-fit:cover;border:2px solid #a07830;margin-bottom:10px;" />'
                    )
            return '<div style="font-size:3rem;margin-bottom:10px;">&#128100;</div>'

        st.markdown("""
<div style="background:linear-gradient(135deg,#1a1208 0%,#2a1e08 100%);
            border:1px solid #a07830; border-top:3px solid #d4a843;
            border-radius:10px; padding:28px 36px; margin-bottom:28px;">
  <div style="color:#d4a843;font-size:1.45rem;font-weight:700;letter-spacing:1px;">
    CRE Intelligence Platform
  </div>
  <div style="color:#a09880;font-size:0.92rem;margin-top:6px;max-width:720px;">
    Built by the Purdue Daniels School of Business MSF cohort for MGMT 690: AI Leadership.
    A real-time commercial real estate intelligence system powered by 20 background agents,
    live market data, and AI-driven investment analysis.
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
  <span style="color:#d4a843; font-size:1.3rem; font-weight:700; letter-spacing:2px;
               text-transform:uppercase;">Meet the Team</span>
  <div style="color:#a09880; font-size:0.85rem; margin-top:6px;">
    MGMT 690: AI Leadership &nbsp;&middot;&nbsp; Purdue Daniels School of Business &nbsp;&middot;&nbsp; MSF Program
  </div>
</div>
""", unsafe_allow_html=True)

        # Render each card in its own column so HTML is isolated per st.markdown call
        _tm_cols = st.columns(len(_TEAM), gap="medium")
        for _col, _tm in zip(_tm_cols, _TEAM):
            _photo = _photo_html(_tm["photo"])
            _col.markdown(
                f'<div style="background:#1e1a0a;border:1px solid #a07830;border-radius:8px;'
                f'padding:24px 16px;text-align:center;">'
                f'{_photo}'
                f'<div style="color:#e8dfc4;font-weight:700;font-size:0.95rem;margin-bottom:4px;">{_tm["name"]}</div>'
                f'<div style="color:#6a5228;font-size:0.75rem;margin-bottom:10px;">{_tm["role"]}</div>'
                f'<a href="{_tm["linkedin"]}" target="_blank" '
                f'style="color:#d4a843;font-size:0.8rem;text-decoration:none;">&#128279; LinkedIn</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section(" Platform Overview")
        _ov_cols = st.columns(3)
        _ov_cols[0].markdown(metric_card("Background Agents", "20", "Auto-updating data sources"), unsafe_allow_html=True)
        _ov_cols[1].markdown(metric_card("Data Sources", "30+", "APIs, RSS feeds, government data"), unsafe_allow_html=True)
        _ov_cols[2].markdown(metric_card("Update Frequency", "30 min", "Fastest agent refresh cycle"), unsafe_allow_html=True)

        st.markdown("""
<div style="background:#16140a;border:1px solid #2a2208;border-radius:8px;
            padding:20px 24px;margin-top:16px;color:#a09880;font-size:0.9rem;line-height:1.8;">
  <div style="color:#d4a843;font-weight:600;margin-bottom:10px;">What This Platform Does</div>
  The CRE Intelligence Platform continuously monitors 20+ commercial real estate market signals —
  population migration, REIT pricing, interest rates, labor markets, inflation, GDP, credit conditions,
  cap rates, rent growth, vacancy, climate risk, and more — and synthesizes them into actionable
  investment intelligence. The AI Investment Advisor combines all live data to score every US metro
  market and generate personalized investment briefs in plain English.
</div>
""", unsafe_allow_html=True)

    # ── System Monitor ────────────────────────────────────────────────────────
    with tab_about_monitor:
        st.markdown("#### Background Agent Monitor")
        st.markdown(
            "20 agents run continuously in background threads, writing to JSON cache files. "
            "Data survives Streamlit reruns. Agents restart automatically if the app restarts."
        )

        section(" Agent Status")
        _about_status = get_status()

        _about_agent_labels = {
            "migration":       ("Agent 1",  "Population & Migration",    "Every 6h"),
            "pricing":         ("Agent 2",  "REIT Pricing",              "Every 1h"),
            "predictions":     ("Agent 3",  "Company Predictions",       "Every 24h"),
            "debugger":        ("Agent 4",  "Debugger / Monitor",        "Every 30min"),
            "news":            ("Agent 5",  "Industry Announcements",    "Every 4h"),
            "rates":           ("Agent 6",  "Interest Rate & Debt",      "Every 1h"),
            "energy":          ("Agent 7",  "Energy & Construction",     "Every 6h"),
            "sustainability":  ("Agent 8",  "Sustainability & ESG",      "Every 6h"),
            "labor_market":    ("Agent 9",  "Labor Market & Demand",     "Every 6h"),
            "gdp":             ("Agent 10", "GDP & Economic Growth",     "Every 6h"),
            "inflation":       ("Agent 11", "Inflation Monitor",         "Every 6h"),
            "credit":          ("Agent 12", "Credit & Capital Markets",  "Every 6h"),
            "vacancy":         ("Agent 13", "Vacancy Monitor",           "Every 12h"),
            "climate_risk":    ("Agent 14", "Climate Risk",              "Every 24h"),
            "cap_rate":        ("Agent 15", "Cap Rate Monitor",          "Every 6h"),
            "rent_growth":     ("Agent 16", "Rent Growth",               "Every 6h"),
            "land_market":     ("Agent 17", "Land & Development",        "Every 12h"),
            "opportunity_zone":("Agent 18", "Opportunity Zones",         "Every 24h"),
            "distressed":      ("Agent 19", "CMBS & Distressed",         "Every 6h"),
            "market_score":    ("Agent 20", "Market Score Composite",    "Every 6h"),
            "manager":         ("Manager",  "System Health Supervisor",  "Every 15min"),
        }

        _cache_key_map = {
            "credit": "credit_data", "energy": "energy_data",
            "gdp": "gdp_data", "inflation": "inflation_data",
            "sustainability": "sustainability_data",
            "manager": "manager_report",
        }

        _about_rows = []
        for _ak, (_anum, _aname, _afreq) in _about_agent_labels.items():
            _as  = _about_status.get(_ak, {})
            _mem_status = _as.get("status", "")   # in-memory (resets on restart)
            _ar  = _as.get("runs", 0)
            _ae  = _as.get("last_error") or ""
            _ck  = _cache_key_map.get(_ak, _ak)
            _cache_age = cache_age_label(_ck)
            _cc  = read_cache(_ck)
            _has_data = _cc.get("data") is not None
            _is_stale = _cc.get("stale", False)

            # Derive status: cache truth takes precedence over stale in-memory flags
            if _mem_status == "running":
                _ast = "RUNNING"                    # always trust live running signal
            elif _has_data and not _is_stale:
                _ast = "OK"                         # fresh cache = healthy regardless of past errors
            elif _mem_status == "error" and (_is_stale or not _has_data):
                _ast = "ERROR"                      # error + bad cache = real problem
            elif _has_data and _is_stale:
                _ast = "STALE"
            else:
                _ast = "MISSING"

            _about_rows.append({
                "key":      _ak,
                "label":    _aname,
                "num":      _anum,
                "name":     _aname,
                "schedule": _afreq,
                "status":   _ast,
                "runs":     _ar,
                "cache_age":_cache_age,
                "has_data": _has_data,
                "error":    _ae[:60] if _ae else "",
            })

        # ── KPI summary strip ──────────────────────────────────────────────────
        _n_total   = len(_about_rows)
        _n_ok      = sum(1 for r in _about_rows if r["status"] in ("OK", "RUNNING"))
        _n_error   = sum(1 for r in _about_rows if r["status"] == "ERROR")
        _n_cached  = sum(1 for r in _about_rows if r["has_data"])
        _health_pct = round(_n_cached / _n_total * 100)

        _kpi_cols = st.columns(4)
        for _kc, (_kl, _kv, _kc2) in zip(_kpi_cols, [
            ("Total Agents",    str(_n_total),              "#c8a040"),
            ("Active / OK",     str(_n_ok),                 "#4caf50"),
            ("Errors",          str(_n_error),              "#f44336" if _n_error else "#4caf50"),
            ("Cache Health",    f"{_health_pct}%",          "#4caf50" if _health_pct >= 80 else "#ff9800"),
        ]):
            _kc.markdown(
                f'<div class="metric-card"><div class="label">{_kl}</div>'
                f'<div class="value" style="color:{_kc2};font-size:1.6rem;font-weight:700;">{_kv}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart row: runs bar + status donut ────────────────────────────────
        _ch_left, _ch_right = st.columns([2, 1], gap="large")

        with _ch_left:
            _status_color_map = {
                "OK":      "#4caf50",
                "RUNNING": "#ff9800",
                "ERROR":   "#f44336",
                "STALE":   "#d4a843",
                "MISSING": "#f44336",
                "IDLE":    "#555544",
            }
            _bar_labels  = [r["label"] for r in _about_rows]
            _bar_runs    = [max(r["runs"], 1) for r in _about_rows]   # min 1 so bar is visible
            _bar_colors  = [_status_color_map.get(r["status"], "#555544") for r in _about_rows]
            _bar_hover   = [
                f"<b>{r['label']}</b><br>Runs: {r['runs']}<br>Status: {r['status']}<br>Schedule: {r['schedule']}<br>Cache: {r['cache_age']}<extra></extra>"
                for r in _about_rows
            ]

            _fig_runs = go.Figure(go.Bar(
                y=_bar_labels,
                x=_bar_runs,
                orientation="h",
                marker=dict(color=_bar_colors, opacity=0.88),
                text=[str(r["runs"]) for r in _about_rows],
                textposition="outside",
                textfont=dict(color="#c8b890", size=10),
                hovertemplate=_bar_hover,
            ))
            _fig_runs.update_layout(
                title=dict(text="Agent Run Count", font=dict(color="#c8a040", size=13), x=0),
                plot_bgcolor="#0d0b04", paper_bgcolor="#13110a",
                margin=dict(t=36, b=20, l=200, r=60),
                height=560,
                xaxis=dict(
                    title="Total Runs Since Last Restart",
                    title_font=dict(color="#7a7050", size=11),
                    tickfont=dict(color="#c8b890", size=10),
                    gridcolor="#1e1c0e",
                ),
                yaxis=dict(tickfont=dict(color="#c8b890", size=10), autorange="reversed"),
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(_fig_runs, use_container_width=True, config={"displayModeBar": False})

        with _ch_right:
            _status_counts = {"OK": 0, "RUNNING": 0, "STALE": 0, "ERROR": 0, "MISSING": 0}
            for r in _about_rows:
                _s = r["status"] if r["status"] in _status_counts else "OK"
                _status_counts[_s] += 1
            # Drop zero-count entries so donut isn't cluttered
            _status_counts = {k: v for k, v in _status_counts.items() if v > 0}

            _donut_labels = list(_status_counts.keys())
            _donut_vals   = list(_status_counts.values())
            _donut_colors = [_status_color_map[s] for s in _donut_labels]

            _fig_donut = go.Figure(go.Pie(
                labels=_donut_labels,
                values=_donut_vals,
                hole=0.62,
                marker=dict(colors=_donut_colors, line=dict(color="#0d0b04", width=2)),
                textfont=dict(color="#c8b890", size=12),
                hovertemplate="<b>%{label}</b><br>%{value} agents<br>%{percent}<extra></extra>",
            ))
            _fig_donut.add_annotation(
                text=f"<b>{_health_pct}%</b><br><span style='font-size:10px'>Healthy</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#c8a040", size=16),
            )
            _fig_donut.update_layout(
                title=dict(text="Status Breakdown", font=dict(color="#c8a040", size=13), x=0),
                plot_bgcolor="#0d0b04", paper_bgcolor="#13110a",
                margin=dict(t=36, b=20, l=10, r=10),
                height=300,
                legend=dict(font=dict(color="#c8b890", size=11), bgcolor="rgba(0,0,0,0)",
                            orientation="h", y=-0.08),
                font=dict(family="Source Sans Pro"),
            )
            st.plotly_chart(_fig_donut, use_container_width=True, config={"displayModeBar": False})

            # ── Cache data presence bar ────────────────────────────────────────
            _cache_labels  = [r["num"] for r in _about_rows]
            _cache_present = [1 if r["has_data"] else 0 for r in _about_rows]
            _cache_colors  = ["#4caf50" if v else "#f44336" for v in _cache_present]

            _fig_cache = go.Figure(go.Bar(
                x=_cache_labels,
                y=_cache_present,
                marker=dict(color=_cache_colors, opacity=0.85),
                hovertemplate=[
                    f"<b>{r['label']}</b><br>Cache: {'OK' if r['has_data'] else 'MISSING'}<br>{r['cache_age']}<extra></extra>"
                    for r in _about_rows
                ],
            ))
            _fig_cache.update_layout(
                title=dict(text="Cache Data Present", font=dict(color="#c8a040", size=13), x=0),
                plot_bgcolor="#0d0b04", paper_bgcolor="#13110a",
                margin=dict(t=36, b=40, l=10, r=10),
                height=220,
                xaxis=dict(tickfont=dict(color="#c8b890", size=9), tickangle=-45),
                yaxis=dict(visible=False),
                font=dict(family="Source Sans Pro"),
                showlegend=False,
            )
            st.plotly_chart(_fig_cache, use_container_width=True, config={"displayModeBar": False})

        # ── Agent Leadership Tree (all 21 agents, 6 tiers) ───────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Agent Leadership Tree")

        _td = {r["key"]: r for r in _about_rows}

        def _get_nd(k):
            r = _td.get(k, {})
            return {
                "name":   r.get("name", k.title()),
                "sched":  r.get("schedule", "\u2014"),
                "runs":   r.get("runs", 0),
                "status": r.get("status", "MISSING"),
                "age":    r.get("cache_age", "\u2014"),
            }

        def _svg_node(x, y, w, h, fill, stroke, title, sub1, status, age, ts=11, rr=10):
            sc = {"OK": "#4caf50", "RUNNING": "#ff9800", "ERROR": "#f44336",
                  "STALE": "#d4a843", "MISSING": "#888"}.get(status, "#888")
            cx = x + w // 2
            # Split long titles onto 2 lines at nearest word boundary to middle
            if len(title) > 16 and " " in title:
                mid = len(title) // 2
                sl = title.rfind(" ", 0, mid)
                sr = title.find(" ", mid)
                if sl == -1:   split = sr
                elif sr == -1: split = sl
                else:          split = sl if (mid - sl) <= (sr - mid) else sr
                l1, l2 = title[:split], title[split + 1:]
                ttl_svg = (
                    f'<text x="{cx}" y="{y + int(h * 0.28)}" text-anchor="middle" fill="{stroke}" '
                    f'font-size="{ts}" font-weight="700" font-family="sans-serif">{l1}</text>'
                    f'<text x="{cx}" y="{y + int(h * 0.46)}" text-anchor="middle" fill="{stroke}" '
                    f'font-size="{ts}" font-weight="700" font-family="sans-serif">{l2}</text>'
                )
                sub_y, st_y = y + int(h * 0.66), y + int(h * 0.84)
            else:
                ttl_svg = (
                    f'<text x="{cx}" y="{y + int(h * 0.38)}" text-anchor="middle" fill="{stroke}" '
                    f'font-size="{ts}" font-weight="700" font-family="sans-serif">{title}</text>'
                )
                sub_y, st_y = y + int(h * 0.60), y + int(h * 0.80)
            return (
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rr}" ry="{rr}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
                + ttl_svg +
                f'<text x="{cx}" y="{sub_y}" text-anchor="middle" fill="#6a6050" '
                f'font-size="9" font-family="sans-serif">{sub1}</text>'
                f'<text x="{cx}" y="{st_y}" text-anchor="middle" fill="{sc}" '
                f'font-size="9" font-family="sans-serif">\u25cf {status} \u00b7 {age}</text>'
            )

        def _bus(upper_cx, lower_cx, y_top, y_bot, lc):
            """Vertical stubs + horizontal bus between two tiers."""
            yb = (y_top + y_bot) // 2
            all_x = sorted(set(upper_cx) | set(lower_cx))
            segs = [f'<line x1="{min(all_x)}" y1="{yb}" x2="{max(all_x)}" y2="{yb}" '
                    f'stroke="{lc}" stroke-width="1.5"/>']
            for x in upper_cx:
                segs.append(f'<line x1="{x}" y1="{y_top}" x2="{x}" y2="{yb}" '
                             f'stroke="{lc}" stroke-width="1.5"/>')
            for x in lower_cx:
                segs.append(f'<line x1="{x}" y1="{yb}" x2="{x}" y2="{y_bot}" '
                             f'stroke="{lc}" stroke-width="1.5"/>')
            return "".join(segs)

        # ── Layout constants ──────────────────────────────────────────────────
        _NW, _NH = 138, 65          # node width / height
        # 5-node tier: margin=61, step=150 → centers 130,280,430,580,730
        _CX5 = [130, 280, 430, 580, 730]
        _NX5 = [61,  211, 361, 511, 661]
        # 2-node tier aligned with T3 positions 1 and 3
        _CX2 = [280, 580]
        _NX2 = [211, 511]
        # Tier Y-tops (T1..T6) with 45px gaps
        _LY  = [10, 120, 230, 340, 450, 560]
        _TB  = [y + _NH for y in _LY]   # tier bottoms
        _lc  = "#5a4818"

        # ── Tier color pairs (fill, stroke) ───────────────────────────────────
        _CA = ("#3a1a00", "#b87020")   # amber     — T1 Infrastructure
        _CT = ("#051e1e", "#1a7870")   # teal      — T2 Real-time
        _CG = ("#161610", "#404038")   # gray      — T3 Periodic
        _CB = ("#080c18", "#283060")   # dark blue — T4 Macro
        _CO = ("#101808", "#405020")   # olive     — T5 CRE Metrics
        _CP = ("#140820", "#483890")   # purple    — T6 Synthesis

        # ── Fetch live data for all 21 agents ─────────────────────────────────
        _nds = {k: _get_nd(k) for k in [
            "manager", "debugger",
            "pricing", "rates",
            "news", "migration", "energy", "sustainability", "labor_market",
            "gdp", "inflation", "credit", "vacancy", "climate_risk",
            "cap_rate", "rent_growth", "land_market", "opportunity_zone", "distressed",
            "market_score", "predictions",
        ]}

        def _n(key, x, y, fill, stroke, ts=11):
            nd = _nds[key]
            return _svg_node(x, y, _NW, _NH, fill, stroke,
                             nd["name"],
                             f"{nd['sched']} \u00b7 {nd['runs']} runs",
                             nd["status"], nd["age"], ts=ts)

        # ── Build all 21 node SVGs ─────────────────────────────────────────────
        _svg_nodes = (
            # T1 — Infrastructure (amber, 2 nodes)
            _n("manager",          _NX2[0], _LY[0], *_CA, ts=12) +
            _n("debugger",         _NX2[1], _LY[0], *_CA, ts=12) +
            # T2 — Real-time (teal, 2 nodes)
            _n("pricing",          _NX2[0], _LY[1], *_CT) +
            _n("rates",            _NX2[1], _LY[1], *_CT) +
            # T3 — Periodic/Contextual (gray, 5 nodes)
            _n("news",             _NX5[0], _LY[2], *_CG) +
            _n("migration",        _NX5[1], _LY[2], *_CG) +
            _n("energy",           _NX5[2], _LY[2], *_CG) +
            _n("sustainability",   _NX5[3], _LY[2], *_CG) +
            _n("labor_market",     _NX5[4], _LY[2], *_CG) +
            # T4 — Macro (dark blue, 5 nodes)
            _n("gdp",              _NX5[0], _LY[3], *_CB) +
            _n("inflation",        _NX5[1], _LY[3], *_CB) +
            _n("credit",           _NX5[2], _LY[3], *_CB) +
            _n("vacancy",          _NX5[3], _LY[3], *_CB) +
            _n("climate_risk",     _NX5[4], _LY[3], *_CB) +
            # T5 — CRE Metrics (olive, 5 nodes)
            _n("cap_rate",         _NX5[0], _LY[4], *_CO) +
            _n("rent_growth",      _NX5[1], _LY[4], *_CO) +
            _n("land_market",      _NX5[2], _LY[4], *_CO) +
            _n("opportunity_zone", _NX5[3], _LY[4], *_CO) +
            _n("distressed",       _NX5[4], _LY[4], *_CO) +
            # T6 — Synthesis (purple, 2 nodes)
            _n("market_score",     _NX2[0], _LY[5], *_CP, ts=12) +
            _n("predictions",      _NX2[1], _LY[5], *_CP, ts=12)
        )

        # ── Build connector lines (bus-style between each tier) ───────────────
        _svg_lines = (
            _bus(_CX2, _CX2, _TB[0], _LY[1], _lc) +   # T1 → T2  (same 2 centers)
            _bus(_CX2, _CX5, _TB[1], _LY[2], _lc) +   # T2 → T3  (fan out)
            _bus(_CX5, _CX5, _TB[2], _LY[3], _lc) +   # T3 → T4
            _bus(_CX5, _CX5, _TB[3], _LY[4], _lc) +   # T4 → T5
            _bus(_CX5, _CX2, _TB[4], _LY[5], _lc)     # T5 → T6  (fan in)
        )

        # ── Legend ────────────────────────────────────────────────────────────
        _leg_items = [
            (_CA, "Infrastructure"), (_CT, "Real-time"),  (_CG, "Periodic"),
            (_CB, "Macro"),          (_CO, "CRE Metrics"), (_CP, "Synthesis"),
        ]
        _leg_svg = ""
        _lx = 100
        for (_lf, _ls), _ll in _leg_items:
            _leg_svg += (
                f'<rect x="{_lx}" y="640" width="11" height="11" rx="2" '
                f'fill="{_lf}" stroke="{_ls}" stroke-width="1"/>'
                f'<text x="{_lx + 15}" y="650" fill="#6a6050" font-size="10" '
                f'font-family="sans-serif">{_ll}</text>'
            )
            _lx += 15 + int(len(_ll) * 6.5) + 18

        _tree_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 665" width="100%"'
            ' style="display:block;margin:0 auto;">'
            + _svg_lines + _svg_nodes + _leg_svg +
            '</svg>'
        )
        st.markdown(
            f'<div style="background:#13110a;border-radius:10px;padding:20px 24px;'
            f'margin-bottom:12px;overflow-x:auto;">'
            f'{_tree_svg}</div>',
            unsafe_allow_html=True,
        )

        # ── Styled agent status table ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Agent Detail")

        _st_hcells = "".join(
            f'<th style="padding:10px 12px 13px;color:#c8a040;font-size:0.76rem;font-weight:700;'
            f'letter-spacing:0.09em;text-align:{al};border-bottom:1px solid #2a2410;">{h}</th>'
            for h, al in [("AGENT","left"),("SCHEDULE","center"),("STATUS","center"),
                          ("RUNS","right"),("CACHE AGE","right"),("LAST ERROR","left")]
        )
        _st_rows_html = ""
        for _r in _about_rows:
            _sc   = _status_color_map.get(_r["status"], "#888")
            _sep  = "border-bottom:1px solid #1e1c0e;"
            _dot  = f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{_sc};margin-right:6px;vertical-align:middle;"></span>'
            _st_rows_html += (
                f'<tr>'
                f'<td style="padding:10px 12px;{_sep}color:#c8b890;font-size:0.88rem;white-space:nowrap;">{_r["label"]}</td>'
                f'<td style="padding:10px 12px;{_sep}text-align:center;color:#7a7050;font-size:0.82rem;">{_r["schedule"]}</td>'
                f'<td style="padding:10px 12px;{_sep}text-align:center;">'
                f'{_dot}<span style="color:{_sc};font-size:0.82rem;font-weight:700;">{_r["status"]}</span></td>'
                f'<td style="padding:10px 12px;{_sep}text-align:right;color:#c8a040;font-size:0.88rem;">{_r["runs"]}</td>'
                f'<td style="padding:10px 12px;{_sep}text-align:right;color:#a09070;font-size:0.82rem;white-space:nowrap;">{_r["cache_age"]}</td>'
                f'<td style="padding:10px 12px;{_sep}color:#ef5350;font-size:0.8rem;">{_r["error"]}</td>'
                f'</tr>'
            )

        _sm_html = f"""
<div style="background:#13110a;border-radius:10px;padding:24px 28px 16px;margin-bottom:8px;">
  <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:'Source Sans Pro',sans-serif;">
      <thead><tr>{_st_hcells}</tr></thead>
      <tbody>{_st_rows_html}</tbody>
    </table>
  </div>
</div>
"""
        st.markdown(_sm_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section(" Cache Health")
        _cache_check_keys = [
            ("migration",           "Every 6h"),
            ("pricing",             "Every 1h"),
            ("rates",               "Every 1h"),
            ("energy_data",         "Every 6h"),
            ("credit_data",         "Every 6h"),
            ("gdp_data",            "Every 6h"),
            ("inflation_data",      "Every 6h"),
            ("labor_market",        "Every 6h"),
            ("cap_rate",            "Every 6h"),
            ("rent_growth",         "Every 6h"),
            ("vacancy",             "Every 12h"),
            ("climate_risk",        "Every 24h"),
            ("market_score",        "Every 6h"),
            ("opportunity_zone",    "Every 24h"),
            ("news",                "Every 4h"),
        ]
        _cc_cols = st.columns(5)
        for _ci, (_ck, _cf) in enumerate(_cache_check_keys):
            _cc = read_cache(_ck)
            _has = _cc["data"] is not None
            _stale = _cc.get("stale", True)
            _label = "OK" if _has and not _stale else ("STALE" if _has else "MISSING")
            _clr   = "#4caf50" if _label == "OK" else ("#ff9800" if _label == "STALE" else "#f44336")
            _cc_cols[_ci % 5].markdown(
                f'<div class="metric-card">'
                f'<div class="label" style="font-size:0.72rem;">{_ck.replace("_", " ").title()}</div>'
                f'<div class="value" style="color:{_clr};font-size:1rem;">{_label}</div>'
                f'<div class="sub">{cache_age_label(_ck)}</div>'
                f'<div style="font-size:0.68rem;color:#555;margin-top:2px;">{_cf}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "All agents run in APScheduler background threads. Caches are JSON files in cache/. "
            "Status reflects the in-memory agent_status dict — resets on app restart."
        )
