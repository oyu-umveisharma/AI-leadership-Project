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


# ═══════════════════════════════════════════════════════════════════════════════
#  WELCOME / ONBOARDING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.onboarding_complete:
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
      html, body, [class*="css"] {{ font-family: 'Source Sans Pro', sans-serif; }}
      .main .block-container {{ max-width: 680px; padding-top: 8vh; }}
      .welcome-box {{
        text-align: center;
        padding: 40px 20px;
      }}
      .welcome-icon {{
        font-size: 3.5rem;
        margin-bottom: 8px;
      }}
      .welcome-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {GOLD};
        margin-bottom: 4px;
      }}
      .welcome-sub {{
        font-size: 1.05rem;
        color: #e8e9ed;
        margin-bottom: 32px;
      }}
      .welcome-prompt {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #e8e9ed;
        margin-bottom: 16px;
      }}
      .welcome-or {{
        color: #999;
        font-size: 0.85rem;
        margin: 18px 0 12px 0;
      }}
      .welcome-footer {{
        margin-top: 48px;
        padding-top: 20px;
        border-top: 3px solid {GOLD};
      }}
      .welcome-footer .purdue {{
        font-size: 0.8rem;
        color: #888;
      }}
      .welcome-footer .purdue b {{
        color: {GOLD_DARK};
      }}
    </style>
    <div class="welcome-box">
      <div class="welcome-icon"></div>
      <div class="welcome-title">CRE Intelligence Platform</div>
      <div class="welcome-sub">Welcome! I'm your AI-powered commercial real estate assistant.</div>
      <div class="welcome-prompt">What are you looking to invest in today?</div>
    </div>
    """, unsafe_allow_html=True)

    # Text input
    user_input = st.text_input(
        "Describe your investment focus",
        placeholder="e.g., Industrial warehouse in Austin, TX",
        label_visibility="collapsed",
    )
    if user_input:
        intent = _parse_intent(user_input)
        _complete_onboarding(**intent)
        st.rerun()

    # Quick-select buttons
    st.markdown('<div style="text-align:center;color:#999;font-size:0.85rem;margin:18px 0 12px 0;">or pick a category</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    quick_options = ["Industrial", "Multifamily", "Office", "Retail", "Just Exploring"]
    for col, opt in zip(cols, quick_options):
        with col:
            if st.button(opt, use_container_width=True, key=f"quick_{opt}"):
                if opt == "Just Exploring":
                    _complete_onboarding()
                else:
                    _complete_onboarding(property_type=opt)
                st.rerun()

    st.markdown(f"""
    <div class="welcome-footer" style="text-align:center;">
      <div class="purdue"><b>Purdue University</b> · Daniels School of Business · MSF Program</div>
    </div>
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
main_tab_re, main_tab_energy, main_tab_macro = st.tabs(["Real Estate", "Energy", "Macro Environment"])

with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab6, tab_vacancy, tab_land, tab_caprate, tab_rent, tab_oz, tab_score = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "System Monitor",
        "Vacancy Monitor",
        "Land & Development",
        "Cap Rate Monitor",
        "Rent Growth",
        "Opportunity Zones",
        "Market Score",
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

                fig_county = go.Figure(go.Choropleth(
                    geojson=counties_geojson,
                    locations=county_df["fips"],
                    z=county_df["migration_score"],
                    colorscale=[
                        [0.0, "#7f0000"], [0.25, "#c62828"], [0.45, "#d4c5a9"],
                        [0.55, "#a5d6a7"], [0.75, "#2e7d32"], [1.0, "#1b5e20"],
                    ],
                    zmin=0, zmax=100,
                    marker_line_width=0.5, marker_line_color="#333",
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
                fig_county.update_geos(fitbounds="locations", visible=False)
                fig_county.update_layout(
                    paper_bgcolor="#0d0b04",
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=500,
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

                _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                for _col, _lbl, _val_str, _sub, _badge_html, _border in [
                    (_kc1, "AVG CAP RATE",     f"{_cap*100:.2f}%",    _cap_sub,  _badge(_cap_d,  "YoY", False), "#d4a843"),
                    (_kc2, "NOI MARGIN",        f"{_noi*100:.1f}%",    _noi_sub,  _badge(_noi_d,  "YoY", True),  "#4a9e58"),
                    (_kc3, "RENT GROWTH YOY",   f"{_rent*100:+.1f}%", _rent_sub, _badge(_rent_d, "YoY", True),  "#4a9e58" if _rent > 0 else "#ef5350"),
                    (_kc4, "AVG VACANCY",       f"{_vac*100:.1f}%",    _vac_sub,  _badge(_vac_d,  "vs prior", False), "#d4a843" if _vac < 0.10 else "#ef5350"),
                ]:
                    _val_color = _border
                    _col.markdown(f"""
<div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:hidden;min-height:130px;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{_border};border-radius:10px 10px 0 0;"></div>
  <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">{_lbl}</div>
  <div style="font-size:34px;font-weight:500;color:{_val_color};line-height:1;margin-bottom:4px;">{_val_str}</div>
  <div style="font-size:11px;color:#6a5228;margin-bottom:2px;">{_sub}</div>
  {_badge_html}
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
            "migration and business growth scores — identifying acquisition opportunities before demand peaks. "
            "Updates every 24 hours alongside company predictions.",
            "predictions",
        )

        cache4 = read_cache("predictions")
        if not stale_banner("predictions") or cache4["data"] is None:
            st.stop()

        pdata4   = cache4["data"]
        listings = pdata4.get("listings", {})
        top3_abbr = pdata4.get("top3_abbr", [])

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
                    _border_clr = "#1b5e20" if _is_city_match else GOLD

                    st.markdown(f"""
                    <div class="listing-card" style="border-left-color:{_border_clr};">
                      <div class="l-price">{price_fmt}</div>
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
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Articles Found", str(ndata.get("article_count", 0)), "Facility announcements"), unsafe_allow_html=True)
        c2.markdown(metric_card("Sources Checked", str(ndata.get("sources_checked", 0)), "News + gov feeds"), unsafe_allow_html=True)
        fetched = ndata.get("fetched_at", "")
        fetched_label = datetime.fromisoformat(fetched).strftime("%b %d, %Y %I:%M %p") if fetched else "N/A"
        c3.markdown(metric_card("Last Scan", fetched_label, "Updates every 4 hours"), unsafe_allow_html=True)

        # ── AI Summary ────────────────────────────────────────────────────────
        section(" Agent 5 — AI Investment Brief: Facility Announcements")
        summary = ndata.get("summary", "")
        if summary:
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-label"> Agent 5 · Industry Announcements · {datetime.today().strftime('%b %d, %Y')}</div>
              <div class="agent-text">{summary}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("News summary is generated every 4 hours. Ensure GROQ_API_KEY is set in .env.")

        # ── Raw Article Feed ─────────────────────────────────────────────────
        raw = ndata.get("raw_articles", [])
        if raw:
            section(f" Raw Announcement Feed ({len(raw)} articles)")

            feed_type_filter = st.selectbox(
                "Filter by source type",
                options=["All", "news", "industry", "press", "government"],
                key="news_filter",
            )

            source_colors = {
                "government": "#1565c0",
                "industry":   "#2e7d32",
                "press":      "#6a1b9a",
                "news":       "#bf360c",
            }

            for art in raw:
                if feed_type_filter != "All" and art.get("feed_type") != feed_type_filter:
                    continue
                ft    = art.get("feed_type", "news")
                color = source_colors.get(ft, "#555")
                link  = art.get("link", "#")
                title = art.get("title", "No title")
                desc  = art.get("description", "")[:280]
                src   = art.get("source", "")
                date  = art.get("pub_date", "")[:22]

                href = f'<a href="{link}" target="_blank" style="color:{color};font-weight:700;text-decoration:none;">{title}</a>' if link and link != "#" else f'<span style="font-weight:700;">{title}</span>'

                st.markdown(f"""
                <div style="border-left:3px solid {color};padding:10px 14px;margin:6px 0;background:#fafafa;border-radius:4px;">
                  <div style="font-size:0.7rem;color:{color};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
                    {src} &nbsp;·&nbsp; {ft.upper()} &nbsp;·&nbsp; {date}
                  </div>
                  <div style="font-size:0.9rem;margin-bottom:4px;">{href}</div>
                  <div style="font-size:0.8rem;color:#555;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(
            "Sources: Reuters, Manufacturing.net, IndustryWeek, PR Newswire, Business Wire, "
            "US Dept of Energy, US Dept of Commerce, EDA, Expansion Solutions, Site Selection Magazine."
        )

        with st.expander("How This Is Calculated"):
            st.markdown("""
**News Relevance Scoring:**

- Articles are pulled from 10+ RSS feeds covering manufacturing, government, and industry press.
- Each article is filtered by CRE-relevant keywords: *plant, warehouse, data center, headquarters, facility, expansion, construction*.
- Articles matching multiple keywords or mentioning specific investment amounts are ranked higher.
- AI extraction (Groq LLM) identifies structured fields: company, location, investment size, job count, facility type, and CRE impact.

**Facility Type Classification:** Manufacturing, Data Center, Warehouse/Distribution, Office/HQ, Training Center, Mixed-Use.

**Data Source:** Reuters, Manufacturing.net, IndustryWeek, PR Newswire, Business Wire, Dept. of Energy, Dept. of Commerce, EDA, Expansion Solutions, Site Selection Magazine.

**Update Frequency:** Company Predictions via Agent 3 (every 24h), Industry Announcements via Agent 5 (every 4h).
""")


    with tab6:
        st.markdown("#### Background Agent Monitor — Agent 4 runs every 30 minutes")
        st.markdown(
            "Agent 4 continuously verifies that all data sources are live, caches are fresh, "
            "and APIs are reachable. This tab shows the live health dashboard."
        )

        # ── Agent Status ────────────────────────────────────────────────────────
        section(" Agent Status")
        status = get_status()

        agent_labels = {
            "migration":      ("Agent 1", "Population & Migration", "Every 6h"),
            "pricing":        ("Agent 2", "REIT Pricing",           "Every 1h"),
            "predictions":    ("Agent 3", "Company Predictions",    "Every 24h"),
            "debugger":       ("Agent 4", "Debugger / Monitor",     "Every 30min"),
            "news":           ("Agent 5", "Industry Announcements", "Every 4h"),
            "rates":          ("Agent 6", "Interest Rate & Debt",   "Every 1h"),
            "energy":         ("Agent 7", "Energy & Construction",  "Every 6h"),
            "sustainability": ("Agent 8", "Sustainability & ESG",   "Every 6h"),
        }

        cols = st.columns(len(agent_labels))
        for col, (agent_key, (num, name, freq)) in zip(cols, agent_labels.items()):
            s = status.get(agent_key, {})
            st_val = s.get("status", "idle")
            runs   = s.get("runs", 0)
            err    = s.get("last_error", None)
            icon   = {"ok": "OK", "running": "...", "error": "ERR", "idle": "--"}.get(st_val, "?")
            color  = {"ok": "status-ok", "running": "status-run", "error": "status-error", "idle": "status-idle"}.get(st_val, "")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{num} · {freq}</div>
              <div class="value">{icon}</div>
              <div class="sub">{name}</div>
              <div class="{color}" style="font-size:0.75rem;margin-top:4px;">{st_val.upper()} · {runs} runs</div>
              {"<div style='color:#b71c1c;font-size:0.7rem;margin-top:4px;'> " + str(err)[:60] + "</div>" if err else ""}
            </div>
            """, unsafe_allow_html=True)

        # ── Cache Ages ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Cache Status")
        cache_keys = [
            ("migration",           "Every 6h",    7),
            ("pricing",             "Every 1h",    2),
            ("predictions",         "Every 24h",   25),
            ("debugger",            "Every 30min", 1),
            ("news",                "Every 4h",    5),
            ("rates",               "Every 1h",    2),
            ("energy_data",         "Every 6h",    7),
            ("sustainability_data", "Every 6h",    7),
        ]
        c_cols = st.columns(len(cache_keys))
        for col, (key, freq, max_h) in zip(c_cols, cache_keys):
            c = read_cache(key)
            age_label = cache_age_label(key)
            stale = c.get("stale", True)
            has_data = c["data"] is not None
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{key.title()} Cache</div>
              <div class="value">{"OK" if has_data and not stale else ("STALE" if has_data else "NONE")}</div>
              <div class="sub">{age_label}</div>
              <div style="font-size:0.72rem;color:#888;margin-top:4px;">Refresh: {freq} · Max age: {max_h}h</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Health Report from Debugger Agent ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Health Report — Agent 4 Output")

        dbg_cache = read_cache("debugger")
        if dbg_cache["data"]:
            dbg = dbg_cache["data"]
            checked = dbg.get("checked_at", "unknown")
            st.caption(f"Last health check: {checked}")

            issues  = dbg.get("issues", [])
            healthy = dbg.get("healthy", [])

            if healthy:
                st.markdown("**Healthy systems:**")
                for h in healthy:
                    st.markdown(f"- {h}")

            if issues:
                st.markdown("**Issues detected:**")
                for i in issues:
                    st.markdown(f"- {i}")
            elif healthy:
                st.success("All systems healthy — no issues detected.")

            # ── Agent sub-status from debugger ────────────────────────────────
            agent_status_in_dbg = dbg.get("agent_status", {})
            if agent_status_in_dbg:
                st.markdown("<br>", unsafe_allow_html=True)
                section(" Last Known Agent States (from Debugger)")
                dbg_df = pd.DataFrame([
                    {
                        "Agent":    k,
                        "Status":   v.get("status", "?"),
                        "Last Run": v.get("last_run", "Never"),
                        "Runs":     v.get("runs", 0),
                        "Last Error": (v.get("last_error") or "")[:80],
                    }
                    for k, v in agent_status_in_dbg.items()
                ])
                st.dataframe(dbg_df, use_container_width=True, hide_index=True)
        else:
            st.info("Debugger agent has not completed its first run yet. Refresh in 30 seconds.")

        st.caption(
            "All agents run independently in background threads managed by APScheduler. "
            "Data is stored in JSON cache files and survives Streamlit reruns. "
            "Requires GROQ_API_KEY in .env for Agent 3 predictions."
        )

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

        # ── Market Heatmap ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Vacancy Rate by Market and Property Type")

        if mkt_rows:
            vac_df = pd.DataFrame(mkt_rows)

            # Pivot for heatmap
            pivot = vac_df.pivot_table(
                index="market", columns="property_type", values="vacancy_rate"
            ).round(1)

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0,  "#1b5e20"],
                    [0.3,  "#66bb6a"],
                    [0.55, "#fff9c4"],
                    [0.75, "#ef5350"],
                    [1.0,  "#7f0000"],
                ],
                zmin=0, zmax=28,
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=11, color="#0d0b04"),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
                colorbar=dict(
                    title=dict(text="Vacancy %", font=dict(color="#c8b890", size=11)),
                    tickfont=dict(color="#c8b890", size=10),
                    thickness=14,
                    bgcolor="#1a1208",
                    bordercolor="#3a3a2a",
                ),
            ))
            fig_heat.update_layout(
                plot_bgcolor="#0d0b04", paper_bgcolor="#1a1208",
                margin=dict(t=20, b=20, l=180, r=20),
                height=620,
                font=dict(family="Source Sans Pro", color="#c8b890"),
                xaxis=dict(side="top", tickfont=dict(color="#c8b890", size=12)),
                yaxis=dict(tickfont=dict(color="#c8b890", size=11)),
            )
            st.plotly_chart(fig_heat, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Green = tight market (low vacancy, strong demand). "
                "Red = soft market (high vacancy, excess supply). "
                "Office and Multifamily face the most pressure heading into 2025."
            )

        # ── Market Detail Table ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Market Detail — Vacancy vs. National Average")

        if mkt_rows:
            detail_df = pd.DataFrame(mkt_rows)
            detail_df["trend_label"] = detail_df["trend"].map(
                lambda t: f"{TREND_ARROW[t]} {t.title()}"
            )
            detail_df["vs_national_fmt"] = detail_df["vs_national"].apply(
                lambda x: f"{x:+.1f}pp"
            )
            detail_df = detail_df.rename(columns={
                "market": "Market", "property_type": "Property Type",
                "vacancy_rate": "Vacancy %", "trend_label": "Trend",
                "vs_national_fmt": "vs. National",
            })[["Market", "Property Type", "Vacancy %", "Trend", "vs. National"]]

            def _color_vs(val):
                try:
                    v = float(val.replace("pp", "").replace("+", ""))
                    if v < -2:   return "color: #66bb6a"
                    elif v > 2:  return "color: #ef5350"
                    else:        return "color: #d4a843"
                except Exception:
                    return ""

            st.dataframe(
                detail_df.style.applymap(_color_vs, subset=["vs. National"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "vs. National = difference in percentage points from the national average "
                "for that property type. Negative (green) = tighter than average. "
                "Positive (red) = looser than average."
            )

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
            section(" Land Market Detail Table")
            _la_display = _la_df.copy()
            _la_display["Avg $/Acre"] = _la_display["Avg $/Acre"].apply(lambda v: f"${v:,}")
            st.dataframe(_la_display, use_container_width=True, hide_index=True)
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
            section(" Market Factor Breakdown — Top 10")
            _ms_table = []
            for r in _ms_rankings[:10]:
                f = r["factors"]
                _ms_table.append({
                    "Rank": r["rank"], "Market": r["market"],
                    "Score": r["composite"], "Grade": r["grade"],
                    "Migration": round(f.get("migration", 0)),
                    "Vacancy":   round(f.get("vacancy", 0)),
                    "Rent":      round(f.get("rent", 0)),
                    "Cap Rate":  round(f.get("cap_rate", 0)),
                    "Land":      round(f.get("land", 0)),
                    "Macro":     round(f.get("macro", 0)),
                })
            st.dataframe(
                pd.DataFrame(_ms_table).set_index("Rank"),
                use_container_width=True,
                height=380,
            )
            st.caption(
                "Composite = weighted sum across 7 factors (0–100 each). "
                "Grade: A ≥ 80, B+ ≥ 70, B ≥ 60, C+ ≥ 50, C ≥ 40, D < 40. "
                "Agent 18 aggregates live caches from all other agents. Q1 2025 benchmarks. Not financial advice."
            )

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

            # ── National cap rates ────────────────────────────────────────────
            section(" National Cap Rates by Property Type")
            _cap_ncols = st.columns(len(_cap_national))
            for _cap_col, (_ptype, _pd) in zip(_cap_ncols, _cap_national.items()):
                _sp_info = _cap_spreads.get(_ptype, {})
                _sig     = _sp_info.get("signal", "")
                _sig_c   = {"attractive": "#66bb6a", "fair": "#d4a843", "compressed": "#ef5350"}.get(_sig, "#a09880")
                _cap_col.markdown(metric_card(
                    _ptype,
                    f"{_pd['rate']}%",
                    f"{_cap_ta.get(_pd['trend'],'')} vs {_pd['prior_year']}% prior year",
                ), unsafe_allow_html=True)
                if _sig:
                    _cap_col.markdown(
                        f"<div style='text-align:center;font-size:0.77rem;color:{_sig_c};"
                        f"margin-top:-6px;margin-bottom:8px;font-weight:700;'>{_sig.upper()}</div>",
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
                _hm_markets = list(_cap_mktcaps.keys())
                _hm_ptypes  = ["Office", "Industrial", "Retail", "Multifamily"]
                _hm_matrix  = [[_cap_mktcaps[m].get(p, 0) for p in _hm_ptypes] for m in _hm_markets]
                fig_cap = go.Figure(go.Heatmap(
                    z=_hm_matrix, x=_hm_ptypes, y=_hm_markets,
                    colorscale=[[0, "#1b4332"], [0.5, "#d4a843"], [1, "#7f1d1d"]],
                    text=[[f"{v:.1f}%" for v in row] for row in _hm_matrix],
                    texttemplate="%{text}",
                    showscale=True,
                    hovertemplate="<b>%{y}</b> · %{x}<br>Cap Rate: %{z:.1f}%<extra></extra>",
                ))
                fig_cap.update_layout(
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    margin=dict(l=160, r=40, t=20, b=40), height=560,
                    xaxis=dict(color="#8a7040"), yaxis=dict(color="#8a7040"),
                )
                st.plotly_chart(fig_cap, use_container_width=True)
                st.caption(
                    "Green = lower cap rate (more expensive / compressed yield). "
                    "Red = higher cap rate (cheaper entry / better yield). "
                    "Source: CoStar / CBRE Q1 2025. Not financial advice."
                )

            # ── Property-type analyst notes ───────────────────────────────────
            section(" Analyst Notes")
            for _ptype, _pd in _cap_national.items():
                _t_c = _cap_tc.get(_pd["trend"], "#d4a843")
                _t_a = _cap_ta.get(_pd["trend"], "")
                st.markdown(
                    f"<div style='background:#171309;border-left:3px solid {_t_c};"
                    f"padding:8px 16px;border-radius:4px;margin-bottom:8px;'>"
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

            # ── National overview ─────────────────────────────────────────────
            section(" National Rent Growth by Property Type (YoY %)")
            _rg_ncols = st.columns(len(_rg_national))
            for _rg_col, (_rg_pt, _rg_d) in zip(_rg_ncols, _rg_national.items()):
                _rg_yoy   = _rg_d["yoy_pct"]
                _rg_color = "#66bb6a" if _rg_yoy > 1 else ("#ef5350" if _rg_yoy < 0 else "#d4a843")
                _rg_col.markdown(metric_card(
                    _rg_pt,
                    f"<span style='color:{_rg_color}'>{_rg_yoy:+.1f}%</span>",
                    f"{_rg_ta.get(_rg_d['trend'],'')} vs {_rg_d['prior_year']:+.1f}% prior year",
                ), unsafe_allow_html=True)

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
                _rg_mkts   = list(_rg_market.keys())
                _rg_ptypes = ["multifamily", "industrial_psf", "office_psf", "retail_psf"]
                _rg_lbls   = ["Multifamily", "Industrial PSF", "Office PSF", "Retail PSF"]
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
                _rg_t_c = _rg_tc.get(_rg_d["trend"], "#d4a843")
                _rg_t_a = _rg_ta.get(_rg_d["trend"], "")
                st.markdown(
                    f"<div style='background:#171309;border-left:3px solid {_rg_t_c};"
                    f"padding:8px 16px;border-radius:4px;margin-bottom:8px;'>"
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
            _oz_si_rows = []
            for _oz_st, _oz_si in _oz_state_inc.items():
                _oz_si_rows.append({
                    "State":    _oz_st,
                    "Program":  _oz_si["program"],
                    "Benefit":  _oz_si["benefit"],
                    "CRE Types": ", ".join(_oz_si["cre_types"]),
                    "Cap":      _oz_si["cap"],
                })
            if _oz_si_rows:
                st.dataframe(pd.DataFrame(_oz_si_rows).set_index("State"), use_container_width=True, height=460)
            st.caption(
                "Source: IRS Revenue Ruling 2018-29, HUD Opportunity Zone designations, "
                "state economic development agencies. Not financial or legal advice. Consult a tax advisor."
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

    # ── Meet the Team ─────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#1a1208; border:1px solid #a07830; border-top:3px solid #d4a843;
            border-radius:8px; padding:32px 36px; margin-top:24px;">
  <div style="text-align:center; margin-bottom:24px;">
    <span style="color:#d4a843; font-size:1.3rem; font-weight:700; letter-spacing:2px;
                 text-transform:uppercase;">Meet the Team</span>
    <div style="color:#a09880; font-size:0.85rem; margin-top:4px;">
      MGMT 690: AI Leadership &nbsp;&middot;&nbsp; Purdue Daniels School of Business
    </div>
  </div>
  <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap;">
    <div style="background:#1e1a0a; border:1px solid #a07830; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Aayman Afzal</div>
      <a href="https://www.linkedin.com/in/aayman-afzal" target="_blank"
         style="color:#d4a843; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1e1a0a; border:1px solid #a07830; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Ajinkya Kodnikar</div>
      <a href="https://www.linkedin.com/in/ajinkya-kodnikar" target="_blank"
         style="color:#d4a843; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1e1a0a; border:1px solid #a07830; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Oyu Amar</div>
      <a href="https://www.linkedin.com/in/oyu-amar/" target="_blank"
         style="color:#d4a843; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1e1a0a; border:1px solid #a07830; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Ricardo Ruiz</div>
      <a href="https://www.linkedin.com/in/ricardoruizjr" target="_blank"
         style="color:#d4a843; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
