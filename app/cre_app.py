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

@st.cache_resource
def _init_scheduler():
    """Called once per server process — keeps one scheduler alive regardless of reruns or new sessions."""
    start_scheduler()
    return True

_init_scheduler()

# ── Brand ────────────────────────────────────────────────────────────────────
GOLD      = "#CFB991"
GOLD_DARK = "#8E6F3E"
BLACK     = "#000000"

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
    background: #111111 !important;
    color: #e8e9ed !important;
  }}

  .main .block-container {{
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
    max-width: 1360px;
  }}

  /* ── Text ── */
  p, li, span, label, div {{ color: #e8e9ed; }}
  .stMarkdown p {{ color: #e8e9ed; }}
  h1, h2, h3, h4, h5, h6 {{ color: #e8e9ed !important; }}
  .stCaption, [data-testid="stCaptionContainer"] p {{ color: #8890a1 !important; }}

  /* ── Metrics ── */
  [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: {GOLD} !important;
  }}
  [data-testid="stMetricLabel"] {{ color: #e8e9ed !important; }}

  /* ── Data tables ── */
  [data-testid="stDataFrame"] {{ background: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 4px; }}
  .stDataFrame th {{ background: #1c1c1c !important; color: {GOLD} !important; font-size: 0.65rem !important; font-weight: 600 !important; text-transform: uppercase !important; }}
  .stDataFrame td {{ background: #1c1c1c !important; color: #e8e9ed !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important; }}

  /* ── Alerts ── */
  [data-testid="stInfo"] {{ background: #1a1f1a; border-color: #2a2a2a; color: #e8e9ed; }}
  [data-testid="stWarning"] {{ background: #1f1a0f; border-color: #2a2a2a; color: #e8e9ed; }}
  [data-testid="stSuccess"] {{ background: #1a1f1a; border-color: #2a2a2a; color: #e8e9ed; }}

  /* ── Cards ── */
  .agent-card {{
    background: #1c1c1c;
    border-radius: 4px;
    padding: 20px 24px;
    margin: 12px 0;
    border: 1px solid #2a2a2a;
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
    color: #e8e9ed;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }}

  .metric-card {{
    background: #1c1c1c;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 16px;
    text-align: center;
    transition: box-shadow 0.2s;
  }}
  .metric-card:hover {{ box-shadow: 0 2px 12px rgba(0,0,0,0.3); border-color: {GOLD}; }}
  .metric-card .label {{ font-size: 0.68rem; color: #8890a1; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-card .value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {GOLD}; margin: 4px 0; }}
  .metric-card .sub   {{ font-size: 0.75rem; color: #8890a1; }}

  .section-header {{
    background: #1c1c1c;
    color: {GOLD};
    padding: 9px 16px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 24px 0 14px 0;
    border-left: 3px solid {GOLD};
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  .listing-card {{
    background: #1c1c1c;
    border: 1px solid #2a2a2a;
    border-left: 3px solid {GOLD};
    border-radius: 4px;
    padding: 14px 18px;
    margin: 8px 0;
  }}
  .listing-card .l-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: {GOLD}; }}
  .listing-card .l-address {{ font-size: 0.9rem; color: #e8e9ed; margin: 2px 0; }}
  .listing-card .l-detail {{ font-size: 0.8rem; color: #8890a1; }}
  .listing-card .l-tag {{
    display: inline-block;
    background: #252525;
    color: {GOLD};
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.65rem;
    font-weight: 600;
    margin-right: 4px;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}

  .status-ok    {{ color: #4caf6e; font-weight: 700; }}
  .status-error {{ color: #e05050; font-weight: 700; }}
  .status-run   {{ color: {GOLD}; font-weight: 700; }}
  .status-idle  {{ color: #555; }}

  /* ── Tabs ── */
  div[data-testid="stTabs"] {{ background: #111111 !important; }}
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ border-bottom: 1px solid #2a2a2a !important; background: #111111 !important; }}
  div[data-testid="stTabs"] button[role="tab"] {{ color: #8890a1 !important; font-weight: 500 !important; font-size: 0.82rem !important; }}
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color: {GOLD} !important; border-bottom-color: {GOLD} !important; font-weight: 700 !important; }}

  /* ── Buttons ── */
  .stButton > button {{
    background: #1c1c1c !important;
    color: {GOLD} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 3px !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important;
    transition: all 0.2s !important;
  }}
  .stButton > button:hover {{ background: {GOLD} !important; color: #000 !important; }}

  /* ── Form controls ── */
  [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background: #1c1c1c !important;
    border-color: #2a2a2a !important;
    color: #e8e9ed !important;
  }}
  .stTextInput input {{ background: #1c1c1c !important; color: #e8e9ed !important; border-color: #2a2a2a !important; }}

  /* ── Plotly dark mode ── */
  .js-plotly-plot text {{ fill: #c8c8c8 !important; }}
  .js-plotly-plot .gtitle {{ fill: {GOLD} !important; }}
  .js-plotly-plot .xtick text, .js-plotly-plot .ytick text {{ fill: #8890a1 !important; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{ background: #111111 !important; }}

  /* ── Footer accent ── */
  body::after {{
    content: '';
    position: fixed;
    bottom: 0; left: 0; width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(207,185,145,0.12) 8%, rgba(207,185,145,0.55) 25%, {GOLD} 50%, rgba(207,185,145,0.55) 75%, rgba(207,185,145,0.12) 92%, transparent 100%);
    z-index: 9998; pointer-events: none;
  }}

  /* push content above footer */
  .main .block-container {{ padding-bottom: 40px; }}
    color: #a09880 !important;
    background: transparent !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: #CFB991 !important;
    border-bottom: 2px solid #CFB991 !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Header banner ────────────────────────────────────────────────────────────
# Pure HTML/CSS banner — matches app dark-gold theme exactly, no image patching
_ICON_B64 = "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"
st.markdown(f"""
<div style="
  background: linear-gradient(135deg, #0f0f0c 0%, #141410 40%, #0f0f0c 100%);
  border: 1px solid #8E6F3E;
  border-bottom: 4px solid #CFB991;
  border-radius: 8px;
  padding: 18px 36px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 110px;
">
  <!-- Left: building icon + title -->
  <div style="display:flex; align-items:center; gap:20px;">
    <img src="data:image/png;base64,{_ICON_B64}"
         style="height:90px; width:90px; object-fit:contain;" />
    <div>
      <div style="color:#CFB991; font-size:2.1rem; font-weight:700;
                  font-family:'DM Sans',-apple-system,sans-serif;
                  letter-spacing:0.5px; line-height:1.1;">
        CRE Intelligence Platform
      </div>
      <div style="color:#8E6F3E; font-size:0.85rem;
                  font-family:'DM Sans',-apple-system,sans-serif;
                  letter-spacing:2px; text-transform:uppercase; margin-top:4px;">
        AI-Powered Commercial Real Estate Intelligence
      </div>
    </div>
  </div>

  <!-- Divider -->
  <div style="width:2px; height:80px; background:#8E6F3E; margin:0 30px; flex-shrink:0;"></div>

  <!-- Right: Purdue branding -->
  <div style="text-align:right;">
    <div style="color:#CFB991; font-size:1.55rem; font-weight:700;
                font-family:'DM Sans',-apple-system,sans-serif; line-height:1.1;">
      Purdue University
    </div>
    <div style="color:#e8dfc4; font-size:1.05rem;
                font-family:'DM Sans',-apple-system,sans-serif; margin-top:4px;">
      Daniels School of Business
    </div>
    <div style="color:#8E6F3E; font-size:0.75rem; letter-spacing:2px;
                text-transform:uppercase; font-family:'DM Sans',-apple-system,sans-serif;
                margin-top:4px;">
      MSF Program
    </div>
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
        f'<div style="background:#1c1c1c;color:{GOLD};padding:8px 16px;border-radius:4px;'
        f'border:1px solid #2a2a2a;border-left:3px solid {GOLD};'
        f'font-size:0.82rem;margin-bottom:16px;letter-spacing:0.04em;text-transform:uppercase;">'
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
                f'<div style="background:#1c1c1c;border-left:3px solid {GOLD};padding:10px 16px;'
                f'border-radius:4px;margin:8px 0 12px 0;font-size:0.88rem;color:#e8e9ed;border:1px solid #2a2a2a;">'
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



# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
main_tab_re, main_tab_energy, main_tab_macro = st.tabs(["Real Estate", "Energy", "Macro Environment"])

with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab6, tab_vacancy = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "System Monitor",
        "Vacancy Monitor",
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

        # ── KPI strip ──────────────────────────────────────────────────────────
        top1      = mig_df.iloc[0]
        top_gain  = mig_df[mig_df["pop_growth_pct"] > 0].shape[0]
        top_loss  = mig_df[mig_df["pop_growth_pct"] < 0].shape[0]
        avg_grow  = mig_df["pop_growth_pct"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("#1 Destination", top1["state_abbr"], top1["state_name"]), unsafe_allow_html=True)
        c2.markdown(metric_card("States Growing", str(top_gain), "Positive net migration"), unsafe_allow_html=True)
        c3.markdown(metric_card("States Shrinking", str(top_loss), "Negative net migration"), unsafe_allow_html=True)
        c4.markdown(metric_card("Avg Pop Growth", f"{avg_grow:.2f}%", "All states YoY"), unsafe_allow_html=True)

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
                                  bgcolor="#1c1c1c", bordercolor="#2a2a2a"),
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
                    paper_bgcolor="#111111",
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
                    paper_bgcolor="#111111",
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
                        title=dict(text="Migration<br>Score", font=dict(size=11, color="#e8dfc4")),
                        tickfont=dict(size=10, color="#e8dfc4"),
                        thickness=14, len=0.65,
                        bgcolor="#16160f",
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
                    bgcolor="#0f0f0c", showland=True, landcolor="#1e2018",
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
                    paper_bgcolor="#16160f",
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=460,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    "<div style='margin-top:24px;padding:14px 12px;background:#16160f;"
                    "border:1px solid #3a3a2a;border-radius:8px;'>"
                    "<div style='font-size:0.78rem;font-weight:700;color:#CFB991;"
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
            textfont=dict(color="#e8dfc4", size=12),
            customdata=top10[["state_name", "pop_growth_pct", "key_companies"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pop Growth: %{customdata[1]:+.2f}%<br>"
                "Key Companies: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
            xaxis=dict(showgrid=True, gridcolor="#2d2d22", range=[0, 110],
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#e8dfc4", size=12)),
            margin=dict(t=20, b=20, l=60, r=60),
            height=320, font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
        metros_disp = metros_df.copy()
        metros_disp["Pop Growth %"] = metros_disp["Pop Growth %"].apply(lambda x: f"{x:+.1f}%")
        metros_disp["Job Growth %"] = metros_disp["Job Growth %"].apply(lambda x: f"{x:+.1f}%")
        metros_disp["Corp HQ Moves"] = metros_disp["Corp HQ Moves"].apply(lambda x: f"{x:+d}")

        def color_demand(val):
            colors = {
                "Very High": "background-color:#c8e6c9;color:#1b5e20;font-weight:700",
                "High":      "background-color:#dcedc8;color:#33691e;font-weight:600",
                "Moderate":  "background-color:#fff9c4;color:#f57f17",
                "Weak":      "background-color:#ffccbc;color:#bf360c",
                "Declining": "background-color:#ffcdd2;color:#b71c1c;font-weight:700",
            }
            return colors.get(val, "")

        def _highlight_metro_row(row):
            if _metro_city and _metro_city.lower() in row["Metro"].lower():
                return ["background-color: #f5f0e6; font-weight: 700"] * len(row)
            return [""] * len(row)

        st.dataframe(
            metros_disp.style.applymap(color_demand, subset=["CRE Demand"]).apply(_highlight_metro_row, axis=1),
            use_container_width=True, hide_index=True
        )

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
        fig_bubble.update_traces(textposition="middle center", textfont=dict(size=9, color="#e8dfc4"))
        fig_bubble.update_layout(
            plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
            xaxis=dict(showgrid=True, gridcolor="#2d2d22", zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            yaxis=dict(showgrid=True, gridcolor="#2d2d22",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            coloraxis_showscale=False, margin=dict(t=20, b=40),
            height=380, font=dict(family="Source Sans Pro", color="#e8dfc4"),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption(
            "Each bubble is a state. Bubble size reflects the composite migration score. "
            "The ideal CRE investment target sits in the upper-right — high population growth AND strong business migration. "
            "States in the lower-left are losing both residents and corporate presence."
        )


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

        from src.cre_pricing import compute_profit_matrix, get_top_opportunities, get_property_type_summary, CAP_RATE_BENCHMARKS

        cache_p = read_cache("pricing")
        if not stale_banner("pricing") or cache_p["data"] is None:
            st.stop()

        pdata    = cache_p["data"]
        reit_df  = pd.DataFrame(pdata["reits"])
        top_opps = pd.DataFrame(pdata["top_opps"])
        pt_summary = pd.DataFrame(pdata["pt_summary"])

        # ── KPI strip ──────────────────────────────────────────────────────────
        best_type  = pt_summary.iloc[0]
        worst_type = pt_summary.iloc[-1]
        best_opp   = top_opps.iloc[0]
        avg_cap    = pt_summary["Cap Rate"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Best Property Type", best_type["Property Type"].split("/")[0].strip(),
                                 f"Margin: {best_type['Eff Profit Margin']*100:.1f}%"), unsafe_allow_html=True)
        c2.markdown(metric_card("Weakest Type", worst_type["Property Type"].split("/")[0].strip(),
                                 "High vacancy / rent decline"), unsafe_allow_html=True)
        c3.markdown(metric_card("Avg Market Cap Rate", f"{avg_cap*100:.2f}%",
                                 "Blended all property types"), unsafe_allow_html=True)
        c4.markdown(metric_card("Top Market", best_opp["Market"].split(",")[0],
                                 f"{best_opp['Property Type'].split('/')[0].strip()} · Score {best_opp['Profit Score']}"),
                    unsafe_allow_html=True)

        # ── Live REIT Prices ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(f" Live REIT Prices — {datetime.today().strftime('%B %d, %Y')}")

        prop_types = reit_df["Property Type"].unique()
        tabs_reit  = st.tabs(list(prop_types))

        for tab_r, pt in zip(tabs_reit, prop_types):
            with tab_r:
                sub   = reit_df[reit_df["Property Type"] == pt].copy()
                col1, col2 = st.columns([2, 1])
                with col1:
                    valid = sub[sub["Price"].notna()]
                    if not valid.empty:
                        colors = [GOLD if r >= 0 else "#c62828" for r in valid["Daily Return"].fillna(0)]
                        fig_p = go.Figure(go.Bar(
                            x=valid["Ticker"], y=valid["Price"],
                            marker_color=colors,
                            text=valid["Price"].apply(lambda x: f"${x:.2f}" if x else "N/A"),
                            textposition="outside",
                            customdata=valid[["Company","Daily Return","Div Yield","Market Cap"]].values,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>Price: $%{y:.2f}<br>"
                                "Daily Return: %{customdata[1]:+.2%}<br>"
                                "Div Yield: %{customdata[2]:.2%}<br>"
                                "Market Cap: $%{customdata[3]:,.0f}<extra></extra>"
                            ),
                        ))
                        fig_p.update_layout(
                            plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
                            yaxis_title="Price ($)", margin=dict(t=30, b=30),
                            height=260, font=dict(family="Source Sans Pro", color="#e8dfc4"),
                        )
                        fig_p.update_xaxes(showgrid=False, tickfont=dict(color="#e8dfc4"))
                        fig_p.update_yaxes(gridcolor="#2d2d22", tickfont=dict(color="#e8dfc4"),
                                           title_font=dict(color="#e8dfc4"))
                        st.plotly_chart(fig_p, use_container_width=True)
                        st.caption(
                            "Live REIT share prices sourced from Yahoo Finance. "
                            "Higher dividend yield indicates more income relative to price — "
                            "useful for comparing income-generating potential across property types."
                        )
                with col2:
                    bench = CAP_RATE_BENCHMARKS.get(pt, {})
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("Cap Rate", f"{bench.get('cap_rate',0)*100:.2f}%", "Market benchmark"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("NOI Margin", f"{bench.get('noi_margin',0)*100:.1f}%", "After opex, before CapEx"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("Rent Growth YoY", f"{bench.get('rent_growth',0)*100:+.1f}%", "Market consensus"), unsafe_allow_html=True)

                disp = sub[["Ticker","Company","Market Focus","Price","Daily Return","Div Yield","Cap Rate","NOI Margin","Vacancy Rate","Rent Growth"]].copy()
                disp["Price"]        = disp["Price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                disp["Daily Return"] = disp["Daily Return"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
                disp["Div Yield"]    = disp["Div Yield"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                disp["Cap Rate"]     = disp["Cap Rate"].apply(lambda x: f"{x*100:.2f}%")
                disp["NOI Margin"]   = disp["NOI Margin"].apply(lambda x: f"{x*100:.1f}%")
                disp["Vacancy Rate"] = disp["Vacancy Rate"].apply(lambda x: f"{x*100:.1f}%")
                disp["Rent Growth"]  = disp["Rent Growth"].apply(lambda x: f"{x*100:+.1f}%")
                st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Profit Margin Heatmap ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Profit Margin Heatmap — Market × Property Type")

        profit_df = compute_profit_matrix()
        pivot = profit_df.pivot_table(
            index="Property Type", columns="Market",
            values="Eff Profit Margin", aggfunc="mean"
        )
        pivot.columns = [c.split(",")[0] for c in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values * 100, x=list(pivot.columns), y=list(pivot.index),
            colorscale=[[0.0, "#b71c1c"], [0.3, "#ef9a9a"], [0.5, "#fff9c4"], [0.7, "#a5d6a7"], [1.0, "#1b5e20"]],
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}", textfont=dict(size=9, color="#e8dfc4"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Margin: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Eff Margin %", thickness=14, len=0.8,
                          tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
        ))
        fig_heat.update_layout(
            paper_bgcolor="#16160f",
            xaxis=dict(tickangle=-35, tickfont=dict(size=9, color="#e8dfc4")),
            yaxis=dict(tickfont=dict(size=9, color="#e8dfc4")),
            margin=dict(t=20, b=100, l=180, r=20),
            height=420, font=dict(family="Source Sans Pro", color="#e8dfc4"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Effective Profit Margin = NOI Margin × (1 − Vacancy) × (1 + Rent Growth). "
            "Green cells are the most profitable combinations of market and property type. "
            "Sunbelt markets (Dallas, Austin, Miami, Nashville) consistently outperform gateway cities "
            "due to lower cap rate compression and stronger rent growth trajectories."
        )

        # ── Top 10 Opportunities ───────────────────────────────────────────────
        section(" Top 10 Highest Profit Margin Opportunities Right Now")
        st.dataframe(top_opps, use_container_width=True, hide_index=True)
        st.caption(
            "Ranked by Effective Profit Margin across all tracked markets and property types. "
            "Industrial and Data Center assets in Sunbelt markets dominate the top rankings due to "
            "low vacancy, high NOI margins, and strong rent growth — key indicators of durable cash flow."
        )

        # ── Property Type Comparison ───────────────────────────────────────────
        section(" Property Type Performance Comparison")
        _user_pt = st.session_state.user_intent.get("property_type")
        colors_pt = []
        for i, row in pt_summary.iterrows():
            if _user_pt and _intent_matches_property_type(_user_pt, row["Property Type"]):
                colors_pt.append("#1b5e20")  # highlight green for user's focus
            elif i == 0:
                colors_pt.append(GOLD)
            elif i < 3:
                colors_pt.append(GOLD_DARK)
            else:
                colors_pt.append("#aaa")
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Bar(
            x=pt_summary["Property Type"].str.split("/").str[0].str.strip(),
            y=pt_summary["Eff Profit Margin"] * 100,
            marker_color=colors_pt,
            text=pt_summary["Eff Profit Margin"].apply(lambda x: f"{x*100:.1f}%"),
            textposition="outside", name="Effective Profit Margin",
        ))
        fig_pt.add_trace(go.Scatter(
            x=pt_summary["Property Type"].str.split("/").str[0].str.strip(),
            y=pt_summary["Cap Rate"] * 100,
            mode="lines+markers", name="Cap Rate",
            line=dict(color=BLACK, width=2, dash="dash"),
            marker=dict(size=7), yaxis="y2",
        ))
        fig_pt.update_layout(
            plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
            yaxis=dict(title="Effective Profit Margin (%)", gridcolor="#2d2d22",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            yaxis2=dict(title="Cap Rate (%)", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4")),
            margin=dict(t=40, b=60),
            height=360, font=dict(family="Source Sans Pro", color="#e8dfc4"),
        )
        fig_pt.update_xaxes(showgrid=False, tickangle=-15, tickfont=dict(color="#e8dfc4"))
        st.plotly_chart(fig_pt, use_container_width=True)
        st.caption(
            "Bars show effective profit margin (left axis); the line shows cap rate (right axis). "
            "Industrial leads on margin due to near-zero vacancy and rapid rent growth driven by e-commerce and reshoring. "
            "Office carries the highest cap rate but the lowest margin — reflecting elevated vacancy and negative rent growth in most markets. "
            "Source: CBRE, JLL, and Green Street 2024–2025 reports. Live REIT prices from Yahoo Finance."
        )

        # ── Rate-Adjusted View ─────────────────────────────────────────────────
        cache_r2 = read_cache("rates")
        rdata2   = cache_r2.get("data") or {}
        cap_adj2 = rdata2.get("cap_rate_adjustments", [])
        cur_10y2 = rdata2.get("current_10y") or 0.0

        if cap_adj2 and cur_10y2:
            st.markdown("<br>", unsafe_allow_html=True)
            section(f" Rate-Adjusted Cap Rates — 10Y at {cur_10y2:.2f}%")
            show_adj = st.toggle("Show rate-adjusted profit matrix", value=True, key="rate_adj_toggle")
            if show_adj:
                adj_df2 = pd.DataFrame(cap_adj2)
                baseline2 = rdata2.get("baseline_10y", 4.0)
                delta2    = cur_10y2 - baseline2
                direction2 = "above" if delta2 > 0 else "below"

                st.caption(
                    f"10Y Treasury ({cur_10y2:.2f}%) is {abs(delta2):.2f}% {direction2} the "
                    f"{baseline2:.1f}% baseline. Cap rates are adjusted using sector-specific betas."
                )

                colors_adj = [
                    "#b71c1c" if r > 0 else "#1b5e20"
                    for r in adj_df2["Rate Adjustment bps"]
                ]
                fig_adj = go.Figure()
                pt_short = adj_df2["Property Type"].str.split("/").str[0].str.strip()
                fig_adj.add_trace(go.Bar(
                    name="Static Cap Rate (%)",
                    x=pt_short,
                    y=adj_df2["Baseline Cap Rate"],
                    marker_color="#CFB991",
                    text=adj_df2["Baseline Cap Rate"].apply(lambda v: f"{v:.2f}%"),
                    textposition="inside",
                ))
                fig_adj.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate (%)",
                    x=pt_short,
                    y=adj_df2["Adjusted Cap Rate"],
                    marker_color=colors_adj,
                    opacity=0.85,
                    text=adj_df2["Rate Adjustment bps"].apply(
                        lambda d: f"{'+' if d > 0 else ''}{d:.0f}bps"
                    ),
                    textposition="outside",
                ))
                fig_adj.update_layout(
                    barmode="group",
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickangle=-15, tickfont=dict(color="#e8dfc4", size=10)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4")),
                    margin=dict(t=40, b=60), height=340,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
                )
                st.plotly_chart(fig_adj, use_container_width=True)
                st.caption(
                    "Shows the static (benchmark) cap rate alongside the rate-adjusted cap rate for each property type. "
                    "When the 10-year Treasury rises above the baseline, cap rates expand — meaning asset values fall "
                    "for the same NOI. Office and retail are most sensitive; industrial and multifamily are more resilient."
                )


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
                <div style="background:#16160f;border:1px solid #8E6F3E;border-radius:8px;
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
        from src.vacancy_agent import NATIONAL_VACANCY, MARKET_VACANCY, TREND_ARROW, TREND_COLOR

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
                textfont=dict(size=11, color="#0f0f0c"),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
                colorbar=dict(
                    title=dict(text="Vacancy %", font=dict(color="#e8dfc4", size=11)),
                    tickfont=dict(color="#e8dfc4", size=10),
                    thickness=14,
                    bgcolor="#16160f",
                    bordercolor="#3a3a2a",
                ),
            ))
            fig_heat.update_layout(
                plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                margin=dict(t=20, b=20, l=180, r=20),
                height=620,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
                xaxis=dict(side="top", tickfont=dict(color="#e8dfc4", size=12)),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=11)),
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
                    else:        return "color: #CFB991"
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
                plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
                yaxis_title="% Above/Below SMA-60",
                yaxis=dict(gridcolor="#2d2d22", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#2d2d22", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                plot_bgcolor="#1a1a14", paper_bgcolor="#16160f",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#2d2d22", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TAB — MACRO ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
with main_tab_macro:
    tab_rates, tab_labor, tab_gdp, tab_inflation, tab_credit = st.tabs([
        "Rate Environment",
        "Labor Market & Tenant Demand",
        "GDP & Economic Growth",
        "Inflation",
        "Credit & Capital Markets",
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
        signal  = env.get("signal", "CAUTIOUS")
        sig_clr = {"BULLISH": "#1b5e20", "CAUTIOUS": "#e65100", "BEARISH": "#b71c1c"}.get(signal, "#333")
        bg_clr  = {"BULLISH": "#e8f5e9", "CAUTIOUS": "#fff3e0", "BEARISH": "#ffebee"}.get(signal, "#f5f5f5")
        st.markdown(f"""
        <div style="background:{bg_clr};border-left:6px solid {sig_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{sig_clr};">
            {env.get('icon','')} Rate Environment: {signal}
          </div>
          <div style="color:#333;margin-top:6px;font-size:0.95rem;">{env.get('summary','')}</div>
          <ul style="margin-top:10px;color:#444;font-size:0.88rem;">
            {"".join(f"<li>{b}</li>" for b in env.get('bullets', []))}
          </ul>
          <div style="font-size:0.75rem;color:#888;margin-top:8px;">Last updated: {cached_at[:19].replace('T',' ')}</div>
        </div>""", unsafe_allow_html=True)

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
                    textfont=dict(size=11, color="#e8dfc4"),
                    hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
                ))
                fig_yc.add_hline(y=values[0] if values else 0, line_dash="dot",
                                  line_color="#aaa", line_width=1)
                fig_yc.update_layout(
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Yield (%)", ticksuffix="%",
                               gridcolor="#2d2d22", tickfont=dict(color="#e8dfc4"),
                               title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=30, l=60, r=20), height=300,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
            paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
            yaxis=dict(title="Rate (%)", ticksuffix="%", gridcolor="#2d2d22",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            xaxis=dict(tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#e8dfc4", size=11)),
            margin=dict(t=40, b=40), height=380,
            font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    marker_color="#CFB991",
                    text=[f"{v:.2f}%" for v in base_caps],
                    textposition="inside", textfont=dict(color="#e8dfc4", size=10),
                ))
                fig_cap.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate",
                    x=pt_labels, y=adj_caps,
                    marker_color=bar_colors,
                    opacity=0.85,
                    text=[f"{v:.2f}%\n({'+' if d>0 else ''}{d:.0f}bps)" for v, d in zip(adj_caps, adj_bps)],
                    textposition="outside", textfont=dict(color="#e8dfc4", size=9),
                ))
                fig_cap.update_layout(
                    barmode="group",
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickangle=-20, tickfont=dict(color="#e8dfc4", size=9)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4")),
                    margin=dict(t=40, b=60), height=360,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    xaxis=dict(title="Near-Term Debt / Total Debt (%)", ticksuffix="%",
                               gridcolor="#2d2d22", tickfont=dict(color="#e8dfc4"),
                               title_font=dict(color="#e8dfc4")),
                    yaxis=dict(tickfont=dict(color="#e8dfc4", size=9)),
                    margin=dict(t=20, b=40, l=60, r=60), height=460,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
        sig_clr_lm = {"STRONG": "#1b5e20", "MODERATE": "#e65100", "SOFT": "#b71c1c"}.get(sig_label, "#555")
        bg_clr_lm  = {"STRONG": "#e8f5e9", "MODERATE": "#fff3e0", "SOFT": "#ffebee"}.get(sig_label, "#f5f5f5")
        st.markdown(f"""
        <div style="background:{bg_clr_lm};border-left:6px solid {sig_clr_lm};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{sig_clr_lm};">
            Tenant Demand Signal: {sig_label}  &nbsp;·&nbsp; Score {sig_score}/100
          </div>
          <div style="color:#555;margin-top:6px;font-size:0.9rem;">
            Synthesized from nonfarm payrolls, job openings, unemployment trend, and sector ETF momentum.
            STRONG ≥ 65 · MODERATE 41–64 · SOFT ≤ 40
          </div>
        </div>""", unsafe_allow_html=True)

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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                xaxis=dict(title="Month-over-Month Change (%)", ticksuffix="%",
                           gridcolor="#2d2d22", tickfont=dict(color="#e8dfc4"),
                           title_font=dict(color="#e8dfc4")),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=10)),
                margin=dict(t=20, b=40, l=220, r=80), height=400,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                yaxis=dict(title="6-Month Return (%)", ticksuffix="%",
                           gridcolor="#2d2d22", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickangle=-20, tickfont=dict(color="#e8dfc4", size=9)),
                margin=dict(t=30, b=80), height=360,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                xaxis=dict(title="Unemployment Rate (%)", ticksuffix="%",
                           gridcolor="#2d2d22", tickfont=dict(color="#e8dfc4"),
                           title_font=dict(color="#e8dfc4")),
                yaxis=dict(tickfont=dict(color="#e8dfc4", size=9)),
                margin=dict(t=20, b=40, l=260, r=80), height=360,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
        cycle_clr = {"EXPANSION": "#1b5e20", "SLOWDOWN": "#e65100", "CONTRACTION": "#b71c1c"}.get(cycle_label, "#555")
        cycle_bg  = {"EXPANSION": "#e8f5e9", "SLOWDOWN": "#fff3e0", "CONTRACTION": "#ffebee"}.get(cycle_label, "#f5f5f5")
        cycle_icon = ""
        st.markdown(f"""
        <div style="background:{cycle_bg};border-left:6px solid {cycle_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{cycle_clr};">
            {cycle_icon} Economic Cycle: {cycle_label} &nbsp;·&nbsp; Score {cycle_score}/100
          </div>
          <div style="color:#555;margin-top:8px;font-size:0.92rem;font-style:italic;">
            {g_cycle.get("cre_implication", "")}
          </div>
          <ul style="margin-top:10px;color:#444;font-size:0.88rem;">
            {"".join(f"<li>{b}</li>" for b in g_cycle.get("bullets", []))}
          </ul>
        </div>""", unsafe_allow_html=True)

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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Annualized %", ticksuffix="%", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=40), height=320,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Index (2017=100)", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=40), height=320,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Index", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=40), height=300,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="Index (0 = trend growth)", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=40), height=300,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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

        # ── Signal Banner ───────────────────────────────────────────────────────
        inf_label = inf_signal.get("label", "UNKNOWN")
        inf_score = inf_signal.get("score", 50)
        inf_clr = {"HOT": "#b71c1c", "MODERATE": "#e65100", "COOLING": "#1b5e20"}.get(inf_label, "#555")
        inf_bg  = {"HOT": "#ffebee", "MODERATE": "#fff3e0", "COOLING": "#e8f5e9"}.get(inf_label, "#f5f5f5")
        inf_icon = ""
        st.markdown(f"""
        <div style="background:{inf_bg};border-left:6px solid {inf_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{inf_clr};">
            {inf_icon} Inflation Regime: {inf_label} &nbsp;·&nbsp; Score {inf_score}/100
          </div>
          <div style="color:#555;margin-top:6px;font-size:0.92rem;">{inf_signal.get("summary", "")}</div>
          <ul style="margin-top:10px;color:#444;font-size:0.88rem;">
            {"".join(f"<li>{b}</li>" for b in inf_signal.get("bullets", []))}
          </ul>
        </div>""", unsafe_allow_html=True)

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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                yaxis=dict(title="YoY %", ticksuffix="%", gridcolor="#2d2d22",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4", size=10)),
                margin=dict(t=40, b=40), height=340,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                yaxis=dict(title="YoY %", ticksuffix="%", gridcolor="#2d2d22",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4", size=10)),
                margin=dict(t=40, b=40), height=340,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
            paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
            yaxis=dict(title="Implied Inflation (%)", ticksuffix="%", gridcolor="#2d2d22",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            xaxis=dict(tickfont=dict(color="#e8dfc4")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#e8dfc4", size=11)),
            margin=dict(t=40, b=40), height=320,
            font=dict(family="Source Sans Pro", color="#e8dfc4"),
        )
        st.plotly_chart(fig_be, use_container_width=True)
        st.caption(
            "Breakeven inflation = nominal Treasury yield minus TIPS yield — the market's consensus inflation forecast. "
            "Elevated breakevens signal that the Fed is unlikely to cut rates soon, keeping cap rates elevated "
            "and compressing CRE asset values."
        )
        st.caption("Data: Federal Reserve Bank of St. Louis (FRED). This is research, not financial advice.")


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

        # ── Signal Banner ───────────────────────────────────────────────────────
        cr_label = cr_signal.get("label", "UNKNOWN")
        cr_score = cr_signal.get("score", 50)
        cr_clr = {"LOOSE": "#1b5e20", "NEUTRAL": "#e65100", "TIGHT": "#b71c1c"}.get(cr_label, "#555")
        cr_bg  = {"LOOSE": "#e8f5e9", "NEUTRAL": "#fff3e0", "TIGHT": "#ffebee"}.get(cr_label, "#f5f5f5")
        cr_icon = ""
        st.markdown(f"""
        <div style="background:{cr_bg};border-left:6px solid {cr_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{cr_clr};">
            {cr_icon} Credit Conditions: {cr_label} &nbsp;·&nbsp; Score {cr_score}/100
          </div>
          <div style="color:#555;margin-top:6px;font-size:0.92rem;">{cr_signal.get("summary", "")}</div>
          <ul style="margin-top:10px;color:#444;font-size:0.88rem;">
            {"".join(f"<li>{b}</li>" for b in cr_signal.get("bullets", []))}
          </ul>
        </div>""", unsafe_allow_html=True)

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
                paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                yaxis=dict(title="IG / BBB Spread (bps)", gridcolor="#2d2d22",
                           tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                yaxis2=dict(title="HY Spread (bps)", overlaying="y", side="right",
                            tickfont=dict(color="#c62828"), title_font=dict(color="#c62828")),
                xaxis=dict(tickfont=dict(color="#e8dfc4")),
                legend=dict(orientation="h", y=1.1, font=dict(color="#e8dfc4", size=10)),
                margin=dict(t=40, b=40), height=360,
                font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
                    paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
                    yaxis=dict(title="VIX Level", gridcolor="#2d2d22",
                               tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
                    xaxis=dict(tickfont=dict(color="#e8dfc4")),
                    margin=dict(t=30, b=40), height=360,
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
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
            paper_bgcolor="#16160f", plot_bgcolor="#1a1a14",
            yaxis=dict(title="Net % Tightening", ticksuffix="%", gridcolor="#2d2d22",
                       zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#e8dfc4"), title_font=dict(color="#e8dfc4")),
            xaxis=dict(tickfont=dict(color="#e8dfc4")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#e8dfc4", size=11)),
            margin=dict(t=40, b=40), height=340,
            font=dict(family="Source Sans Pro", color="#e8dfc4"),
        )
        st.plotly_chart(fig_ls, use_container_width=True)
        st.caption(
            "Fed Senior Loan Officer Survey (quarterly). Positive = banks tightening loan standards (less credit supply). "
            "Negative = easing (more credit supply). CRE loan tightening directly restricts acquisition and development financing."
        )
        st.caption("Data: Federal Reserve Bank of St. Louis (FRED). This is research, not financial advice.")

    # ── Meet the Team ─────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#16160f; border:1px solid #8E6F3E; border-top:3px solid #CFB991;
            border-radius:8px; padding:32px 36px; margin-top:24px;">
  <div style="text-align:center; margin-bottom:24px;">
    <span style="color:#CFB991; font-size:1.3rem; font-weight:700; letter-spacing:2px;
                 text-transform:uppercase;">Meet the Team</span>
    <div style="color:#a09880; font-size:0.85rem; margin-top:4px;">
      MGMT 690: AI Leadership &nbsp;&middot;&nbsp; Purdue Daniels School of Business
    </div>
  </div>
  <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap;">
    <div style="background:#1a1a14; border:1px solid #8E6F3E; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Aayman Afzal</div>
      <a href="https://www.linkedin.com/in/aayman-afzal" target="_blank"
         style="color:#CFB991; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1a1a14; border:1px solid #8E6F3E; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Ajinkya Kodnikar</div>
      <a href="https://www.linkedin.com/in/ajinkya-kodnikar" target="_blank"
         style="color:#CFB991; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1a1a14; border:1px solid #8E6F3E; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Oyu Amar</div>
      <a href="https://www.linkedin.com/in/oyuamar" target="_blank"
         style="color:#CFB991; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
    <div style="background:#1a1a14; border:1px solid #8E6F3E; border-radius:8px;
                padding:20px 28px; text-align:center; min-width:160px;">
      <div style="font-size:2rem; margin-bottom:8px;">&#128100;</div>
      <div style="color:#e8dfc4; font-weight:700; font-size:1rem;">Ricardo Ruiz</div>
      <a href="https://www.linkedin.com/in/ricardoruizjr" target="_blank"
         style="color:#CFB991; font-size:0.78rem; text-decoration:none;">LinkedIn</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
