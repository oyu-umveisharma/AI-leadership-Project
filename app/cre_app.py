"""
Commercial Real Estate Intelligence Platform
Purdue MSF | Group AI Project

Eight background agents update independently on schedules:
  Agent 1 — Population & Migration    (every 6h)
  Agent 2 — CRE Pricing & Profit      (every 1h)
  Agent 3 — Company Predictions       (every 24h, LLM)
  Agent 4 — Debugger / Monitor        (every 30min)
  Agent 5 — News & Announcements      (every 4h)
  Agent 6 — Interest Rate & Debt      (every 1h, requires FRED_API_KEY)
  Agent 7 — Energy & Construction     (every 6h)
  Agent 8 — Sustainability & ESG      (every 6h)

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


def _normalize_input(text: str) -> str:
    """Normalize user input to proper title case with special handling."""
    if not text:
        return text
    text = text.strip()
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
    "los angeles": "CA", "san francisco": "CA", "san diego": "CA", "san jose": "CA",
    "sacramento": "CA", "irvine": "CA", "oakland": "CA", "long beach": "CA",
    "new york": "NY", "brooklyn": "NY", "queens": "NY", "bronx": "NY", "buffalo": "NY",
    "chicago": "IL", "aurora": "IL", "naperville": "IL",
    "houston": "TX", "dallas": "TX", "austin": "TX", "san antonio": "TX", "fort worth": "TX", "el paso": "TX",
    "miami": "FL", "tampa": "FL", "orlando": "FL", "jacksonville": "FL", "fort lauderdale": "FL", "st. petersburg": "FL",
    "phoenix": "AZ", "tucson": "AZ", "mesa": "AZ", "scottsdale": "AZ",
    "seattle": "WA", "tacoma": "WA", "spokane": "WA", "bellevue": "WA",
    "denver": "CO", "colorado springs": "CO",
    "atlanta": "GA", "savannah": "GA", "augusta": "GA",
    "boston": "MA", "cambridge": "MA", "worcester": "MA",
    "portland": "OR", "salem": "OR", "eugene": "OR",
    "las vegas": "NV", "reno": "NV", "henderson": "NV",
    "nashville": "TN", "memphis": "TN", "knoxville": "TN", "chattanooga": "TN",
    "charlotte": "NC", "raleigh": "NC", "durham": "NC",
    "salt lake city": "UT", "provo": "UT",
    "minneapolis": "MN", "st. paul": "MN",
    "indianapolis": "IN", "fort wayne": "IN",
    "columbus": "OH", "cleveland": "OH", "cincinnati": "OH",
    "detroit": "MI", "grand rapids": "MI", "ann arbor": "MI",
    "philadelphia": "PA", "pittsburgh": "PA",
    "kansas city": "MO", "st. louis": "MO",
    "richmond": "VA", "virginia beach": "VA", "arlington": "VA",
    "baltimore": "MD", "columbia": "MD",
    "charleston": "SC", "greenville": "SC",
    "sioux falls": "SD", "rapid city": "SD", "aberdeen": "SD",
    "fargo": "ND", "bismarck": "ND",
    "omaha": "NE", "lincoln": "NE",
    "boise": "ID", "meridian": "ID",
}


def _get_location_scope(location_str: str) -> dict:
    """Parse a location string into city/state components."""
    if not location_str:
        return {"city": None, "state": None, "scope": "national"}

    loc = location_str.strip()

    # "City, ST" pattern
    if "," in loc:
        parts = [p.strip() for p in loc.split(",", 1)]
        city = parts[0]
        st_part = parts[1].upper().rstrip(".") if len(parts) > 1 else ""
        state_name = _US_STATES.get(st_part, st_part.title())
        return {"city": city, "state": state_name, "state_abbr": st_part if st_part in _US_STATES else None, "scope": "city"}

    # Check if it's a full state name
    if loc.lower() in _STATE_NAME_TO_ABBR:
        abbr = _STATE_NAME_TO_ABBR[loc.lower()]
        return {"city": None, "state": _US_STATES[abbr], "state_abbr": abbr, "scope": "state"}

    # Check if it's a state abbreviation
    if loc.upper() in _US_STATES:
        return {"city": None, "state": _US_STATES[loc.upper()], "state_abbr": loc.upper(), "scope": "state"}

    # City-to-state lookup for bare city names
    abbr = _CITY_TO_STATE.get(loc.lower())
    if abbr:
        return {"city": loc, "state": _US_STATES[abbr], "state_abbr": abbr, "scope": "city"}

    # Unknown city — no state resolved
    return {"city": loc, "state": None, "state_abbr": None, "scope": "city"}


# Synonyms that map to canonical property types
_PT_SYNONYMS = {
    "warehouse": "Industrial", "logistics": "Industrial", "distribution": "Industrial",
    "fulfillment": "Industrial", "manufacturing": "Industrial", "factory": "Industrial",
    "apartment": "Multifamily", "apartments": "Multifamily", "residential": "Multifamily",
    "shop": "Retail", "shopping": "Retail", "mall": "Retail", "store": "Retail",
    "medical": "Healthcare", "hospital": "Healthcare", "clinic": "Healthcare",
    "hotel": "Hospitality", "motel": "Hospitality",
    "storage": "Self-Storage", "self-storage": "Self-Storage",
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


def _parse_intent(raw: str) -> dict:
    """Parse free-text input into property_type and location."""
    raw_lower = raw.lower().strip()

    # Detect property type — check synonyms first, then canonical names
    prop_type = None
    for syn, canonical in _PT_SYNONYMS.items():
        if syn in raw_lower:
            prop_type = canonical
            break
    if not prop_type:
        for pt in ["industrial", "multifamily", "office", "retail", "data center", "healthcare", "hospitality"]:
            if pt in raw_lower:
                prop_type = _normalize_input(pt)
                break

    # Detect region keywords
    region = None
    region_states = None
    for rname, rstates in _REGION_STATES.items():
        if rname in raw_lower:
            region = rname.title()
            region_states = rstates
            break

    # Heuristic: everything after "in " or "on the " is the location
    location = None
    for prep in [" in the ", " in ", " on the ", " on "]:
        if prep in raw_lower:
            location = raw[raw_lower.index(prep) + len(prep):].strip().rstrip(".")
            break
    if not location and "," in raw:
        parts = raw.rsplit(",", 1)
        if len(parts) == 2 and len(parts[1].strip()) <= 3:
            location = raw.strip()

    # If we detected a region, don't treat it as a city/state location
    if region and not location:
        location = region
    elif region and location and location.lower() == region.lower():
        pass  # already set

    # Normalize
    location = _normalize_input(location) if location else None
    loc_scope = _get_location_scope(location)

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
        color: {BLACK};
        margin-bottom: 4px;
      }}
      .welcome-sub {{
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 32px;
      }}
      .welcome-prompt {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {BLACK};
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
      <div class="welcome-icon">🏢</div>
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
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Source Sans Pro', sans-serif; }}

  .cre-header {{
    background: {BLACK};
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
  }}
  .cre-header h1 {{ color: {GOLD}; font-size: 1.6rem; font-weight: 700; margin: 0; }}
  .cre-header span {{ color: white; font-size: 0.85rem; opacity: 0.8; }}
  .gold-bar {{ height: 4px; background: linear-gradient(90deg, {GOLD_DARK}, {GOLD}, {GOLD_DARK}); margin-bottom: 24px; }}

  .agent-card {{
    background: {BLACK};
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
    border-left: 5px solid {GOLD};
  }}
  .agent-card .agent-label {{
    color: {GOLD};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }}
  .agent-card .agent-text {{
    color: #eee;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }}

  .metric-card {{
    background: white;
    border: 1px solid #e0e0e0;
    border-top: 4px solid {GOLD};
    border-radius: 6px;
    padding: 16px;
    text-align: center;
  }}
  .metric-card .label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: {BLACK}; margin: 4px 0; }}
  .metric-card .sub   {{ font-size: 0.78rem; color: #888; }}

  .section-header {{
    background: {BLACK};
    color: {GOLD};
    padding: 9px 16px;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 700;
    margin: 24px 0 14px 0;
    border-left: 5px solid {GOLD};
  }}

  .listing-card {{
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-left: 4px solid {GOLD};
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
  }}
  .listing-card .l-price {{ font-size: 1.4rem; font-weight: 700; color: {BLACK}; }}
  .listing-card .l-address {{ font-size: 0.9rem; color: #333; margin: 2px 0; }}
  .listing-card .l-detail {{ font-size: 0.8rem; color: #666; }}
  .listing-card .l-tag {{
    display: inline-block;
    background: {GOLD};
    color: {BLACK};
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 4px;
    margin-top: 4px;
  }}

  .status-ok    {{ color: #2e7d32; font-weight: 700; }}
  .status-error {{ color: #b71c1c; font-weight: 700; }}
  .status-run   {{ color: #e65100; font-weight: 700; }}
  .status-idle  {{ color: #757575; }}

  div[data-testid="stTabs"] button {{
    font-weight: 600 !important;
    font-size: 0.92rem !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cre-header">
  <h1> CRE Intelligence Platform</h1>
  <span>Purdue University · Daniels School of Business · MSF Program &nbsp;|&nbsp; {datetime.today().strftime('%B %d, %Y')}</span>
</div>
<div class="gold-bar"></div>
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
        f'<div style="background:{BLACK};color:{GOLD};padding:8px 16px;border-radius:4px;'
        f'font-size:0.9rem;margin-bottom:16px;display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:1.1rem;">🎯</span>'
        f'<span>Currently analyzing: <b>{_focus_label}</b></span>'
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
                f'<div style="background:#f5f0e6;border-left:4px solid {GOLD};padding:10px 16px;'
                f'border-radius:4px;margin:8px 0 12px 0;font-size:0.9rem;color:#333;">'
                f'<b>🎯 Insight for {label}:</b> {insight}</div>',
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
        st.warning(f"⏳ Agent is fetching {cache_key} data for the first time — please wait ~30 seconds and refresh.")
        return False
    age = cache_age_label(cache_key)
    if c.get("stale"):
        st.warning(f"⚠ Data is stale (last updated {age}). Agent may be restarting.")
    else:
        st.caption(f" Last updated: {age} · Auto-refreshes in background")
    return True

def agent_last_updated(agent_name: str):
    st.caption(f"Last updated: {cache_age_label(agent_name)}")


# ── Meet the Team (fixed footer) ──────────────────────────────────────────────
st.markdown(f"""
<style>
  .meet-team-footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background: {BLACK};
    padding: 14px 32px;
    border-top: 3px solid {GOLD};
    text-align: center;
  }}
  .meet-team-footer .label {{
    color: {GOLD};
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
  }}
  .meet-team-footer .names {{
    display: flex;
    justify-content: center;
    gap: 36px;
    flex-wrap: wrap;
  }}
  .meet-team-footer a {{
    color: {GOLD} !important;
    text-decoration: underline !important;
    font-size: 0.9rem;
    font-weight: 600;
  }}
  .meet-team-footer .course {{
    color: #888;
    font-size: 0.7rem;
    margin-top: 6px;
  }}
  /* Push page content up so footer doesn't overlap last element */
  .main .block-container {{ padding-bottom: 90px; }}
</style>
<div class="meet-team-footer">
  <div class="label">Meet the Team</div>
  <div class="names">
    <a href="https://www.linkedin.com/in/aayman-afzal/" target="_blank">Aayman Afzal</a>
    <a href="https://www.linkedin.com/in/ajinkyakodnikar/" target="_blank">Ajinkya Kodnikar</a>
    <a href="https://www.linkedin.com/in/oyu-amar/" target="_blank">Oyu Amar</a>
    <a href="https://www.linkedin.com/in/ricardo-ruiz1/" target="_blank">Ricardo Ruiz</a>
  </div>
  <div class="course">MGMT 690 · AI Leadership · Purdue Daniels School of Business</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
main_tab_re, main_tab_energy = st.tabs(["Real Estate", "Energy"])

with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "System Monitor",
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

        if _map_city:
            coords = _CITY_COORDS.get(_map_city.lower())
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

            fig_map = go.Figure(go.Choropleth(
                locations=mig_df["state_abbr"],
                z=mig_df["composite_score"],
                locationmode="USA-states",
                colorscale=[
                    [0.0,  "#b71c1c"],
                    [0.25, "#ef5350"],
                    [0.45, "#ffcdd2"],
                    [0.55, "#fff9c4"],
                    [0.70, "#a5d6a7"],
                    [0.85, "#388e3c"],
                    [1.0,  "#1b5e20"],
                ],
                zmin=0, zmax=100,
                marker=dict(line=dict(width=_marker_line_widths, color=_marker_line_colors)),
                colorbar=dict(
                    title=dict(text="Migration<br>Score", font=dict(size=11)),
                    tickfont=dict(size=10), thickness=14, len=0.7,
                ),
                text=mig_df.apply(
                    lambda r: f"<b>{r['state_name']}</b><br>"
                              f"Pop Growth: {r['pop_growth_pct']:+.2f}%<br>"
                              f"Business Score: {r['biz_score']}<br>"
                              f"Composite: {r['composite_score']}<br>"
                              f"Key Companies: {r['key_companies']}<br>"
                              f"<i>{r['growth_drivers']}</i>",
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
            if _zoom_coords and _zoom_scale:
                fig_map.update_layout(
                    geo=dict(scope="usa", showlakes=True, lakecolor="lightblue",
                             bgcolor="white", showland=True, landcolor="#f5f5f5",
                             projection_scale=_zoom_scale,
                             center=dict(lat=_zoom_coords[0], lon=_zoom_coords[1])),
                )
            else:
                fig_map.update_layout(
                    geo=dict(scope="usa", showlakes=True, lakecolor="lightblue",
                             bgcolor="white", showland=True, landcolor="#f5f5f5",
                             projection_scale=1, center=dict(lat=38, lon=-96)),
                )

            fig_map.update_layout(
                paper_bgcolor="white",
                margin=dict(t=10, b=10, l=0, r=0),
                height=460,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
                dragmode=False,
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
            st.caption(
                "Darker green indicates states with the strongest combined population inflow and business migration. "
                "These markets historically see the earliest and sharpest increases in CRE demand — "
                "particularly for multifamily, industrial, and mixed-use properties."
            )

        with legend_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for color, label in [
                ("#1b5e20", " High Growth (70–100)"),
                ("#388e3c", " Growing (55–70)"),
                ("#fff9c4", " Stable (45–55)"),
                ("#ef5350", " Declining (25–45)"),
                ("#b71c1c", "⛔ High Outflow (<25)"),
            ]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin:6px 0;font-size:0.82rem;'>"
                    f"<div style='width:16px;height:16px;background:{color};border-radius:3px;margin-right:8px;flex-shrink:0;'></div>"
                    f"{label}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Score = 60% Pop Growth + 40% Business Migration Index")

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
            textfont=dict(color="#1a1a1a", size=12),
            customdata=top10[["state_name", "pop_growth_pct", "key_companies"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pop Growth: %{customdata[1]:+.2f}%<br>"
                "Key Companies: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[0, 110],
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#1a1a1a", size=12)),
            margin=dict(t=20, b=20, l=60, r=60),
            height=320, font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
        )
        fig_bubble.update_traces(textposition="middle center", textfont=dict(size=9, color="#1a1a1a"))
        fig_bubble.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            coloraxis_showscale=False, margin=dict(t=20, b=40),
            height=380, font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis_title="Price ($)", margin=dict(t=30, b=30),
                            height=260, font=dict(family="Source Sans Pro", color="#1a1a1a"),
                        )
                        fig_p.update_xaxes(showgrid=False, tickfont=dict(color="#1a1a1a"))
                        fig_p.update_yaxes(gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                                           title_font=dict(color="#1a1a1a"))
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
            texttemplate="%{text}", textfont=dict(size=9, color="#1a1a1a"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Margin: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Eff Margin %", thickness=14, len=0.8,
                          tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
        ))
        fig_heat.update_layout(
            paper_bgcolor="white",
            xaxis=dict(tickangle=-35, tickfont=dict(size=9, color="#1a1a1a")),
            yaxis=dict(tickfont=dict(size=9, color="#1a1a1a")),
            margin=dict(t=20, b=100, l=180, r=20),
            height=420, font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Effective Profit Margin (%)", gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis2=dict(title="Cap Rate (%)", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
            margin=dict(t=40, b=60),
            height=360, font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        fig_pt.update_xaxes(showgrid=False, tickangle=-15, tickfont=dict(color="#1a1a1a"))
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
        cur_10y2 = rdata2.get("current_10y")

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
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                               tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickangle=-15, tickfont=dict(color="#1a1a1a", size=10)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
                    margin=dict(t=40, b=60), height=340,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                            padding:16px 20px;margin-bottom:12px;border-left:4px solid {badge_fg};">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="background:{badge_bg};color:{badge_fg};font-size:0.72rem;
                                 font-weight:700;padding:2px 8px;border-radius:4px;
                                 text-transform:uppercase;letter-spacing:0.5px;">{atype}</span>
                    <span style="font-size:1rem;font-weight:700;color:#1a1a1a;">{co}{ticker_str}</span>
                    <span style="font-size:0.85rem;color:#555;margin-left:auto;">{loc}</span>
                  </div>
                  <div style="font-size:0.9rem;color:#333;margin-bottom:6px;">{detail}</div>
                  {"<div style='font-size:0.82rem;color:#555;margin-bottom:6px;'>" + meta_line + "</div>" if meta_line else ""}
                  {"<div style='font-size:0.82rem;color:#1b5e20;'><b>CRE Opportunity:</b> " + impact + "</div>" if impact else ""}
                  <div style="font-size:0.72rem;color:#aaa;margin-top:6px;">Source: {source}</div>
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
                from src.cre_listings import get_cheapest_buildings
                listings[_user_abbr_b] = get_cheapest_buildings(_user_abbr_b, n=10)
                if _user_abbr_b not in top3_abbr:
                    top3_abbr = [_user_abbr_b] + list(top3_abbr)
            except Exception:
                pass

        if not listings:
            st.info("Listings will appear after the first scheduled agent run (every 24 hours).")
        else:
            from src.cre_listings import format_listing_card

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
                    price_fmt  = f"${listing.get('price', 0):,}"
                    sqft_fmt   = f"{listing.get('sqft', 0):,} sqft"
                    ppsf_fmt   = f"${listing.get('price_per_sqft', 0):.0f}/sqft"
                    cap_fmt    = f"{listing.get('cap_rate', 0):.2f}% cap rate"
                    noi_fmt    = f"${listing.get('noi_annual', 0):,}/yr NOI"
                    dom_fmt    = f"{listing.get('days_on_market', 0)}d on market"
                    built_fmt  = f"Built {listing.get('year_built', 'N/A')}"
                    pt_fmt     = listing.get("property_type", "")
                    addr_fmt   = f"{listing.get('address', '')}, {listing.get('city', '')}, {listing.get('state', '')}"
                    highlights = listing.get("highlights", "")

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
            icon   = {"ok": "", "running": "⏳", "error": "", "idle": ""}.get(st_val, "")
            color  = {"ok": "status-ok", "running": "status-run", "error": "status-error", "idle": "status-idle"}.get(st_val, "")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{num} · {freq}</div>
              <div class="value">{icon}</div>
              <div class="sub">{name}</div>
              <div class="{color}" style="font-size:0.75rem;margin-top:4px;">{st_val.upper()} · {runs} runs</div>
              {"<div style='color:#b71c1c;font-size:0.7rem;margin-top:4px;'>⚠ " + str(err)[:60] + "</div>" if err else ""}
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
              <div class="value">{"" if has_data and not stale else ("⚠" if has_data else "")}</div>
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



with main_tab_energy:
    tab_rates, tab_energy, tab_esg = st.tabs([
        "Rate Environment",
        "Energy & Construction Costs",
        "Sustainability",
    ])


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — RATE ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════════════
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
                st.warning(f"⚠ {err}")
            else:
                st.info("Rate data is fetched every hour. Check that `FRED_API_KEY` is set in `.env`.")
            st.stop()

        rates       = rdata.get("rates", {})
        env         = rdata.get("environment", {})
        cap_adj     = rdata.get("cap_rate_adjustments", [])
        debt_risk   = rdata.get("reit_debt_risk", [])
        yc          = rdata.get("yield_curve", {})
        current_10y = rdata.get("current_10y")
        baseline    = rdata.get("baseline_10y", 4.0)
        cached_at   = rdata.get("cached_at", "")

        # ── Signal Banner ───────────────────────────────────────────────────────
        signal  = env.get("signal", "CAUTIOUS")
        sig_clr = {"BULLISH": "#1b5e20", "CAUTIOUS": "#e65100", "BEARISH": "#b71c1c"}.get(signal, "#333")
        bg_clr  = {"BULLISH": "#e8f5e9", "CAUTIOUS": "#fff3e0", "BEARISH": "#ffebee"}.get(signal, "#f5f5f5")
        st.markdown(f"""
        <div style="background:{bg_clr};border-left:6px solid {sig_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{sig_clr};">
            {env.get('icon','⚪')} Rate Environment: {signal}
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
                    textfont=dict(size=11, color="#1a1a1a"),
                    hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
                ))
                fig_yc.add_hline(y=values[0] if values else 0, line_dash="dot",
                                  line_color="#aaa", line_width=1)
                fig_yc.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Yield (%)", ticksuffix="%",
                               gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                               title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    margin=dict(t=30, b=30, l=60, r=20), height=300,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
                    annotations=[dict(
                        text="⚠ INVERTED" if inverted else "Normal slope",
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
                def _fmt(v):
                    if v is None: return "—"
                    return f"{v:+.2f}{unit}" if unit == "%" else f"{v:+.0f}{unit}"
                tbl_rows.append({
                    "Rate":   name,
                    "Now":    f"{r['current']:.2f}{unit}" if unit == "%" else f"{r['current']:.0f}{unit}",
                    "1W Δ":   _fmt(r.get("delta_1w")),
                    "1M Δ":   _fmt(r.get("delta_1m")),
                    "1Y Δ":   _fmt(r.get("delta_1y")),
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
            series = r["series"]
            # Filter to last 365 calendar days
            cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            pts = [o for o in series if o["date"] >= cutoff]
            if not pts:
                continue
            dates  = [o["date"] for o in pts]
            values = [o["value"] for o in pts]
            fig_tr.add_trace(go.Scatter(
                x=dates, y=values, name=sname,
                mode="lines", line=dict(color=clr, width=2),
                hovertemplate=f"{sname}: %{{y:.3f}}%<br>%{{x}}<extra></extra>",
            ))
        # Shade inverted periods (2Y > 10Y)
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
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title="Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#1a1a1a", size=11)),
            margin=dict(t=40, b=40), height=380,
            font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                    textposition="inside", textfont=dict(color="#1a1a1a", size=10),
                ))
                fig_cap.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate",
                    x=pt_labels, y=adj_caps,
                    marker_color=bar_colors,
                    opacity=0.85,
                    text=[f"{v:.2f}%\n({'+' if d>0 else ''}{d:.0f}bps)" for v, d in zip(adj_caps, adj_bps)],
                    textposition="outside", textfont=dict(color="#1a1a1a", size=9),
                ))
                fig_cap.update_layout(
                    barmode="group",
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                               tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickangle=-20, tickfont=dict(color="#1a1a1a", size=9)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
                    margin=dict(t=40, b=60), height=360,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(title="Near-Term Debt / Total Debt (%)", ticksuffix="%",
                               gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                               title_font=dict(color="#1a1a1a")),
                    yaxis=dict(tickfont=dict(color="#1a1a1a", size=9)),
                    margin=dict(t=20, b=40, l=60, r=60), height=460,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                    f"⚠ **High refinancing risk:** {names} — these REITs have ≥25% of debt maturing "
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

        # ── Rate-Adjusted Pricing Note ───────────────────────────────────────────
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


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — ENERGY & CONSTRUCTION COSTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_energy:
        _show_tab_header(
            "energy",
            "Agent 7 tracks **oil, natural gas, copper, and steel** prices to derive a "
            "**Construction Cost Signal** that indicates whether building costs are rising or easing. Updates every 6 hours.",
            "energy",
        )

        cache_e = read_cache("energy_data")
        if cache_e["data"] is None:
            st.warning("⏳ Energy agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_e = cache_age_label("energy_data")
        st.caption(f" Last updated: {age_e} · Auto-refreshes in background")

        edata = cache_e["data"]
        commodities = edata.get("commodities", [])
        cost_signal = edata.get("construction_cost_signal", "UNKNOWN")
        avg_momentum = edata.get("avg_momentum_pct", 0)

        # ── KPI strip ──────────────────────────────────────────────────────────
        signal_color = {"HIGH": "", "MODERATE": "", "LOW": ""}.get(cost_signal, "⚪")
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
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="% Above/Below SMA-60",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
        _show_tab_header(
            "sustainability",
            "Agent 8 monitors **clean-energy ETFs** and **green REITs** to gauge ESG momentum "
            "relative to the broad market (SPY). Updates every 6 hours.",
            "sustainability",
        )
        # Map property types to relevant green REIT
        _ESG_REIT_MAP = {
            "industrial": "Prologis (PLD) — world's largest industrial REIT with LEED-certified logistics portfolio",
            "data center": "Equinix (EQIX) — powers data centers with 90%+ renewable energy",
            "office": "Alexandria (ARE) — carbon-neutral life science campuses",
            "healthcare": "Alexandria (ARE) — carbon-neutral life science campuses",
        }
        _esg_pt = st.session_state.user_intent.get("property_type")
        _esg_match = _ESG_REIT_MAP.get((_esg_pt or "").lower())
        if _esg_match:
            st.info(f"For **{_esg_pt}** investors, the most relevant green REIT is **{_esg_match}**.")

        cache_s = read_cache("sustainability_data")
        if cache_s["data"] is None:
            st.warning("⏳ Sustainability agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_s = cache_age_label("sustainability_data")
        st.caption(f" Last updated: {age_s} · Auto-refreshes in background")

        sdata = cache_s["data"]
        clean_energy = sdata.get("clean_energy", [])
        green_reits = sdata.get("green_reits", [])
        esg_signal = sdata.get("esg_momentum_signal", "UNKNOWN")
        bench_ret = sdata.get("benchmark_return_pct", 0)
        avg_clean_ret = sdata.get("avg_clean_energy_return_pct", 0)

        # ── KPI strip ──────────────────────────────────────────────────────────
        esg_icon = {"STRONG": "", "NEUTRAL": "", "WEAK": ""}.get(esg_signal, "⚪")
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
        section("⚡ Clean Energy ETF Performance (ICLN, TAN, QCLN)")

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
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
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
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
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


