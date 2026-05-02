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
if "nav_to_tab" not in st.session_state:
    st.session_state.nav_to_tab = None
if "show_brief_for" not in st.session_state:
    st.session_state.show_brief_for = None

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

# ── University → (city, state_abbr) ─────────────────────────────────────────
_UNIVERSITY_LOCATIONS = {
    # Alabama
    "university of alabama": ("Tuscaloosa", "AL"), "ua": ("Tuscaloosa", "AL"),
    "auburn": ("Auburn", "AL"), "auburn university": ("Auburn", "AL"),
    "samford": ("Birmingham", "AL"),
    # Alaska
    "university of alaska": ("Fairbanks", "AK"), "uas": ("Juneau", "AK"),
    # Arizona
    "arizona state": ("Tempe", "AZ"), "arizona state university": ("Tempe", "AZ"), "asu": ("Tempe", "AZ"),
    "university of arizona": ("Tucson", "AZ"),
    "northern arizona": ("Flagstaff", "AZ"), "nau": ("Flagstaff", "AZ"),
    # Arkansas
    "university of arkansas": ("Fayetteville", "AR"), "u of a fayetteville": ("Fayetteville", "AR"),
    "arkansas state": ("Jonesboro", "AR"),
    # California
    "uc berkeley": ("Berkeley", "CA"), "cal berkeley": ("Berkeley", "CA"), "berkeley": ("Berkeley", "CA"),
    "ucla": ("Los Angeles", "CA"), "university of california los angeles": ("Los Angeles", "CA"),
    "usc": ("Los Angeles", "CA"), "university of southern california": ("Los Angeles", "CA"),
    "stanford": ("Palo Alto", "CA"), "stanford university": ("Palo Alto", "CA"),
    "uc davis": ("Davis", "CA"),
    "uc san diego": ("San Diego", "CA"), "ucsd": ("San Diego", "CA"),
    "uc santa barbara": ("Santa Barbara", "CA"), "ucsb": ("Santa Barbara", "CA"),
    "uc irvine": ("Irvine", "CA"), "uci": ("Irvine", "CA"),
    "uc riverside": ("Riverside", "CA"),
    "uc santa cruz": ("Santa Cruz", "CA"),
    "caltech": ("Pasadena", "CA"), "california institute of technology": ("Pasadena", "CA"),
    "san jose state": ("San Jose", "CA"), "sjsu": ("San Jose", "CA"),
    "cal poly slo": ("San Luis Obispo", "CA"), "cal poly": ("San Luis Obispo", "CA"),
    "san diego state": ("San Diego", "CA"), "sdsu": ("San Diego", "CA"),
    "loyola marymount": ("Los Angeles", "CA"), "lmu": ("Los Angeles", "CA"),
    "santa clara university": ("Santa Clara", "CA"),
    "university of san francisco": ("San Francisco", "CA"), "usf sf": ("San Francisco", "CA"),
    "pepperdine": ("Malibu", "CA"),
    # Colorado
    "university of colorado": ("Boulder", "CO"), "cu boulder": ("Boulder", "CO"),
    "colorado state": ("Fort Collins", "CO"), "csu": ("Fort Collins", "CO"),
    "university of denver": ("Denver", "CO"), "du": ("Denver", "CO"),
    "colorado school of mines": ("Golden", "CO"),
    "air force academy": ("Colorado Springs", "CO"),
    # Connecticut
    "yale": ("New Haven", "CT"), "yale university": ("New Haven", "CT"),
    "university of connecticut": ("Storrs", "CT"), "uconn": ("Storrs", "CT"),
    "fairfield university": ("Fairfield", "CT"),
    "wesleyan": ("Middletown", "CT"), "wesleyan university": ("Middletown", "CT"),
    # DC
    "george washington": ("Washington", "DC"), "george washington university": ("Washington", "DC"), "gwu": ("Washington", "DC"),
    "georgetown": ("Washington", "DC"), "georgetown university": ("Washington", "DC"),
    "american university": ("Washington", "DC"),
    "howard": ("Washington", "DC"), "howard university": ("Washington", "DC"),
    "catholic university": ("Washington", "DC"),
    # Delaware
    "university of delaware": ("Newark", "DE"), "udel": ("Newark", "DE"),
    # Florida
    "university of florida": ("Gainesville", "FL"), "uf": ("Gainesville", "FL"),
    "florida state": ("Tallahassee", "FL"), "fsu": ("Tallahassee", "FL"),
    "university of miami": ("Coral Gables", "FL"),
    "ucf": ("Orlando", "FL"), "university of central florida": ("Orlando", "FL"),
    "usf": ("Tampa", "FL"), "university of south florida": ("Tampa", "FL"),
    "fau": ("Boca Raton", "FL"), "florida atlantic": ("Boca Raton", "FL"),
    "fiu": ("Miami", "FL"), "florida international": ("Miami", "FL"),
    "florida gulf coast": ("Fort Myers", "FL"), "fgcu": ("Fort Myers", "FL"),
    # Georgia
    "university of georgia": ("Athens", "GA"), "uga": ("Athens", "GA"),
    "georgia tech": ("Atlanta", "GA"), "georgia institute of technology": ("Atlanta", "GA"),
    "emory": ("Atlanta", "GA"), "emory university": ("Atlanta", "GA"),
    "georgia state": ("Atlanta", "GA"), "gsu": ("Atlanta", "GA"),
    "kennesaw state": ("Kennesaw", "GA"), "ksu": ("Kennesaw", "GA"),
    # Hawaii
    "university of hawaii": ("Honolulu", "HI"), "uh manoa": ("Honolulu", "HI"),
    # Idaho
    "boise state": ("Boise", "ID"),
    "university of idaho": ("Moscow", "ID"),
    "idaho state": ("Pocatello", "ID"),
    # Illinois
    "university of illinois": ("Champaign", "IL"), "uiuc": ("Champaign", "IL"), "u of i": ("Champaign", "IL"),
    "northwestern": ("Evanston", "IL"), "northwestern university": ("Evanston", "IL"),
    "university of chicago": ("Chicago", "IL"),
    "depaul": ("Chicago", "IL"), "depaul university": ("Chicago", "IL"),
    "loyola chicago": ("Chicago", "IL"), "loyola university chicago": ("Chicago", "IL"),
    "illinois state": ("Normal", "IL"),
    "southern illinois": ("Carbondale", "IL"), "siu": ("Carbondale", "IL"),
    # Indiana
    "purdue": ("West Lafayette", "IN"), "purdue university": ("West Lafayette", "IN"),
    "indiana university": ("Bloomington", "IN"), "iu": ("Bloomington", "IN"), "iu bloomington": ("Bloomington", "IN"),
    "notre dame": ("South Bend", "IN"), "university of notre dame": ("South Bend", "IN"),
    "ball state": ("Muncie", "IN"),
    "butler": ("Indianapolis", "IN"), "butler university": ("Indianapolis", "IN"),
    "iupui": ("Indianapolis", "IN"),
    # Iowa
    "iowa state": ("Ames", "IA"),
    "university of iowa": ("Iowa City", "IA"),
    # Kansas
    "kansas state": ("Manhattan", "KS"), "k state": ("Manhattan", "KS"),
    "university of kansas": ("Lawrence", "KS"), "ku": ("Lawrence", "KS"),
    "wichita state": ("Wichita", "KS"),
    # Kentucky
    "university of kentucky": ("Lexington", "KY"), "uk lexington": ("Lexington", "KY"),
    "university of louisville": ("Louisville", "KY"), "u of l": ("Louisville", "KY"),
    "western kentucky": ("Bowling Green", "KY"), "wku": ("Bowling Green", "KY"),
    # Louisiana
    "lsu": ("Baton Rouge", "LA"), "louisiana state": ("Baton Rouge", "LA"),
    "tulane": ("New Orleans", "LA"), "tulane university": ("New Orleans", "LA"),
    "loyola new orleans": ("New Orleans", "LA"),
    "university of louisiana": ("Lafayette", "LA"), "ul lafayette": ("Lafayette", "LA"),
    # Maine
    "university of maine": ("Orono", "ME"),
    # Maryland
    "johns hopkins": ("Baltimore", "MD"), "johns hopkins university": ("Baltimore", "MD"), "jhu": ("Baltimore", "MD"),
    "university of maryland": ("College Park", "MD"), "umd": ("College Park", "MD"),
    "towson": ("Towson", "MD"), "towson university": ("Towson", "MD"),
    "loyola maryland": ("Baltimore", "MD"),
    # Massachusetts
    "mit": ("Cambridge", "MA"), "massachusetts institute of technology": ("Cambridge", "MA"),
    "harvard": ("Cambridge", "MA"), "harvard university": ("Cambridge", "MA"),
    "boston university": ("Boston", "MA"), "bu": ("Boston", "MA"),
    "boston college": ("Chestnut Hill", "MA"), "bc": ("Chestnut Hill", "MA"),
    "northeastern": ("Boston", "MA"), "northeastern university": ("Boston", "MA"),
    "tufts": ("Medford", "MA"), "tufts university": ("Medford", "MA"),
    "umass amherst": ("Amherst", "MA"), "university of massachusetts": ("Amherst", "MA"), "umass": ("Amherst", "MA"),
    "wpi": ("Worcester", "MA"), "worcester polytechnic": ("Worcester", "MA"),
    "brandeis": ("Waltham", "MA"),
    "suffolk university": ("Boston", "MA"),
    # Michigan
    "michigan state": ("East Lansing", "MI"), "msu": ("East Lansing", "MI"),
    "university of michigan": ("Ann Arbor", "MI"), "u of m": ("Ann Arbor", "MI"), "umich": ("Ann Arbor", "MI"),
    "wayne state": ("Detroit", "MI"), "wayne state university": ("Detroit", "MI"),
    "western michigan": ("Kalamazoo", "MI"), "wmu": ("Kalamazoo", "MI"),
    "central michigan": ("Mount Pleasant", "MI"), "cmu michigan": ("Mount Pleasant", "MI"),
    "grand valley state": ("Allendale", "MI"), "gvsu": ("Allendale", "MI"),
    # Minnesota
    "university of minnesota": ("Minneapolis", "MN"), "umn": ("Minneapolis", "MN"), "u of m minnesota": ("Minneapolis", "MN"),
    "minnesota state": ("Mankato", "MN"),
    "st olaf": ("Northfield", "MN"),
    "macalester": ("St. Paul", "MN"),
    # Mississippi
    "university of mississippi": ("Oxford", "MS"), "ole miss": ("Oxford", "MS"),
    "mississippi state": ("Starkville", "MS"),
    # Missouri
    "university of missouri": ("Columbia", "MO"), "mizzou": ("Columbia", "MO"),
    "washington university in st louis": ("St. Louis", "MO"), "wustl": ("St. Louis", "MO"),
    "saint louis university": ("St. Louis", "MO"), "slu": ("St. Louis", "MO"),
    "missouri state": ("Springfield", "MO"),
    # Montana
    "montana state": ("Bozeman", "MT"),
    "university of montana": ("Missoula", "MT"),
    # Nebraska
    "university of nebraska": ("Lincoln", "NE"), "unl": ("Lincoln", "NE"),
    "creighton": ("Omaha", "NE"), "creighton university": ("Omaha", "NE"),
    "nebraska wesleyan": ("Lincoln", "NE"),
    # Nevada
    "university of nevada reno": ("Reno", "NV"), "unr": ("Reno", "NV"),
    "unlv": ("Las Vegas", "NV"), "university of nevada las vegas": ("Las Vegas", "NV"),
    # New Hampshire
    "dartmouth": ("Hanover", "NH"), "dartmouth college": ("Hanover", "NH"),
    "university of new hampshire": ("Durham", "NH"), "unh": ("Durham", "NH"),
    # New Jersey
    "rutgers": ("New Brunswick", "NJ"), "rutgers university": ("New Brunswick", "NJ"),
    "princeton": ("Princeton", "NJ"), "princeton university": ("Princeton", "NJ"),
    "seton hall": ("South Orange", "NJ"), "seton hall university": ("South Orange", "NJ"),
    "montclair state": ("Montclair", "NJ"),
    "njit": ("Newark", "NJ"),
    # New Mexico
    "university of new mexico": ("Albuquerque", "NM"), "unm": ("Albuquerque", "NM"),
    "new mexico state": ("Las Cruces", "NM"), "nmsu": ("Las Cruces", "NM"),
    # New York
    "columbia": ("New York", "NY"), "columbia university": ("New York", "NY"),
    "nyu": ("New York", "NY"), "new york university": ("New York", "NY"),
    "cornell": ("Ithaca", "NY"), "cornell university": ("Ithaca", "NY"),
    "fordham": ("New York", "NY"), "fordham university": ("New York", "NY"),
    "suny buffalo": ("Buffalo", "NY"), "university at buffalo": ("Buffalo", "NY"), "ub": ("Buffalo", "NY"),
    "suny stony brook": ("Stony Brook", "NY"), "stony brook": ("Stony Brook", "NY"),
    "suny albany": ("Albany", "NY"), "university at albany": ("Albany", "NY"),
    "rpi": ("Troy", "NY"), "rensselaer polytechnic": ("Troy", "NY"),
    "rochester": ("Rochester", "NY"), "university of rochester": ("Rochester", "NY"),
    "syracuse": ("Syracuse", "NY"), "syracuse university": ("Syracuse", "NY"),
    "vassar": ("Poughkeepsie", "NY"),
    "hofstra": ("Hempstead", "NY"),
    "new york tech": ("Old Westbury", "NY"),
    # North Carolina
    "unc": ("Chapel Hill", "NC"), "university of north carolina": ("Chapel Hill", "NC"), "unc chapel hill": ("Chapel Hill", "NC"),
    "nc state": ("Raleigh", "NC"), "north carolina state": ("Raleigh", "NC"), "ncsu": ("Raleigh", "NC"),
    "duke": ("Durham", "NC"), "duke university": ("Durham", "NC"),
    "wake forest": ("Winston-Salem", "NC"), "wake forest university": ("Winston-Salem", "NC"),
    "unc charlotte": ("Charlotte", "NC"), "uncc": ("Charlotte", "NC"),
    "appalachian state": ("Boone", "NC"), "app state": ("Boone", "NC"),
    "ecu": ("Greenville", "NC"), "east carolina": ("Greenville", "NC"),
    # North Dakota
    "university of north dakota": ("Grand Forks", "ND"), "und": ("Grand Forks", "ND"),
    "north dakota state": ("Fargo", "ND"), "ndsu": ("Fargo", "ND"),
    # Ohio
    "ohio state": ("Columbus", "OH"), "osu": ("Columbus", "OH"), "the ohio state": ("Columbus", "OH"),
    "university of cincinnati": ("Cincinnati", "OH"), "uc cincinnati": ("Cincinnati", "OH"),
    "miami university": ("Oxford", "OH"),
    "ohio university": ("Athens", "OH"),
    "case western": ("Cleveland", "OH"), "case western reserve": ("Cleveland", "OH"), "cwru": ("Cleveland", "OH"),
    "bowling green state": ("Bowling Green", "OH"), "bgsu": ("Bowling Green", "OH"),
    "kent state": ("Kent", "OH"),
    # Oklahoma
    "university of oklahoma": ("Norman", "OK"), "ou": ("Norman", "OK"),
    "oklahoma state": ("Stillwater", "OK"), "osu oklahoma": ("Stillwater", "OK"),
    "oral roberts": ("Tulsa", "OK"), "oru": ("Tulsa", "OK"),
    # Oregon
    "university of oregon": ("Eugene", "OR"), "u of o": ("Eugene", "OR"),
    "oregon state": ("Corvallis", "OR"), "osu oregon": ("Corvallis", "OR"),
    "portland state": ("Portland", "OR"), "psu oregon": ("Portland", "OR"),
    "reed college": ("Portland", "OR"),
    # Pennsylvania
    "penn state": ("State College", "PA"), "pennsylvania state": ("State College", "PA"), "psu": ("State College", "PA"),
    "upenn": ("Philadelphia", "PA"), "university of pennsylvania": ("Philadelphia", "PA"),
    "temple": ("Philadelphia", "PA"), "temple university": ("Philadelphia", "PA"),
    "drexel": ("Philadelphia", "PA"), "drexel university": ("Philadelphia", "PA"),
    "carnegie mellon": ("Pittsburgh", "PA"), "cmu": ("Pittsburgh", "PA"),
    "university of pittsburgh": ("Pittsburgh", "PA"), "pitt": ("Pittsburgh", "PA"),
    "lehigh": ("Bethlehem", "PA"), "lehigh university": ("Bethlehem", "PA"),
    "villanova": ("Villanova", "PA"), "villanova university": ("Villanova", "PA"),
    "la salle": ("Philadelphia", "PA"),
    "swarthmore": ("Swarthmore", "PA"),
    # Rhode Island
    "brown": ("Providence", "RI"), "brown university": ("Providence", "RI"),
    "university of rhode island": ("Kingston", "RI"), "uri": ("Kingston", "RI"),
    "providence college": ("Providence", "RI"),
    # South Carolina
    "clemson": ("Clemson", "SC"), "clemson university": ("Clemson", "SC"),
    "university of south carolina": ("Columbia", "SC"), "usc columbia": ("Columbia", "SC"),
    "college of charleston": ("Charleston", "SC"),
    "citadel": ("Charleston", "SC"),
    # South Dakota
    "south dakota state": ("Brookings", "SD"), "sdsu sd": ("Brookings", "SD"),
    "university of south dakota": ("Vermillion", "SD"), "usd": ("Vermillion", "SD"),
    # Tennessee
    "university of tennessee": ("Knoxville", "TN"), "ut knoxville": ("Knoxville", "TN"), "utk": ("Knoxville", "TN"),
    "vanderbilt": ("Nashville", "TN"), "vanderbilt university": ("Nashville", "TN"),
    "tennessee state": ("Nashville", "TN"), "tsu": ("Nashville", "TN"),
    "middle tennessee": ("Murfreesboro", "TN"), "mtsu": ("Murfreesboro", "TN"),
    "memphis": ("Memphis", "TN"), "university of memphis": ("Memphis", "TN"),
    "belmont": ("Nashville", "TN"), "belmont university": ("Nashville", "TN"),
    # Texas
    "university of texas": ("Austin", "TX"), "ut austin": ("Austin", "TX"), "longhorns": ("Austin", "TX"),
    "texas am": ("College Station", "TX"), "texas a&m": ("College Station", "TX"), "tamu": ("College Station", "TX"),
    "texas tech": ("Lubbock", "TX"), "texas tech university": ("Lubbock", "TX"), "ttu": ("Lubbock", "TX"),
    "smu": ("Dallas", "TX"), "southern methodist": ("Dallas", "TX"), "southern methodist university": ("Dallas", "TX"),
    "rice": ("Houston", "TX"), "rice university": ("Houston", "TX"),
    "baylor": ("Waco", "TX"), "baylor university": ("Waco", "TX"),
    "ut dallas": ("Richardson", "TX"), "utd": ("Richardson", "TX"),
    "university of houston": ("Houston", "TX"), "uh houston": ("Houston", "TX"),
    "tcu": ("Fort Worth", "TX"), "texas christian": ("Fort Worth", "TX"),
    "utsa": ("San Antonio", "TX"), "ut san antonio": ("San Antonio", "TX"),
    "sam houston state": ("Huntsville", "TX"),
    "stephen f austin": ("Nacogdoches", "TX"),
    # Utah
    "university of utah": ("Salt Lake City", "UT"), "u of u": ("Salt Lake City", "UT"),
    "byu": ("Provo", "UT"), "brigham young": ("Provo", "UT"), "brigham young university": ("Provo", "UT"),
    "utah state": ("Logan", "UT"), "usu": ("Logan", "UT"),
    "weber state": ("Ogden", "UT"),
    # Vermont
    "university of vermont": ("Burlington", "VT"), "uvm": ("Burlington", "VT"),
    "middlebury": ("Middlebury", "VT"),
    # Virginia
    "university of virginia": ("Charlottesville", "VA"), "uva": ("Charlottesville", "VA"),
    "virginia tech": ("Blacksburg", "VA"), "vt": ("Blacksburg", "VA"),
    "george mason": ("Fairfax", "VA"), "george mason university": ("Fairfax", "VA"), "gmu": ("Fairfax", "VA"),
    "william and mary": ("Williamsburg", "VA"),
    "vcu": ("Richmond", "VA"), "virginia commonwealth": ("Richmond", "VA"),
    "liberty university": ("Lynchburg", "VA"), "liberty": ("Lynchburg", "VA"),
    "james madison": ("Harrisonburg", "VA"), "jmu": ("Harrisonburg", "VA"),
    "old dominion": ("Norfolk", "VA"), "odu": ("Norfolk", "VA"),
    # Washington
    "university of washington": ("Seattle", "WA"), "uw seattle": ("Seattle", "WA"), "uw": ("Seattle", "WA"),
    "washington state": ("Pullman", "WA"), "wsu": ("Pullman", "WA"),
    "seattle university": ("Seattle", "WA"),
    "gonzaga": ("Spokane", "WA"), "gonzaga university": ("Spokane", "WA"),
    "western washington": ("Bellingham", "WA"),
    # West Virginia
    "west virginia university": ("Morgantown", "WV"), "wvu": ("Morgantown", "WV"),
    "marshall university": ("Huntington", "WV"),
    # Wisconsin
    "university of wisconsin": ("Madison", "WI"), "uw madison": ("Madison", "WI"), "wisconsin": ("Madison", "WI"),
    "marquette": ("Milwaukee", "WI"), "marquette university": ("Milwaukee", "WI"),
    "uwm": ("Milwaukee", "WI"), "uw milwaukee": ("Milwaukee", "WI"),
    # Wyoming
    "university of wyoming": ("Laramie", "WY"), "uwyo": ("Laramie", "WY"),
}
# Sorted longest-first to prevent short keys matching inside longer names
_UNI_KEYS_SORTED = sorted(_UNIVERSITY_LOCATIONS.keys(), key=len, reverse=True)


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
        'would it look', 'what would', 'how would', 'what does it',
        'can you analyze', 'analyze this', 'give me a', 'show me',
    ])
    # Natural-language intent phrases (first-person or interrogative investment intent)
    has_intent   = any(w in t for w in [
        'i want to', 'i am looking', "i'm looking", 'i would like',
        'we want to', 'we are looking', "we're looking",
        'what would it look like', 'what would it cost',
        'how much would', 'what if i', 'tell me about investing',
    ])
    # Trigger advisor if: (financials + action) OR (budget + sqft) OR (3+ signals)
    # OR natural-language intent paired with any action/investment word
    signals = sum([has_budget, has_sqft, has_timeline, has_action, has_intent])
    return (
        (has_budget and has_action)
        or (has_budget and has_sqft)
        or signals >= 3
        or (has_action and has_intent)     # e.g. "I want to build near UT"
        or (has_intent and len(t.split()) >= 8)  # long natural-language question
    )


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

    # Fallback 1: proximity phrase — "near/around/close to/by [X]"
    if not location:
        _prox_m = _re.search(
            r'(?:^|\b)(?:near|around|by|close\s+to|next\s+to|adjacent\s+to)\s+'
            r'(?:the\s+|a\s+)?(.+?)(?:\s*$|[,.])',
            raw, _re.IGNORECASE,
        )
        if _prox_m:
            _cand = _prox_m.group(1).strip()
            _cand_l = _cand.lower()
            # University lookup (longest key first to avoid partial matches)
            for _ukey in _UNI_KEYS_SORTED:
                if _re.search(r'\b' + _re.escape(_ukey) + r'\b', _cand_l):
                    _ucity, _usa = _UNIVERSITY_LOCATIONS[_ukey]
                    location = f"{_ucity}, {_US_STATES[_usa]}"
                    break
            if not location:
                # "X of [State]" pattern
                _om = _re.search(r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', _cand)
                if _om and _om.group(1).lower() in _STATE_NAME_TO_ABBR:
                    location = _US_STATES[_STATE_NAME_TO_ABBR[_om.group(1).lower()]]
                elif _cand_l in _STATE_NAME_TO_ABBR:
                    location = _US_STATES[_STATE_NAME_TO_ABBR[_cand_l]]
                elif _cand.upper() in _US_STATES:
                    location = _US_STATES[_cand.upper()]
                elif _cand_l in _CITY_TO_STATE:
                    _ca = _CITY_TO_STATE[_cand_l]
                    location = _cand.title() + ", " + _US_STATES[_ca]
                else:
                    location = _cand

    # Fallback 2: university name anywhere in input
    if not location:
        for _ukey in _UNI_KEYS_SORTED:
            if _re.search(r'\b' + _re.escape(_ukey) + r'\b', raw_lower):
                _ucity, _usa = _UNIVERSITY_LOCATIONS[_ukey]
                location = f"{_ucity}, {_US_STATES[_usa]}"
                break

    # Fallback 3a: "<city> <state>" combo — e.g., "Houston Texas", "Austin TX",
    # "Miami Florida". Catches the no-comma case before the state-only fallback
    # would discard the city.
    if not location:
        # Sort cities longest-first so "los angeles" beats "los"
        for _cn, _ca in sorted(_CITY_TO_STATE.items(), key=lambda x: -len(x[0])):
            if not _re.search(r'\b' + _re.escape(_cn) + r'\b', raw_lower):
                continue
            _state_full = _US_STATES[_ca].lower()
            # Match either the full state name OR the 2-letter abbr after the city
            if (_re.search(r'\b' + _re.escape(_state_full) + r'\b', raw_lower)
                    or _re.search(r'\b' + _ca.lower() + r'\b', raw_lower)):
                location = _cn.title() + ", " + _ca
                break

    # Fallback 3b: state name anywhere in input (no city detected)
    if not location:
        for _sn, _sa in _STATE_NAME_TO_ABBR.items():
            if _re.search(r'\b' + _re.escape(_sn) + r'\b', raw_lower):
                location = _US_STATES[_sa]
                break

    # Fallback 4: known city name anywhere in input
    if not location:
        for _cn, _ca in _CITY_TO_STATE.items():
            if _re.search(r'\b' + _re.escape(_cn) + r'\b', raw_lower):
                location = _cn.title() + ", " + _US_STATES[_ca]
                break

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
#  QUICK INVESTMENT BRIEF — unified AI panel shown after any query
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _quick_brief_insight(property_type: str, location: str, data_summary: str) -> str:
    """Short Groq-generated market insight for the Quick Brief panel."""
    import os as _os
    key = _os.getenv("GROQ_API_KEY", "")
    if not key:
        return ""
    try:
        from groq import Groq as _Groq
        client = _Groq(api_key=key)
        pt  = property_type or "commercial real estate"
        loc = location or "the US"
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": (
                    "You are a CRE investment analyst. Write exactly 2 sentences of actionable "
                    "market insight based on the data provided. Be specific with numbers. "
                    "No preamble, no headers, no bullets."
                )},
                {"role": "user", "content": (
                    f"Investor is focused on {pt} in {loc}.\n\nLive market data:\n"
                    f"{data_summary[:600]}\n\nGive a concise 2-sentence insight."
                )},
            ],
            max_tokens=140,
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# Canonical property-type → cap-rate / rent-growth cache keys
_BRIEF_PT_KEYS = {
    "Industrial":  ("Industrial",   "industrial_psf"),
    "Multifamily": ("Multifamily",  "multifamily"),
    "Office":      ("Office",       "office_psf"),
    "Retail":      ("Retail",       "retail_psf"),
    "Healthcare":  ("Office",       "office_psf"),   # fallback — no separate cache slice
}


def _build_brief_data(intent: dict) -> dict:
    """Compose a Quick Brief payload: top 3 markets + an AI insight."""
    pt       = (intent or {}).get("property_type")
    location = (intent or {}).get("location")
    user_mkt = (intent or {}).get("city") or location

    caprate_key, rg_key = _BRIEF_PT_KEYS.get(pt, ("Industrial", "industrial_psf"))

    ms_cache = read_cache("market_score") or {}
    ms_rank  = (ms_cache.get("data") or {}).get("rankings", []) or []
    cr_cache = read_cache("cap_rate") or {}
    cr_mkts  = (cr_cache.get("data") or {}).get("market_cap_rates", {}) or {}
    rg_cache = read_cache("rent_growth") or {}
    rg_mkts  = (rg_cache.get("data") or {}).get("market_rent_growth", {}) or {}

    ms_by_mkt = {m.get("market"): m for m in ms_rank if m.get("market")}
    ranked    = sorted(ms_rank, key=lambda m: m.get("composite", 0), reverse=True)

    # If the user named a location, surface its market first (if we score it)
    seen: set[str] = set()
    chosen: list[str] = []
    if user_mkt:
        um = user_mkt.lower()
        for m in ms_rank:
            mk = m.get("market", "")
            if um in mk.lower():
                chosen.append(mk)
                seen.add(mk)
                break
    for m in ranked:
        mk = m.get("market", "")
        if mk and mk not in seen:
            chosen.append(mk)
            seen.add(mk)
            if len(chosen) >= 3:
                break

    top3 = []
    for mk in chosen[:3]:
        row = ms_by_mkt.get(mk, {})
        cap = (cr_mkts.get(mk) or {}).get(caprate_key)
        rg  = (rg_mkts.get(mk) or {}).get(rg_key)
        top3.append({
            "market":      mk,
            "score":       float(row.get("composite", 0) or 0),
            "grade":       row.get("grade", ""),
            "cap_rate":    cap,
            "rent_growth": rg,
        })

    # AI insight — summarize the 3 rows as grounded context
    lines = []
    for r in top3:
        parts = [f"{r['market']} score {r['score']:.1f}"]
        if r["cap_rate"] is not None:
            parts.append(f"{pt or 'CRE'} cap {r['cap_rate']:.1f}%")
        if r["rent_growth"] is not None:
            parts.append(f"rent growth {r['rent_growth']:+.1f}%")
        lines.append(", ".join(parts))
    insight = _quick_brief_insight(pt or "", location or "", "\n".join(lines))

    return {
        "property_type": pt,
        "location":      location,
        "user_market":   user_mkt,
        "top3":          top3,
        "insight":       insight,
        "raw_input":     (intent or {}).get("raw_input", ""),
    }


def _render_quick_brief(intent: dict, context_key: str) -> None:
    """Render the Quick Brief panel + CTA buttons. `context_key` disambiguates button keys."""
    brief = _build_brief_data(intent)
    pt  = brief.get("property_type") or "All property types"
    loc = brief.get("location") or "United States"
    header = f"{pt.upper()} IN {loc.upper()} — QUICK BRIEF"

    # ── Panel ────────────────────────────────────────────────────────────────
    rows_html = ""
    for i, r in enumerate(brief["top3"], start=1):
        cap_str = f"{r['cap_rate']:.1f}%" if r["cap_rate"] is not None else "—"
        rg_raw  = r["rent_growth"]
        if rg_raw is None:
            rg_str = "—"
            rg_col = "#a09880"
        else:
            rg_str = f"{rg_raw:+.1f}%"
            rg_col = "#4caf50" if rg_raw >= 0 else "#ef5350"
        score_col = ("#4caf50" if r["score"] >= 75 else
                     "#d4a843" if r["score"] >= 60 else
                     "#ff9800" if r["score"] >= 45 else "#ef5350")
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:10px 14px;border-bottom:1px solid #2a2208;">'
            f'  <div style="color:#e8dfc4;font-size:0.92rem;">'
            f'    <span style="color:#d4a843;font-weight:700;margin-right:10px;">{i}.</span>'
            f'    <b>{r["market"]}</b>'
            f'    <span style="color:#6a5228;font-size:0.78rem;margin-left:8px;">Grade {r["grade"] or "—"}</span>'
            f'  </div>'
            f'  <div style="color:#a09880;font-size:0.82rem;display:flex;gap:18px;">'
            f'    <span>Score: <b style="color:{score_col};">{r["score"]:.1f}</b></span>'
            f'    <span>Cap: <b style="color:#c8a040;">{cap_str}</b></span>'
            f'    <span>Growth: <b style="color:{rg_col};">{rg_str}</b></span>'
            f'  </div>'
            f'</div>'
        )
    if not rows_html:
        rows_html = (
            '<div style="padding:18px;color:#a09880;font-size:0.9rem;">'
            'Market scoring data is still loading — try again in ~30 seconds.</div>'
        )

    insight_html = ""
    if brief.get("insight"):
        insight_html = (
            f'<div style="background:#16140a;border-left:3px solid #d4a843;'
            f'padding:12px 16px;margin-top:14px;border-radius:4px;'
            f'color:#e8dfc4;font-size:0.9rem;line-height:1.6;font-style:italic;">'
            f'<span style="color:#d4a843;font-style:normal;font-weight:700;">AI Insight:</span> '
            f'{brief["insight"]}'
            f'</div>'
        )

    # ── Header bar — visible immediately after submit, no scroll required ──
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1208 0%,#221a0a 100%);'
        f'border:1px solid #a07830;border-top:3px solid #d4a843;'
        f'border-radius:10px 10px 0 0;border-bottom:none;'
        f'padding:18px 26px 14px 26px;margin-top:20px;">'
        f'  <div style="color:#d4a843;font-size:0.82rem;letter-spacing:0.15em;'
        f'      font-weight:700;margin-bottom:4px;">&#128200; {header}</div>'
        f'  <div style="color:#a09880;font-size:0.8rem;">'
        f'      Top 3 markets ranked on live composite scoring, cap rates, and rent growth — '
        f'      pick an action below or scroll for the AI insight.'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── CTAs FIRST — surfaced above the fold so the user never has to scroll ──
    _c1, _c2, _c3 = st.columns([1.2, 1.4, 0.8], gap="small")
    with _c1:
        explore = st.button(
            "Explore Dashboard →",
            use_container_width=True,
            key=f"brief_explore_{context_key}",
        )
    with _c2:
        full = st.button(
            "Get Full P&L Analysis →",
            use_container_width=True,
            type="primary",
            key=f"brief_advisor_{context_key}",
        )
    with _c3:
        dismiss = st.button(
            "Dismiss",
            use_container_width=True,
            key=f"brief_dismiss_{context_key}",
        )

    # ── Detail panel (markets table + AI insight) — supporting context ───────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1208 0%,#221a0a 100%);'
        f'border:1px solid #a07830;border-top:none;border-radius:0 0 10px 10px;'
        f'padding:8px 26px 22px 26px;margin-bottom:20px;">'
        f'  <div style="background:#0f0d06;border:1px solid #2a2208;border-radius:6px;'
        f'      overflow:hidden;">{rows_html}</div>'
        f'  {insight_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if explore:
        st.session_state.show_brief_for = None
        if not st.session_state.onboarding_complete:
            _complete_onboarding(**intent)
        st.rerun()
    if full:
        _prompt = intent.get("raw_input") or (
            f"{intent.get('property_type') or 'Commercial real estate'} in "
            f"{intent.get('location') or 'the US'}"
        )
        st.session_state.adv_home_prompt   = _prompt
        st.session_state.adv_auto_generate = True
        st.session_state.adv_navigate      = True
        st.session_state.show_brief_for    = None
        if not st.session_state.onboarding_complete:
            _complete_onboarding(raw_input=_prompt)
        st.rerun()
    if dismiss:
        st.session_state.show_brief_for = None
        st.rerun()


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
        f'<div class="prop-card" onclick="window.top.location.href=\'?select={name}\'">'
        f'{icon}<div class="prop-card-lbl">{"BROWSE ALL" if name == "Exploring" else name.upper()}</div></div>'
        for name, icon in _PROP_ICONS.items()
    ])

    # ── Recent searches HTML ──────────────────────────────────────────────────
    _recent_html = ""
    if st.session_state.recent_searches:
        _rs_items = "".join([
            f'<span class="rs-item" onclick="window.top.location.href=\'?q={_r.replace(" ","+")}\'">'
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

    # ── Houston migration map as dim fullscreen background ────────────────────
    # Pull the same neighborhood data the metro drill-down uses so the welcome
    # screen previews exactly what the user sees after they search "Houston Texas".
    try:
        from src.zip_migration import _METRO_DATA as _BG_METRO_DATA
        _bg_zones = _BG_METRO_DATA["Houston"]["zones"]
        _bg_lats   = [z[1] for z in _bg_zones]
        _bg_lons   = [z[2] for z in _bg_zones]
        _bg_scores = [z[3] for z in _bg_zones]
        _bg_names  = [z[0] for z in _bg_zones]
    except Exception:
        _bg_lats, _bg_lons, _bg_scores, _bg_names = [29.76], [-95.37], [70], ["Houston"]

    import json as _bg_json
    _bg_payload = _bg_json.dumps({
        "lat":    _bg_lats,
        "lon":    _bg_lons,
        "score":  _bg_scores,
        "name":   _bg_names,
    })

    # The Houston map renders inside a self-contained iframe — NO window.parent
    # access, since Streamlit's iframe sandbox on Render strips
    # `allow-same-origin` and any cross-origin DOM access throws SecurityError.
    # All positioning/sizing/clipping is done from parent-document CSS via the
    # `iframe[srcdoc*="{_BG_MAP_TAG}"]` attribute selector below.
    _BG_MAP_TAG = "home-bg-map-marker-7f3c"
    components.html(
        f"""
<!doctype html><html><head>
<!-- {_BG_MAP_TAG} -->
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>
  html, body {{ margin:0; padding:0; height:100vh; width:100vw;
                background:#0d0b04; overflow:hidden; }}
  /* Dark fallback so the dots stay readable if tiles fail to load */
  #m {{ width:100vw; height:100vh; background:#0d0b04; }}
  /* Tint the bright OSM raster tiles toward our dark/gold palette without a
     custom style URL — light enough that road grid + city labels stay visible
     as a wallpaper. */
  #m .mapboxgl-canvas-container,
  #m .maplibregl-canvas-container {{
    filter: invert(0.88) hue-rotate(180deg) brightness(1.05) saturate(0.45) contrast(0.85);
  }}
</style>
</head><body><div id='m'></div>
<script>
const d = {_bg_payload};
Plotly.newPlot('m', [{{
  type: 'scattermapbox',
  lat: d.lat, lon: d.lon, mode: 'markers',
  marker: {{
    size: d.score.map(s => Math.max(10, s/2.8)),
    color: d.score,
    colorscale: [
      [0.0, '#7f0000'], [0.35, '#c62828'], [0.55, '#d4a843'],
      [0.80, '#4caf50'], [1.00, '#1b5e20']
    ],
    cmin: 0, cmax: 100, opacity: 0.95,
  }},
  hoverinfo: 'skip'
}}], {{
  mapbox: {{
    /* open-street-map uses standard OSM raster tiles — no Mapbox token,
       no CartoDB CORS issue. The CSS filter above re-tints them dark. */
    style: 'open-street-map',
    center: {{ lat: 29.78, lon: -95.45 }},
    zoom: 8.6
  }},
  paper_bgcolor: '#0d0b04',
  plot_bgcolor:  '#0d0b04',
  margin: {{ t: 0, b: 0, l: 0, r: 0 }},
  showlegend: false
}}, {{ displayModeBar: false, staticPlot: true, responsive: true }});

// Plotly auto-resizes when the iframe's CSS-imposed dimensions change.
// We just need to keep responsive=true above and trust the parent CSS.
window.addEventListener('resize', () => {{
  try {{ Plotly.Plots.resize('m'); }} catch (e) {{}}
}});
</script></body></html>
        """,
        height=900,
    )

    # Vignette overlay rendered as a parent-document div so we don't have to
    # cross the iframe boundary (which fails with SecurityError on Render).
    st.markdown('<div class="home-bg-vignette" aria-hidden="true"></div>',
                unsafe_allow_html=True)

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

      /* ── Houston map as fullscreen background ──────────────────────────── */
      /* Target the iframe DIRECTLY by the unique marker tag inside its srcdoc.
         Attribute substring selectors work in every browser and don't depend
         on Streamlit's container test-ids (which differ between versions and
         break on Render). No window.parent JS — purely parent-document CSS. */
      iframe[srcdoc*="home-bg-map-marker-7f3c"] {{
        position: fixed !important;
        top: 0 !important; left: 0 !important;
        width: 100vw !important; height: 100vh !important;
        border: none !important;
        z-index: 0 !important;
        opacity: 0.85 !important;
        pointer-events: none !important;
      }}
      /* Collapse every Streamlit wrapper that contains this iframe so its
         900px-tall in-flow placeholder doesn't push the rest of the page down. */
      [data-testid="stElementContainer"]:has(iframe[srcdoc*="home-bg-map-marker-7f3c"]),
      [data-testid="element-container"]:has(iframe[srcdoc*="home-bg-map-marker-7f3c"]),
      [data-testid="stIFrame"]:has(iframe[srcdoc*="home-bg-map-marker-7f3c"]) {{
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
      }}
      /* Marker container is also collapsed (it's an empty anchor div) */
      [data-testid="stElementContainer"]:has(.home-bg-map-anchor),
      [data-testid="element-container"]:has(.home-bg-map-anchor) {{
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      /* Vignette overlay rendered as a real div in the parent document
         (no JS injection needed, no cross-origin risk) */
      .home-bg-vignette {{
        position: fixed; inset: 0;
        background: radial-gradient(ellipse at center,
                    rgba(13,11,4,0.10) 0%,
                    rgba(13,11,4,0.35) 55%,
                    rgba(13,11,4,0.70) 100%);
        z-index: 1;
        pointer-events: none;
      }}
      /* Foreground welcome content sits above the map and vignette */
      .cre-nav, .cre-ticker, .cre-hero, .cre-wrap,
      [data-testid="stForm"], .s-examples,
      [data-testid="stColumn"] {{
        position: relative;
        z-index: 10;
      }}

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
      .ey-text   {{ color:{GOLD}; font-size:.8rem; font-weight:500; letter-spacing:3.5px; }}
      .hero-title {{
        font-size:4.2rem; font-weight:700; color:{GOLD};
        line-height:1.08; margin-bottom:22px; letter-spacing:-.5px;
      }}
      .hero-sub {{
        font-size:1.2rem; color:#7a6840; line-height:1.65;
        max-width:620px; margin:0 auto 36px;
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
      /* Nuke every border/background on Streamlit's nested form containers
         so only the input/button styles render — fixes the cut-off outer box */
      [data-testid="stForm"],
      [data-testid="stForm"] > div,
      [data-testid="stForm"] [data-testid="stVerticalBlock"],
      [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
        border:none !important;
        box-shadow:none !important;
        background:transparent !important;
        outline:none !important;
      }}
      [data-testid="stForm"] {{ padding:0 !important; }}
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
      [data-testid="stTextInput"] input::placeholder {{
        color:#a89260 !important;
        opacity:0.85 !important;
      }}
      [data-testid="stTextInput"] > div {{ border:none !important; background:transparent !important; }}
      [data-testid="InputInstructions"] {{ display:none !important; }}

      /* Breathing room below the search row */
      [data-testid="stForm"] {{ margin-bottom:18px !important; }}

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

      /* Override Streamlit's salmon primary-button color → gold theme */
      .stButton > button[kind="primary"],
      [data-testid="stBaseButton-primary"] {{
        background:{GOLD} !important;
        color:#0d0b04 !important;
        border:1px solid {GOLD} !important;
        font-weight:700 !important;
      }}
      .stButton > button[kind="primary"]:hover,
      [data-testid="stBaseButton-primary"]:hover {{
        background:#e8c060 !important;
        border-color:#e8c060 !important;
        color:#0d0b04 !important;
      }}

      /* ── Search examples ────────────────────────────────────────────────── */
      .s-examples {{
        text-align:center; color:#a89260; font-size:.78rem; margin-bottom:52px;
      }}
      .s-ex {{
        color:{GOLD}; cursor:pointer;
        text-decoration:underline; text-decoration-color:rgba(200,160,64,.5);
      }}
      .s-ex-static {{ color:{GOLD}; font-weight:500; }}

      /* ── Property type section ──────────────────────────────────────────── */
      .cre-wrap {{ max-width:1160px; margin:0 auto; padding:0 48px; }}
      .prop-hdr {{
        text-align:center; color:#3a2e1a; font-size:.8rem; font-weight:600;
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
            _NAV_KEYWORDS = {
                "team":                "About",
                "meet the team":       "About",
                "about":               "About",
                "our team":            "About",
                "system monitor":      "About",
                "monitor":             "About",
                "investment advisor":  "Investment Advisor",
                "advisor":             "Investment Advisor",
                "energy":              "Energy",
                "macro":               "Macro Environment",
                "economy":             "Macro Environment",
                "gdp":                 "Macro Environment",
            }
            _ui = user_input.lower().strip()
            _nav = next((v for k, v in _NAV_KEYWORDS.items() if k == _ui or _ui == k), None)
            if _nav:
                st.session_state.nav_to_tab = _nav
                _complete_onboarding()   # no raw_input → won't pollute recent searches
            elif _is_advisor_query(user_input):
                _complete_onboarding(raw_input=user_input)
                st.session_state.adv_home_prompt   = user_input
                st.session_state.adv_auto_generate = True
                st.session_state.adv_navigate      = True
            else:
                # Show Quick Brief first — user picks a CTA to navigate
                _intent = _parse_intent(user_input)
                _intent["raw_input"] = user_input
                st.session_state.show_brief_for = _intent
            st.rerun()

    # ── Quick Investment Brief (inline, after query) ──────────────────────────
    if st.session_state.get("show_brief_for"):
        _render_quick_brief(st.session_state.show_brief_for, context_key="welcome")

    # ── Example queries ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="s-examples">
      Try:&nbsp;
      <span class="s-ex-static">&ldquo;Multifamily in Nashville&rdquo;</span>
      &nbsp;&middot;&nbsp;
      <span class="s-ex-static">&ldquo;Office cap rates Chicago&rdquo;</span>
      &nbsp;&middot;&nbsp;
      <span class="s-ex-static">&ldquo;Best Sunbelt markets 2026&rdquo;</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Property type cards (native buttons for reliable click handling) ───────
    import base64 as _b64

    def _svg_uri(s: str) -> str:
        return "data:image/svg+xml;base64," + _b64.b64encode(s.strip().encode()).decode()

    _SVG_INDUSTRIAL = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="20" width="38" height="21" rx="1.5"/>
  <rect x="7" y="24" width="7" height="7"/><rect x="18" y="24" width="7" height="7"/><rect x="29" y="24" width="7" height="7"/>
  <rect x="7" y="34" width="7" height="7"/><rect x="18" y="34" width="7" height="7"/><rect x="29" y="34" width="7" height="7"/>
  <path d="M3 20L13 13v7M13 20L23 13v7M23 20L33 13v7"/>
  <line x1="3" y1="20" x2="41" y2="20"/>
</svg>''')

    _SVG_MULTIFAMILY = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="3" width="11" height="11" rx="2"/><rect x="17" y="3" width="11" height="11" rx="2"/><rect x="30" y="3" width="11" height="11" rx="2"/>
  <rect x="3" y="17" width="11" height="11" rx="2"/><rect x="17" y="17" width="11" height="11" rx="2"/><rect x="30" y="17" width="11" height="11" rx="2"/>
  <rect x="3" y="30" width="11" height="11" rx="2"/><rect x="17" y="30" width="11" height="11" rx="2"/><rect x="30" y="30" width="11" height="11" rx="2"/>
</svg>''')

    _SVG_OFFICE = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="5" y="4" width="34" height="36" rx="2.5"/>
  <line x1="15" y1="4" x2="15" y2="40"/>
  <line x1="5" y1="15" x2="39" y2="15"/>
  <line x1="5" y1="25" x2="39" y2="25"/>
  <line x1="5" y1="35" x2="39" y2="35"/>
</svg>''')

    _SVG_RETAIL = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="4" y="20" width="36" height="21" rx="1"/>
  <path d="M2 11h40l2 9H0l4-9z"/>
  <rect x="17" y="28" width="10" height="13"/>
  <rect x="6" y="25" width="8" height="7" rx="1"/>
  <rect x="30" y="25" width="8" height="7" rx="1"/>
</svg>''')

    _SVG_HEALTHCARE = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="4" y="16" width="36" height="24" rx="3"/>
  <path d="M15 16v-5a2 2 0 012-2h10a2 2 0 012 2v5"/>
  <line x1="22" y1="23" x2="22" y2="33"/>
  <line x1="17" y1="28" x2="27" y2="28"/>
</svg>''')

    _SVG_GLOBE = _svg_uri('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44" fill="none" stroke="#a07828" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="22" cy="22" r="19"/>
  <ellipse cx="22" cy="22" rx="9" ry="19"/>
  <line x1="3" y1="22" x2="41" y2="22"/>
  <path d="M6 13h32M6 31h32"/>
</svg>''')

    _PROP_LABELS = {
        "Industrial":  ("INDUSTRIAL",  _SVG_INDUSTRIAL),
        "Multifamily": ("MULTIFAMILY", _SVG_MULTIFAMILY),
        "Office":      ("OFFICE",      _SVG_OFFICE),
        "Retail":      ("RETAIL",      _SVG_RETAIL),
        "Healthcare":  ("HEALTHCARE",  _SVG_HEALTHCARE),
        "Exploring":   ("BROWSE\nALL", _SVG_GLOBE),
    }

    st.markdown(f"""
<style>
  .prop-card-btn > div[data-testid="stButton"] > button {{
    background: #161006 !important;
    border: 1px solid rgba(180,145,50,.38) !important;
    border-radius: 12px !important;
    padding: 32px 10px 22px !important;
    color: #a07828 !important;
    font-size: .58rem !important;
    font-weight: 500 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    min-height: 148px !important;
    transition: border-color .2s, box-shadow .2s, background .2s !important;
    cursor: pointer !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 0 !important;
    line-height: 1.5 !important;
    box-shadow: none !important;
    position: relative !important;
  }}
  .prop-card-btn > div[data-testid="stButton"] > button::before {{
    content: "" !important;
    display: block !important;
    width: 44px !important;
    height: 44px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: contain !important;
    flex-shrink: 0 !important;
    margin-bottom: 18px !important;
  }}
  /* Per-card icon injection */
  .prop-card-Industrial > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_INDUSTRIAL}") !important;
  }}
  .prop-card-Multifamily > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_MULTIFAMILY}") !important;
  }}
  .prop-card-Office > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_OFFICE}") !important;
  }}
  .prop-card-Retail > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_RETAIL}") !important;
  }}
  .prop-card-Healthcare > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_HEALTHCARE}") !important;
  }}
  .prop-card-Exploring > div[data-testid="stButton"] > button::before {{
    background-image: url("{_SVG_GLOBE}") !important;
  }}
  .prop-card-btn > div[data-testid="stButton"] > button:hover,
  .prop-card-btn > div[data-testid="stButton"] > button:focus {{
    background: #1e1608 !important;
    border-color: rgba(200,160,64,.7) !important;
    color: #c8a040 !important;
    box-shadow: 0 0 18px rgba(180,145,50,.12) !important;
    outline: none !important;
  }}
</style>
<div class="cre-wrap" style="padding-bottom:0;">
  <div class="prop-hdr">OR SELECT A PROPERTY TYPE</div>
</div>
""", unsafe_allow_html=True)

    _pcols = st.columns(len(_PROP_LABELS))
    for _pcol, (_pname, (_plbl, _psvg)) in zip(_pcols, _PROP_LABELS.items()):
        with _pcol:
            st.markdown(f'<div class="prop-card-btn prop-card-{_pname}">', unsafe_allow_html=True)
            if st.button(_plbl, key=f"propcard_{_pname}", use_container_width=True):
                if _pname == "Exploring":
                    _complete_onboarding()
                else:
                    _complete_onboarding(property_type=_pname)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Recent searches ───────────────────────────────────────────────────────
    st.markdown(f"{_recent_html}", unsafe_allow_html=True)

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
    overflow: visible;
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
    background: #13110a !important;
    border: 1px solid #3a3010 !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
  }}
  [data-testid="stExpander"] summary {{
    color: #c8a040 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    letter-spacing: 0.03em !important;
  }}
  [data-testid="stExpander"] summary:hover {{
    color: #e8c860 !important;
    background: rgba(200,160,64,0.06) !important;
  }}
  [data-testid="stExpander"] svg {{
    fill: #c8a040 !important;
    stroke: #c8a040 !important;
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

  /* ── Tooltips ── */
  .cre-tt {{
    position: relative; display: inline-flex; align-items: center;
    gap: 4px; cursor: default;
  }}
  .cre-tt-icon {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 13px; height: 13px; border-radius: 50%;
    background: rgba(200,160,64,.15); border: 1px solid rgba(200,160,64,.3);
    color: {GOLD}; font-size: 8px; font-weight: 700;
    cursor: help; flex-shrink: 0; line-height: 1;
  }}
  .cre-tt .cre-tt-box {{
    visibility: hidden; opacity: 0;
    position: absolute; z-index: 9999; bottom: calc(100% + 6px); left: 0;
    min-width: 220px; max-width: 300px;
    background: #1e1a0a; border: 1px solid rgba(200,160,64,.3);
    border-radius: 8px; padding: 10px 13px;
    font-size: 11px; color: #c8b890; line-height: 1.55;
    font-weight: 400; text-transform: none; letter-spacing: 0;
    box-shadow: 0 4px 20px rgba(0,0,0,.5);
    transition: opacity .15s; pointer-events: none;
  }}
  .cre-tt:hover .cre-tt-box {{ visibility: visible; opacity: 1; }}

  /* ── Signal pills ── */
  .sig-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 600; letter-spacing: .05em;
  }}
  .sig-EXPANDING  {{ background: rgba(27,94,32,.25);  color: #4caf50; border: 1px solid rgba(76,175,80,.3); }}
  .sig-STRONG     {{ background: rgba(27,94,32,.25);  color: #4caf50; border: 1px solid rgba(76,175,80,.3); }}
  .sig-TIGHT      {{ background: rgba(27,94,32,.25);  color: #4caf50; border: 1px solid rgba(76,175,80,.3); }}
  .sig-MODERATE   {{ background: rgba(230,81,0,.15);  color: #ffb74d; border: 1px solid rgba(255,152,0,.3); }}
  .sig-BALANCED   {{ background: rgba(230,81,0,.15);  color: #ffb74d; border: 1px solid rgba(255,152,0,.3); }}
  .sig-FLAT       {{ background: rgba(230,81,0,.15);  color: #ffb74d; border: 1px solid rgba(255,152,0,.3); }}
  .sig-SOFT       {{ background: rgba(183,28,28,.2);  color: #ef5350; border: 1px solid rgba(239,83,80,.3); }}
  .sig-CONTRACTING{{ background: rgba(183,28,28,.2);  color: #ef5350; border: 1px solid rgba(239,83,80,.3); }}
  .sig-LOOSE      {{ background: rgba(183,28,28,.2);  color: #ef5350; border: 1px solid rgba(239,83,80,.3); }}

  /* ── Freshness badge ── */
  .fresh-dot {{
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
  }}
  .fresh-ok   {{ background: #4caf50; box-shadow: 0 0 4px rgba(76,175,80,.5); }}
  .fresh-warn {{ background: #ffb74d; box-shadow: 0 0 4px rgba(255,183,77,.4); }}
  .fresh-stale{{ background: #ef5350; box-shadow: 0 0 4px rgba(239,83,80,.4); }}
</style>
""", unsafe_allow_html=True)

# ── Header banner with live data ticker ─────────────────────────────────────

# Read live cache values for ticker
_hdr_rates = read_cache("rates") or {}
_hdr_rd    = _hdr_rates.get("data", _hdr_rates)
try:   _hdr_tsy = f"{float(_hdr_rd.get('DGS10') or _hdr_rd.get('ten_year_yield') or _hdr_rd.get('treasury_10yr') or 4.5):.2f}%"
except Exception: _hdr_tsy = "4.50%"

_hdr_cap = read_cache("cap_rate") or {}
_hdr_cd  = _hdr_cap.get("data", _hdr_cap)
try:   _hdr_cap_str = f"{float(_hdr_cd.get('national_avg_cap_rate', _hdr_cd.get('cap_rate', 5.6))):.2f}%"
except Exception: _hdr_cap_str = "5.60%"

_hdr_ms   = read_cache("market_score") or {}
_hdr_top  = "Austin, TX"
try:
    _hdr_sc = _hdr_ms.get("scores") or (_hdr_ms.get("data") or {}).get("rankings") or []
    if _hdr_sc: _hdr_top = _hdr_sc[0].get("market", "Austin, TX")
except Exception: pass

_hdr_vac = read_cache("vacancy") or {}
_hdr_vd  = _hdr_vac.get("data", {}) or {}
try:
    _hdr_nat_vac = _hdr_vd.get("national", {})
    _hdr_ind_vac = _hdr_nat_vac.get("Industrial", {}).get("rate", 4.5)
    _hdr_vac_str = f"{_hdr_ind_vac:.1f}%"
except Exception: _hdr_vac_str = "4.5%"

_hdr_rg = read_cache("rent_growth") or {}
_hdr_rgd = (_hdr_rg.get("data") or {}).get("national", {})
try:
    _hdr_ind_rg  = _hdr_rgd.get("Industrial", {}).get("yoy_pct", 8.0)
    _hdr_rg_str  = f"{_hdr_ind_rg:+.1f}%"
    _hdr_rg_up   = _hdr_ind_rg >= 0
except Exception:
    _hdr_rg_str = "+8.0%"
    _hdr_rg_up  = True

# Active focus pill — computed here, rendered as a native Streamlit row AFTER the header
_hdr_intent    = st.session_state.user_intent
_hdr_focus_pt  = _hdr_intent.get("property_type")
_hdr_focus_loc = _hdr_intent.get("city") or _hdr_intent.get("state") or ""

st.markdown(f"""
<div style="background:#0d0b04; border-bottom:1px solid #2a2208; margin-bottom:0;">

  <!-- Brand row -->
  <div style="display:flex; align-items:center; justify-content:space-between;
              padding:10px 24px 10px;">
    <div style="display:flex; align-items:center; gap:12px;">
      <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
        <rect width="32" height="32" rx="6" fill="#2a2008"/>
        <rect x="6" y="18" width="4" height="8" rx="1" fill="#d4a843"/>
        <rect x="12" y="12" width="4" height="14" rx="1" fill="#e8c060"/>
        <rect x="18" y="6" width="4" height="20" rx="1" fill="#d4a843"/>
        <rect x="24" y="14" width="2" height="12" rx="1" fill="#a07830"/>
        <path d="M6 18 L14 12 L20 6 L26 14" stroke="#f0d080" stroke-width="1.5" fill="none" stroke-linecap="round"/>
      </svg>
      <div>
        <div style="font-size:15px; font-weight:600; color:#d4a843; letter-spacing:0.02em; line-height:1.2;">CRE Intelligence Platform</div>
        <div style="font-size:9px; color:#4a3820; letter-spacing:0.14em; text-transform:uppercase; margin-top:1px;">AI-Powered Commercial Real Estate Intelligence</div>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:12px; font-weight:500; color:#c8a040;">Purdue University</div>
      <div style="font-size:10px; color:#5a4020; letter-spacing:0.08em; text-transform:uppercase; margin-top:1px;">Daniels School · MSF</div>
    </div>
  </div>

  <!-- Live data ticker row -->
  <div style="background:#090700; border-top:1px solid rgba(200,160,64,.1);
              display:flex; align-items:center; height:34px; overflow:hidden;">
    <div style="display:flex; align-items:center; gap:0; height:100%; width:100%;">
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;border-right:1px solid rgba(200,160,64,.08);height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Ind. Cap Rate</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">{_hdr_cap_str}</span>
        <span style="color:#ef5350;font-size:.62rem;">&#9660; 20bps</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;border-right:1px solid rgba(200,160,64,.08);height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Ind. Rent Growth</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">{_hdr_rg_str}</span>
        <span style="color:{"#4caf50" if _hdr_rg_up else "#ef5350"};font-size:.62rem;">{"&#9650;" if _hdr_rg_up else "&#9660;"} YoY</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;border-right:1px solid rgba(200,160,64,.08);height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Ind. Vacancy</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">{_hdr_vac_str}</span>
        <span style="color:#4caf50;font-size:.62rem;">&#9650; tight</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;border-right:1px solid rgba(200,160,64,.08);height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Top Market</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">{_hdr_top}</span>
        <span style="background:rgba(76,175,80,.15);border:1px solid rgba(76,175,80,.3);color:#4caf50;font-size:.58rem;padding:1px 7px;border-radius:10px;font-weight:600;">High</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;border-right:1px solid rgba(200,160,64,.08);height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">10Y Treasury</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">{_hdr_tsy}</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:0 20px;height:100%;white-space:nowrap;">
        <span style="color:#3a3020;font-size:.6rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">DSCR Min</span>
        <span style="color:#d8d0b8;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;">1.25x</span>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

# ── Focus pill row (native Streamlit — rendered below header, never inside f-string) ─
if _hdr_focus_pt:
    _pill_txt = _hdr_focus_pt + (f" · {_hdr_focus_loc}" if _hdr_focus_loc else "")
    _pill_cols = st.columns([4, 2, 1, 1])
    with _pill_cols[1]:
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:flex-end;height:38px;">'
            f'<div style="background:#0d1a0d;border:1px solid #2a5a2a;border-radius:20px;'
            f'padding:3px 12px;display:flex;align-items:center;gap:6px;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:#4caf50;display:inline-block;"></span>'
            f'<span style="font-size:0.7rem;font-weight:600;color:#a0d0a0;letter-spacing:0.06em;">'
            f'FOCUS: {_pill_txt.upper()}</span></div></div>',
            unsafe_allow_html=True,
        )
    with _pill_cols[2]:
        if st.button("Change Focus", key="hdr_change_focus", use_container_width=True):
            st.session_state.onboarding_complete = False
            st.rerun()
    with _pill_cols[3]:
        if st.button("✕ Clear", key="hdr_clear_focus", use_container_width=True):
            st.session_state.user_intent["property_type"] = None
            st.rerun()

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
            _new_intent["raw_input"] = _new_query
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

# Quick brief only shown on the startup/welcome page, not on the dashboard.

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


# ── Custom table renderers (flex-div layout matching Top Metro Areas style) ───

_ROW_BORDER  = "border-bottom:1px solid #1e1a08;"
_COL_HDR     = "font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;"
_RANK_STYLE  = "width:24px;font-size:12px;color:#4a3e18;flex-shrink:0;text-align:center;"
_NAME_STYLE  = "font-size:13px;font-weight:500;color:#d4a843;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
_MUTED_STYLE = "font-size:11px;color:#6a5228;"

def _tbl_container(title: str, count_label: str, col_hdr_html: str, rows_html: str) -> str:
    return (
        f'<div style="background:#131008;border:1px solid #221e0a;border-radius:10px;overflow:hidden;margin:4px 0 12px 0;">'
        # Header bar
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;{_ROW_BORDER}">'
        f'  <div style="display:flex;align-items:center;gap:10px;">'
        f'    <div style="width:3px;height:20px;background:#d4a843;border-radius:2px;flex-shrink:0;"></div>'
        f'    <span style="font-size:11px;font-weight:600;color:#d4a843;letter-spacing:0.08em;text-transform:uppercase;">{title}</span>'
        f'  </div>'
        f'  <span style="font-size:11px;color:#6a5228;">{count_label}</span>'
        f'</div>'
        # Column headers
        f'<div style="display:flex;align-items:center;padding:8px 16px;{_ROW_BORDER}gap:8px;">'
        f'  {col_hdr_html}'
        f'</div>'
        # Rows
        f'{rows_html}'
        f'</div>'
    )

def _dot_indicator(score: float, total: int = 5, max_val: float = 100.0) -> str:
    filled = max(0, min(total, round(score / max(max_val, 1) * total)))
    return "".join(
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:{"#4a9e58" if i < filled else "#2a2208"};margin-right:2px;"></span>'
        for i in range(total)
    )

def _bar_cell(value: float, max_abs: float, show_pct: bool = True) -> str:
    color = "#4a9e58" if value >= 0 else "#ef5350"
    w = min(100, abs(value) / max(max_abs, 0.01) * 100)
    sign = "+" if value > 0 else ""
    label = f"{sign}{value:.1f}%" if show_pct else f"{value:.0f}"
    return (
        f'<div style="display:flex;flex-direction:column;gap:3px;">'
        f'  <span style="font-size:12px;color:#c8b890;">{label}</span>'
        f'  <div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">'
        f'    <div style="height:100%;width:{w:.0f}%;background:linear-gradient(90deg,{"#2a7a38,#4a9e58" if value >= 0 else "#8b2a2a,#ef5350"});border-radius:2px;"></div>'
        f'  </div>'
        f'</div>'
    )

def _score_bar_cell(score: float) -> str:
    color_g = "#2a7a38,#4a9e58" if score >= 60 else ("#7a6a20,#c8a040" if score >= 40 else "#8b2a2a,#ef5350")
    w = min(100, score)
    return (
        f'<div style="display:flex;flex-direction:column;gap:3px;">'
        f'  <span style="font-size:12px;color:#c8b890;">{score:.0f}</span>'
        f'  <div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">'
        f'    <div style="height:100%;width:{w:.0f}%;background:linear-gradient(90deg,{color_g});border-radius:2px;"></div>'
        f'  </div>'
        f'</div>'
    )

def _zone_pill(zone: str) -> str:
    styles = {
        "Best Fit": "background:#0d2a12;color:#4a9e58",
        "Moderate": "background:#2a1a04;color:#a07830",
        "Low":      "background:#2a0d0d;color:#9e4a4a",
    }
    s = styles.get(zone, "background:#1a1a0a;color:#6a6a4a")
    return f'<span style="{s};font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;">{zone}</span>'

def _driver_tag(text: str) -> str:
    return f'<span style="background:#1a1505;color:#6a5228;font-size:11px;padding:3px 10px;border-radius:20px;">{text}</span>'


def _render_county_table(df, active_pt=None, title="County Rankings", start_rank=1) -> str:
    if df.empty:
        return ""
    max_pop = max(df["pop_growth_pct"].abs().max(), 0.1)

    # Column header row
    col_hdrs = (
        f'<div style="width:24px;"></div>'
        f'<div style="flex:2.5;{_COL_HDR}">COUNTY</div>'
        f'<div style="flex:1.2;{_COL_HDR}">POPULATION</div>'
        f'<div style="flex:1.5;{_COL_HDR}">POP GROWTH</div>'
    )
    if active_pt:
        col_hdrs += f'<div style="flex:1.5;{_COL_HDR}">{active_pt.upper()} SCORE</div>'
    col_hdrs += (
        f'<div style="flex:1.2;{_COL_HDR}">MIGRATION</div>'
        f'<div style="flex:2;{_COL_HDR}">KEY DRIVER</div>'
    )

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        rank = start_rank + i
        r = (
            f'<div style="display:flex;align-items:center;padding:10px 16px;{_ROW_BORDER}gap:8px;">'
            f'  <div style="{_RANK_STYLE}">{rank}</div>'
            f'  <div style="flex:2.5;{_NAME_STYLE}">{row["name"]}</div>'
            f'  <div style="flex:1.2;font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#8a7040;">{int(row["population"]):,}</div>'
            f'  <div style="flex:1.5;">{_bar_cell(row["pop_growth_pct"], max_pop)}</div>'
        )
        if active_pt:
            r += f'<div style="flex:1.5;">{_score_bar_cell(row["pt_score"])}</div>'
        r += (
            f'  <div style="flex:1.2;display:flex;align-items:center;gap:2px;">{_dot_indicator(row["migration_score"])}</div>'
            f'  <div style="flex:2;">{_driver_tag(str(row["top_driver"]))}</div>'
            f'</div>'
        )
        rows_html += r

    return _tbl_container(title, f'{len(df)} counties tracked', col_hdrs, rows_html)


def _render_neighborhood_table(df, active_pt=None, title="Neighborhood Rankings", start_rank=1) -> str:
    if df.empty:
        return ""
    max_pop  = max(df["pop_growth_pct"].abs().max(), 0.1)
    max_rent = max(df["median_rent_growth_pct"].abs().max(), 0.1)

    col_hdrs = (
        f'<div style="width:24px;"></div>'
        f'<div style="flex:2;{_COL_HDR}">NEIGHBORHOOD</div>'
        f'<div style="flex:1;{_COL_HDR}">TYPE</div>'
        f'<div style="flex:1;{_COL_HDR}">ZONE FIT</div>'
    )
    if active_pt:
        col_hdrs += f'<div style="flex:1.2;{_COL_HDR}">{active_pt.upper()} SCORE</div>'
    col_hdrs += (
        f'<div style="flex:1;{_COL_HDR}">MIGRATION</div>'
        f'<div style="flex:1.2;{_COL_HDR}">POP GROWTH</div>'
        f'<div style="flex:1.2;{_COL_HDR}">RENT GROWTH</div>'
    )

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        rank = start_rank + i
        r = (
            f'<div style="display:flex;align-items:center;padding:10px 16px;{_ROW_BORDER}gap:8px;">'
            f'  <div style="{_RANK_STYLE}">{rank}</div>'
            f'  <div style="flex:2;{_NAME_STYLE}">{row["name"]}</div>'
            f'  <div style="flex:1;{_MUTED_STYLE}">{row["neighborhood_type"]}</div>'
            f'  <div style="flex:1;">{_zone_pill(str(row.get("zone_fit","—")))}</div>'
        )
        if active_pt:
            r += f'<div style="flex:1.2;">{_score_bar_cell(row["pt_score"])}</div>'
        r += (
            f'  <div style="flex:1;display:flex;align-items:center;gap:2px;">{_dot_indicator(row["migration_score"])}</div>'
            f'  <div style="flex:1.2;">{_bar_cell(row["pop_growth_pct"], max_pop)}</div>'
            f'  <div style="flex:1.2;">{_bar_cell(row["median_rent_growth_pct"], max_rent)}</div>'
            f'</div>'
        )
        rows_html += r

    return _tbl_container(title, f'{len(df)} neighborhoods tracked', col_hdrs, rows_html)


def _render_generic_table(df, title, count_label="", hints=None, scrollable=False, max_height=400) -> str:
    """
    Universal flex-div table renderer matching the Top Metro Areas style.

    hints: dict of col_name -> {
      type: "text"|"name"|"pct_bar"|"score_bar"|"badge"|"tag"|"colored"|"price"|"dots"
      flex: float (default 1)
      badge_map: {value: "css_string"}
      bar_max: float
    }
    scrollable: wrap rows in a scrollable div
    max_height: height in px for scrollable
    """
    if hints is None:
        hints = {}
    if df.empty:
        return ""

    _RGT_ROW_BORDER = "border-bottom:1px solid #1e1a08;"
    _RGT_COL_HDR    = "font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;"
    _RGT_RANK_STYLE = "width:24px;font-size:12px;color:#4a3e18;flex-shrink:0;text-align:center;"

    cols = list(df.columns)

    # Auto-compute bar_max for pct_bar columns
    auto_bar_max = {}
    for c in cols:
        h = hints.get(c, {})
        if h.get("type") == "pct_bar" and "bar_max" not in h:
            try:
                vals = df[c].apply(lambda v: abs(float(
                    str(v).replace("%","").replace("pp","").replace("+","").strip()
                )))
                auto_bar_max[c] = max(vals.max(), 0.01)
            except Exception:
                auto_bar_max[c] = 1.0

    def _rgt_cell_html(col, val):
        h = hints.get(col, {})
        t = h.get("type", "text")
        raw = str(val) if val is not None else "—"

        if t == "name":
            return f'<span style="font-size:13px;font-weight:500;color:#d4a843;">{raw}</span>'

        if t == "pct_bar":
            try:
                clean = raw.replace("%","").replace("pp","").strip()
                num = float(clean)
                mx = h.get("bar_max", auto_bar_max.get(col, 1.0))
                color_g = "#2a7a38,#4a9e58" if num >= 0 else "#8b2a2a,#ef5350"
                w = min(100, abs(num)/max(mx,0.01)*100)
                sign = "+" if num > 0 else ""
                unit = "%" if "%" in raw else "pp"
                label = f"{sign}{num:.1f}{unit}"
                return (f'<div style="display:flex;flex-direction:column;gap:3px;">'
                        f'<span style="font-size:12px;color:#c8b890;">{label}</span>'
                        f'<div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">'
                        f'<div style="height:100%;width:{w:.0f}%;background:linear-gradient(90deg,{color_g});border-radius:2px;"></div>'
                        f'</div></div>')
            except Exception:
                pass

        if t == "score_bar":
            try:
                num = float(str(val).split("/")[0])
                mx = h.get("bar_max", 100.0)
                color_g = "#2a7a38,#4a9e58" if num >= mx*0.6 else ("#7a6a20,#c8a040" if num >= mx*0.4 else "#8b2a2a,#ef5350")
                w = min(100, num/max(mx,1)*100)
                return (f'<div style="display:flex;flex-direction:column;gap:3px;">'
                        f'<span style="font-size:12px;color:#c8b890;">{num:.0f}</span>'
                        f'<div style="height:4px;background:#1e1a08;border-radius:2px;width:80px;">'
                        f'<div style="height:100%;width:{w:.0f}%;background:linear-gradient(90deg,{color_g});border-radius:2px;"></div>'
                        f'</div></div>')
            except Exception:
                pass

        if t == "badge":
            bmap = h.get("badge_map", {})
            s = bmap.get(raw, "background:#1a1a0a;color:#6a6a4a")
            return f'<span style="{s};font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;">{raw}</span>'

        if t == "tag":
            return f'<span style="background:#1a1505;color:#6a5228;font-size:11px;padding:3px 10px;border-radius:20px;">{raw}</span>'

        if t == "colored":
            try:
                clean = str(val).replace("%","").replace("pp","").replace("bps","").strip()
                num = float(clean)
                color = "#4a9e58" if num > 0 else ("#ef5350" if num < 0 else "#8a7040")
                return f'<span style="font-size:12px;color:{color};font-weight:500;">{raw}</span>'
            except Exception:
                pass

        if t == "price":
            return f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#c8b890;">{raw}</span>'

        if t == "dots":
            try:
                num = float(str(val))
                mx = h.get("bar_max", 100.0)
                filled = max(0, min(5, round(num/max(mx,1)*5)))
                dots = "".join(
                    f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                    f'background:{"#4a9e58" if i < filled else "#2a2208"};margin-right:2px;"></span>'
                    for i in range(5)
                )
                return f'<div style="display:flex;align-items:center;">{dots}</div>'
            except Exception:
                pass

        # Default plain text — first column gets gold, rest muted
        is_first = (col == cols[0])
        clr = "#d4a843" if is_first else "#8a7040"
        return f'<span style="font-size:12px;color:{clr};">{raw}</span>'

    # Column headers
    col_hdr_html = '<div style="width:24px;"></div>'
    for c in cols:
        flex = hints.get(c, {}).get("flex", 1)
        col_hdr_html += f'<div style="flex:{flex};{_RGT_COL_HDR}">{c.upper()}</div>'

    # Data rows
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        r = f'<div style="display:flex;align-items:center;padding:10px 16px;{_RGT_ROW_BORDER}gap:8px;">'
        r += f'<div style="{_RGT_RANK_STYLE}">{i+1}</div>'
        for c in cols:
            flex = hints.get(c, {}).get("flex", 1)
            r += f'<div style="flex:{flex};">{_rgt_cell_html(c, row[c])}</div>'
        r += '</div>'
        rows_html += r

    if not count_label:
        count_label = f"{len(df)} records"

    if scrollable:
        scroll_rows = f'<div style="max-height:{max_height}px;overflow-y:auto;">{rows_html}</div>'
        return (
            f'<div style="background:#131008;border:1px solid #221e0a;border-radius:10px;overflow:hidden;margin:4px 0 12px 0;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #1e1a08;">'
            f'  <div style="display:flex;align-items:center;gap:10px;">'
            f'    <div style="width:3px;height:20px;background:#d4a843;border-radius:2px;flex-shrink:0;"></div>'
            f'    <span style="font-size:11px;font-weight:600;color:#d4a843;letter-spacing:0.08em;text-transform:uppercase;">{title}</span>'
            f'  </div>'
            f'  <span style="font-size:11px;color:#6a5228;">{count_label}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;padding:8px 16px;border-bottom:1px solid #1e1a08;gap:8px;">'
            f'  {col_hdr_html}'
            f'</div>'
            f'{scroll_rows}'
            f'</div>'
        )
    return _tbl_container(title, count_label, col_hdr_html, rows_html)


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
        st.warning(f"Agent is fetching {cache_key} data for the first time — please wait ~30 seconds and refresh.")
        return False
    age = cache_age_label(cache_key)
    if c.get("stale"):
        dot = '<span class="fresh-dot fresh-stale"></span>'
        st.markdown(
            f'<div style="font-size:11px;color:#8a6030;padding:6px 0;">'
            f'{dot}Data is stale (last updated {age}) — agent may be restarting</div>',
            unsafe_allow_html=True,
        )
    else:
        dot = '<span class="fresh-dot fresh-ok"></span>'
        st.markdown(
            f'<div style="font-size:11px;color:#5a4820;padding:4px 0;">'
            f'{dot}Live data · last updated {age} · auto-refreshes in background</div>',
            unsafe_allow_html=True,
        )
    return True

def agent_last_updated(agent_name: str):
    age = cache_age_label(agent_name)
    dot = '<span class="fresh-dot fresh-ok"></span>'
    st.markdown(
        f'<div style="font-size:11px;color:#5a4820;padding:4px 0;">{dot}Last updated: {age}</div>',
        unsafe_allow_html=True,
    )


# ── Tooltip helper ────────────────────────────────────────────────────────────
def _tt(label: str, tip: str) -> str:
    """Wrap a label with a hoverable ? tooltip. Returns HTML string."""
    return (
        f'<span class="cre-tt">{label}'
        f'<span class="cre-tt-icon">?</span>'
        f'<span class="cre-tt-box">{tip}</span>'
        f'</span>'
    )


# ── Signal pill helper ────────────────────────────────────────────────────────
_SIG_DESCRIPTIONS = {
    "EXPANDING":   ("▲", "Sector ETF up >2% — corporate hiring is growing, driving more leasing demand"),
    "FLAT":        ("→", "Sector ETF within ±2% — demand is stable, no strong expansion or contraction"),
    "CONTRACTING": ("▼", "Sector ETF down >2% — hiring is slowing, leasing demand may soften"),
    "STRONG":      ("▲", "Score ≥65 — low unemployment, rising payrolls, high job openings. Landlords hold pricing power"),
    "MODERATE":    ("→", "Score 41–64 — mixed signals. Demand present but softening. Be selective"),
    "SOFT":        ("▼", "Score ≤40 — rising unemployment or falling payrolls. Tenants downsizing, concessions rising"),
    "TIGHT":       ("▲", "Unemployment <4% — strong local economy, higher occupier demand and rent growth potential"),
    "BALANCED":    ("→", "Unemployment 4–6% — stable absorption, neutral for CRE demand"),
    "LOOSE":       ("▼", "Unemployment >6% — weaker absorption, potential vacancy risk and lease concessions"),
    "HIGH":        ("▲", "Costs rising faster than average — construction budgets at risk of overrun"),
    "LOW":         ("▼", "Costs below average — favorable for development margins"),
    "HOT":         ("▲", "CPI well above 2% — replacement cost support for values, but rate uncertainty rising"),
    "COOLING":     ("▼", "Inflation trending back toward target — rate pressure may ease"),
    "LOOSE_CREDIT":("▲", "Spreads narrow, banks easing — debt cheap and available, deal volume high"),
    "TIGHT_CREDIT":("▼", "Spreads wide, banks tightening — higher equity requirements, fewer loans closing"),
}

def _sig_pill(signal: str, credit_context: bool = False) -> str:
    """Return a colored HTML pill with a plain-English tooltip for a signal label."""
    key = signal.upper()
    if credit_context and key == "LOOSE":
        key = "LOOSE_CREDIT"
    if credit_context and key == "TIGHT":
        key = "TIGHT_CREDIT"
    arrow, desc = _SIG_DESCRIPTIONS.get(key, ("·", signal))
    css_key = key.replace("_CREDIT", "").replace("_", "")
    return (
        f'<span class="cre-tt">'
        f'<span class="sig-pill sig-{css_key}">{arrow} {signal}</span>'
        f'<span class="cre-tt-box">{desc}</span>'
        f'</span>'
    )


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

# ── Auto-navigate to requested tab via JS click ───────────────────────────────
if st.session_state.get("nav_to_tab"):
    _target_tab = st.session_state.nav_to_tab
    st.session_state.nav_to_tab = None
    st.components.v1.html(f"""
    <script>
      (function() {{
        function clickTab() {{
          var tabs = window.parent.document.querySelectorAll('[data-testid="stTab"] button, button[role="tab"]');
          for (var i = 0; i < tabs.length; i++) {{
            if (tabs[i].textContent.trim().toLowerCase().includes('{_target_tab.lower()}')) {{
              tabs[i].click();
              return;
            }}
          }}
        }}
        setTimeout(clickTab, 300);
        setTimeout(clickTab, 700);
      }})();
    </script>
    """, height=0)


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

    # Pre-fill from global intent if text area is still empty
    _adv_intent    = st.session_state.user_intent
    _adv_intent_pt = _adv_intent.get("property_type")
    _adv_intent_loc = _adv_intent.get("city") or _adv_intent.get("state") or _adv_intent.get("location")

    if _adv_intent_pt and not st.session_state.get("adv_prompt_text") and not st.session_state.get("adv_result"):
        _adv_pt_hints = {
            "Industrial":  ("warehouse / industrial facility", "10,000 sq ft", "$5M"),
            "Multifamily": ("multifamily / apartment complex", "24 units",     "$8M"),
            "Office":      ("office building",                  "8,000 sq ft",  "$4M"),
            "Retail":      ("retail strip center",              "5,000 sq ft",  "$3M"),
            "Healthcare":  ("medical office / healthcare",       "6,000 sq ft",  "$4M"),
        }
        _adv_hint = _adv_pt_hints.get(_adv_intent_pt, (_adv_intent_pt.lower(), "10,000 sq ft", "$5M"))
        _adv_loc_str = f" in {_adv_intent_loc}" if _adv_intent_loc else ""
        st.session_state["adv_prompt_text"] = (
            f"I want to invest in a {_adv_hint[0]}{_adv_loc_str} — "
            f"approximately {_adv_hint[1]} with a {_adv_hint[2]} budget over a 5-year hold"
        )

    # Context banner when intent is populated
    if _adv_intent_pt:
        _adv_ctx_loc = f" · {_adv_intent_loc}" if _adv_intent_loc else ""
        st.markdown(
            f'<div style="background:#0d1a0d;border:1px solid #2a4a2a;border-radius:7px;'
            f'padding:8px 14px;margin-bottom:8px;font-size:0.8rem;color:#7ab07a;">'
            f'Session focus: <b style="color:#a0d0a0;">{_adv_intent_pt}{_adv_ctx_loc}</b> — '
            f'prompt pre-filled from your selection. Edit freely.</div>',
            unsafe_allow_html=True,
        )

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
                import traceback as _tb
                st.error(f"Error generating recommendation: {_adv_err}")
                st.code(_tb.format_exc(), language="python")

    # ── Data source freshness banner ──────────────────────────────────────────
    _mgr_report = (read_cache("manager_report").get("data") or {})
    _adv_deps   = _mgr_report.get("advisor_dependencies", [])
    _dep_issues = [d for d in _adv_deps if d.get("health") not in ("OK", "RUNNING")]
    _backed_off = _mgr_report.get("backed_off_agents", [])
    if _dep_issues:
        _dep_names  = ", ".join(d["agent"].replace("_", " ").title() for d in _dep_issues)
        _hints      = list({d["hint"] for d in _dep_issues if d.get("hint")})
        _hint_txt   = f" — {_hints[0]}" if _hints else ""
        st.warning(
            f"Some data sources used by the Investment Advisor are stale or missing: "
            f"**{_dep_names}**{_hint_txt}. "
            f"Results may be less accurate until they refresh. "
            f"Go to the **About** tab → System Health to trigger a manual refresh.",
            icon="⚠️",
        )
    if _backed_off:
        _bo_names = ", ".join(a.replace("_", " ").title() for a in _backed_off)
        st.error(
            f"**{_bo_names}** failed 3+ times and need manual attention. "
            f"Check your API keys in the .env file or verify network access.",
            icon="❌",
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  REPORT OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    _adv_result = st.session_state.adv_result
    # Validate stored result has required keys (clears stale results from old code versions)
    if _adv_result and "error" not in _adv_result:
        _required_keys = {"primary", "financials", "runners", "weights", "narrative", "params"}
        if not _required_keys.issubset(_adv_result.keys()):
            st.session_state.adv_result = None
            _adv_result = None
    if _adv_result and "error" not in _adv_result:
      try:
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

        # ── "What does this mean for me?" AI explanation ─────────────────────
        if "adv_plain_english" not in st.session_state:
            st.session_state.adv_plain_english = None

        _wtm_col, _wtm_btn_col = st.columns([5, 1])
        with _wtm_btn_col:
            if st.button("What does this mean for me?", key="wtm_btn",
                         help="Get a plain-English explanation of these results — no jargon"):
                st.session_state.adv_plain_english = None   # clear previous
                with st.spinner("Generating plain-English explanation..."):
                    try:
                        import os as _wtm_os
                        from groq import Groq as _WGroq
                        _wtm_key = _wtm_os.getenv("GROQ_API_KEY", "")
                        if not _wtm_key:
                            st.session_state.adv_plain_english = "GROQ_API_KEY not set — add it to .env to enable AI explanations."
                        else:
                            _wtm_client = _WGroq(api_key=_wtm_key)
                            _wtm_fin    = _adv_result.get("financials", {})
                            _wtm_prim   = _adv_result.get("primary", {})
                            _wtm_params = _adv_result.get("params", {})
                            _wtm_prompt = (
                                f"An investor asked: '{st.session_state.get('adv_prompt_submitted', _wtm_params.get('location_raw',''))}'\n\n"
                                f"The AI recommended: {_wtm_prim.get('market','N/A')} for {_wtm_params.get('property_type','CRE')} development.\n"
                                f"Key numbers:\n"
                                f"- Total project cost: ${_wtm_fin.get('total_cost',0)/1e6:.1f}M\n"
                                f"- Estimated IRR: {_wtm_fin.get('irr_est',0):.1f}%\n"
                                f"- Annual NOI: ${_wtm_fin.get('annual_noi',0)/1e3:.0f}K\n"
                                f"- Total profit over hold: ${_wtm_fin.get('total_profit',0)/1e6:.1f}M\n"
                                f"- Opportunity score: {_wtm_prim.get('opportunity_score',0):.0f}/100\n"
                                f"- Market grade: {_wtm_prim.get('grade','N/A')}\n"
                                f"- Climate risk: {_wtm_prim.get('climate_label','N/A')}\n"
                                f"- Hold period: {_wtm_params.get('timeline_years',5)} years\n"
                            )
                            _wtm_resp = _wtm_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": (
                                        "You are a patient, friendly real estate advisor explaining a CRE investment "
                                        "recommendation to someone with no finance background. Use plain English — no jargon. "
                                        "Structure your response as:\n"
                                        "1. What the AI is recommending and why (2 sentences)\n"
                                        "2. What the numbers mean in simple terms (3-4 sentences — explain IRR, NOI, profit in everyday language)\n"
                                        "3. The biggest opportunity (1 sentence)\n"
                                        "4. The biggest risk to watch (1 sentence)\n"
                                        "Be encouraging but honest. Write like you're talking to a friend, not writing a report."
                                    )},
                                    {"role": "user", "content": _wtm_prompt},
                                ],
                                max_tokens=400,
                                temperature=0.4,
                            )
                            st.session_state.adv_plain_english = _wtm_resp.choices[0].message.content.strip()
                    except Exception as _wtm_err:
                        st.session_state.adv_plain_english = f"Could not generate explanation: {_wtm_err}"

        if st.session_state.adv_plain_english:
            with _wtm_col:
                st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1a0d 0%,#111a08 100%);
            border:1px solid #2a4020;border-left:4px solid #4caf50;
            border-radius:8px;padding:18px 22px;margin-bottom:16px;">
  <div style="color:#80c858;font-size:0.78rem;font-weight:600;letter-spacing:.08em;
              text-transform:uppercase;margin-bottom:10px;">Plain-English Explanation</div>
  <div style="color:#d4e8c4;font-size:0.92rem;line-height:1.65;white-space:pre-wrap;">{st.session_state.adv_plain_english}</div>
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

        for _col, (_lbl, _tip, _val, _clr) in zip(
            [_c1, _c2, _c3, _c4, _c5],
            [
                ("Opportunity Score",  "Composite 0–100 score weighing migration (25%), vacancy (20%), rent growth (15%), labor (15%), cap rate (10%), land (10%), and macro (5%). ≥75 = A-grade market, ≥60 = B, below 50 = avoid.",
                 f"{_opp:.1f}/100",                             _adv_score_color(_opp)),
                ("Est. Total Cost",    "All-in development cost: land purchase + hard construction + soft costs (architecture, permits, financing fees, contingency). This is the total capital needed before the property generates any income.",
                 f"${financials['total_cost']/1e6:.2f}M",         "#e8dfc4"),
                ("Estimated ROI",      "Return on Investment over the full hold period: (Total Profit ÷ Total Cost) × 100. Includes rental income and estimated exit sale proceeds. Does NOT account for leverage. Target: 20%+ is strong for CRE development.",
                 f"{_roi}%",                                     _roi_color),
                ("Buildout Timeline",  "Estimated months from land acquisition to Certificate of Occupancy. Includes entitlement, design, and construction phases. Longer buildout = more capital at risk before income begins.",
                 f"{financials['buildout_months']} mo",           "#e8dfc4"),
                ("Est. Exit Value",    "Projected sale price at end of hold period, calculated as: stabilized NOI ÷ exit cap rate. A lower exit cap rate (strong market) = higher sale price. This drives most of the total return.",
                 f"${financials['exit_value']/1e6:.2f}M",          "#d4a843"),
            ]
        ):
            _col.markdown(
                f'<div class="metric-card"><div class="label">{_tt(_lbl, _tip)}</div>'
                f'<div class="value" style="color:{_clr};font-size:1.5rem;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cross-Agent Signal Correlator ─────────────────────────────────────
        section(" Cross-Agent Market Signal")
        try:
            from src.signal_correlator import run_signal_correlator as _run_corr
            _corr = _run_corr()
            _ov   = _corr["overall"]
            _ov_c = _corr["color"]
            _conf = _corr["confidence"]
            _align = _corr["alignment"]
            _sigs  = _corr["signals"]
            _reasons = _corr["top_reasons"]
            _risks   = _corr["top_risks"]

            # Direction icon per signal
            _dir_icon = {1: "▲", -1: "▼", 0: "→"}
            _dir_clr  = {1: "#4caf50", -1: "#f44336", 0: "#ffb74d"}

            _sig_rows = "".join([
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid #1e1a08;">'
                f'<span style="color:{_dir_clr[s["direction"]]};font-size:1rem;flex-shrink:0;width:14px;">{_dir_icon[s["direction"]]}</span>'
                f'<div>'
                f'<span style="color:#c8b890;font-size:0.82rem;font-weight:600;">{s["dimension"]}</span>'
                f'<span style="color:{_dir_clr[s["direction"]]};font-size:0.76rem;margin-left:8px;">{s["verdict"]}'
                + (f' · {s["label"]}' if s.get("label") and s["label"] not in ("N/A", "") else "")
                + f'</span>'
                f'<div style="color:#6a5630;font-size:0.76rem;margin-top:2px;">{s["reason"]}</div>'
                f'</div></div>'
                for s in _sigs if s.get("label") != "N/A"
            ])

            _reasons_html = "".join([f'<div style="color:#80c858;font-size:0.8rem;padding:3px 0;">+ {r}</div>' for r in _reasons])
            _risks_html   = "".join([f'<div style="color:#ef9a9a;font-size:0.8rem;padding:3px 0;">− {r}</div>' for r in _risks])

            st.markdown(f"""
<div style="background:#13110a;border:1px solid #2a2208;border-radius:10px;padding:20px 24px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:20px;margin-bottom:16px;flex-wrap:wrap;">
    <div style="font-size:1.6rem;font-weight:700;color:{_ov_c};letter-spacing:1px;">{_ov}</div>
    <div>
      <div style="color:#a09880;font-size:0.82rem;">Confidence: <span style="color:{_ov_c};font-weight:600;">{_conf}%</span></div>
      <div style="color:#5a4820;font-size:0.76rem;">{_align}</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:16px;">
      <div style="text-align:center;"><div style="color:#4caf50;font-size:1.2rem;font-weight:700;">{_corr['n_positive']}</div><div style="color:#3a3010;font-size:0.7rem;">Bullish</div></div>
      <div style="text-align:center;"><div style="color:#ffb74d;font-size:1.2rem;font-weight:700;">{_corr['n_neutral']}</div><div style="color:#3a3010;font-size:0.7rem;">Neutral</div></div>
      <div style="text-align:center;"><div style="color:#f44336;font-size:1.2rem;font-weight:700;">{_corr['n_negative']}</div><div style="color:#3a3010;font-size:0.7rem;">Cautionary</div></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>{_sig_rows}</div>
    <div>
      {"<div style='color:#6a5228;font-size:0.7rem;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;'>Top Reasons</div>" + _reasons_html if _reasons_html else ""}
      {"<div style='color:#6a5228;font-size:0.7rem;letter-spacing:.08em;text-transform:uppercase;margin:10px 0 6px;'>Key Risks</div>" + _risks_html if _risks_html else ""}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        except Exception as _ce:
            st.caption(f"Signal correlator unavailable: {_ce}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        #  LOCATION MARKET INTELLIGENCE
        # ══════════════════════════════════════════════════════════════════════
        _loc_raw   = (params.get("location_raw") or "").lower().strip()
        _prop_type = params.get("property_type", "Industrial")

        # State name → abbr lookup for display and filtering
        _STATE_ABBR = {
            "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
            "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
            "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS",
            "kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA",
            "michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT",
            "nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ","new mexico":"NM",
            "new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
            "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
            "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT",
            "virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY",
            "sunbelt":"TX","southeast":"TN","midwest":"IL","northeast":"NY","southwest":"AZ",
        }
        _STATE_TAX = {
            "TX":("0%","No state income tax — highest net operating income retention in the US"),
            "FL":("0%","No state income tax — strong for REIT structures and passive investors"),
            "NV":("0%","No state income tax — favorable for holding entities"),
            "TN":("0%","No state income tax on wages — eliminated Hall Tax in 2021"),
            "WY":("0%","No state income tax — minimal regulatory burden"),
            "SD":("0%","No state income tax"),
            "AK":("0%","No state income tax"),
            "WA":("0%","No state personal income tax (capital gains tax enacted 2023)"),
            "AZ":("2.5%","Flat 2.5% state income tax — among the lowest rates nationally"),
            "CO":("4.4%","Flat 4.4% rate — Enterprise Zone credits available for development"),
            "NC":("4.5%","Competitive flat tax with phased reductions through 2026"),
            "GA":("5.49%","Flat tax trajectory; QOZ-heavy state with strong Job Tax Credits"),
            "UT":("4.65%","Flat 4.65% — strong OZ pipeline in Salt Lake corridor"),
            "SC":("6.4%","Standard rate but aggressive OZ and county incentive programs"),
            "IN":("3.05%","One of the lowest flat rates in the Midwest"),
            "OH":("3.75%","Progressive up to 3.75%; municipal taxes apply in metro areas"),
            "IL":("4.95%","Flat rate; higher property taxes offset income tax savings"),
            "CA":("13.3%","Highest marginal rate nationally — factor into hold period net returns"),
            "NY":("10.9%","Combined city + state can exceed 14% in NYC"),
        }

        # Pull from caches
        _lmi_lm   = (read_cache("labor_market").get("data") or {})
        _lmi_rg   = (read_cache("rent_growth").get("data") or {})
        _lmi_cap  = (read_cache("cap_rate").get("data") or {})
        _lmi_vac  = (read_cache("vacancy").get("data") or {})
        _lmi_oz   = (read_cache("opportunity_zone").get("data") or {})
        _lmi_ms   = (read_cache("market_score").get("data") or {})

        # Match state
        _matched_abbr = None
        for _sname, _sabb in _STATE_ABBR.items():
            if _sname in _loc_raw:
                _matched_abbr = _sabb; break

        # Prop-type → cache key mapping
        _pt_vac_key = {
            "Industrial":"Industrial","Multifamily":"Multifamily",
            "Office":"Office","Retail":"Retail","Healthcare":"Office",
        }.get(_prop_type, "Industrial")
        _pt_rg_key = {
            "Industrial":"Industrial","Multifamily":"Multifamily",
            "Office":"Office","Retail":"Retail","Healthcare":"Office",
        }.get(_prop_type, "Industrial")

        section(f" Location Intelligence — {_location or 'Target Market'}")

        # ── Row 1: live KPI stats ─────────────────────────────────────────────
        _li1, _li2, _li3, _li4, _li5 = st.columns(5)

        # Unemployment
        _unemp_data = _lmi_lm.get("metro_unemployment", [])
        _unemp_match = next((r for r in _unemp_data
                             if _loc_raw and any(w in r["market"].lower() for w in _loc_raw.split()
                             if len(w) > 3)), None)
        _unemp_val = f"{_unemp_match['unemp_rate']:.1f}%" if _unemp_match else (
            _lmi_lm.get("fred_labor", {}).get("Unemployment Rate", {}).get("current") or "—")
        if isinstance(_unemp_val, float): _unemp_val = f"{_unemp_val:.1f}%"
        _unemp_sig = _unemp_match.get("signal","") if _unemp_match else ""
        _unemp_clr = "#4caf50" if _unemp_sig == "TIGHT" else ("#ff9800" if _unemp_sig == "BALANCED" else "#c8b890")

        # Rent growth for property type
        _rg_nat = _lmi_rg.get("national", {})
        _rg_val = _rg_nat.get(_pt_rg_key, {}).get("yoy_pct") or _rg_nat.get(_pt_rg_key, {}).get("yoy") or "—"
        _rg_clr = "#4caf50" if isinstance(_rg_val, (int,float)) and _rg_val > 2 else ("#ff9800" if isinstance(_rg_val, (int,float)) and _rg_val > 0 else "#ef5350")
        _rg_fmt = f"{_rg_val:+.1f}%" if isinstance(_rg_val, (int,float)) else str(_rg_val)

        # Cap rate
        _cap_nat = _lmi_cap.get("national", {})
        _cap_val = _cap_nat.get(_prop_type, {}).get("cap_rate") or _cap_nat.get(_prop_type, {}).get("rate")
        if not _cap_val:
            for _k, _v in _cap_nat.items():
                if _prop_type.lower() in _k.lower():
                    _cap_val = _v.get("cap_rate") or _v.get("rate"); break
        _cap_fmt = f"{_cap_val:.2f}%" if isinstance(_cap_val, (int,float)) else f"{primary.get('cap_rate',0):.2f}%"

        # Vacancy
        _vac_nat = _lmi_vac.get("national", NATIONAL_VACANCY)
        _vac_entry = _vac_nat.get(_pt_vac_key, {})
        _vac_val = _vac_entry.get("rate") or _vac_entry.get("vacancy_rate") if isinstance(_vac_entry, dict) else _vac_entry
        _vac_fmt = f"{_vac_val:.1f}%" if isinstance(_vac_val, (int,float)) else "—"

        # Migration score
        _mig_score = primary.get("mig_score", 0)

        _li_cards = [
            (
                _tt("Unemployment", "Share of the labor force actively seeking work. Below 4% (TIGHT) signals a strong local economy with high tenant demand and rent growth potential. Above 6% (LOOSE) signals weaker absorption and higher vacancy risk. Source: FRED / BLS (UNRATE series), updated every 6 hours."),
                _unemp_val, _unemp_clr, _unemp_sig or "State rate"
            ),
            (
                _tt(f"{_prop_type} Rent Growth", f"Year-over-year change in asking rents for {_prop_type.lower()} properties nationwide. Positive growth above 3% signals strong landlord pricing power. Negative growth signals oversupply or weak demand. Source: Zillow / CoStar benchmarks via Agent 15, updated every 6 hours."),
                _rg_fmt, _rg_clr, "YoY national"
            ),
            (
                _tt(f"{_prop_type} Cap Rate", f"Capitalization rate = Net Operating Income ÷ Property Value. A lower cap rate means investors are paying more for each dollar of income (bullish sentiment). Compare to the 10Y Treasury yield — a spread above 150bps is generally considered attractive. Source: CoStar / Agent 5, updated every 6 hours."),
                _cap_fmt, "#d4a843", "National benchmark"
            ),
            (
                _tt(f"{_prop_type} Vacancy", f"Percentage of {_prop_type.lower()} space currently unoccupied. Below 7% gives landlords pricing power; above 14% signals oversupply and rent pressure. This is the national average — local markets may vary significantly. Source: CoStar / Agent 4, updated every 12 hours."),
                _vac_fmt, "#c8b890", "National avg"
            ),
            (
                _tt("Migration Score", "Composite 0–100 score measuring net population inflow into this state. Combines Census Bureau population estimates, IRS migration data, and BLS job growth. Higher scores (70+) indicate strong in-migration that drives housing and commercial demand. Source: US Census Bureau / Agent 1, updated every 6 hours."),
                f"{_mig_score:.0f}/100", "#d4a843", "Demand proxy"
            ),
        ]
        for _col, (_lbl, _val, _clr, _sub) in zip([_li1, _li2, _li3, _li4, _li5], _li_cards):
            _col.markdown(
                f'<div class="metric-card"><div class="label">{_lbl}</div>'
                f'<div class="value" style="color:{_clr};font-size:1.35rem;">{_val}</div>'
                f'<div class="sub">{_sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Rent growth chart + Suggested markets ──────────────────────
        _li_left, _li_right = st.columns([1, 1], gap="large")

        with _li_left:
            st.markdown(
                f'<div style="color:#d4a843;font-size:0.82rem;font-weight:600;'
                f'letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;">'
                f'Rent Growth by Property Type (National YoY)</div>',
                unsafe_allow_html=True,
            )
            _rg_chart_data = {
                k: v.get("yoy_pct") or v.get("yoy") or 0
                for k, v in _rg_nat.items()
                if isinstance(v, dict) and (v.get("yoy_pct") or v.get("yoy")) is not None
            }
            if _rg_chart_data:
                _rg_types  = list(_rg_chart_data.keys())
                _rg_vals   = [_rg_chart_data[t] for t in _rg_types]
                _rg_colors = ["#4caf50" if v > 2 else ("#ff9800" if v > 0 else "#ef5350") for v in _rg_vals]
                _highlight  = [1.0 if t == _pt_rg_key else 0.5 for t in _rg_types]
                _fig_rg = go.Figure(go.Bar(
                    x=_rg_types, y=_rg_vals,
                    marker=dict(color=_rg_colors, opacity=_highlight),
                    text=[f"{v:+.1f}%" for v in _rg_vals],
                    textposition="outside",
                    textfont=dict(color="#c8b890", size=11),
                    hovertemplate="<b>%{x}</b><br>YoY: %{y:+.1f}%<extra></extra>",
                ))
                _fig_rg.add_hline(y=0, line_color="#555", line_width=1)
                _fig_rg.update_layout(
                    plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
                    margin=dict(t=30, b=10, l=10, r=10), height=240,
                    xaxis=dict(tickfont=dict(color="#c8b890", size=10), tickangle=-15),
                    yaxis=dict(ticksuffix="%", gridcolor="#2a2a1a", tickfont=dict(color="#c8b890", size=10)),
                    showlegend=False,
                )
                st.plotly_chart(_fig_rg, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Highlighted bar = {_prop_type} (your target type). Source: Agent 15 / Zillow–CoStar benchmarks.")
            else:
                st.info("Rent growth data populating — refresh in ~30 seconds.")

        with _li_right:
            # Top markets from all_scored filtered to location, or just top overall
            st.markdown(
                f'<div style="color:#d4a843;font-size:0.82rem;font-weight:600;'
                f'letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;">'
                f'Top Scored Markets{f" — {_location}" if _location else ""}</div>',
                unsafe_allow_html=True,
            )
            # Filter all_scored by location if possible, else show top 6
            _loc_words = [w for w in _loc_raw.split() if len(w) > 3]
            _loc_filtered = [
                m for m in all_scored
                if any(w in m["market"].lower() for w in _loc_words)
            ] if _loc_words else []
            _show_markets = (_loc_filtered or all_scored)[:6]

            if _show_markets:
                _sm_rows = "".join([
                    f'<tr style="border-bottom:1px solid #1e1a08;">'
                    f'<td style="padding:7px 10px;color:#c8b890;font-size:12px;font-weight:{"700" if i==0 else "400"};">'
                    f'{"★ " if i==0 else f"{i+1}. "}{m["market"]}</td>'
                    f'<td style="padding:7px 10px;text-align:right;font-family:monospace;font-size:12px;'
                    f'color:{"#4caf50" if m["opportunity_score"]>=75 else "#ff9800" if m["opportunity_score"]>=55 else "#ef5350"};">'
                    f'{m["opportunity_score"]:.0f}</td>'
                    f'<td style="padding:7px 10px;text-align:right;font-family:monospace;font-size:11px;color:#d4a843;">'
                    f'{m.get("cap_rate",0):.2f}%</td>'
                    f'<td style="padding:7px 10px;text-align:right;font-family:monospace;font-size:11px;'
                    f'color:{"#4caf50" if m.get("rent_growth",0)>2 else "#ff9800"};">'
                    f'{m.get("rent_growth",0):+.1f}%</td>'
                    f'</tr>'
                    for i, m in enumerate(_show_markets)
                ])
                st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#171309;border-radius:8px;overflow:hidden;border:1px solid #2a2208;">
  <thead><tr style="border-bottom:1px solid #2a2208;">
    <th style="padding:8px 10px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Market</th>
    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Score</th>
    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Cap Rate</th>
    <th style="padding:8px 10px;text-align:right;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Rent Δ</th>
  </tr></thead>
  <tbody>{_sm_rows}</tbody>
</table>""", unsafe_allow_html=True)
                st.caption("★ = top recommended market for your prompt. Score 0–100 composite.")
            else:
                st.info("Market scoring data loading.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tax Advantages ─────────────────────────────────────────────────────
        section(" Tax Advantages for This Location")

        _tax_left, _tax_right = st.columns([1, 1], gap="large")

        with _tax_left:
            # State tax info
            _st_tax_info = _STATE_TAX.get(_matched_abbr, None)
            _st_display  = _matched_abbr or _location or "Target State"
            if _st_tax_info:
                _st_rate, _st_note = _st_tax_info
                _st_color = "#4caf50" if _st_rate == "0%" else ("#ff9800" if float(_st_rate.replace("%","")) < 5 else "#ef5350")
                st.markdown(f"""
<div style="background:#13110a;border:1px solid #2a2208;border-radius:10px;padding:18px 20px;margin-bottom:12px;">
  <div style="color:#d4a843;font-size:0.82rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:12px;">State Tax Environment — {_st_display}</div>
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <div style="font-size:2rem;font-weight:700;color:{_st_color};font-family:monospace;">{_st_rate}</div>
    <div style="color:#a09880;font-size:0.88rem;">State Income Tax Rate</div>
  </div>
  <div style="color:#c8b890;font-size:0.84rem;line-height:1.6;">{_st_note}</div>
</div>""", unsafe_allow_html=True)

            # Federal tax benefits
            _fed_tax_rows = [
                ("Cost Segregation",       "Accelerate depreciation on personal property (5-yr) and land improvements (15-yr). Year-1 deduction can offset 20–30% of building value."),
                ("Bonus Depreciation",     "60% bonus depreciation in 2025 on qualifying assets — phases to 40% in 2026. Front-loads the tax shield into early hold years."),
                ("1031 Exchange",          "Defer 100% of capital gains tax by rolling proceeds into a like-kind property within 45-day identification / 180-day close windows."),
                ("Depreciation Recapture Strategy", "Structure the exit via an installment sale or OZ re-investment to manage 25% recapture tax on straight-line depreciation."),
            ]
            for _fed_name, _fed_desc in _fed_tax_rows:
                st.markdown(
                    f'<div style="background:#16140a;border-left:3px solid #705020;border-radius:4px;'
                    f'padding:10px 14px;margin-bottom:8px;">'
                    f'<div style="color:#d4a843;font-size:0.82rem;font-weight:600;margin-bottom:4px;">{_fed_name}</div>'
                    f'<div style="color:#a09880;font-size:0.82rem;line-height:1.5;">{_fed_desc}</div></div>',
                    unsafe_allow_html=True,
                )

        with _tax_right:
            # OZ markets near the location
            _oz_mkts = _lmi_oz.get("oz_markets", {})
            _oz_ranked_all = _lmi_oz.get("top_markets_by_score", [])

            # Filter OZ markets by location
            _loc_oz = [
                (name, info) for name, info in _oz_mkts.items()
                if any(w in name.lower() for w in _loc_words) or
                   (_matched_abbr and _matched_abbr.lower() in name.lower())
            ] if (_loc_words or _matched_abbr) else []
            _show_oz = sorted(_loc_oz, key=lambda x: x[1].get("opportunity_score", 0), reverse=True)[:4]

            # Fall back to top OZ markets if no location match
            if not _show_oz and _oz_ranked_all:
                _show_oz = [(r[0], {"opportunity_score": r[1], "key_zones": [], "property_focus": [], "highlights": []})
                            for r in _oz_ranked_all[:4]]

            if _show_oz:
                st.markdown(
                    f'<div style="color:#d4a843;font-size:0.82rem;font-weight:600;'
                    f'letter-spacing:.05em;text-transform:uppercase;margin-bottom:10px;">'
                    f'Opportunity Zone Markets{f" — {_location}" if _location else ""}</div>',
                    unsafe_allow_html=True,
                )
                for _oz_name, _oz_info in _show_oz:
                    _oz_sc   = _oz_info.get("opportunity_score", 0)
                    _oz_sc_c = "#4caf50" if _oz_sc >= 80 else ("#d4a843" if _oz_sc >= 70 else "#ef5350")
                    _oz_zones = " · ".join((_oz_info.get("key_zones") or [])[:2])
                    _oz_focus = ", ".join((_oz_info.get("property_focus") or [])[:2])
                    _oz_hi    = (_oz_info.get("highlights") or [""])[0]

                    # ── Market header card ─────────────────────────────────────
                    st.markdown(f"""
<div style="background:#0a1e0a;border:1px solid #2a4a2a;border-left:3px solid #4caf50;
            border-radius:6px;padding:12px 16px;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="color:#80c858;font-weight:600;font-size:0.88rem;">{_oz_name}</div>
    <div style="color:{_oz_sc_c};font-weight:700;font-family:monospace;font-size:1rem;">{_oz_sc}</div>
  </div>
  {"<div style='color:#5a9050;font-size:0.78rem;margin-top:4px;'>Zones: " + _oz_zones + "</div>" if _oz_zones else ""}
  {"<div style='color:#5a9050;font-size:0.78rem;'>Focus: " + _oz_focus + "</div>" if _oz_focus else ""}
  {"<div style='color:#a09880;font-size:0.78rem;margin-top:4px;'>" + _oz_hi + "</div>" if _oz_hi else ""}
</div>""", unsafe_allow_html=True)

                    # ── Expandable listings panel ──────────────────────────────
                    _oz_state = _oz_name.split(", ")[-1] if ", " in _oz_name else "TX"
                    _oz_city  = _oz_name.split(", ")[0] if ", " in _oz_name else _oz_name
                    _oz_city_slug = _oz_name.lower().replace(", ", "-").replace(" ", "-")

                    with st.expander(f"View properties for sale in {_oz_name}"):
                        from src.cre_listings import get_cheapest_buildings
                        _oz_listings = get_cheapest_buildings(_oz_state, n=8)

                        _loopnet_base = f"https://www.loopnet.com/search/commercial-real-estate/{_oz_city_slug}/for-sale/?sk=opportunityzone"
                        st.caption(f"Showing 8 representative listings · [See all on LoopNet ↗]({_loopnet_base})")

                        _pt_loopnet = {
                            "Industrial": "industrial-distribution-warehousing",
                            "Retail":     "retail",
                            "Office":     "office",
                            "Multifamily":"apartments",
                            "Mixed-Use":  "specialty",
                        }

                        for _lst in _oz_listings:
                            _lst_pt   = _lst.get("property_type", "Commercial")
                            _lst_slug = _pt_loopnet.get(_lst_pt, "commercial-real-estate")
                            _lst_url  = f"https://www.loopnet.com/search/{_lst_slug}/{_oz_city_slug}/for-sale/?sk=opportunityzone"
                            _lst_dom_c = "#ef5350" if _lst.get("days_on_market",0) > 90 else ("#d4a843" if _lst.get("days_on_market",0) > 30 else "#4caf50")

                            st.markdown(f"""
<div style="background:#111a0a;border:1px solid #243424;border-radius:6px;
            padding:14px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
    <div>
      <div style="color:#c8b890;font-weight:600;font-size:0.9rem;">
        {_lst.get("address","")}, {_oz_city}, {_oz_state}
      </div>
      <div style="color:#5a7a5a;font-size:0.75rem;margin-top:2px;">{_lst_pt}</div>
    </div>
    <div style="text-align:right;">
      <div style="color:#d4a843;font-weight:700;font-size:1rem;">${_lst.get("price",0):,.0f}</div>
      <div style="color:#5a7a5a;font-size:0.72rem;">${_lst.get("price_per_sqft",0):.0f}/sqft</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;">
    <div style="background:#0d150d;border-radius:4px;padding:6px 8px;text-align:center;">
      <div style="color:#5a7a5a;font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;">Size</div>
      <div style="color:#c8b890;font-size:0.82rem;font-weight:600;">{_lst.get("sqft",0):,} sqft</div>
    </div>
    <div style="background:#0d150d;border-radius:4px;padding:6px 8px;text-align:center;">
      <div style="color:#5a7a5a;font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;">Cap Rate</div>
      <div style="color:#4caf50;font-size:0.82rem;font-weight:600;">{_lst.get("cap_rate",0):.2f}%</div>
    </div>
    <div style="background:#0d150d;border-radius:4px;padding:6px 8px;text-align:center;">
      <div style="color:#5a7a5a;font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;">NOI / yr</div>
      <div style="color:#c8b890;font-size:0.82rem;font-weight:600;">${_lst.get("noi_annual",0):,.0f}</div>
    </div>
    <div style="background:#0d150d;border-radius:4px;padding:6px 8px;text-align:center;">
      <div style="color:#5a7a5a;font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;">Days Listed</div>
      <div style="color:{_lst_dom_c};font-size:0.82rem;font-weight:600;">{_lst.get("days_on_market",0)}d</div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="color:#4a6a4a;font-size:0.73rem;">
      Built {_lst.get("year_built","")} &nbsp;·&nbsp; {_lst.get("highlights","")}
    </div>
    <a href="{_lst_url}" target="_blank"
       style="background:#0d2a0d;border:1px solid #2a5a2a;color:#4caf50;
              font-size:0.73rem;font-weight:600;padding:4px 10px;
              border-radius:4px;text-decoration:none;white-space:nowrap;">
      Search similar on LoopNet ↗
    </a>
  </div>
</div>""", unsafe_allow_html=True)

                st.markdown(
                    '<div style="background:#16140a;border:1px solid #3a3020;border-radius:6px;'
                    'padding:10px 14px;margin-top:4px;">'
                    '<div style="color:#d4a843;font-size:0.78rem;font-weight:600;margin-bottom:4px;">OZ 10-Year Benefit</div>'
                    '<div style="color:#a09880;font-size:0.78rem;line-height:1.5;">'
                    'Hold a QOF investment for 10+ years and pay <strong style="color:#4caf50;">zero capital gains tax</strong> '
                    'on all appreciation inside the fund. A $5M gain on a $10M development = $0 federal tax at exit. '
                    'Must invest unrealized gains within 180 days.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Opportunity Zone data loading — check the Opportunity Zones tab for full detail.")

        st.markdown(
            '<hr style="border:none;border-top:1px solid #2a2208;margin:28px 0 24px;">',
            unsafe_allow_html=True,
        )

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

        # ── Market Score Breakdown ────────────────────────────────────────────
        _ms_cache_adv = read_cache("market_score")
        _ms_data_adv  = (_ms_cache_adv.get("data") or {})
        _ms_rankings  = _ms_data_adv.get("rankings", [])
        _ms_entry     = next((r for r in _ms_rankings if r.get("market") == primary.get("market")), None)
        if _ms_entry and _ms_entry.get("breakdown"):
            _bd2 = _ms_entry["breakdown"]
            _strengths = _bd2.get("strengths", [])
            _weaknesses = _bd2.get("weaknesses", [])
            _drag  = _bd2.get("drag_factor", "")
            _drag_note = _bd2.get("drag_note", "")
            _lift  = _bd2.get("lift_factor", "")
            _lift_note = _bd2.get("lift_note", "")
            _factor_notes = _bd2.get("notes", {})

            _str_pills  = "".join([f'<span style="background:rgba(76,175,80,.2);color:#80c858;border-radius:12px;padding:3px 10px;font-size:0.75rem;margin:2px;">{s}</span>' for s in _strengths])
            _weak_pills = "".join([f'<span style="background:rgba(244,67,54,.15);color:#ef9a9a;border-radius:12px;padding:3px 10px;font-size:0.75rem;margin:2px;">{w}</span>' for w in _weaknesses])

            _factor_rows = "".join([
                f'<tr><td style="color:#a09880;padding:4px 0;font-size:0.8rem;">{_factor_notes.get(k, "").split(" — ")[0] if " — " not in str(_factor_notes.get(k,"")) else k.title()}</td>'
                f'<td style="text-align:right;"><div style="display:inline-block;background:{"#4caf50" if v>=65 else "#f44336" if v<=40 else "#ffb74d"};'
                f'height:6px;width:{max(4,int(v*0.6))}px;border-radius:3px;"></div>'
                f'<span style="color:#a09880;font-size:0.75rem;margin-left:6px;">{v:.0f}</span></td></tr>'
                for k, v in _ms_entry.get("factors", {}).items()
            ])

            st.markdown(f"""
<div style="background:#13110a;border:1px solid #2a2208;border-radius:10px;padding:18px 22px;margin:12px 0;">
  <div style="color:#d4a843;font-weight:600;font-size:0.9rem;margin-bottom:12px;">
    Market Score Breakdown — {_ms_entry['market']}
    <span style="color:#a09880;font-weight:400;font-size:0.82rem;margin-left:10px;">Composite: {_ms_entry['composite']}/100 · Grade: {_ms_entry['grade']}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>
      <table style="width:100%;border-collapse:collapse;">{_factor_rows}</table>
      {"<div style='margin-top:10px;'><span style='color:#6a5228;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;'>Strengths</span><div style='margin-top:4px;'>" + _str_pills + "</div></div>" if _strengths else ""}
      {"<div style='margin-top:8px;'><span style='color:#6a5228;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;'>Weaknesses</span><div style='margin-top:4px;'>" + _weak_pills + "</div></div>" if _weaknesses else ""}
    </div>
    <div>
      {"<div style='background:#0d2a12;border-left:3px solid #4caf50;border-radius:4px;padding:10px 14px;margin-bottom:8px;'><div style='color:#4caf50;font-size:0.76rem;font-weight:600;margin-bottom:4px;'>Biggest Lift: " + _lift + "</div><div style='color:#a09880;font-size:0.78rem;'>" + _lift_note + "</div></div>" if _lift else ""}
      {"<div style='background:#2a0d0d;border-left:3px solid #f44336;border-radius:4px;padding:10px 14px;'><div style='color:#f44336;font-size:0.76rem;font-weight:600;margin-bottom:4px;'>Biggest Drag: " + _drag + "</div><div style='color:#a09880;font-size:0.78rem;'>" + _drag_note + "</div></div>" if _drag else ""}
      {"<div style='margin-top:8px;background:#1a1208;border-left:3px solid #ff9800;border-radius:4px;padding:8px 14px;'><div style='color:#ff9800;font-size:0.74rem;font-weight:600;'>Climate Adjustment</div><div style='color:#a09880;font-size:0.76rem;'>−" + str(_ms_entry.get('climate_penalty', 0)) + " pts applied to composite score</div></div>" if _ms_entry.get("climate_penalty", 0) > 0 else ""}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Financials breakdown ──────────────────────────────────────────────
        section(" Financial Estimates")

        _fc1, _fc2, _fc3, _fc4 = st.columns(4)
        for _fc, (_lbl, _tip, _val, _clr) in zip(
            [_fc1, _fc2, _fc3, _fc4],
            [
                ("Land Cost",          "Estimated cost to acquire the land parcel. Based on $/sqft land comps for the selected market and property type.",
                 f"${financials['land_cost']/1e6:.2f}M",         "#e8dfc4"),
                ("Construction",       "Hard construction cost — materials, labor, and contractor fees. Adjusted by the platform's energy cost signal (LOW/MODERATE/HIGH).",
                 f"${financials['construction_cost']/1e6:.2f}M", "#e8dfc4"),
                ("Soft Costs",         "Architecture, engineering, permits, legal, financing fees, and contingency (typically 15–20% of hard costs).",
                 f"${financials['soft_costs']/1e6:.2f}M",         "#e8dfc4"),
                ("Total Project Cost", "All-in development cost: Land + Construction + Soft Costs. This is the capital you need to deploy before the project generates income.",
                 f"${financials['total_cost']/1e6:.2f}M",         "#d4a843"),
            ]
        ):
            _fc.markdown(
                f'<div class="metric-card"><div class="label">{_tt(_lbl, _tip)}</div>'
                f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
        _fc5, _fc6, _fc7, _fc8 = st.columns(4)
        _irr_c = "#4caf50" if financials["irr_est"] > 10 else "#ff9800"
        _pft_c = "#4caf50" if financials["total_profit"] > 0 else "#f44336"
        for _fc, (_lbl, _tip, _val, _clr) in zip(
            [_fc5, _fc6, _fc7, _fc8],
            [
                ("Annual NOI",     "Net Operating Income per year: Effective Gross Income minus operating expenses (management, insurance, taxes, maintenance). Does NOT include debt service.",
                 f"${financials['annual_noi']/1e3:.0f}K",     "#e8dfc4"),
                ("Cumulative NOI", "Total NOI earned over the full hold period (typically 10 years). Measures how much income the property generates before financing costs.",
                 f"${financials['total_noi']/1e6:.2f}M",       "#e8dfc4"),
                ("Estimated IRR",  "Internal Rate of Return — annualized return on your total investment including both cash flows and projected exit proceeds. A target of 10%+ is generally considered strong for CRE development.",
                 f"{financials['irr_est']:.1f}%",              _irr_c),
                ("Total Profit",   "Total cash profit over the hold period: Cumulative NOI + Exit Value − Total Project Cost. A positive number means the deal made money; negative means a loss.",
                 f"${financials['total_profit']/1e6:.2f}M",    _pft_c),
            ]
        ):
            _fc.markdown(
                f'<div class="metric-card"><div class="label">{_tt(_lbl, _tip)}</div>'
                f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                unsafe_allow_html=True,
            )

        _esig = financials.get("energy_signal", "MODERATE")
        _emult = {"LOW": "0.88×", "MODERATE": "1.0×", "HIGH": "1.20×"}.get(_esig, "1.0×")
        st.caption(
            f"Construction cost signal: **{_esig}** (platform energy agent). "
            f"Cost multiplier: {_emult}. Buildout estimate: {financials['buildout_months']} months."
        )

        # ── Income vs Expense Line Chart ──────────────────────────────────────
        st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
        st.markdown(
            _tt("Income vs Expense Projection",
                "Year-by-year operating forecast (Years 1–N). Year 0 is the construction/equity phase shown above.<br><br>"
                "<b>Gold — Revenue (EGI)</b>: Gross rent collected minus vacancy loss. This is your top-line income.<br>"
                "<b>Red — Expenses</b>: Operating costs (management, taxes, insurance, maintenance) PLUS annual loan payments.<br>"
                "<b>Green dotted — NOI</b>: Net Operating Income = Revenue minus operating costs only (before debt).<br>"
                "<b>Blue dashed — Cash Flow</b>: What you actually pocket each year = NOI minus loan payment.<br><br>"
                "When the <b>gold line is above red</b>, the property generates more income than it costs — it's profitable. "
                "The green shaded area shows the profit margin; red shading shows a loss period."),
            unsafe_allow_html=True,
        )
        _hold_yrs   = int(financials.get("hold_years", params.get("timeline_years", 5)) or 5)
        _base_noi   = financials.get("annual_noi", 0)
        _total_cost = financials.get("total_cost", 0)
        _equity     = _adv_result.get("financing", {}).get("equity_required", _total_cost)
        _base_egi   = _base_noi / 0.65 if _base_noi else 0
        _base_opex  = _base_egi - _base_noi
        _ann_ds     = _adv_result.get("financing", {}).get("annual_debt_service", 0)

        # Build data for Years 1–N only (Year 0 shown as callout, not on chart)
        _chart_yrs = []
        _chart_egi = []
        _chart_exp = []
        _chart_noi = []
        _chart_cf  = []
        _chart_opex_only = []
        _chart_ds_only   = []

        for _yr in range(1, _hold_yrs + 1):
            _g    = 1.03 ** (_yr - 1)
            _egi  = round(_base_egi  * _g)
            _opex = round(_base_opex * (1.025 ** (_yr - 1)))
            _noi  = _egi - _opex
            _cf   = _noi - _ann_ds
            _chart_yrs.append(_yr)
            _chart_egi.append(_egi)
            _chart_exp.append(_opex + _ann_ds)
            _chart_noi.append(_noi)
            _chart_cf.append(_cf)
            _chart_opex_only.append(_opex)
            _chart_ds_only.append(_ann_ds)

        # Year 0 equity callout above chart
        _yr1_cf_color = "#4caf50" if (_chart_cf[0] if _chart_cf else 0) >= 0 else "#ef5350"
        _yr1_cf_label = "Cash Flow Positive" if (_chart_cf[0] if _chart_cf else 0) >= 0 else "Cash Flow Negative"
        st.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:10px;flex-wrap:wrap;">
  <div style="background:#1a1208;border:1px solid #3a2510;border-left:3px solid #ef5350;
              border-radius:6px;padding:8px 14px;font-size:0.8rem;">
    <span style="color:#6a4020;">Year 0 — Equity Deployed:</span>
    <span style="color:#ef5350;font-weight:700;margin-left:6px;">${_equity/1e6:.2f}M</span>
    <span style="color:#5a4020;font-size:0.72rem;margin-left:6px;">(construction + close)</span>
  </div>
  <div style="background:#0d1a0d;border:1px solid #1a3010;border-left:3px solid {_yr1_cf_color};
              border-radius:6px;padding:8px 14px;font-size:0.8rem;">
    <span style="color:#4a6030;">Year 1 Cash Flow:</span>
    <span style="color:{_yr1_cf_color};font-weight:700;margin-left:6px;">${(_chart_cf[0] if _chart_cf else 0)/1e3:.0f}K/yr</span>
    <span style="color:#3a5020;font-size:0.72rem;margin-left:6px;">({_yr1_cf_label})</span>
  </div>
</div>
""", unsafe_allow_html=True)

        _fin_fig = go.Figure()

        # Profit fill: green where revenue > expenses, red otherwise
        _fill_egi = _chart_egi[:]
        _fill_exp = _chart_exp[:]
        _fin_fig.add_trace(go.Scatter(
            x=_chart_yrs + _chart_yrs[::-1],
            y=_fill_egi + _fill_exp[::-1],
            fill="toself",
            fillcolor="rgba(76,175,80,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
            name="_profit_fill",
        ))

        # Revenue line
        _fin_fig.add_trace(go.Scatter(
            x=_chart_yrs, y=_chart_egi,
            name="Revenue (EGI = Gross Rent − Vacancy)",
            line=dict(color="#d4a843", width=2.5),
            mode="lines+markers", marker=dict(size=5),
            customdata=list(zip(
                [round(v / 0.92) for v in _chart_egi],
                [round(v * 0.08) for v in _chart_egi],
            )),
            hovertemplate=(
                "<b>Year %{x} — Revenue</b><br>"
                "Gross Rent: $%{customdata[0]:,.0f}<br>"
                "− Vacancy Loss: $%{customdata[1]:,.0f}<br>"
                "<b>= EGI: $%{y:,.0f}</b><extra></extra>"
            ),
        ))
        # Expense line
        _fin_fig.add_trace(go.Scatter(
            x=_chart_yrs, y=_chart_exp,
            name="Expenses (OpEx + Debt Service)",
            line=dict(color="#ef5350", width=2.5),
            mode="lines+markers", marker=dict(size=5),
            customdata=list(zip(_chart_opex_only, _chart_ds_only)),
            hovertemplate=(
                "<b>Year %{x} — Expenses</b><br>"
                "Operating Costs: $%{customdata[0]:,.0f}<br>"
                "+ Debt Service: $%{customdata[1]:,.0f}<br>"
                "<b>= Total: $%{y:,.0f}</b><extra></extra>"
            ),
        ))
        # NOI line
        _fin_fig.add_trace(go.Scatter(
            x=_chart_yrs, y=_chart_noi,
            name="NOI (Revenue − OpEx)",
            line=dict(color="#4caf50", width=2, dash="dot"),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="<b>Year %{x} — NOI</b><br>$%{y:,.0f}<extra></extra>",
        ))
        # Cash flow line
        _fin_fig.add_trace(go.Scatter(
            x=_chart_yrs, y=_chart_cf,
            name="Cash Flow After Debt",
            line=dict(color="#64b5f6", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate="<b>Year %{x} — Cash Flow</b><br>$%{y:,.0f}<extra></extra>",
        ))

        # Zero reference line (break-even)
        _fin_fig.add_hline(
            y=0, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
            annotation_text="Break-even", annotation_position="right",
            annotation_font=dict(color="rgba(255,255,255,0.35)", size=9),
        )

        _fin_fig.update_layout(
            plot_bgcolor="#0f0f0c", paper_bgcolor="#16160f",
            font=dict(color="#a09880", size=11),
            xaxis=dict(
                title=_tt("Hold Year", "Each year of property ownership after completion. Year 1 = first full operating year."),
                gridcolor="#2a2a20", tickmode="linear", dtick=1,
                title_font=dict(color="#a09880"),
            ),
            yaxis=dict(
                title=_tt("Annual $ Amount", "Dollar amounts per year. Positive = money coming in. Negative = money going out. The gap between gold (revenue) and red (expenses) is your profit."),
                gridcolor="#2a2a20", tickformat="$~s",
                title_font=dict(color="#a09880"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h", y=-0.28),
            margin=dict(l=10, r=10, t=20, b=80),
            height=380,
        )
        st.plotly_chart(_fin_fig, use_container_width=True, key="fin_proj_chart", config={"displayModeBar": False})
        st.caption(
            "Chart shows Years 1–" + str(_hold_yrs) + " (operating phase). "
            "Revenue (gold) above Expenses (red) = profitable year — green shading shows the profit margin. "
            "NOI = revenue minus operating costs only (before loan payments). "
            "Cash Flow (blue) = what you pocket after paying the loan each year."
        )


        # ── Financing Structure ───────────────────────────────────────────────
        financing = _adv_result.get("financing", {})
        if financing:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Financing Structure")
            _fn1, _fn2, _fn3, _fn4, _fn5 = st.columns(5)
            _dscr_c = "#4caf50" if financing.get("dscr", 0) >= 1.25 else ("#ff9800" if financing.get("dscr", 0) >= 1.0 else "#f44336")
            _coc_c  = "#4caf50" if financing.get("cash_on_cash_pct", 0) >= 6 else "#ff9800"
            _irrl_c = "#4caf50" if financing.get("leveraged_irr_pct", 0) >= 15 else "#ff9800"
            for _fc, (_lbl, _tip, _val, _clr) in zip(
                [_fn1, _fn2, _fn3, _fn4, _fn5],
                [
                    ("LTV",               "Loan-to-Value ratio: how much of the project cost is financed by debt. 65% LTV = lender covers 65%, you put in 35% equity. Higher LTV = more leverage but more risk.",
                     f"{financing['ltv_pct']:.0f}%",                         "#e8dfc4"),
                    ("Loan Amount",        "Total debt drawn from the construction/permanent lender. This is the amount you borrow, not the total project cost.",
                     f"${financing['loan_amount']/1e6:.2f}M",                "#e8dfc4"),
                    ("Equity Required",    "Your out-of-pocket cash investment: Total Project Cost minus the loan. This is the minimum you need to have available at closing.",
                     f"${financing['equity_required']/1e6:.2f}M",            "#d4a843"),
                    ("Annual Debt Service","Total loan payments per year (principal + interest). This is the fixed cost that must be covered by NOI before you see any cash flow.",
                     f"${financing['annual_debt_service']/1e3:.0f}K",        "#e8dfc4"),
                    ("DSCR",               "Debt Service Coverage Ratio: NOI ÷ Annual Debt Service. ≥1.25x = lenders are comfortable (green). 1.0–1.25x = marginal (yellow). <1.0x = property can't cover its debt (red).",
                     f"{financing['dscr']:.2f}x",                            _dscr_c),
                ]
            ):
                _fc.markdown(
                    f'<div class="metric-card"><div class="label">{_tt(_lbl, _tip)}</div>'
                    f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<br style='margin:4px 0'>", unsafe_allow_html=True)
            _fn6, _fn7, _fn8, _fn9 = st.columns(4)
            for _fc, (_lbl, _tip, _val, _clr) in zip(
                [_fn6, _fn7, _fn8, _fn9],
                [
                    ("Loan Rate",          "Interest rate on the construction/permanent loan and amortization period. A 30yr amortization spreads payments over 30 years, reducing annual debt service.",
                     f"{financing['loan_rate_pct']:.2f}% / {financing['amort_years']}yr", "#e8dfc4"),
                    ("Cash Flow After DS", "Annual cash profit after paying all operating expenses AND loan payments. This is the actual money available to distribute to investors each year.",
                     f"${financing['cash_flow_after_ds']/1e3:.0f}K/yr",                  _coc_c),
                    ("Cash-on-Cash",       "Cash-on-Cash return: Annual Cash Flow ÷ Equity Invested × 100. Measures the annual cash yield on your equity. Target: 6%+ is generally considered good.",
                     f"{financing['cash_on_cash_pct']:.1f}%",                            _coc_c),
                    ("Leveraged IRR",      "IRR including the effect of debt — since you only put in equity (not the full cost), returns are amplified by leverage. Target: 15%+ for development deals.",
                     f"{financing['leveraged_irr_pct']:.1f}%",                           _irrl_c),
                ]
            ):
                _fc.markdown(
                    f'<div class="metric-card"><div class="label">{_tt(_lbl, _tip)}</div>'
                    f'<div class="value" style="color:{_clr};">{_val}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── 10-Year P&L Pro Forma ─────────────────────────────────────────────
        proforma = _adv_result.get("proforma", [])
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
        tax = _adv_result.get("tax_benefits", {})
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

        # ── Forecasting (Agent 22 · FRED projections) ─────────────────────────
        _fc_cache = read_cache("forecast") or {}
        _fc_data  = _fc_cache.get("data") or {}
        _fc_proj  = _fc_data.get("projections", {}) or {}
        _fc_hist  = _fc_data.get("historical",  {}) or {}

        if _fc_proj:
            section(" Forecasting — Macro Projections")
            st.markdown(
                '<div style="color:#a09880;font-size:0.86rem;margin-top:-6px;margin-bottom:14px;">'
                'Forward-looking macro drivers: Atlanta Fed GDPNow, FOMC Summary of Economic Projections '
                '(real GDP & Fed funds), and market-implied 10-year breakeven inflation. '
                'Feed into exit cap rate, debt service, and rent growth assumptions.'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Projections table ─────────────────────────────────────────────
            _fc_header = (
                '<tr style="background:#1a1408;">'
                '  <th style="text-align:left;padding:10px 14px;color:#d4a843;'
                '      font-size:0.8rem;letter-spacing:0.06em;border-bottom:1px solid #2a2208;">'
                '      INDICATOR</th>'
                '  <th style="text-align:right;padding:10px 14px;color:#d4a843;'
                '      font-size:0.8rem;border-bottom:1px solid #2a2208;">CURRENT</th>'
                '  <th style="text-align:right;padding:10px 14px;color:#d4a843;'
                '      font-size:0.8rem;border-bottom:1px solid #2a2208;">Q2 2026</th>'
                '  <th style="text-align:right;padding:10px 14px;color:#d4a843;'
                '      font-size:0.8rem;border-bottom:1px solid #2a2208;">Q3 2026</th>'
                '  <th style="text-align:right;padding:10px 14px;color:#d4a843;'
                '      font-size:0.8rem;border-bottom:1px solid #2a2208;">Q4 2026</th>'
                '</tr>'
            )
            _fc_rows = []
            for _name, _p in _fc_proj.items():
                _cur = _p.get("current")
                _q2  = _p.get("q2_2026")
                _q3  = _p.get("q3_2026")
                _q4  = _p.get("q4_2026")
                _unit = _p.get("unit", "%")
                _delta = (_q4 - _cur) if (_q4 is not None and _cur is not None) else 0
                _dir_col = ("#4caf50" if _delta > 0.05 else
                            "#ef5350" if _delta < -0.05 else "#c8b890")
                def _fmt(v):
                    return f"{v:.2f}{_unit}" if v is not None else "—"
                _fc_rows.append(
                    f'<tr>'
                    f'  <td style="padding:10px 14px;color:#e8dfc4;font-size:0.9rem;'
                    f'      border-bottom:1px solid #2a2208;">{_name}'
                    f'      <div style="color:#6a5228;font-size:0.72rem;">FRED {_p.get("series_id","")}</div></td>'
                    f'  <td style="padding:10px 14px;text-align:right;color:#c8b890;'
                    f'      font-size:0.92rem;border-bottom:1px solid #2a2208;">{_fmt(_cur)}</td>'
                    f'  <td style="padding:10px 14px;text-align:right;color:{_dir_col};'
                    f'      font-size:0.92rem;border-bottom:1px solid #2a2208;">{_fmt(_q2)}</td>'
                    f'  <td style="padding:10px 14px;text-align:right;color:{_dir_col};'
                    f'      font-size:0.92rem;border-bottom:1px solid #2a2208;">{_fmt(_q3)}</td>'
                    f'  <td style="padding:10px 14px;text-align:right;color:{_dir_col};'
                    f'      font-size:0.92rem;border-bottom:1px solid #2a2208;font-weight:700;">{_fmt(_q4)}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="background:#0f0d06;border:1px solid #2a2208;border-radius:8px;'
                f'overflow:hidden;margin-bottom:14px;">'
                f'  <table style="width:100%;border-collapse:collapse;">'
                f'    <thead>{_fc_header}</thead>'
                f'    <tbody>{"".join(_fc_rows)}</tbody>'
                f'  </table>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── History + forecast line chart ─────────────────────────────────
            try:
                import plotly.graph_objects as _go
                _fig_fc = _go.Figure()
                _line_colors = {
                    "GDP Nowcast (Atlanta Fed)":  "#d4a843",
                    "GDP Projection (FOMC)":      "#c8a040",
                    "10Y Breakeven Inflation":    "#80a848",
                    "Fed Funds Projection (FOMC)":"#9868b8",
                }
                for _name, _h in _fc_hist.items():
                    _pts = _h.get("points", []) or []
                    if not _pts:
                        continue
                    _hx = [p["date"] for p in _pts]
                    _hy = [p["value"] for p in _pts]
                    _clr = _line_colors.get(_name, "#d4a843")

                    # Historical solid line
                    _fig_fc.add_trace(_go.Scatter(
                        x=_hx, y=_hy, mode="lines", name=_name,
                        line=dict(color=_clr, width=2),
                        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
                    ))
                    # Forecast dotted extension (last historical → Q4 2026)
                    _p = _fc_proj.get(_name, {})
                    _fx = [_hx[-1], "2026-06-30", "2026-09-30", "2026-12-31"]
                    _fy = [_hy[-1], _p.get("q2_2026"), _p.get("q3_2026"), _p.get("q4_2026")]
                    _fig_fc.add_trace(_go.Scatter(
                        x=_fx, y=_fy, mode="lines", name=f"{_name} (forecast)",
                        line=dict(color=_clr, width=2, dash="dot"),
                        showlegend=False,
                        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
                    ))
                    # Confidence interval shaded band (±15% around point estimate)
                    _fy_high = [_hy[-1],
                                _p.get("q2_2026_high"), _p.get("q3_2026_high"), _p.get("q4_2026_high")]
                    _fy_low  = [_hy[-1],
                                _p.get("q2_2026_low"),  _p.get("q3_2026_low"),  _p.get("q4_2026_low")]
                    if all(v is not None for v in _fy_high + _fy_low):
                        import re as _re
                        _rgb = _re.findall(r"[\da-fA-F]{2}", _clr.lstrip("#"))
                        _rgba_fill = (f"rgba({int(_rgb[0],16)},{int(_rgb[1],16)},{int(_rgb[2],16)},0.12)"
                                      if len(_rgb) == 3 else "rgba(212,168,67,0.12)")
                        _fig_fc.add_trace(_go.Scatter(
                            x=_fx + _fx[::-1],
                            y=_fy_high + _fy_low[::-1],
                            fill="toself",
                            fillcolor=_rgba_fill,
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False,
                            hoverinfo="skip",
                            name=f"{_name} CI",
                        ))

                _fig_fc.update_layout(
                    plot_bgcolor="#0f0f0c",
                    paper_bgcolor="#16160f",
                    margin=dict(t=20, b=20, l=30, r=20),
                    height=360,
                    xaxis=dict(tickfont=dict(color="#e8dfc4", size=10),
                               gridcolor="#2a2a1a", title=""),
                    yaxis=dict(tickfont=dict(color="#e8dfc4", size=10),
                               gridcolor="#2a2a1a", title="Rate (%)",
                               titlefont=dict(color="#a09880")),
                    font=dict(family="Source Sans Pro", color="#e8dfc4"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="left", x=0,
                                font=dict(color="#c8b890", size=10),
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(_fig_fc, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption("Solid lines = historical (FRED). Dotted = projected Q2–Q4 2026. Shaded band = ±15% confidence interval.")
            except Exception as _fc_err:
                st.caption(f"Chart unavailable: {_fc_err}")

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

            st.markdown(_render_generic_table(
                _cmp_df,
                title="Primary vs Runner-Up Markets",
                count_label=f"{len(_cmp_df)} markets",
                hints={
                    "Rank":             {"type": "badge", "flex": 0.9, "badge_map": {
                        "Primary": "background:#2a1a04;color:#d4a843",
                    }},
                    "Market":           {"type": "name",      "flex": 1.8},
                    "Opp. Score":       {"type": "score_bar", "flex": 1.2},
                    "Cap Rate":         {"type": "text",      "flex": 0.8},
                    "Rent Growth":      {"type": "pct_bar",   "flex": 1.2},
                    "Climate Risk":     {"type": "text",      "flex": 1.2},
                    "Mkt Fundamentals": {"type": "score_bar", "flex": 1},
                    "Migration Score":  {"type": "score_bar", "flex": 1},
                },
            ), unsafe_allow_html=True)

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
                st.markdown(_render_generic_table(
                    _all_df,
                    title="All Candidate Markets — Scored & Ranked",
                    count_label=f"{len(_all_df)} markets",
                    scrollable=True, max_height=400,
                    hints={
                        "Rank":          {"type": "text",      "flex": 0.5},
                        "Market":        {"type": "name",      "flex": 1.8},
                        "Score":         {"type": "score_bar", "flex": 1.2},
                        "Cap Rate":      {"type": "text",      "flex": 0.8},
                        "Rent Growth":   {"type": "pct_bar",   "flex": 1.2},
                        "Climate Score": {"type": "score_bar", "flex": 1},
                        "Migration":     {"type": "text",      "flex": 0.8},
                    },
                ), unsafe_allow_html=True)

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
                st.markdown(_render_generic_table(
                    pd.DataFrame(_wt_rows),
                    title="Factor Weights & Scores",
                    count_label=f"{len(_wt_rows)} factors",
                    hints={
                        "Factor":       {"type": "name",    "flex": 1.2},
                        "Weight":       {"type": "pct_bar", "flex": 0.8},
                        "Raw Score":    {"type": "text",    "flex": 0.8},
                        "Contribution": {"type": "text",    "flex": 0.8},
                        "Rationale":    {"type": "text",    "flex": 2.5},
                    },
                ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

      except Exception as _render_err:
        import traceback as _tb2
        st.error(f"**Render error** — this usually means a stale result is cached. Click **Generate Recommendation** to refresh.")
        with st.expander("Technical details"):
            st.code(_tb2.format_exc(), language="python")
        if st.button("Clear cached result", key="adv_clear_cache"):
            st.session_state.adv_result = None
            st.rerun()

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
            "rentcast":        ("Agent 21", "RentCast Property DB",      "Every 24h"),
            "forecast":        ("Agent 22", "Economic Forecast (FRED)",  "Every 6h"),
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

        with st.expander(" Agent Detail", expanded=False):
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
            ("forecast",            "Every 6h"),
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

        # ── Manager Agent Report ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" System Health Supervisor — Last Report")

        _mgr_col, _mgr_btn_col = st.columns([5, 1])
        with _mgr_btn_col:
            if st.button("Run Now", key="run_manager_now"):
                from src.cre_agents import force_run as _force_run
                _force_run("manager")
                st.toast("Manager agent triggered — refresh in ~10 seconds", icon="✅")

        _mgr_cache = read_cache("manager_report")
        _mgr_data  = _mgr_cache.get("data") or {}

        if not _mgr_data:
            with _mgr_col:
                st.info("No manager report yet — click 'Run Now' or wait up to 15 minutes for the first scheduled run.")
        else:
            with _mgr_col:
                _mgr_age = cache_age_label("manager_report")
                _mgr_pct = _mgr_data.get("health_pct", 0)
                _mgr_ok  = _mgr_data.get("ok", 0)
                _mgr_tot = _mgr_data.get("total_agents", 0)
                _mgr_heal= _mgr_data.get("healed", 0)
                _mgr_iss = _mgr_data.get("issues", 0)
                _mgr_keys= _mgr_data.get("key_issues", 0)

                _pct_clr = "#4caf50" if _mgr_pct >= 80 else ("#ff9800" if _mgr_pct >= 50 else "#f44336")
                st.markdown(f"""
<div style="background:#13110a;border:1px solid #2a2208;border-radius:10px;padding:18px 24px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px;">
    <div style="font-size:2.2rem;font-weight:700;color:{_pct_clr};font-family:monospace;">{_mgr_pct}%</div>
    <div>
      <div style="color:#c8b890;font-size:0.95rem;font-weight:600;">System Health</div>
      <div style="color:#5a4820;font-size:0.78rem;">Last checked: {_mgr_age} · {_mgr_ok}/{_mgr_tot} agents OK</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:16px;flex-wrap:wrap;">
      <div style="text-align:center;"><div style="color:#4caf50;font-size:1.3rem;font-weight:700;">{_mgr_heal}</div><div style="color:#5a4820;font-size:0.72rem;">Auto-healed</div></div>
      <div style="text-align:center;"><div style="color:{"#f44336" if _mgr_iss else "#4caf50"};font-size:1.3rem;font-weight:700;">{_mgr_iss}</div><div style="color:#5a4820;font-size:0.72rem;">Unresolved</div></div>
      <div style="text-align:center;"><div style="color:{"#f44336" if _mgr_keys else "#4caf50"};font-size:1.3rem;font-weight:700;">{_mgr_keys}</div><div style="color:#5a4820;font-size:0.72rem;">Missing API keys</div></div>
    </div>
  </div>
""", unsafe_allow_html=True)

                # API key status
                _api_keys = _mgr_data.get("api_keys", [])
                if _api_keys:
                    _key_rows = "".join([
                        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1e1a08;">'
                        f'<span style="width:8px;height:8px;border-radius:50%;background:{"#4caf50" if k["status"]=="OK" else "#f44336"};display:inline-block;flex-shrink:0;"></span>'
                        f'<span style="color:#c8b890;font-size:0.82rem;font-family:monospace;">{k["key"]}</span>'
                        f'<span style="color:#5a4820;font-size:0.78rem;margin-left:4px;">{k["description"]}</span>'
                        f'<span style="margin-left:auto;color:{"#4caf50" if k["status"]=="OK" else "#f44336"};font-size:0.78rem;font-weight:600;">{k["status"]}</span>'
                        f'</div>'
                        for k in _api_keys
                    ])
                    st.markdown(f'<div style="margin-top:8px;"><div style="color:#6a5228;font-size:0.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">API Keys</div>{_key_rows}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('</div>', unsafe_allow_html=True)

                # Healed / issue agents
                _healed_agents = _mgr_data.get("healed_agents", [])
                _unresolved    = _mgr_data.get("unresolved", [])
                if _healed_agents:
                    st.markdown(
                        f'<div style="background:#0d2a12;border:1px solid #1a5020;border-radius:8px;padding:10px 14px;margin-top:8px;">'
                        f'<span style="color:#4caf50;font-size:0.78rem;font-weight:600;">Auto-healed: </span>'
                        f'<span style="color:#c8b890;font-size:0.82rem;">{", ".join(_healed_agents)}</span></div>',
                        unsafe_allow_html=True,
                    )
                if _unresolved:
                    st.markdown(
                        f'<div style="background:#2a0d0d;border:1px solid #5a1010;border-radius:8px;padding:10px 14px;margin-top:6px;">'
                        f'<span style="color:#f44336;font-size:0.78rem;font-weight:600;">Unresolved: </span>'
                        f'<span style="color:#c8b890;font-size:0.82rem;">{", ".join(_unresolved)}</span></div>',
                        unsafe_allow_html=True,
                    )

                # Backed-off agents
                _backed_off_agents = _mgr_data.get("backed_off_agents", [])
                if _backed_off_agents:
                    st.markdown(
                        f'<div style="background:#2a1a00;border:1px solid #7a4000;border-radius:8px;padding:10px 14px;margin-top:6px;">'
                        f'<span style="color:#ff9800;font-size:0.78rem;font-weight:600;">Needs manual fix (failed 3x): </span>'
                        f'<span style="color:#c8b890;font-size:0.82rem;">{", ".join(_backed_off_agents)}</span>'
                        f'<div style="color:#7a5820;font-size:0.74rem;margin-top:4px;">Check your API keys in .env or verify network access, then click Run Now.</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Restart verification results
                _verif = _mgr_data.get("verification", {})
                if _verif:
                    _conf  = [k for k, v in _verif.items() if v == "CONFIRMED"]
                    _unconf = [k for k, v in _verif.items() if v != "CONFIRMED"]
                    if _conf:
                        st.markdown(
                            f'<div style="background:#0d2a12;border:1px solid #1a5020;border-radius:8px;padding:8px 14px;margin-top:6px;">'
                            f'<span style="color:#4caf50;font-size:0.76rem;font-weight:600;">Restart verified: </span>'
                            f'<span style="color:#a09880;font-size:0.78rem;">{", ".join(_conf)}</span></div>',
                            unsafe_allow_html=True,
                        )
                    if _unconf:
                        st.markdown(
                            f'<div style="background:#1a1400;border:1px solid #3a3000;border-radius:8px;padding:8px 14px;margin-top:4px;">'
                            f'<span style="color:#ffb74d;font-size:0.76rem;font-weight:600;">Restart unconfirmed: </span>'
                            f'<span style="color:#a09880;font-size:0.78rem;">{", ".join(_unconf)} — may still be running</span></div>',
                            unsafe_allow_html=True,
                        )

                # Investment Advisor dependency chain
                _adv_dep_checks = _mgr_data.get("advisor_dependencies", [])
                _adv_ok_count   = _mgr_data.get("advisor_ok", 0)
                if _adv_dep_checks:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="color:#6a5228;font-size:0.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">'
                        f'Investment Advisor Data Sources ({_adv_ok_count}/{len(_adv_dep_checks)} healthy)</div>',
                        unsafe_allow_html=True,
                    )
                    _dep_rows = ""
                    for _dep in _adv_dep_checks:
                        _dh  = _dep.get("health", "?")
                        _dot_c = "#4caf50" if _dh == "OK" else ("#ff9800" if _dh in ("STALE", "RUNNING") else "#f44336")
                        _hint  = _dep.get("hint", "")
                        _dep_rows += (
                            f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1e1a08;">'
                            f'<span style="width:8px;height:8px;border-radius:50%;background:{_dot_c};display:inline-block;flex-shrink:0;"></span>'
                            f'<span style="color:#c8b890;font-size:0.82rem;">{_dep["agent"].replace("_"," ").title()}</span>'
                            f'<span style="margin-left:auto;color:{_dot_c};font-size:0.76rem;font-weight:600;">{_dh}</span>'
                            + (f'<span style="color:#7a5820;font-size:0.72rem;margin-left:8px;">{_hint}</span>' if _hint else '')
                            + '</div>'
                        )
                    st.markdown(
                        f'<div style="background:#13110a;border:1px solid #2a2208;border-radius:8px;padding:12px 16px;">'
                        f'{_dep_rows}</div>',
                        unsafe_allow_html=True,
                    )

                # Per-agent detail: show any with hints or consecutive failures
                _problem_agents = [a for a in _mgr_data.get("agents", [])
                                   if a.get("consecutive_failures", 0) > 0 or a.get("missing_fields") or a.get("hint")]
                if _problem_agents:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="color:#6a5228;font-size:0.7rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Agent Diagnostics</div>',
                        unsafe_allow_html=True,
                    )
                    for _pa in _problem_agents:
                        _pa_c = "#ff9800" if _pa.get("consecutive_failures", 0) < 3 else "#f44336"
                        _mf   = ", ".join(_pa.get("missing_fields", []))
                        _hint = _pa.get("hint", "")
                        _cf   = _pa.get("consecutive_failures", 0)
                        _detail = []
                        if _cf:   _detail.append(f"{_cf} consecutive failure{'s' if _cf>1 else ''}")
                        if _mf:   _detail.append(f"missing fields: {_mf}")
                        if _hint: _detail.append(_hint)
                        st.markdown(
                            f'<div style="background:#1a1208;border-left:3px solid {_pa_c};border-radius:4px;'
                            f'padding:7px 12px;margin-bottom:4px;font-size:0.8rem;">'
                            f'<span style="color:{_pa_c};font-weight:600;">{_pa["agent"].replace("_"," ").title()}</span>'
                            f'<span style="color:#7a6040;margin-left:10px;">{" · ".join(_detail)}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
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
        else:
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
                st.markdown(_render_generic_table(
                    disp_comm,
                    title="Commodity Detail",
                    count_label=f"{len(disp_comm)} commodities",
                    hints={
                        "Commodity":    {"type": "name",    "flex": 1.5},
                        "Latest Price": {"type": "price",   "flex": 1},
                        "SMA-60":       {"type": "price",   "flex": 1},
                        "% vs SMA":     {"type": "colored", "flex": 1},
                    },
                ), unsafe_allow_html=True)

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
        else:
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
                st.markdown(_render_generic_table(
                    disp_esg,
                    title="Full Detail — Clean Energy & Green REITs",
                    count_label=f"{len(disp_esg)} securities",
                    hints={
                        "Security":  {"type": "name",    "flex": 1.5},
                        "Price":     {"type": "price",   "flex": 0.8},
                        "6mo Return":{"type": "pct_bar", "flex": 1.2},
                        "SMA-60":    {"type": "price",   "flex": 0.8},
                        "% vs SMA":  {"type": "colored", "flex": 0.8},
                    },
                ), unsafe_allow_html=True)

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
    tab_rates, tab_labor, tab_gdp, tab_inflation, tab_credit, tab_distressed, tab_reit = st.tabs([
        "Rate Environment",
        "Labor Market & Tenant Demand",
        "GDP & Economic Growth",
        "Inflation",
        "Credit & Capital Markets",
        "CMBS & Distressed",
        "REIT Sectors",
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
        else:

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
            _re_sum   = env.get("summary", "")
            _re_conf  = env.get("confidence", "High")
            # Normalize score: agent may store raw accumulation int (-5..+5) in
            # older cache entries; clamp to 0-100 matching the signal band.
            _re_score_raw = env.get("score", 0)
            if _re_score_raw > 10:
                _re_score = _re_score_raw  # already normalized (new cache)
            elif signal == "BULLISH":
                _re_score = min(99, 75 + max(0, _re_score_raw - 2) * 8)
            elif signal == "BEARISH":
                _re_score = max(0, 24 + min(0, _re_score_raw + 2) * 8)
            else:
                _re_score = max(25, min(74, 50 + _re_score_raw * 12))
            st.markdown(gauge_card(
                title       = "RATE ENVIRONMENT",
                label       = signal,
                score       = _re_score,
                summary     = _re_sum,
                agent_num   = "A6  Agent 6",
                age_label   = cache_age_label("rates"),
                confidence  = _re_conf,
                low_good    = False,
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
                    st.markdown(_render_generic_table(
                        tbl_df,
                        title="Interest Rate Monitor",
                        count_label=f"{len(tbl_df)} rates tracked",
                        hints={
                            "Rate": {"type": "name",    "flex": 2},
                            "Now":  {"type": "price",   "flex": 0.8},
                            "1W Δ": {"type": "colored", "flex": 0.8},
                            "1M Δ": {"type": "colored", "flex": 0.8},
                            "1Y Δ": {"type": "colored", "flex": 0.8},
                        },
                    ), unsafe_allow_html=True)

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

                    st.markdown(_render_generic_table(
                        disp_adj,
                        title="Profit Margin Impact by Property Type",
                        count_label=f"{len(disp_adj)} property types",
                        hints={
                            "Property Type":      {"type": "name",    "flex": 1.5},
                            "Baseline Cap Rate":  {"type": "text",    "flex": 1},
                            "Adjusted Cap Rate":  {"type": "text",    "flex": 1},
                            "Rate Adjustment bps":{"type": "colored", "flex": 1},
                            "Static Margin %":    {"type": "text",    "flex": 1},
                            "Adj Margin %":       {"type": "text",    "flex": 1},
                            "Margin Delta bps":   {"type": "colored", "flex": 1},
                        },
                    ), unsafe_allow_html=True)
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
                    st.markdown(_render_generic_table(
                        debt_df,
                        title="REIT Refinancing Risk",
                        count_label=f"{len(debt_df)} REITs",
                        scrollable=True, max_height=440,
                        hints={
                            "Ticker":           {"type": "name",    "flex": 0.7},
                            "Name":             {"type": "text",    "flex": 2},
                            "Risk %":           {"type": "pct_bar", "flex": 1},
                            "Near-Term Debt $B":{"type": "price",   "flex": 1},
                            "Total Debt $B":    {"type": "price",   "flex": 1},
                            "Risk Level":       {"type": "badge",   "flex": 0.8, "badge_map": {
                                "High":   "background:#2a0d0d;color:#ef5350",
                                "Medium": "background:#2a1500;color:#ffa726",
                                "Low":    "background:#0d2a12;color:#4a9e58",
                            }},
                        },
                    ), unsafe_allow_html=True)

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
        else:

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
            _KPI_TIP = {
                "Unemployment Rate":        "Share of the labor force actively looking for work. Below 4% = tight market → strong CRE demand. Above 6% = slack → weaker tenant absorption.",
                "Nonfarm Payrolls":         "Total US jobs excluding farm workers. Monthly gains >150K = strong economy. Rising payrolls directly drive office, industrial and retail leasing.",
                "Job Openings (JOLTS)":     "Job Openings and Labor Turnover Survey — number of unfilled positions. High openings (>8M) = companies expanding → more space needed.",
                "Quits Rate":               "Percentage of workers voluntarily leaving jobs. High quits = confident workers, tight labor market, wage pressure. Good for multifamily (wage growth → rent growth).",
                "Labor Force Participation":"Share of working-age population employed or seeking work. Rising = more potential tenants and consumers. Declining = structural demand headwinds.",
                "Avg Hourly Earnings":      "Average pay per hour in the private sector. Rising wages → stronger household balance sheets → supports multifamily rent growth and retail spending.",
            }
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
              <div class="label">{_tt(key, _KPI_TIP.get(key, key))}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_html or sub}</div>
            </div>""", unsafe_allow_html=True)

            # ── Job Postings Index (leading indicator) ──────────────────────────────
            _jpi = demand_sig.get("job_postings_index")
            if _jpi is not None:
                _jpi_color = ("#4caf50" if _jpi >= 65 else ("#ef5350" if _jpi <= 40 else "#d4a843"))
                _jpi_label = "STRONG" if _jpi >= 65 else ("SOFT" if _jpi <= 40 else "MODERATE")
                _jpi_tip = ("Composite leading indicator derived from employer hiring signals. "
                            "Leads official BLS payroll data by approximately 4–6 weeks. "
                            "Score 0–100: ≥65 = Strong hiring intent · 41–64 = Moderate · ≤40 = Soft.")
                st.markdown(
                    f'<div class="metric-card" style="max-width:280px;border-left:3px solid {_jpi_color};">'
                    f'<div class="label">{_tt("Job Postings Index", _jpi_tip)}</div>'
                    f'<div class="value" style="color:{_jpi_color};">{_jpi:.0f}</div>'
                    f'<div class="sub" style="color:{_jpi_color};">{_jpi_label} · Leads payroll ~4–6 weeks</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
                st.markdown(_render_generic_table(
                    disp_sec,
                    title="Employment by Sector",
                    count_label=f"{len(disp_sec)} sectors",
                    hints={
                        "Sector":             {"type": "name",    "flex": 2},
                        "Employment (K)":     {"type": "price",   "flex": 1},
                        "MoM %":              {"type": "pct_bar", "flex": 1.2},
                        "CRE Demand Driver":  {"type": "tag",     "flex": 1.5},
                        "Period":             {"type": "text",    "flex": 0.8},
                    },
                ), unsafe_allow_html=True)
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
                pt_rows = [(pt, f"{sum(v)/len(v):+.1f}%",
                            "EXPANDING" if sum(v)/len(v) > 2 else ("CONTRACTING" if sum(v)/len(v) < -2 else "FLAT"))
                           for pt, v in sorted(pt_summary.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)]
                _etf_rows_html = "".join([
                    f'<tr><td style="padding:7px 12px;color:#c8b890;font-size:12px;">{pt}</td>'
                    f'<td style="padding:7px 12px;font-family:monospace;font-size:12px;color:{("#4caf50" if "+" in ret else "#ef5350")};">{ret}</td>'
                    f'<td style="padding:7px 12px;">{_sig_pill(sig)}</td></tr>'
                    for pt, ret, sig in pt_rows
                ])
                st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#171309;border-radius:8px;overflow:hidden;border:1px solid #2a2208;">
  <thead><tr style="border-bottom:1px solid #2a2208;">
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">CRE Property Type</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Avg Sector Return","6-month total return of sector ETFs linked to this property type. Rising ETFs signal expanding corporate employment and leasing demand.")}</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Demand Signal","EXPANDING = avg return >+2% · FLAT = ±2% · CONTRACTING = <-2%. Hover a pill for plain-English meaning.")}</th>
  </tr></thead>
  <tbody>{_etf_rows_html}</tbody>
</table>""", unsafe_allow_html=True)
                st.caption("Data: Yahoo Finance (6-month trailing). Hover column headers or signal pills for definitions.")
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

                # Table with signal pills
                _mu_rows_html = "".join([
                    f'<tr style="border-bottom:1px solid #1e1a08;">'
                    f'<td style="padding:7px 12px;color:#c8b890;font-size:12px;">{row["market"]}</td>'
                    f'<td style="padding:7px 12px;font-family:monospace;font-size:12px;color:#d4a843;">{row["unemp_rate"]:.1f}%</td>'
                    f'<td style="padding:7px 12px;font-family:monospace;font-size:12px;color:{"#ef5350" if row["delta_1m"]>0 else "#4caf50"};">{row["delta_1m"]:+.1f}pp</td>'
                    f'<td style="padding:7px 12px;">{_sig_pill(row["signal"])}</td>'
                    f'<td style="padding:7px 12px;color:#5a4820;font-size:11px;">{row["period"]}</td>'
                    f'</tr>'
                    for row in metro_unemp
                ])
                st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#171309;border-radius:8px;overflow:hidden;border:1px solid #2a2208;">
  <thead><tr style="border-bottom:1px solid #2a2208;">
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">State / Key Metros</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Unemployment %","Share of the labor force actively seeking work but unemployed. Below 4% = tight market, above 6% = loose.")}</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("MoM Δ","Month-over-month change in unemployment rate, in percentage points. Green = falling (improving), red = rising.")}</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Labor Market","Hover the signal pill for a plain-English explanation of what this means for CRE demand.")}</th>
    <th style="padding:8px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Period</th>
  </tr></thead>
  <tbody>{_mu_rows_html}</tbody>
</table>""", unsafe_allow_html=True)
                st.caption("Data: FRED state unemployment rates (BLS LAUS). Updated monthly. Hover column headers or signal pills for definitions.")
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
        else:

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
            st.markdown(_render_generic_table(
                outlook_df,
                title="CRE Property Type Outlook by Cycle Phase",
                count_label=f"{len(outlook_df)} property types",
                hints={
                    "Property Type":  {"type": "name", "flex": 1.8},
                    "Expansion":      {"type": "text", "flex": 1},
                    "Slowdown":       {"type": "text", "flex": 1},
                    "Contraction":    {"type": "text", "flex": 1},
                    "Current Outlook":{"type": "text", "flex": 1},
                },
            ), unsafe_allow_html=True)
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
        else:

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
        else:

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
                    st.markdown(_render_generic_table(
                        pd.DataFrame(_dst_dlq_rows),
                        title="CMBS Delinquency Rates by Property Type",
                        count_label=f"{len(_dst_dlq_rows)} property types",
                        hints={
                            "Property Type": {"type": "name",    "flex": 1.2},
                            "Rate":          {"type": "pct_bar", "flex": 1},
                            "Prior Year":    {"type": "text",    "flex": 0.8},
                            "YoY":           {"type": "colored", "flex": 0.8},
                            "Trend":         {"type": "tag",     "flex": 1},
                        },
                    ), unsafe_allow_html=True)
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
                st.markdown(_render_generic_table(
                    pd.DataFrame(_dst_pipe_rows),
                    title="Known Distressed Asset Pipeline",
                    count_label=f"{len(_dst_pipe_rows)} assets",
                    scrollable=True, max_height=360,
                    hints={
                        "Asset":       {"type": "name",  "flex": 2},
                        "Type":        {"type": "tag",   "flex": 0.8},
                        "Loan":        {"type": "price", "flex": 0.8},
                        "Status":      {"type": "badge", "flex": 1, "badge_map": {
                            "Matured / Non-Performing": "background:#2a0d0d;color:#ef5350",
                            "90+ Days Delinquent":      "background:#2a0d0d;color:#ef5350",
                            "REO":                      "background:#2a0d2a;color:#ce93d8",
                            "Watchlist":                "background:#2a1500;color:#ffa726",
                            "Performing":               "background:#0d2a12;color:#4a9e58",
                        }},
                        "Market":      {"type": "text",  "flex": 1},
                        "Opportunity": {"type": "text",  "flex": 2},
                    },
                ), unsafe_allow_html=True)

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

    # ═══════════════════════════════════════════════════════════════════════════
    #  TAB — REIT SECTORS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_reit:
        st.markdown("#### What are institutional investors pricing into real estate sectors?")
        st.markdown(
            "Public REIT prices **lead private CRE values by 6–12 months** — when institutional investors "
            "rotate out of REIT sectors, private market cap rate expansion typically follows. "
            "Tracking REIT momentum gives early warning of shifting CRE capital flows. "
            "Data from Yahoo Finance via yfinance. Updates every hour."
        )
        agent_last_updated("reit")

        cache_reit = read_cache("reit")
        _reit_data = (cache_reit.get("data") or {}) if cache_reit else {}

        if not _reit_data:
            st.info("📡 REIT agent is fetching data — refresh in ~30 seconds.")
        else:
            _reit_tickers = _reit_data.get("tickers", [])
            _reit_sectors = _reit_data.get("sectors", {})
            _reit_best    = _reit_data.get("best_sector_3m", "—")
            _reit_worst   = _reit_data.get("worst_sector_3m", "—")
            _reit_mom     = _reit_data.get("cre_momentum", "NEUTRAL")
            _reit_spy3m   = _reit_data.get("spy_return_3m", 0)
            _reit_total   = _reit_data.get("total_tickers_tracked", 0)

            # ── Signal banner ─────────────────────────────────────────────────
            _mom_colors = {"BULLISH": "#4a9e58", "NEUTRAL": "#d4a843", "BEARISH": "#ef5350"}
            _mom_bg     = {"BULLISH": "#0d2a12", "NEUTRAL": "#2a1a04", "BEARISH": "#2a0d0d"}
            _mom_c = _mom_colors.get(_reit_mom, "#d4a843")
            _mom_b = _mom_bg.get(_reit_mom, "#2a1a04")
            st.markdown(
                f'<div style="background:{_mom_b};border:1px solid {_mom_c};border-radius:8px;'
                f'padding:12px 20px;margin-bottom:12px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
                f'<span style="font-size:0.8rem;color:#888;letter-spacing:0.1em;text-transform:uppercase;">CRE Momentum Signal</span>'
                f'<span style="background:{_mom_c};color:#0d0b04;font-size:0.8rem;font-weight:700;'
                f'padding:4px 14px;border-radius:20px;letter-spacing:0.08em;">{_reit_mom}</span>'
                f'<span style="color:#888;font-size:0.82rem;">Best Sector (3m): '
                f'<b style="color:#c8b890;">{_reit_best}</b></span>'
                f'<span style="color:#888;font-size:0.82rem;">Worst Sector (3m): '
                f'<b style="color:#ef5350;">{_reit_worst}</b></span>'
                f'<span style="color:#888;font-size:0.82rem;">SPY 3m: '
                f'<b style="color:{("#4a9e58" if _reit_spy3m > 0 else "#ef5350")};">{_reit_spy3m:+.1f}%</b></span>'
                f'<span style="color:#888;font-size:0.82rem;">Tracking: <b style="color:#c8b890;">{_reit_total} tickers</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Sector metric cards ───────────────────────────────────────────
            if _reit_sectors:
                section(" Sector Performance (3-Month Return)")
                _sec_cols = st.columns(len(_reit_sectors))
                for _sec_col, (sector, sec_data) in zip(_sec_cols, _reit_sectors.items()):
                    _s3m = sec_data["return_3m"]
                    _s_c = "#4a9e58" if _s3m > 0 else "#ef5350"
                    _sec_col.markdown(
                        metric_card(
                            sector,
                            f"<span style='color:{_s_c}'>{_s3m:+.1f}%</span>",
                            f"3m · {sec_data['ticker_count']} tickers",
                        ),
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Grouped bar chart — sector returns ────────────────────────
                section(" Sector Returns: 1M / 3M / 6M vs SPY")
                _sec_names = list(_reit_sectors.keys())
                _sec_1m = [_reit_sectors[s]["return_1m"] for s in _sec_names]
                _sec_3m = [_reit_sectors[s]["return_3m"] for s in _sec_names]
                _sec_6m = [_reit_sectors[s]["return_6m"] for s in _sec_names]

                fig_sec = go.Figure()
                fig_sec.add_trace(go.Bar(
                    name="1M Return", x=_sec_names, y=_sec_1m,
                    marker_color="#4fc3f7",
                    hovertemplate="<b>%{x}</b><br>1M: %{y:+.1f}%<extra></extra>",
                ))
                fig_sec.add_trace(go.Bar(
                    name="3M Return", x=_sec_names, y=_sec_3m,
                    marker_color=GOLD,
                    hovertemplate="<b>%{x}</b><br>3M: %{y:+.1f}%<extra></extra>",
                ))
                fig_sec.add_trace(go.Bar(
                    name="6M Return", x=_sec_names, y=_sec_6m,
                    marker_color="#a5d6a7",
                    hovertemplate="<b>%{x}</b><br>6M: %{y:+.1f}%<extra></extra>",
                ))
                fig_sec.add_hline(
                    y=_reit_spy3m, line_color="#ef5350", line_dash="dash", line_width=1.5,
                    annotation_text=f"SPY 3M: {_reit_spy3m:+.1f}%",
                    annotation_position="top right",
                    annotation_font=dict(color="#ef5350", size=11),
                )
                fig_sec.update_layout(
                    barmode="group",
                    plot_bgcolor="#1e1a0a",
                    paper_bgcolor="#1a1208",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    xaxis=dict(gridcolor="#2a2208", color="#8a7040", tickangle=-20),
                    yaxis=dict(gridcolor="#2a2208", color="#8a7040", title="Return (%)", ticksuffix="%"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8b890")),
                    margin=dict(t=40, b=80, l=60, r=20),
                    height=380,
                )
                st.plotly_chart(fig_sec, use_container_width=True)

            # ── Individual REIT table ─────────────────────────────────────────
            if _reit_tickers:
                st.markdown("<br>", unsafe_allow_html=True)
                section(" Individual REIT Performance")
                _reit_df = pd.DataFrame(_reit_tickers)
                _reit_disp = _reit_df[["ticker", "name", "sector", "current_price",
                                        "return_1m", "return_3m", "vs_spy_3m", "momentum"]].copy()
                _reit_disp.columns = ["Ticker", "Name", "Sector", "Price", "1M Return", "3M Return", "vs SPY (3m)", "Momentum"]
                _reit_disp["Price"] = _reit_disp["Price"].apply(lambda x: f"${x:.2f}")
                _reit_disp["1M Return"] = _reit_disp["1M Return"].apply(lambda x: f"{x:+.1f}%")
                _reit_disp["3M Return"] = _reit_disp["3M Return"].apply(lambda x: f"{x:+.1f}%")
                _reit_disp["vs SPY (3m)"] = _reit_disp["vs SPY (3m)"].apply(lambda x: f"{x:+.1f}%")

                st.markdown(_render_generic_table(
                    _reit_disp,
                    title="REIT Universe Performance",
                    count_label=f"{len(_reit_disp)} tickers",
                    scrollable=True, max_height=480,
                    hints={
                        "Ticker":      {"type": "name",    "flex": 0.7},
                        "Name":        {"type": "text",    "flex": 2},
                        "Sector":      {"type": "tag",     "flex": 1.2},
                        "Price":       {"type": "price",   "flex": 0.8},
                        "1M Return":   {"type": "colored", "flex": 0.8},
                        "3M Return":   {"type": "colored", "flex": 0.8},
                        "vs SPY (3m)": {"type": "colored", "flex": 0.9},
                        "Momentum":    {"type": "badge",   "flex": 0.9, "badge_map": {
                            "BULLISH": "background:#0d2a12;color:#4a9e58",
                            "NEUTRAL": "background:#2a1a04;color:#d4a843",
                            "BEARISH": "background:#2a0d0d;color:#ef5350",
                        }},
                    },
                ), unsafe_allow_html=True)

                st.download_button(
                    "⬇ Download CSV",
                    data=_reit_disp.to_csv(index=False),
                    file_name="reit_performance.csv",
                    mime="text/csv",
                    key="dl_reit_tickers",
                )

            with st.expander("How to Read REIT Signals"):
                st.markdown("""
**Why Public REITs Lead Private CRE by 6–12 Months**

Public REITs trade on exchanges with daily liquidity, meaning institutional investors can immediately
reprice their view of real estate when fundamentals shift. Private CRE transactions take months to
close and appraise, so private market values lag public market signals.

**How to use REIT signals:**
- **BULLISH momentum (3M return > +5%)**: Institutions are pricing in improving fundamentals — rents rising, vacancy falling. Private cap rates may compress 6–12 months ahead.
- **NEUTRAL momentum (±5%)**: Mixed signals. Monitor for direction change.
- **BEARISH momentum (3M return < -5%)**: Institutions anticipate deteriorating fundamentals — rising vacancy, softening rents, or higher cap rates ahead.

**Key sectors to watch:**
- **Office REITs (BXP, VNO)**: Sensitive to remote work trends and downtown foot traffic
- **Industrial (PLD, STAG)**: Driven by e-commerce, nearshoring, and supply chain logistics
- **Data Centers (EQIX, DLR)**: AI/cloud infrastructure build-out; strong secular tailwind
- **Multifamily (EQR, AVB)**: Housing affordability and migration patterns

**vs SPY**: Shows how each REIT is performing relative to the broad market.
Positive = outperforming equities; negative = real estate-specific headwinds.

**Data Source:** Yahoo Finance (yfinance). Updated every hour via Agent 24.
""")


with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab_supply, tab_returns, tab_oz, tab_score, tab_climate, tab_supply_pipeline = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "Supply & Demand",
        "Returns & Fundamentals",
        "Opportunity Zones",
        "Market Score",
        "Climate Risk",
        "Supply Pipeline",
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

        # ── Property-type scoring helpers ──────────────────────────────────────
        # State-level: re-weight composite score by property type
        _PT_STATE_W = {
            "Industrial":  {"biz": 0.60, "pop": 0.20, "comp": 0.20},
            "Multifamily": {"biz": 0.20, "pop": 0.60, "comp": 0.20},
            "Office":      {"biz": 0.50, "pop": 0.20, "comp": 0.30},
            "Retail":      {"biz": 0.30, "pop": 0.50, "comp": 0.20},
            "Healthcare":  {"biz": 0.25, "pop": 0.55, "comp": 0.20},
        }
        # Neighborhood zone suitability: 0=poor, 1=ok, 2=best
        _PT_ZONE_RANK = {
            "Industrial":  {"Urban Core": 0, "Suburban": 1, "Exurban": 2},
            "Multifamily": {"Urban Core": 2, "Suburban": 1, "Exurban": 0},
            "Office":      {"Urban Core": 2, "Suburban": 1, "Exurban": 0},
            "Retail":      {"Urban Core": 2, "Suburban": 1, "Exurban": 0},
            "Healthcare":  {"Urban Core": 1, "Suburban": 2, "Exurban": 0},
        }
        _PT_ZONE_LABEL = {0: "Low", 1: "Moderate", 2: "Best Fit"}

        def _pt_state_score(row, pt):
            biz  = float(row.get("biz_score", 50))
            pop_n = max(0.0, min(100.0, (float(row.get("pop_growth_pct", 0)) + 3) / 8 * 100))
            comp = float(row.get("composite_score", 50))
            if not pt or pt not in _PT_STATE_W:
                return comp
            w = _PT_STATE_W[pt]
            return round(w["biz"] * biz + w["pop"] * pop_n + w["comp"] * comp, 1)

        def _pt_neighborhood_score(row, pt):
            base = float(row.get("migration_score", 50))
            pop  = float(row.get("pop_growth_pct", 0))
            rent = float(row.get("median_rent_growth_pct", 0))
            zone = row.get("neighborhood_type", "Suburban")
            if not pt:
                return base
            pop_n  = max(0.0, min(100.0, (pop + 3) / 8 * 100))
            rent_n = max(0.0, min(100.0, (rent + 5) / 15 * 100))
            zr     = _PT_ZONE_RANK.get(pt, {}).get(zone, 1)
            zone_n = zr / 2 * 100
            if pt == "Industrial":
                s = base*0.30 + pop_n*0.20 + rent_n*0.20 + zone_n*0.30
            elif pt == "Multifamily":
                s = base*0.30 + pop_n*0.35 + rent_n*0.20 + zone_n*0.15
            elif pt == "Office":
                s = base*0.35 + pop_n*0.20 + rent_n*0.15 + zone_n*0.30
            elif pt == "Retail":
                s = base*0.30 + pop_n*0.30 + rent_n*0.20 + zone_n*0.20
            elif pt == "Healthcare":
                s = base*0.25 + pop_n*0.40 + rent_n*0.10 + zone_n*0.25
            else:
                s = base
            return round(max(0.0, min(100.0, s)), 1)

        def _pt_county_score(row, pt):
            base = float(row.get("migration_score", 50))
            pop  = float(row.get("pop_growth_pct", 0))
            if not pt:
                return base
            pop_n = max(0.0, min(100.0, (pop + 3) / 8 * 100))
            driver = str(row.get("top_driver", "")).lower()
            # Driver affinity bonus (+10 if driver matches property type)
            _driver_affinity = {
                "Industrial":  ["manufacturing", "logistics", "industrial", "warehouse", "port"],
                "Multifamily": ["population", "housing", "migration", "university", "renter"],
                "Office":      ["tech", "finance", "corporate", "hq", "professional"],
                "Retail":      ["retail", "consumer", "tourism", "mixed-use", "commercial"],
                "Healthcare":  ["medical", "healthcare", "hospital", "biotech", "senior"],
            }
            affinity = any(k in driver for k in _driver_affinity.get(pt, []))
            bonus = 8.0 if affinity else 0.0
            w = _PT_STATE_W.get(pt, {"biz": 0.33, "pop": 0.33, "comp": 0.34})
            s = base * (w.get("comp", 0.3) + w.get("biz", 0.3)) + pop_n * w.get("pop", 0.3) + bonus
            return round(max(0.0, min(100.0, s / (1 + bonus/100))), 1)

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

        # ── Apply property-type re-weighting to state rankings ─────────────────
        mig_df["pt_score"] = mig_df.apply(lambda r: _pt_state_score(r, _mig_pt), axis=1)
        if _mig_pt:
            # Preserve any location-based sort within the pt-reranked order
            if _mig_abbr:
                mig_df["_loc"] = (mig_df["state_abbr"] == _mig_abbr).astype(int)
                mig_df = mig_df.sort_values(["_loc", "pt_score"], ascending=[False, False]).reset_index(drop=True)
                mig_df = mig_df.drop(columns=["_loc"])
            else:
                mig_df = mig_df.sort_values("pt_score", ascending=False).reset_index(drop=True)
            _pt_weights_desc = {
                "Industrial":  "Business growth 60% · Pop growth 20% · Composite 20%",
                "Multifamily": "Pop growth 60% · Business growth 20% · Composite 20%",
                "Office":      "Business growth 50% · Composite 30% · Pop growth 20%",
                "Retail":      "Pop growth 50% · Business growth 30% · Composite 20%",
                "Healthcare":  "Pop growth 55% · Business growth 25% · Composite 20%",
            }
            st.markdown(
                f'<div style="background:#0d1a0d;border:1px solid #2a4a2a;border-radius:8px;'
                f'padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
                f'<span style="color:#4caf50;font-size:1rem;">⬡</span>'
                f'<span style="color:#c8d8c8;font-size:0.85rem;">'
                f'Rankings re-weighted for <b style="color:#a0d0a0;">{_mig_pt}</b> — '
                f'<span style="color:#5a8a5a;font-size:0.78rem;">{_pt_weights_desc.get(_mig_pt,"")}</span>'
                f'</span></div>',
                unsafe_allow_html=True,
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

        # ── Corporate HQ Move Tracker (from cache) ─────────────────────────────
        _hq_moves = data.get("hq_moves", [])
        if _hq_moves:
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Corporate HQ Relocations — CRE Demand Tracker")
            # Filter to selected/analyzed state if intent is set
            _hq_filter_state = _mig_abbr
            _hq_displayed = (
                [m for m in _hq_moves if m.get("to_state") == _hq_filter_state]
                if _hq_filter_state else _hq_moves
            )
            _hq_scope_label = (f"Showing moves **to {_hq_filter_state}**"
                               if _hq_filter_state and _hq_displayed else
                               "Showing all tracked corporate HQ relocations")
            if _hq_filter_state and not _hq_displayed:
                _hq_displayed = _hq_moves
                _hq_scope_label = f"No tracked HQ moves to {_hq_filter_state} — showing all markets"
            st.caption(_hq_scope_label)
            _hq_rows_html = ""
            for _hm in _hq_displayed:
                _hq_rows_html += (
                    f'<tr style="border-bottom:1px solid #1e1a08;">'
                    f'<td style="padding:8px 12px;color:#c8b890;font-size:12px;font-weight:600;">{_hm["company"]}</td>'
                    f'<td style="padding:8px 12px;color:#7a6840;font-size:12px;">{_hm["from_city"]}</td>'
                    f'<td style="padding:8px 12px;color:#4a9e58;font-size:12px;font-weight:600;">{_hm["to_city"]}, {_hm["to_state"]}</td>'
                    f'<td style="padding:8px 12px;color:#d4a843;font-size:12px;text-align:center;">{_hm["year"]}</td>'
                    f'<td style="padding:8px 12px;color:#c8b890;font-size:12px;text-align:right;">{_hm["employees"]:,}</td>'
                    f'<td style="padding:8px 12px;color:#d4a843;font-size:12px;text-align:right;">{_hm["sqft_demand_ksf"]:,} ksf</td>'
                    f'</tr>'
                )
            st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:#171309;border-radius:8px;overflow:hidden;border:1px solid #2a2208;">
  <thead><tr style="background:#1a1408;border-bottom:1px solid #2a2208;">
    <th style="padding:9px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Company</th>
    <th style="padding:9px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">From</th>
    <th style="padding:9px 12px;text-align:left;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">To</th>
    <th style="padding:9px 12px;text-align:center;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">Year</th>
    <th style="padding:9px 12px;text-align:right;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Employees","Approximate employees relocating with the HQ move.")}</th>
    <th style="padding:9px 12px;text-align:right;font-size:10px;color:#d4a843;letter-spacing:.1em;text-transform:uppercase;">{_tt("Est. Demand","Estimated office/industrial space demand generated by the relocation, in thousands of square feet.")}</th>
  </tr></thead>
  <tbody>{_hq_rows_html}</tbody>
</table>""", unsafe_allow_html=True)
            st.caption("Source: Public corporate filings and announcements. Estimated sq ft demand based on employees × avg space-per-employee benchmarks.")

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

                # Inline property-type selector for county view
                _cty_PT_OPTIONS = ["None", "Industrial", "Multifamily", "Office", "Retail", "Healthcare"]
                _cty_default_idx = _cty_PT_OPTIONS.index(_mig_pt) if _mig_pt in _cty_PT_OPTIONS else 0
                _cty_sel_cols = st.columns([3, 1])
                with _cty_sel_cols[0]:
                    _cty_pt = st.selectbox(
                        "Property Type Lens",
                        _cty_PT_OPTIONS,
                        index=_cty_default_idx,
                        key="cty_pt_sel",
                        help="Re-scores counties for the selected property type.",
                        label_visibility="collapsed",
                    )
                with _cty_sel_cols[1]:
                    if _cty_pt != "None" and _cty_pt != _mig_pt:
                        if st.button("Apply", key="cty_pt_apply", use_container_width=True):
                            st.session_state.user_intent["property_type"] = _cty_pt
                            st.rerun()
                    elif _cty_pt == "None" and _mig_pt:
                        if st.button("Clear", key="cty_pt_clear", use_container_width=True):
                            st.session_state.user_intent["property_type"] = None
                            st.rerun()

                _active_cty_pt = _cty_pt if _cty_pt != "None" else None

                county_df["pt_score"] = county_df.apply(lambda r: _pt_county_score(r, _active_cty_pt), axis=1)
                _cty_map_z    = county_df["pt_score"] if _active_cty_pt else county_df["migration_score"]
                _cty_cbar_lbl = f"{_active_cty_pt}\nScore" if _active_cty_pt else "Score"

                fig_county = go.Figure(go.Choroplethmapbox(
                    geojson=counties_geojson,
                    locations=county_df["fips"],
                    z=_cty_map_z,
                    featureidkey="id",
                    colorscale=[
                        [0.0, "#7f0000"], [0.25, "#c62828"], [0.45, "#d4c5a9"],
                        [0.55, "#a5d6a7"], [0.75, "#2e7d32"], [1.0, "#1b5e20"],
                    ],
                    zmin=0, zmax=100,
                    marker_line_width=0.5, marker_line_color="#333",
                    marker_opacity=0.85,
                    colorbar=dict(title=dict(text=_cty_cbar_lbl, font=dict(size=10, color="#e8e9ed")),
                                  tickfont=dict(size=9, color="#e8e9ed"), thickness=12, len=0.6,
                                  bgcolor="#171309", bordercolor="#2a2208"),
                    text=county_df.apply(
                        lambda r: (
                            f"<b>{r['name']}</b><br>"
                            + (f"{_active_cty_pt} Score: {r['pt_score']:.0f}<br>Migration Score: {r['migration_score']}<br>"
                               if _active_cty_pt else f"Score: {r['migration_score']}<br>")
                            + f"Pop Growth: {r['pop_growth_pct']:+.1f}%<br>"
                            f"Pop: {r['population']:,}<br>{r['top_driver']}"
                        ), axis=1),
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig_county.update_geos(fitbounds="locations", visible=False,
                    bgcolor="#111111", landcolor="#1a1a1a", lakecolor="#0a0a0a")
                fig_county.update_layout(
                    paper_bgcolor="#111111", plot_bgcolor="#111111",
                    geo=dict(bgcolor="#111111"),
                    margin=dict(t=10, b=10, l=0, r=0),
                    height=520, font=dict(family="DM Sans", color="#e8e9ed"),
                )
                st.plotly_chart(fig_county, use_container_width=True, config={"displayModeBar": False})

                # County rankings table
                section(f"County Rankings — {_US_STATES.get(_sel_state_abbr, _sel_state_abbr)}")

                if _active_cty_pt:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#5a8a5a;margin:-4px 0 8px;">'
                        f'Ranked by <b style="color:#a0d0a0;">{_active_cty_pt} Score</b> · '
                        f'{_pt_weights_desc.get(_active_cty_pt,"")}</div>',
                        unsafe_allow_html=True,
                    )

                _county_sort_col = "pt_score" if _active_cty_pt else "migration_score"
                _cdisp = county_df.sort_values(_county_sort_col, ascending=False).copy().reset_index(drop=True)

                _n_cty_total = len(_cdisp)
                _n_cty_top   = max(1, (_n_cty_total + 1) // 2)
                _cty_top     = _cdisp.iloc[:_n_cty_top]
                _cty_rest    = _cdisp.iloc[_n_cty_top:]

                _cty_title = f"{_active_cty_pt} County Rankings" if _active_cty_pt else "County Rankings — Migration & Growth"
                st.markdown(
                    _render_county_table(_cty_top, active_pt=_active_cty_pt, title=_cty_title, start_rank=1),
                    unsafe_allow_html=True,
                )

                if not _cty_rest.empty:
                    with st.expander(
                        f"Show lower-ranked counties ({len(_cty_rest)} counties — bottom 50%)",
                        expanded=False,
                    ):
                        st.markdown(
                            _render_county_table(_cty_rest, active_pt=_active_cty_pt, title="Lower-Ranked Counties", start_rank=_n_cty_top + 1),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "These counties scored in the bottom half for "
                            + (f"**{_active_cty_pt}** demand signals." if _active_cty_pt else "overall migration strength.")
                        )

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

                # Compute pt_score on zip_df before map (may not have been done yet if fallback path)
                if "pt_score" not in zip_df.columns:
                    zip_df["pt_score"] = zip_df.apply(lambda r: _pt_neighborhood_score(r, _mig_pt), axis=1)
                _map_bubble_score = zip_df["pt_score"] if _mig_pt else zip_df["migration_score"]
                _map_cbar_title   = f"{_mig_pt}\nScore" if _mig_pt else "Score"
                fig_metro = go.Figure(go.Scattermapbox(
                    lat=zip_df["lat"], lon=zip_df["lon"],
                    mode="markers+text",
                    marker=dict(
                        size=_map_bubble_score.clip(10, 95) / 5,
                        color=_map_bubble_score,
                        colorscale=[
                            [0.0, "#7f0000"], [0.3, "#c62828"], [0.5, "#d4c5a9"],
                            [0.7, "#2e7d32"], [1.0, "#1b5e20"],
                        ],
                        cmin=0, cmax=100,
                        showscale=True,
                        colorbar=dict(title=dict(text=_map_cbar_title, font=dict(color="#e8e9ed")),
                                      thickness=10, len=0.5,
                                      tickfont=dict(color="#e8e9ed")),
                    ),
                    text=zip_df["name"],
                    textposition="top center",
                    textfont=dict(size=9, color="#e8e9ed"),
                    customdata=zip_df[["name", "pt_score", "migration_score", "pop_growth_pct", "median_rent_growth_pct", "neighborhood_type"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + (f"{_mig_pt} Score: %{{customdata[1]:.0f}}<br>Migration Score: %{{customdata[2]:.0f}}<br>" if _mig_pt else "Score: %{customdata[2]:.0f}<br>")
                        + "Pop Growth: %{customdata[3]:+.1f}%<br>"
                        "Rent Growth: %{customdata[4]:+.1f}%<br>"
                        "Type: %{customdata[5]}<extra></extra>"
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

                # ── Neighborhood Rankings table ────────────────────────────────
                section(f"Neighborhood Rankings — {_sel_metro}")

                # Inline property-type selector (reads from + writes to global intent)
                _PT_OPTIONS = ["None", "Industrial", "Multifamily", "Office", "Retail", "Healthcare"]
                _pt_default_idx = _PT_OPTIONS.index(_mig_pt) if _mig_pt in _PT_OPTIONS else 0
                _nbhd_sel_cols = st.columns([3, 1])
                with _nbhd_sel_cols[0]:
                    _nbhd_pt = st.selectbox(
                        "Property Type Lens",
                        _PT_OPTIONS,
                        index=_pt_default_idx,
                        key="nbhd_pt_sel",
                        help="Re-scores and re-ranks neighborhoods for the selected property type. "
                             "Linked to your global session focus.",
                        label_visibility="collapsed",
                    )
                with _nbhd_sel_cols[1]:
                    if _nbhd_pt != "None" and _nbhd_pt != _mig_pt:
                        if st.button("Apply", key="nbhd_pt_apply", use_container_width=True):
                            st.session_state.user_intent["property_type"] = _nbhd_pt
                            st.rerun()
                    elif _nbhd_pt == "None" and _mig_pt:
                        if st.button("Clear", key="nbhd_pt_clear", use_container_width=True):
                            st.session_state.user_intent["property_type"] = None
                            st.rerun()

                # Use the inline selector value immediately (no rerun needed for display)
                _nbhd_active_pt = _nbhd_pt if _nbhd_pt != "None" else None

                zip_df = zip_df.copy()
                zip_df["pt_score"] = zip_df.apply(lambda r: _pt_neighborhood_score(r, _nbhd_active_pt), axis=1)
                # Zone fit: always show suitability label (generic if no type)
                zip_df["zone_fit"] = zip_df["neighborhood_type"].apply(
                    lambda z: _PT_ZONE_LABEL.get(_PT_ZONE_RANK.get(_nbhd_active_pt, {}).get(z, 1), "Moderate")
                    if _nbhd_active_pt else z   # fall back to just the zone name
                )
                _nbhd_sort_col = "pt_score" if _nbhd_active_pt else "migration_score"
                zip_df = zip_df.sort_values(_nbhd_sort_col, ascending=False).reset_index(drop=True)

                if _nbhd_active_pt:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#5a8a5a;margin:-4px 0 8px;">'
                        f'Ranked by <b style="color:#a0d0a0;">{_nbhd_active_pt} Score</b> · '
                        f'{_pt_weights_desc.get(_nbhd_active_pt,"")}</div>',
                        unsafe_allow_html=True,
                    )

                # Split top 50% / bottom 50%
                _n_total_nbhd = len(zip_df)
                _n_top = max(1, (_n_total_nbhd + 1) // 2)
                _df_top  = zip_df.iloc[:_n_top]
                _df_rest = zip_df.iloc[_n_top:]

                _nbhd_title = f"{_nbhd_active_pt} Neighborhood Rankings — {_sel_metro}" if _nbhd_active_pt else f"Neighborhood Rankings — {_sel_metro}"
                st.markdown(
                    _render_neighborhood_table(_df_top, active_pt=_nbhd_active_pt, title=_nbhd_title, start_rank=1),
                    unsafe_allow_html=True,
                )

                if not _df_rest.empty:
                    with st.expander(
                        f"Show lower-ranked neighborhoods ({len(_df_rest)} areas — bottom 50%)",
                        expanded=False,
                    ):
                        st.markdown(
                            _render_neighborhood_table(_df_rest, active_pt=_nbhd_active_pt, title="Lower-Ranked Neighborhoods", start_rank=_n_top + 1),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "These neighborhoods scored in the bottom half for "
                            + (f"**{_nbhd_active_pt}** demand signals." if _nbhd_active_pt else "overall migration strength.")
                        )

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

                _map_z     = mig_df["pt_score"] if _mig_pt else mig_df["composite_score"]
                _map_clbl  = f"{_mig_pt}<br>Score" if _mig_pt else "Migration<br>Score"
                fig_map = go.Figure(go.Choropleth(
                    locations=mig_df["state_abbr"],
                    z=_map_z,
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
                        title=dict(text=_map_clbl, font=dict(size=11, color="#c8b890")),
                        tickfont=dict(size=10, color="#c8b890"),
                        thickness=14, len=0.65,
                        bgcolor="#1a1208",
                        bordercolor="#3a3a2a",
                        borderwidth=1,
                    ),
                    text=mig_df.apply(
                        lambda r: (
                            f"<b>{r['state_name']}</b><br>"
                            + (f"{_mig_pt} Score: {r['pt_score']:.0f}<br>Migration Score: {r['composite_score']:.0f}<br>" if _mig_pt else f"Migration Score: {r['composite_score']:.0f}<br>")
                            + f"Classification: {_classify(r['pt_score'] if _mig_pt else r['composite_score'])}<br>"
                            f"Pop Growth: {r['pop_growth_pct']:+.1f}% · Biz Score: {r['biz_score']:.0f}<br>"
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
                _map_cap = (
                    f"Map re-scored for **{_mig_pt}** demand — color reflects the property-type-adjusted score, "
                    f"not the generic migration composite. Darker green = stronger fit for {_mig_pt} investment."
                ) if _mig_pt else (
                    "Darker green indicates states with the strongest combined population inflow and business migration. "
                    "These markets historically see the earliest and sharpest increases in CRE demand — "
                    "particularly for multifamily, industrial, and mixed-use properties."
                )
                st.caption(_map_cap)

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
        _top10_score_col = "pt_score" if _mig_pt else "composite_score"
        _top10_title = (
            f" Top 10 States — {_mig_pt} Migration Score"
            if _mig_pt else
            " Top 10 States for CRE Investment (Migration Score)"
        )
        section(_top10_title)
        top10 = mig_df.head(10)
        _bar_scores = top10[_top10_score_col]
        _bar_custom = top10.apply(
            lambda r: [r["state_name"], r["pop_growth_pct"], r["key_companies"],
                       r["composite_score"], r.get("biz_score", 0)], axis=1
        ).tolist()
        fig_bar = go.Figure(go.Bar(
            x=_bar_scores, y=top10["state_abbr"],
            orientation="h",
            marker=dict(color=_bar_scores, colorscale=[[0, GOLD_DARK], [1, GOLD]], showscale=False),
            text=_bar_scores.apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            textfont=dict(color="#c8b890", size=12),
            customdata=_bar_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + (f"{_mig_pt} Score: %{{x:.0f}}<br>Base Migration: %{{customdata[3]:.0f}}<br>" if _mig_pt else "Score: %{x:.0f}<br>")
                + "Pop Growth: %{customdata[1]:+.2f}%<br>"
                "Biz Score: %{customdata[4]:.0f}<br>"
                "Key Companies: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            plot_bgcolor="#1e1a0a", paper_bgcolor="#1a1208",
            xaxis=dict(showgrid=True, gridcolor="#2a2208", range=[0, 115],
                       tickfont=dict(color="#c8b890"), title_font=dict(color="#c8b890")),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#c8b890", size=12)),
            margin=dict(t=20, b=20, l=60, r=60),
            height=320, font=dict(family="Source Sans Pro", color="#c8b890"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        if _mig_pt:
            st.caption(
                f"Rankings re-weighted for **{_mig_pt}** demand. "
                f"{_pt_weights_desc.get(_mig_pt, '')}. "
                "Higher scores indicate stronger macro conditions for this property type."
            )
        else:
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

        # ── Compare Two Markets ──────────────────────────────────────────────
        _ms_cache = read_cache("market_score")
        _ms_data = (_ms_cache.get("data") or {}) if _ms_cache else {}
        _rankings = _ms_data.get("rankings", [])
        _all_markets = [r["market"] for r in _rankings] if _rankings else []

        if _all_markets:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Compare Two Markets"):
                cmp_c1, cmp_c2 = st.columns(2)
                cmp_a = cmp_c1.selectbox("Market A", _all_markets, key="cmp_mkt_a")
                cmp_b = cmp_c2.selectbox("Market B", _all_markets, index=min(1, len(_all_markets)-1), key="cmp_mkt_b")

                _ra = next((r for r in _rankings if r["market"] == cmp_a), None)
                _rb = next((r for r in _rankings if r["market"] == cmp_b), None)

                if _ra and _rb:
                    _factor_labels = ["Migration", "Vacancy", "Rent", "Cap Rate", "Land", "Macro"]
                    _factor_keys   = ["migration", "vacancy", "rent", "cap_rate", "land", "macro"]

                    _fig_radar = go.Figure()
                    for _r, _col in [(_ra, GOLD), (_rb, "#4fc3f7")]:
                        _vals = [_r["factors"].get(k, 50) for k in _factor_keys]
                        _vals_closed = _vals + [_vals[0]]
                        _labs_closed = _factor_labels + [_factor_labels[0]]
                        _fig_radar.add_trace(go.Scatterpolar(
                            r=_vals_closed, theta=_labs_closed,
                            fill="toself", name=_r["market"],
                            line_color=_col,
                            opacity=0.8,
                        ))
                    _fig_radar.update_layout(
                        polar=dict(
                            bgcolor="#1e1a0a",
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333", tickfont=dict(color="#888")),
                            angularaxis=dict(gridcolor="#333", tickfont=dict(color="#c8b890")),
                        ),
                        paper_bgcolor="#1a1208", plot_bgcolor="#1e1a0a",
                        legend=dict(font=dict(color="#c8b890"), bgcolor="#1a1208"),
                        margin=dict(t=40, b=40), height=380,
                    )
                    st.plotly_chart(_fig_radar, use_container_width=True)

                    _sc1, _sc2 = st.columns(2)
                    _sc1.markdown(metric_card(cmp_a, f"{_ra['composite']}/100 · Grade {_ra['grade']}", f"Rank #{_ra['rank']}"), unsafe_allow_html=True)
                    _sc2.markdown(metric_card(cmp_b, f"{_rb['composite']}/100 · Grade {_rb['grade']}", f"Rank #{_rb['rank']}"), unsafe_allow_html=True)


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
                    "AVG CAP RATE":   ("NOI ÷ Property Value",       "A lower cap rate = higher asset value (investors paying more per dollar of income). Compressed cap rates signal strong demand. Rises when interest rates climb or asset values fall."),
                    "NOI MARGIN":     ("NOI ÷ Gross Revenue",        "Net Operating Income margin — what's left after operating expenses (insurance, maintenance, mgmt fees) but before CapEx and debt payments. Higher = more efficient property."),
                    "RENT GROWTH YOY":("(Rent Now − Rent Yr Ago) ÷ Rent Yr Ago", "How much rents have risen over 12 months. Positive rent growth means existing leases rolling to market rates generate higher NOI over time."),
                    "AVG VACANCY":    ("Unleased SF ÷ Total Rentable SF", "Share of space sitting empty. Lower vacancy = landlords have pricing power. Rising vacancy = oversupply or weak demand — a leading indicator of cap rate expansion."),
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
<div style="background:#171309;border:1px solid #2a2208;border-radius:10px;padding:18px 16px;position:relative;overflow:visible;min-height:130px;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{_border};border-radius:10px 10px 0 0;"></div>
  <div style="font-size:10px;color:#6a5228;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">{_tt(_lbl, _exp_desc)}</div>
  <div style="font-size:34px;font-weight:500;color:{_border};line-height:1;margin-bottom:4px;">{_val_str}</div>
  <div style="font-size:11px;color:#6a5228;margin-bottom:2px;font-family:monospace;">{_exp_formula}</div>
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
            st.markdown(_render_generic_table(
                top5_df,
                title="Top 5 States Attracting Corporate Investment",
                count_label=f"{len(top5_df)} states",
                hints={
                    "State":                  {"type": "name", "flex": 1.5},
                    "Abbr":                   {"type": "text", "flex": 0.5},
                    "Pop Growth":             {"type": "pct_bar", "flex": 1.2},
                    "Business Score":         {"type": "score_bar", "flex": 1.2},
                    "Recent Corporate Moves": {"type": "text", "flex": 2},
                    "Growth Drivers":         {"type": "text", "flex": 2},
                },
            ), unsafe_allow_html=True)
            st.download_button(
                "⬇ Download CSV",
                data=top5_df.to_csv(index=False),
                file_name="migration_top_states.csv",
                mime="text/csv",
                key="dl_migration_states",
            )
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

                _exp_label = f" {state_name} ({abbr}) — {len(group)} {_pt_label}listings"
                _exp_open  = abbr == (_user_abbr4 or (_sorted_abbr4[0] if _sorted_abbr4 else None))
                with st.expander(_exp_label, expanded=_exp_open):
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

            # ── $/sqft Trend Chart (RentCast sparklines) ─────────────────────
            _rc_ppsf_trend = _rc_data.get("price_per_sqft_trend", {})
            _trend_states = [abbr for abbr in _sorted_abbr4 if abbr in _rc_ppsf_trend]
            if _trend_states:
                st.markdown("<br>", unsafe_allow_html=True)
                section(" $/sqft Trend by Market (6-Month)")
                try:
                    import plotly.graph_objects as _go_rc
                    _fig_ppsf = _go_rc.Figure()
                    _sparkline_colors = ["#d4a843", "#80a848", "#9868b8", "#c84848", "#4888c8"]
                    for _ci, _abbr in enumerate(_trend_states[:5]):
                        _pts = _rc_ppsf_trend[_abbr]
                        _sx = [p["month"] for p in _pts]
                        _sy = [p["ppsf"] for p in _pts]
                        _clr = _sparkline_colors[_ci % len(_sparkline_colors)]
                        _fig_ppsf.add_trace(_go_rc.Scatter(
                            x=_sx, y=_sy, mode="lines+markers", name=_abbr,
                            line=dict(color=_clr, width=2),
                            marker=dict(size=5, color=_clr),
                            hovertemplate=f"<b>{_abbr}</b><br>%{{x}}<br>${{y:.0f}}/sqft<extra></extra>",
                        ))
                    _fig_ppsf.update_layout(
                        plot_bgcolor="#0f0d06", paper_bgcolor="#16160f",
                        margin=dict(t=20, b=20, l=40, r=20), height=240,
                        xaxis=dict(tickfont=dict(color="#e8dfc4", size=9), gridcolor="#2a2a1a", title=""),
                        yaxis=dict(tickfont=dict(color="#e8dfc4", size=9), gridcolor="#2a2a1a",
                                   title="$/sqft", titlefont=dict(color="#a09880"), tickprefix="$"),
                        font=dict(family="Source Sans Pro", color="#e8dfc4"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                    font=dict(color="#c8b890", size=10), bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(_fig_ppsf, use_container_width=True, config={"displayModeBar": False})
                    st.caption("6-month simulated $/sqft trend per market. Based on current listing averages with forward-looking drift.")
                except Exception as _ppsf_err:
                    st.caption(f"$/sqft trend chart unavailable: {_ppsf_err}")

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

            # Bucket articles by credibility tier
            _cred_buckets = {"VERIFIED": [], "HIGH": [], "MODERATE": [], "LOW": []}
            for art in raw:
                if feed_type_filter != "All" and art.get("feed_type") != feed_type_filter:
                    continue
                if art.get("credibility_score", 0) < _cred_min_score:
                    continue
                _bl = art.get("credibility_label", "LOW")
                _cred_buckets.setdefault(_bl, []).append(art)

            _tier_meta = {
                "VERIFIED": ("★ Verified",        True,  "#4caf50"),
                "HIGH":     ("✓ High Credibility", True,  "#8bc34a"),
                "MODERATE": ("~ Moderate",         False, "#ff9800"),
                "LOW":      ("⚠ Low / Unverified", False, "#f44336"),
            }
            shown = 0
            for _tier, _arts in _cred_buckets.items():
                if not _arts:
                    continue
                _tlabel, _topen, _tclr = _tier_meta.get(_tier, (_tier, False, "#888"))
                with st.expander(f"{_tlabel} ({len(_arts)})", expanded=_topen):
                    for art in _arts:
                        _render_article(art)
                shown += len(_arts)

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
    with tab_supply:
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
        st.caption("Vacancy rate & trend by market and property type — CBRE / JLL / CoStar Q1 2025")

        with st.expander("Show detail table", expanded=False):
            if mkt_rows:
                detail_df = pd.DataFrame(mkt_rows)

                # Build header (flex-div)
                _md_col_styles = [
                    ("MARKET",        "flex:2;"),
                    ("PROPERTY TYPE", "flex:2;"),
                    ("VACANCY %",     "flex:1;text-align:right;"),
                    ("TREND",         "flex:1;text-align:right;"),
                    ("VS. NATIONAL",  "flex:1;text-align:right;"),
                ]
                _md_hcells = "".join(
                    f'<div style="{fw}font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">{h}</div>'
                    for h, fw in _md_col_styles
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
                    _md_rows_html += (
                        f'<div style="display:flex;align-items:center;padding:10px 16px;border-bottom:1px solid #1e1a08;background:{_row_bg};opacity:{_row_opacity};">'
                        f'<div style="flex:2;color:#c8b890;font-size:0.9rem;white-space:nowrap;">{_row["market"]}</div>'
                        f'<div style="flex:2;color:#a09070;font-size:0.87rem;">{_row["property_type"]}</div>'
                        f'<div style="flex:1;text-align:right;color:#c8a040;font-size:0.92rem;font-weight:600;letter-spacing:0.05em;">{_vac_val:.1f}%</div>'
                        f'<div style="flex:1;text-align:right;color:{_trend_c};font-size:0.87rem;">{_trend_str}</div>'
                        f'<div style="flex:1;text-align:right;color:{_vs_c};font-size:0.87rem;font-weight:600;">{_vs_str}</div>'
                        f'</div>'
                    )

                _md_html = f"""<div style="background:#13110a;border-radius:10px;padding:20px 32px 16px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;padding:10px 16px;border-bottom:1px solid #2a2410;">{_md_hcells}</div>
  {_md_rows_html}
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">vs. National = pp difference from property-type national average &nbsp;·&nbsp; <span style="color:#66bb6a;">Green</span> = tighter than avg &nbsp;·&nbsp; <span style="color:#ef5350;">Red</span> = looser than avg &nbsp;·&nbsp; Not financial advice.</div>
</div>"""
                st.markdown(_md_html, unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download CSV",
                    data=detail_df.to_csv(index=False),
                    file_name="vacancy_rates_by_market.csv",
                    mime="text/csv",
                    key="dl_vacancy",
                )

        # ── Property Demand Score ────────────────────────────────────────────
        if _vac_focus_col and mkt_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            section(f" Property Demand Score — {_vac_focus_col}")
            st.markdown(
                f"Composite market health score for **{_vac_focus_col}** — combines vacancy tightness "
                f"and net absorption signal. Higher = tighter supply/demand balance.",
                unsafe_allow_html=False,
            )

            # Per-type weights: (vacancy_weight, absorption_weight)
            _pds_weights = {
                "Industrial":  (0.50, 0.50),
                "Multifamily": (0.60, 0.40),
                "Office":      (0.70, 0.30),
                "Retail":      (0.55, 0.45),
                "Mixed-Use":   (0.55, 0.45),
            }
            _w_vac, _w_abs = _pds_weights.get(_vac_focus_col, (0.60, 0.40))

            # Build market→vacancy map for focused type
            _pds_vac: dict = {}
            for _r in mkt_rows:
                if _r.get("property_type") == _vac_focus_col:
                    _pds_vac[_r["market"]] = float(_r.get("vacancy_rate", 10.0))

            # Build market→absorption map for focused type
            _pds_abs: dict = {}
            for _r in vac_data.get("absorption_rows", []):
                if _r.get("property_type") == _vac_focus_col:
                    _pds_abs[_r["market"]] = float(_r.get("net_absorption_ksf", 0))

            # Normalize helpers
            def _norm(v, lo, hi, inv=False):
                if hi == lo: return 50.0
                s = max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))
                return round(100.0 - s if inv else s, 1)

            _all_markets_pds = sorted(set(list(_pds_vac.keys()) + list(_pds_abs.keys())))
            _pds_rows = []
            for _m in _all_markets_pds:
                _vv = _pds_vac.get(_m, 10.0)
                _av = _pds_abs.get(_m, 0)
                _vs = _norm(_vv, 4.0, 25.0, inv=True)
                _as = _norm(_av, -2000, 8000)
                _ds = round(_w_vac * _vs + _w_abs * _as, 1)
                _grade = "A" if _ds >= 75 else ("B" if _ds >= 55 else ("C" if _ds >= 40 else "D"))
                _pds_rows.append({"market": _m, "score": _ds, "vac_score": _vs, "abs_score": _as,
                                   "vacancy": _vv, "absorption": _av, "grade": _grade})

            _pds_rows.sort(key=lambda x: x["score"], reverse=True)

            # Render as horizontal bar chart
            _pds_df = pd.DataFrame(_pds_rows)
            _pds_colors = [
                "#66bb6a" if r >= 70 else ("#d4a843" if r >= 50 else "#ef5350")
                for r in _pds_df["score"]
            ]
            fig_pds = go.Figure(go.Bar(
                x=_pds_df["score"], y=_pds_df["market"], orientation="h",
                marker_color=_pds_colors,
                text=[f"{r['grade']}  {r['score']:.0f}" for r in _pds_rows],
                textposition="outside", textfont=dict(color="#c8b890", size=10),
                customdata=_pds_df[["vacancy", "absorption", "vac_score", "abs_score"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Demand Score: %{{x:.0f}}<br>"
                    f"Vacancy Rate: %{{customdata[0]:.1f}}%<br>"
                    f"Absorption: %{{customdata[1]:,.0f}} ksf<br>"
                    f"Vac Component: %{{customdata[2]:.0f}} (×{_w_vac:.0%})<br>"
                    f"Abs Component: %{{customdata[3]:.0f}} (×{_w_abs:.0%})"
                    "<extra></extra>"
                ),
            ))
            fig_pds.update_layout(
                xaxis=dict(range=[0, 115], gridcolor="#2a2208", tickfont=dict(color="#c8b890"),
                           title="Demand Score (0–100)", title_font=dict(color="#7a7050")),
                yaxis=dict(autorange="reversed", tickfont=dict(color="#c8b890", size=11)),
                plot_bgcolor="#0d0b04", paper_bgcolor="#13110a",
                margin=dict(t=10, b=20, l=180, r=60), height=460,
                font=dict(family="Source Sans Pro", color="#c8b890"),
            )
            st.plotly_chart(fig_pds, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"Vacancy component (×{_w_vac:.0%}): inverted vacancy rate — lower vacancy = higher score. "
                f"Absorption component (×{_w_abs:.0%}): positive net absorption = higher score. "
                "Weights are property-type specific."
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
            _abs_default_idx = _pt_options.index(_vac_focus_col) if _vac_focus_col in _pt_options else 0
            _sel_pt = st.selectbox("Property Type", _pt_options, index=_abs_default_idx, key="abs_pt_sel")
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
            st.caption("Developable acreage, pricing & pipeline activity — CoStar / CBRE Q1 2025")

            with st.expander("Show detail table", expanded=False):
                _LND_C = {"Industrial": "#2bbfb0", "Mixed-Use": "#a09040", "Residential": "#a07830"}

                # Legend dots
                _legend_html = " &nbsp;&nbsp; ".join(
                    f'<span style="display:inline-flex;align-items:center;gap:6px;font-size:0.82rem;color:#c8b890;">'
                    f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{c};"></span>{lbl}</span>'
                    for lbl, c in _LND_C.items()
                )

                # Header (flex-div)
                _lnd_col_styles = [
                    ("MARKET",       "flex:2;"),
                    ("INDUSTRIAL",   "flex:1;text-align:right;"),
                    ("MIXED-USE",    "flex:1;text-align:right;"),
                    ("RESIDENTIAL",  "flex:1;text-align:right;"),
                    ("TOTAL (AC)",   "flex:1;text-align:right;"),
                    ("MIX",          "flex:2;padding-left:8px;"),
                ]
                _lnd_hcells = "".join(
                    f'<div style="{fw}font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">{h}</div>'
                    for h, fw in _lnd_col_styles
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
                    _lnd_rows_html += (
                        f'<div style="display:flex;align-items:center;padding:10px 14px;border-bottom:1px solid #1e1a08;">'
                        f'<div style="flex:2;color:#c8b890;font-size:0.9rem;white-space:nowrap;">{_lr["Market"]}</div>'
                        f'<div style="flex:1;text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_ind:,}</div>'
                        f'<div style="flex:1;text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_mix:,}</div>'
                        f'<div style="flex:1;text-align:right;color:#c8a040;font-size:0.9rem;letter-spacing:0.04em;">{_res:,}</div>'
                        f'<div style="flex:1;text-align:right;color:#c8a040;font-size:0.9rem;font-weight:700;letter-spacing:0.04em;">{_tot:,}</div>'
                        f'<div style="flex:2;padding-left:8px;">{_bar}</div>'
                        f'</div>'
                    )

                _lnd_table_html = f"""<div style="background:#13110a;border-radius:10px;padding:20px 32px 16px;margin-bottom:8px;">
  <div style="margin-bottom:16px;">{_legend_html}</div>
  <div style="display:flex;align-items:center;padding:10px 14px;border-bottom:1px solid #2a2410;">{_lnd_hcells}</div>
  {_lnd_rows_html}
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">Acreage = entitled or shovel-ready developable land actively available &nbsp;·&nbsp; MIX bar shows Industrial / Mixed-Use / Residential proportion &nbsp;·&nbsp; Source: CoStar Land / CBRE Q1 2025 &nbsp;·&nbsp; Not financial advice.</div>
</div>"""
                st.markdown(_lnd_table_html, unsafe_allow_html=True)
                st.caption(
                    "Entitlement timeline = estimated months from land purchase to permitted/shovel-ready status. "
                    "Markets like New York and Los Angeles require 3-5 years; Sun Belt typically 12-20 months."
                )

        st.markdown(
            '<hr style="border:none;border-top:1px solid #2a2208;margin:32px 0 24px;">',
            unsafe_allow_html=True,
        )
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
            st.markdown(_render_generic_table(
                _ent_df,
                title="Recent Entitlement Filings — Q1 2025",
                count_label=f"{len(_ent_df)} filings",
                hints={
                    "Market":                    {"type": "name", "flex": 1.2},
                    "Acres":                     {"type": "text", "flex": 0.7},
                    "Zoning":                    {"type": "tag",  "flex": 1},
                    "Applicant":                 {"type": "text", "flex": 1},
                    "Est. Months to Shovel-Ready": {"type": "text", "flex": 1.2},
                    "Filed":                     {"type": "text", "flex": 0.8},
                },
            ), unsafe_allow_html=True)
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

            # ── Property-type re-weighting ────────────────────────────────────
            _ms_active_pt = _active_pt()
            _MS_PT_WEIGHTS = {
                "Industrial":  {"migration": 0.25, "vacancy": 0.28, "rent": 0.18, "cap_rate": 0.10, "land": 0.14, "macro": 0.05},
                "Multifamily": {"migration": 0.35, "vacancy": 0.22, "rent": 0.25, "cap_rate": 0.10, "land": 0.03, "macro": 0.05},
                "Office":      {"migration": 0.18, "vacancy": 0.30, "rent": 0.15, "cap_rate": 0.17, "land": 0.05, "macro": 0.15},
                "Retail":      {"migration": 0.28, "vacancy": 0.20, "rent": 0.22, "cap_rate": 0.15, "land": 0.05, "macro": 0.10},
                "Healthcare":  {"migration": 0.30, "vacancy": 0.18, "rent": 0.15, "cap_rate": 0.15, "land": 0.10, "macro": 0.12},
            }

            def _ms_pt_composite(ranking_row, pt):
                f  = ranking_row.get("factors", {})
                w  = _MS_PT_WEIGHTS.get(pt, {})
                if not w:
                    return float(ranking_row.get("composite", 50))
                raw = sum(float(f.get(k, 50)) * wt for k, wt in w.items())
                penalty = float(ranking_row.get("climate_penalty", 0) or 0)
                return round(max(0.0, raw - penalty), 1)

            if _ms_active_pt and _ms_rankings:
                # Compute pt-weighted scores
                _ms_pt_rows = []
                for _r in _ms_rankings:
                    _pts = _ms_pt_composite(_r, _ms_active_pt)
                    _ms_pt_rows.append({**_r, "pt_composite": _pts,
                                        "pt_grade": "A" if _pts >= 80 else ("B+" if _pts >= 70 else ("B" if _pts >= 60 else ("C+" if _pts >= 50 else ("C" if _pts >= 40 else "D"))))})
                _ms_pt_rows.sort(key=lambda x: x["pt_composite"], reverse=True)
                _ms_pt_top3  = [r["market"] for r in _ms_pt_rows[:3]]
                _ms_pt_avoid = [r["market"] for r in _ms_pt_rows[-3:]]
                _ms_pt_avg   = round(sum(r["pt_composite"] for r in _ms_pt_rows) / len(_ms_pt_rows), 1)

                st.markdown(
                    f'<div style="background:#0d1a0d;border:1px solid #2a4a2a;border-radius:8px;'
                    f'padding:10px 16px;margin-bottom:8px;">'
                    f'<span style="color:#4caf50;font-size:0.85rem;">Rankings re-weighted for '
                    f'<b style="color:#a0d0a0;">{_ms_active_pt}</b></span>'
                    f'<span style="color:#3a6a3a;font-size:0.75rem;margin-left:10px;">'
                    + " · ".join(f"{k.title()} {int(v*100)}%" for k, v in _MS_PT_WEIGHTS[_ms_active_pt].items())
                    + f'</span></div>',
                    unsafe_allow_html=True,
                )

                section(f" {_ms_active_pt} Market Rankings")
                _pt_c1, _pt_c2, _pt_c3 = st.columns(3)
                _pt_c1.markdown(metric_card(f"Avg {_ms_active_pt} Score", f"{_ms_pt_avg}/100", "19 tracked markets"), unsafe_allow_html=True)
                _pt_c2.markdown(metric_card("Top Markets", " · ".join(_ms_pt_top3[:2]), f"Best {_ms_active_pt} opportunity"), unsafe_allow_html=True)
                _pt_c3.markdown(metric_card("Avoid", " · ".join(_ms_pt_avoid[:2]), f"Weakest {_ms_active_pt} signals"), unsafe_allow_html=True)

                _ms_pt_df = pd.DataFrame(_ms_pt_rows)
                _ms_pt_colors = ["#66bb6a" if s >= 70 else ("#d4a843" if s >= 50 else "#ef5350") for s in _ms_pt_df["pt_composite"]]
                fig_ms_pt = go.Figure(go.Bar(
                    x=_ms_pt_df["pt_composite"], y=_ms_pt_df["market"], orientation="h",
                    marker_color=_ms_pt_colors,
                    text=[f"{r['pt_grade']}  {r['pt_composite']}" for r in _ms_pt_rows],
                    textposition="outside",
                    customdata=_ms_pt_df[["composite", "market"]].values,
                    hovertemplate="<b>%{y}</b><br>"
                                  + f"{_ms_active_pt} Score: %{{x:.1f}}<br>"
                                  + "Base Score: %{customdata[0]:.1f}<extra></extra>",
                ))
                fig_ms_pt.update_layout(
                    xaxis=dict(range=[0, 110], title=f"{_ms_active_pt} Score (0–100)",
                               gridcolor="#2a2208", color="#8a7040"),
                    yaxis=dict(categoryorder="total ascending", color="#8a7040"),
                    plot_bgcolor="#0d0b04", paper_bgcolor="#0d0b04",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    margin=dict(l=160, r=80, t=10, b=40), height=580,
                )
                st.plotly_chart(fig_ms_pt, use_container_width=True)
                with st.expander("Show base composite rankings", expanded=False):
                    pass  # falls through to the existing chart below

            # ── Summary cards ─────────────────────────────────────────────────
            _base_section_label = "Base Composite Rankings" if _ms_active_pt else " Composite Market Rankings"
            section(_base_section_label)
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

            # Grade badge colors
            def _grade_badge(g):
                _gc = {"A": "#2bbfb0", "B+": "#c8a040", "B": "#7a6830", "C+": "#6a5828", "C": "#4a3820", "D": "#3a2010"}.get(g, "#3a3020")
                _tc = "#fff" if g in ("A",) else "#c8b870"
                return (
                    f'<span style="display:inline-block;background:{_gc};color:{_tc};'
                    f'font-size:0.78rem;font-weight:700;padding:3px 10px;border-radius:5px;'
                    f'letter-spacing:0.04em;">{g}</span>'
                )

            # Header (flex-div) — fixed widths to keep columns aligned in scrollable container
            _mfb_fcols_hdr = "".join(
                f'<div style="width:64px;flex-shrink:0;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;text-align:center;">{lbl.upper()}</div>'
                for lbl, _, _ in _FACTOR_COLS
            )
            _mfb_hdr = (
                f'<div style="display:flex;align-items:center;padding:10px 14px;border-bottom:1px solid #2a2410;min-width:900px;">'
                f'<div style="width:32px;flex-shrink:0;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;text-align:center;">#</div>'
                f'<div style="flex:1;min-width:130px;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">MARKET</div>'
                f'<div style="width:90px;flex-shrink:0;"></div>'
                f'<div style="width:70px;flex-shrink:0;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;text-align:right;">SCORE</div>'
                f'<div style="width:64px;flex-shrink:0;font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;text-align:center;">GRADE</div>'
                f'{_mfb_fcols_hdr}'
                f'<div style="width:80px;flex-shrink:0;font-size:10px;color:#ef5350;letter-spacing:0.08em;text-transform:uppercase;text-align:center;">CLIMATE ADJ.</div>'
                f'</div>'
            )

            # Rows
            _mfb_rows = ""
            for _ri, _rr in enumerate(_ms_rankings[:10]):
                _f       = _rr["factors"]
                _sc      = float(_rr["composite"])
                _penalty = float(_rr.get("climate_penalty", 0) or 0)
                _raw_sc  = float(_rr.get("raw_composite", _sc) or _sc)

                # Mini score bar (80px track)
                _bar_fill = min(_raw_sc / 100 * 80, 80)
                _score_bar = (
                    f'<div style="width:80px;height:6px;background:#2a2410;border-radius:3px;overflow:hidden;">'
                    f'<div style="width:{_bar_fill:.1f}px;height:6px;background:#c8a040;border-radius:3px;"></div>'
                    f'</div>'
                )

                # Composite score cell
                if _penalty > 0:
                    _score_cell = (
                        f'<div style="font-size:1.05rem;font-weight:700;color:#c8a040;">{_sc:.1f}</div>'
                        f'<div style="font-size:0.72rem;color:#ef5350;margin-top:2px;">−{_penalty:.1f} climate</div>'
                    )
                else:
                    _score_cell = f'<div style="font-size:1.05rem;font-weight:700;color:#c8a040;">{_sc:.1f}</div>'

                # Factor cells (flex children)
                _fcells = ""
                for _, _fkey, _fcolor in _FACTOR_COLS:
                    _fv = float(_f.get(_fkey, 0) or 0)
                    _fw = min(_fv / 100 * 44, 44)
                    _fcells += (
                        f'<div style="width:64px;flex-shrink:0;text-align:center;">'
                        f'<div style="font-size:1rem;font-weight:700;color:{_fcolor};letter-spacing:0.02em;">{round(_fv)}</div>'
                        f'<div style="margin:4px auto 0;width:44px;height:3px;background:#2a2410;border-radius:2px;">'
                        f'<div style="width:{_fw:.1f}px;height:3px;background:{_fcolor};border-radius:2px;"></div></div>'
                        f'</div>'
                    )

                # Climate adjustment cell
                _clim_cell = (
                    f'<div style="width:80px;flex-shrink:0;text-align:center;">'
                    + (f'<span style="color:#ef5350;font-size:0.88rem;font-weight:700;">−{_penalty:.1f}</span>' if _penalty > 0 else '<span style="color:#4a4530;font-size:0.88rem;">—</span>')
                    + f'</div>'
                )

                _rank_c = "#c8a040" if _ri < 3 else "#7a7050"
                _mfb_rows += (
                    f'<div style="display:flex;align-items:center;padding:10px 14px;border-bottom:1px solid #1e1a08;min-width:900px;">'
                    f'<div style="width:32px;flex-shrink:0;text-align:center;color:{_rank_c};font-size:0.88rem;">{_rr["rank"]}</div>'
                    f'<div style="flex:1;min-width:130px;color:#c8b870;font-size:0.95rem;font-weight:600;white-space:nowrap;">{_rr["market"]}</div>'
                    f'<div style="width:90px;flex-shrink:0;">{_score_bar}</div>'
                    f'<div style="width:70px;flex-shrink:0;text-align:right;">{_score_cell}</div>'
                    f'<div style="width:64px;flex-shrink:0;text-align:center;">{_grade_badge(_rr["grade"])}</div>'
                    f'{_fcells}'
                    f'{_clim_cell}'
                    f'</div>'
                )

            _mfb_html = f"""<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:16px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">MARKET FACTOR BREAKDOWN &mdash; TOP 10</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Composite investment score by market &mdash; CoStar / CBRE Q1 2025</div>
  </div>
  <div style="font-size:0.78rem;color:#7a7050;font-style:italic;margin-bottom:18px;">Factor scores out of 100. Migration is the primary differentiator across markets. <span style="color:#ef5350;">Climate Adj.</span> = points deducted for high physical climate risk (state score &ge; 60).</div>
  <div style="overflow-x:auto;">{_mfb_hdr}{_mfb_rows}</div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">Scores are composite index values. Not financial advice.</div>
</div>"""
            st.markdown(_mfb_html, unsafe_allow_html=True)

            # ── Download button — market score rankings ───────────────────────
            _ms_dl_rows = []
            for _r in _ms_rankings:
                _f = _r.get("factors", {})
                _ms_dl_rows.append({
                    "Rank": _r.get("rank"),
                    "Market": _r.get("market"),
                    "Composite Score": _r.get("composite"),
                    "Grade": _r.get("grade"),
                    "Migration": _f.get("migration"),
                    "Vacancy": _f.get("vacancy"),
                    "Rent": _f.get("rent"),
                    "Cap Rate": _f.get("cap_rate"),
                    "Land": _f.get("land"),
                    "Macro": _f.get("macro"),
                    "Climate Penalty": _r.get("climate_penalty", 0),
                })
            if _ms_dl_rows:
                _ms_dl_df = pd.DataFrame(_ms_dl_rows)
                st.download_button(
                    "⬇ Download CSV",
                    data=_ms_dl_df.to_csv(index=False),
                    file_name="market_score_rankings.csv",
                    mime="text/csv",
                    key="dl_market_score",
                )

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — CAP RATE MONITOR
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_returns:
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

            # ── Best Markets For Your Type callout ───────────────────────────
            if _cap_focus_col and _cap_mktcaps and _cap_t10y:
                _bm_spreads = []
                for _bm_mkt, _bm_caps in _cap_mktcaps.items():
                    _bm_cr = _bm_caps.get(_cap_focus_col)
                    if _bm_cr:
                        _bm_spread = round(float(_bm_cr) - float(_cap_t10y), 2)
                        _bm_sig    = "ATTRACTIVE" if _bm_spread > 2.5 else ("FAIR" if _bm_spread > 1.5 else "COMPRESSED")
                        _bm_sig_c  = {"ATTRACTIVE": "#66bb6a", "FAIR": "#d4a843", "COMPRESSED": "#ef5350"}[_bm_sig]
                        _bm_spreads.append({"market": _bm_mkt, "cap_rate": _bm_cr, "spread": _bm_spread,
                                            "signal": _bm_sig, "sig_c": _bm_sig_c})
                _bm_spreads.sort(key=lambda x: x["spread"], reverse=True)
                _bm_top5 = _bm_spreads[:5]

                if _bm_top5:
                    _bm_cards = ""
                    for _rank, _bm in enumerate(_bm_top5, 1):
                        _bm_cards += (
                            f'<div style="flex:1;min-width:140px;background:#0d1a0d;border:1px solid #2a4a2a;'
                            f'border-top:3px solid {_bm["sig_c"]};border-radius:8px;padding:14px 12px;">'
                            f'<div style="font-size:0.65rem;color:#4a7a4a;letter-spacing:0.1em;margin-bottom:6px;">#{_rank}</div>'
                            f'<div style="font-size:0.88rem;font-weight:600;color:#c8d8c8;margin-bottom:8px;line-height:1.2;">{_bm["market"]}</div>'
                            f'<div style="font-size:1.1rem;font-weight:700;color:#c8a040;">{_bm["cap_rate"]:.1f}%</div>'
                            f'<div style="font-size:0.75rem;color:#7a9a7a;margin-top:2px;">Spread: {_bm["spread"]:+.2f}pp</div>'
                            f'<div style="font-size:0.68rem;font-weight:700;color:{_bm["sig_c"]};'
                            f'margin-top:6px;letter-spacing:0.08em;">{_bm["signal"]}</div>'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div style="background:#0a120a;border:1px solid #2a4020;border-radius:10px;'
                        f'padding:16px 20px;margin-bottom:16px;">'
                        f'<div style="font-size:0.72rem;color:#4a7a4a;letter-spacing:0.12em;'
                        f'text-transform:uppercase;margin-bottom:12px;">'
                        f'Best Markets for {_cap_focus_col} — Ranked by Cap Rate Spread vs. {_cap_t10y:.2f}% Treasury</div>'
                        f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{_bm_cards}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

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
                st.markdown(_render_generic_table(
                    pd.DataFrame(_sp_rows),
                    title="Treasury Spread Analysis",
                    count_label=f"{len(_sp_rows)} property types",
                    hints={
                        "Property Type": {"type": "name", "flex": 1.2},
                        "Cap Rate":      {"type": "text", "flex": 0.8},
                        "10Y Spread":    {"type": "colored", "flex": 0.8},
                        "Signal":        {"type": "badge", "flex": 1, "badge_map": {
                            "ATTRACTIVE":  "background:#0d2a12;color:#4a9e58",
                            "FAIR":        "background:#2a1a04;color:#a07830",
                            "COMPRESSED":  "background:#2a0d0d;color:#9e4a4a",
                        }},
                    },
                ), unsafe_allow_html=True)
                st.caption("Spread > 2.5pp = Attractive · 1.5–2.5pp = Fair · < 1.5pp = Compressed vs. current 10Y Treasury.")
                st.download_button(
                    "⬇ Download CSV",
                    data=pd.DataFrame(_sp_rows).to_csv(index=False),
                    file_name="cap_rate_spreads.csv",
                    mime="text/csv",
                    key="dl_cap_rate",
                )

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

            # ── Spread Compression / Expansion Trend ─────────────────────────
            _cap_spread_trend = _cap_data.get("spread_trend", [])
            if _cap_spread_trend:
                section(" Spread Trend — Avg Cap Rate vs. 10Y Treasury (6 Months)")
                try:
                    import plotly.graph_objects as _go_cr
                    _st_x = [p["month"] for p in _cap_spread_trend]
                    _st_y = [p["spread"] for p in _cap_spread_trend]
                    # Determine compression/expansion label
                    _latest_spread = _st_y[-1]
                    _three_mo_avg  = sum(_st_y[-3:]) / 3 if len(_st_y) >= 3 else _latest_spread
                    if _latest_spread < _three_mo_avg - 0.05:
                        _trend_lbl = "Compressing"
                        _trend_clr = "#ef5350"
                    elif _latest_spread > _three_mo_avg + 0.05:
                        _trend_lbl = "Expanding"
                        _trend_clr = "#4caf50"
                    else:
                        _trend_lbl = "Stable"
                        _trend_clr = "#d4a843"
                    _fig_spr = _go_cr.Figure()
                    _fig_spr.add_trace(_go_cr.Scatter(
                        x=_st_x, y=_st_y, mode="lines+markers",
                        name="Avg Spread (Cap Rate − 10Y Treasury)",
                        line=dict(color=_trend_clr, width=2),
                        marker=dict(size=6, color=_trend_clr),
                        hovertemplate="%{x}<br>Spread: %{y:.2f}pp<extra></extra>",
                    ))
                    _fig_spr.update_layout(
                        plot_bgcolor="#0f0d06", paper_bgcolor="#16160f",
                        margin=dict(t=30, b=20, l=40, r=120), height=220,
                        xaxis=dict(tickfont=dict(color="#e8dfc4", size=9), gridcolor="#2a2a1a", title=""),
                        yaxis=dict(tickfont=dict(color="#e8dfc4", size=9), gridcolor="#2a2a1a",
                                   title="Spread (pp)", titlefont=dict(color="#a09880"), ticksuffix="pp"),
                        font=dict(family="Source Sans Pro", color="#e8dfc4"),
                        annotations=[dict(
                            text=f"<b>{_trend_lbl}</b>",
                            x=1.01, y=_latest_spread, xref="paper", yref="y",
                            showarrow=False,
                            font=dict(size=13, color=_trend_clr, family="Source Sans Pro"),
                        )],
                    )
                    st.plotly_chart(_fig_spr, use_container_width=True, config={"displayModeBar": False})
                    st.caption(
                        f"Spread = avg national cap rate − 10Y Treasury yield. "
                        f"Current: {_latest_spread:.2f}pp · 3-month avg: {_three_mo_avg:.2f}pp · "
                        f"Trend: **{_trend_lbl}** "
                        f"(Compressing = spread narrowing = CRE pricing less attractive vs. treasuries)"
                    )
                except Exception as _spr_err:
                    st.caption(f"Spread trend chart unavailable: {_spr_err}")

        st.markdown(
            '<hr style="border:none;border-top:1px solid #2a2208;margin:32px 0 24px;">',
            unsafe_allow_html=True,
        )
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

            # ── One-line insight for focused type ─────────────────────────────
            if _rg_pt_focus and _rg_focus_key and _rg_market:
                # Build ranked list of markets by focused type's rent growth
                _rg_insight_rows = []
                for _im, _id in _rg_market.items():
                    _iv = _id.get(_rg_focus_key)
                    if _iv is not None:
                        _rg_insight_rows.append((_im, float(_iv)))
                _rg_insight_rows.sort(key=lambda x: x[1], reverse=True)

                if _rg_insight_rows:
                    _ig_top_mkt, _ig_top_val = _rg_insight_rows[0]
                    _ig_nat_val = (_rg_national.get(_rg_pt_focus) or {}).get("yoy_pct", 0)
                    _ig_rank_of = len(_rg_insight_rows)
                    _ig_vs_nat  = round(_ig_top_val - _ig_nat_val, 1)
                    _ig_color   = "#66bb6a" if _ig_top_val > 0 else "#ef5350"
                    _ig_vs_c    = "#66bb6a" if _ig_vs_nat > 0 else "#ef5350"
                    _ig_unit    = "%" if _rg_pt_focus == "Multifamily" else "% PSF"
                    st.markdown(
                        f'<div style="background:#0d1a0d;border:1px solid #2a4a2a;border-radius:8px;'
                        f'padding:12px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px;'
                        f'flex-wrap:wrap;">'
                        f'<span style="font-size:1.4rem;font-weight:700;color:{_ig_color};">'
                        f'{_ig_top_val:+.1f}{_ig_unit}</span>'
                        f'<span style="color:#c8d8c8;font-size:0.9rem;">'
                        f'<b>{_ig_top_mkt}</b> leads {_rg_pt_focus} rent growth — '
                        f'<span style="color:{_ig_vs_c};">{_ig_vs_nat:+.1f}pp vs. national avg</span>'
                        f'</span>'
                        f'<span style="margin-left:auto;font-size:0.75rem;color:#4a7a4a;">'
                        f'#1 of {_ig_rank_of} tracked markets</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Bottom market insight
                    _ig_bot_mkt, _ig_bot_val = _rg_insight_rows[-1]
                    _ig_bot_c = "#ef5350" if _ig_bot_val < 0 else "#d4a843"
                    st.markdown(
                        f'<div style="background:#1a0d0d;border:1px solid #4a2a2a;border-radius:8px;'
                        f'padding:10px 18px;margin-bottom:12px;font-size:0.82rem;color:#c8a8a8;">'
                        f'Weakest: <b>{_ig_bot_mkt}</b> at '
                        f'<span style="color:{_ig_bot_c};font-weight:700;">{_ig_bot_val:+.1f}{_ig_unit}</span>'
                        f' — #{_ig_rank_of} of {_ig_rank_of} markets'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

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
                # Build downloadable rent growth DataFrame
                _rg_dl_rows = []
                for _m in _rg_mkts:
                    _rd = _rg_market[_m]
                    _rg_dl_rows.append({
                        "Market":         _m,
                        "Multifamily %":  _rd.get("multifamily", ""),
                        "Industrial PSF %": _rd.get("industrial_psf", ""),
                        "Office PSF %":   _rd.get("office_psf", ""),
                        "Retail PSF %":   _rd.get("retail_psf", ""),
                    })
                st.download_button(
                    "⬇ Download CSV",
                    data=pd.DataFrame(_rg_dl_rows).to_csv(index=False),
                    file_name="rent_growth_by_market.csv",
                    mime="text/csv",
                    key="dl_rent_growth",
                )

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

                # Header (flex-div) — scrollable for narrow screens
                _ti_col_styles = [
                    ("STATE",     "width:70px;flex-shrink:0;text-align:center;"),
                    ("PROGRAM",   "flex:2;min-width:160px;"),
                    ("BENEFIT",   "flex:3;min-width:180px;"),
                    ("CRE TYPES", "flex:2;min-width:150px;"),
                    ("CAP",       "flex:1;min-width:120px;"),
                ]
                _ti_hcells = "".join(
                    f'<div style="{fw}font-size:10px;color:#4a3e18;letter-spacing:0.08em;text-transform:uppercase;">{h}</div>'
                    for h, fw in _ti_col_styles
                )

                # Rows
                _ti_rows_html = ""
                for _ti_abbr, _ti_si in _ti_filtered.items():
                    _ti_fallback = _TI_CHIP["Mixed-Use"]
                    _ti_type_chips = "".join(
                        f'<span style="{_chip_css}{_TI_CHIP.get(t, _ti_fallback)}">{t}</span>'
                        for t in _ti_si["cre_types"]
                    )
                    _ti_rows_html += (
                        f'<div style="display:flex;align-items:flex-start;padding:12px 16px;border-bottom:1px solid #1e1a08;">'
                        f'<div style="width:70px;flex-shrink:0;text-align:center;padding-top:2px;">'
                        f'<span style="display:inline-block;background:#2a2410;border:1px solid #3a3020;'
                        f'color:#c8a040;font-size:0.8rem;font-weight:700;padding:4px 8px;border-radius:5px;'
                        f'letter-spacing:0.06em;">{_ti_abbr}</span></div>'
                        f'<div style="flex:2;min-width:160px;color:#c8b870;font-size:0.92rem;font-weight:600;padding-right:12px;">{_ti_si["program"]}</div>'
                        f'<div style="flex:3;min-width:180px;color:#9a9070;font-size:0.86rem;line-height:1.5;padding-right:12px;">{_ti_si["benefit"]}</div>'
                        f'<div style="flex:2;min-width:150px;padding-right:12px;">{_ti_type_chips}</div>'
                        f'<div style="flex:1;min-width:120px;color:#c8a040;font-size:0.86rem;font-family:monospace;white-space:nowrap;">{_ti_si["cap"]}</div>'
                        f'</div>'
                    )

                _ti_html = f"""<div style="background:#13110a;border-radius:10px;padding:28px 32px 20px;margin-bottom:8px;">
  <div style="border-left:4px solid #c8a040;padding-left:14px;margin-bottom:20px;">
    <div style="font-size:1.15rem;font-weight:700;color:#c8a040;letter-spacing:0.06em;">STATE CRE TAX INCENTIVE PROGRAMS</div>
    <div style="font-size:0.8rem;color:#7a7050;margin-top:3px;">Federal &amp; state-level incentives by property type &mdash; IRS / State Policy 2024&ndash;25</div>
  </div>
  <div style="font-size:0.8rem;color:#7a7050;margin-bottom:16px;">Showing {_ti_count} program{"s" if _ti_count != 1 else ""}</div>
  <div style="overflow-x:auto;"><div style="display:flex;align-items:center;padding:10px 16px;border-bottom:1px solid #2a2410;min-width:700px;">{_ti_hcells}</div><div style="min-width:700px;">{_ti_rows_html}</div></div>
  <div style="margin-top:14px;font-size:0.75rem;color:#4a4530;">Source: IRS, HUD Opportunity Zone designations, state economic development agencies. Not financial or legal advice. Consult a tax advisor.</div>
</div>"""
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

            st.markdown(_render_generic_table(
                tbl_df,
                title="Metro Climate Risk Scores",
                count_label=f"{len(tbl_df)} markets",
                scrollable=True, max_height=480,
                hints={
                    "Metro":         {"type": "name",      "flex": 2},
                    "State":         {"type": "text",      "flex": 0.6},
                    "Score":         {"type": "score_bar", "flex": 1},
                    "Risk Level":    {"type": "badge",     "flex": 1, "badge_map": {
                        "Low":      "background:#0d2a12;color:#4a9e58",
                        "Moderate": "background:#2a1a04;color:#a07830",
                        "High":     "background:#2a0d0d;color:#ef5350",
                        "Severe":   "background:#2a0d2a;color:#ce93d8",
                        "Extreme":  "background:#1a0d2a;color:#ce93d8",
                    }},
                    "Flood":         {"type": "score_bar", "flex": 0.8},
                    "Wildfire":      {"type": "score_bar", "flex": 0.8},
                    "Heat":          {"type": "score_bar", "flex": 0.8},
                    "Wind":          {"type": "score_bar", "flex": 0.8},
                    "Sea Level Rise":{"type": "score_bar", "flex": 0.9},
                },
            ), unsafe_allow_html=True)
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

    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — SUPPLY PIPELINE (Building Permits)
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_supply_pipeline:
        st.markdown("#### How much new supply is entering each market?")
        st.markdown(
            "Building permits are a **leading indicator** of future supply — typically 12–24 months before "
            "units hit the market. High permit volumes in a market signal incoming supply pressure that "
            "may compress rents and occupancy. Low permit volumes indicate constrained supply, supporting "
            "rent growth. Data sourced from FRED (U.S. Census Bureau Building Permits Survey)."
        )
        agent_last_updated("building_permits")

        cache_bp = read_cache("building_permits")
        _bp_data = (cache_bp.get("data") or {}) if cache_bp else {}

        if not _bp_data:
            st.info("📡 Building permits agent is fetching data — refresh in ~30 seconds.")
        else:
            _bp_markets    = _bp_data.get("markets", [])
            _bp_nat_trend  = _bp_data.get("national_trend", [])
            _bp_top        = _bp_data.get("top_supply_markets", [])
            _bp_low        = _bp_data.get("low_supply_markets", [])

            # ── KPI row ───────────────────────────────────────────────────────
            _bp_high_count = sum(1 for m in _bp_markets if m["supply_pressure"] == "HIGH")
            _bp_nat_latest = _bp_nat_trend[-1]["value"] if _bp_nat_trend else 0
            _bp_top_market = _bp_top[0] if _bp_top else "—"

            _kpi1, _kpi2, _kpi3, _kpi4 = st.columns(4)
            _kpi1.markdown(metric_card("Markets Tracked", str(len(_bp_markets)), "building permit series"), unsafe_allow_html=True)
            _kpi2.markdown(metric_card("High Supply Markets", str(_bp_high_count), ">20,000 permits/12mo"), unsafe_allow_html=True)
            _kpi3.markdown(metric_card("National Permits (Latest)", f"{int(_bp_nat_latest):,}", "total US monthly (000s)"), unsafe_allow_html=True)
            _kpi4.markdown(metric_card("Top Supply Market", _bp_top_market, "highest 12mo permit volume"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── National trend line chart ─────────────────────────────────────
            if _bp_nat_trend:
                section(" US National Building Permits Trend")
                _nat_df = pd.DataFrame(_bp_nat_trend)
                fig_nat = go.Figure(go.Scatter(
                    x=_nat_df["date"],
                    y=_nat_df["value"],
                    mode="lines+markers",
                    name="US Total Permits",
                    line=dict(color=GOLD, width=2),
                    marker=dict(size=5, color=GOLD),
                    hovertemplate="<b>%{x}</b><br>Permits: %{y:,.0f} (000s)<extra></extra>",
                ))
                fig_nat.update_layout(
                    title=dict(text="US Total Building Permits (Monthly, 000s)", font=dict(color="#c8b890", size=13)),
                    plot_bgcolor="#1e1a0a",
                    paper_bgcolor="#1a1208",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    xaxis=dict(gridcolor="#2a2208", color="#8a7040"),
                    yaxis=dict(gridcolor="#2a2208", color="#8a7040", title="Permits (000s)"),
                    margin=dict(t=40, b=40, l=60, r=20),
                    height=300,
                )
                st.plotly_chart(fig_nat, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Market supply pressure table ──────────────────────────────────
            if _bp_markets:
                section(" Market Supply Pressure Rankings")
                _bp_df = pd.DataFrame(_bp_markets)
                _bp_display = _bp_df[["market", "permits_12mo", "yoy_change_pct", "supply_pressure", "trend"]].copy()
                _bp_display.columns = ["Market", "Permits (12mo)", "YoY Change", "Supply Pressure", "Trend"]
                _bp_display["Permits (12mo)"] = _bp_display["Permits (12mo)"].apply(lambda x: f"{int(x):,}")
                _bp_display["YoY Change"] = _bp_display["YoY Change"].apply(lambda x: f"{x:+.1f}%")
                _bp_display = _bp_display.sort_values("Supply Pressure", ascending=True)

                st.markdown(_render_generic_table(
                    _bp_display,
                    title="Market Building Permit Supply Pressure",
                    count_label=f"{len(_bp_display)} markets",
                    hints={
                        "Market":          {"type": "name",    "flex": 1.8},
                        "Permits (12mo)":  {"type": "text",    "flex": 1.2},
                        "YoY Change":      {"type": "colored", "flex": 1},
                        "Supply Pressure": {"type": "badge",   "flex": 1.2, "badge_map": {
                            "HIGH":     "background:#2a0d0d;color:#ef5350",
                            "MODERATE": "background:#2a1a04;color:#d4a843",
                            "LOW":      "background:#0d2a12;color:#4a9e58",
                        }},
                        "Trend": {"type": "badge", "flex": 1.2, "badge_map": {
                            "ACCELERATING": "background:#2a0d0d;color:#ef5350",
                            "STABLE":       "background:#2a1a04;color:#d4a843",
                            "DECELERATING": "background:#0d2a12;color:#4a9e58",
                        }},
                    },
                    scrollable=True, max_height=480,
                ), unsafe_allow_html=True)

                st.download_button(
                    "⬇ Download CSV",
                    data=_bp_display.to_csv(index=False),
                    file_name="building_permits_supply_pressure.csv",
                    mime="text/csv",
                    key="dl_building_permits",
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Horizontal bar chart — top 10 markets ─────────────────────
                section(" Top 10 Markets by 12-Month Permit Volume")
                _bp_sorted = sorted(_bp_markets, key=lambda x: x["permits_12mo"], reverse=True)[:10]
                _bp_bar_colors = {
                    "HIGH":     "#ef5350",
                    "MODERATE": GOLD,
                    "LOW":      "#4a9e58",
                }
                fig_bar = go.Figure(go.Bar(
                    x=[m["permits_12mo"] for m in _bp_sorted],
                    y=[m["market"] for m in _bp_sorted],
                    orientation="h",
                    marker_color=[_bp_bar_colors.get(m["supply_pressure"], GOLD) for m in _bp_sorted],
                    text=[f"{m['permits_12mo']:,}" for m in _bp_sorted],
                    textposition="outside",
                    textfont=dict(color="#c8b890", size=11),
                    hovertemplate="<b>%{y}</b><br>12mo Permits: %{x:,}<extra></extra>",
                ))
                fig_bar.update_layout(
                    plot_bgcolor="#1e1a0a",
                    paper_bgcolor="#1a1208",
                    font=dict(family="Source Sans Pro", color="#c8b890"),
                    xaxis=dict(gridcolor="#2a2208", color="#8a7040", title="12-Month Permits"),
                    yaxis=dict(categoryorder="total ascending", color="#8a7040"),
                    margin=dict(l=180, r=100, t=20, b=40),
                    height=380,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption(
                    "Color: RED = High Supply (>20K permits/12mo) · GOLD = Moderate (8K–20K) · GREEN = Low (<8K). "
                    "High supply markets may face rent compression 12–24 months ahead."
                )

            with st.expander("About Building Permit Data"):
                st.markdown("""
**Building Permits** are issued by local governments before construction begins and are the earliest
available signal of incoming housing supply. The U.S. Census Bureau Building Permits Survey (BPS)
collects this data monthly.

**Why permits matter for CRE:**
- Industrial parks, warehouses, and multifamily projects all require building permits
- High permit volumes today = supply hitting the market 12–24 months from now
- Supply waves compress vacancy rates and limit rent growth for existing owners
- Low-permit markets (constrained geography or regulation) maintain higher rents

**How to use this data:**
- **HIGH supply pressure** markets (>20K permits/12mo): Underwrite conservative rent growth assumptions; focus on value-add plays vs. core
- **MODERATE** markets: Balanced supply/demand; standard underwriting applies
- **LOW supply** markets: Landlords have pricing power; supports premium cap rate compression

**Data Source:** Federal Reserve Economic Data (FRED), U.S. Census Bureau BPS. National series: PERMIT.
Updated every 24 hours via Agent 23.
""")


