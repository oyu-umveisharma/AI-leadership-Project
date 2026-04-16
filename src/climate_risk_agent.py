"""
Climate Risk Agent
==================
Tracks climate hazard risk for US CRE markets.

Data Sources:
  - OpenFEMA Disaster Declarations API (no key): flood, hurricane/wind, wildfire
    declaration counts per state (2015–present)
  - NIFC WFIGS ArcGIS API (no key): wildfire acres burned per state (2019–present)
  - Static NOAA 1991–2020 Climate Normals: extreme heat baseline by state
  - Static NOAA 2022 Sea Level Rise Technical Report: coastal exposure by state

Composite risk score 0–100 with labeled bands:
  Low (0–25) | Moderate (26–50) | High (51–75) | Severe (76–100)

Factor weights:
  Flood 25% | Wildfire 20% | Extreme Heat 20% | Wind/Hurricane 20% | Sea Level Rise 15%

Geographic levels:
  - State:  all 50 states + DC
  - Metro:  40 major US metros (state score + local adjustment)

Cache: cache/climate_risk.json
Schedule: Every 24 hours
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Factor weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHTS = {
    "flood":     0.25,
    "wildfire":  0.20,
    "heat":      0.20,
    "wind":      0.20,
    "sea_level": 0.15,
}

WEIGHT_LABELS = {
    "flood":     "Flood Risk",
    "wildfire":  "Wildfire Risk",
    "heat":      "Extreme Heat",
    "wind":      "Wind / Hurricane",
    "sea_level": "Sea Level Rise",
}

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# ── Score helpers ─────────────────────────────────────────────────────────────

def score_label(score: float) -> str:
    if score >= 76:
        return "Severe"
    if score >= 51:
        return "High"
    if score >= 26:
        return "Moderate"
    return "Low"


def label_color(label: str) -> str:
    return {
        "Low":      "#4caf50",
        "Moderate": "#ff9800",
        "High":     "#f44336",
        "Severe":   "#7b1fa2",
    }.get(label, "#888888")


# ── Static: Extreme Heat (NOAA 1991–2020 Climate Normals, mean annual TMAX °F) ──
# Source: NOAA Climate Normals https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
HEAT_TMAX = {
    "FL": 82.1, "HI": 82.4, "LA": 79.2, "TX": 76.8, "AZ": 75.9,
    "GA": 74.3, "AL": 73.8, "MS": 73.5, "SC": 73.0, "AR": 71.5,
    "OK": 71.4, "NC": 70.2, "TN": 68.9, "NV": 68.9, "NM": 68.3,
    "KY": 67.2, "MO": 67.8, "KS": 67.5, "VA": 66.5, "DC": 66.2,
    "CA": 65.5, "MD": 64.8, "NE": 64.3, "DE": 64.2, "IL": 63.5,
    "IN": 63.2, "NJ": 63.1, "OH": 62.8, "WV": 62.3, "UT": 62.1,
    "IA": 61.1, "CO": 60.2, "PA": 60.5, "NY": 59.8, "SD": 59.2,
    "MI": 58.9, "ID": 58.5, "OR": 58.2, "CT": 58.1, "MA": 57.2,
    "RI": 57.5, "WA": 57.1, "WY": 56.8, "WI": 56.4, "ND": 55.1,
    "MT": 55.0, "NH": 54.8, "MN": 54.2, "VT": 53.2, "ME": 52.1,
    "AK": 39.5,
}
_HEAT_MIN = min(HEAT_TMAX.values())
_HEAT_MAX = max(HEAT_TMAX.values())


def _heat_score(state: str) -> float:
    t = HEAT_TMAX.get(state, _HEAT_MIN)
    return round((t - _HEAT_MIN) / (_HEAT_MAX - _HEAT_MIN) * 100, 1)


# ── Static: Sea Level Rise (NOAA 2022 SLR Technical Report, Intermediate-High) ─
# Source: https://oceanservice.noaa.gov/hazards/sealevelrise/sealevelrise-tech-report.html
# Inland states = 0; coastal states scaled by projected 2050 rise (inches)
SEA_LEVEL_SCORES = {
    "FL": 95, "LA": 95, "MS": 70, "AL": 65, "TX": 80,
    "GA": 65, "SC": 70, "NC": 75, "VA": 72, "MD": 72,
    "DE": 70, "NJ": 68, "NY": 65, "CT": 60, "RI": 60,
    "MA": 58, "NH": 48, "ME": 45, "DC": 50,
    "CA": 55, "OR": 35, "WA": 40, "HI": 75, "AK": 30,
}


def _sea_score(state: str) -> float:
    return float(SEA_LEVEL_SCORES.get(state, 0))


# ── Metro-level risk adjustments (delta vs state score per factor) ────────────
# Positive = higher local risk than state avg; negative = lower
METRO_ADJUSTMENTS = {
    "Miami":          {"flood": 20,  "wind": 20,  "sea_level": 20, "heat": 10},
    "New Orleans":    {"flood": 25,  "wind": 15,  "sea_level": 25},
    "Houston":        {"flood": 20,  "wind": 10,  "sea_level": 10},
    "Tampa":          {"flood": 15,  "wind": 15,  "sea_level": 15},
    "Jacksonville":   {"flood": 10,  "wind": 10,  "sea_level": 10},
    "Orlando":        {"flood": 5,   "wind": 5},
    "Phoenix":        {"heat": 20,   "wildfire": 10},
    "Las Vegas":      {"heat": 15},
    "Los Angeles":    {"wildfire": 20, "wind": 5},
    "San Diego":      {"wildfire": 15, "sea_level": 5},
    "San Francisco":  {"wildfire": 5,  "sea_level": 10},
    "Sacramento":     {"wildfire": 20, "heat": 10},
    "Seattle":        {"flood": 5,   "wildfire": -5},
    "Portland":       {"wildfire": 10},
    "Denver":         {"wildfire": 10},
    "Dallas":         {"wind": 5,    "heat": 5},
    "Austin":         {"heat": 5,    "flood": 5},
    "San Antonio":    {"heat": 10},
    "Chicago":        {"wind": 10},
    "New York City":  {"flood": 10,  "wind": 5,   "sea_level": 10},
    "Boston":         {"sea_level": 5, "flood": 5},
    "Washington DC":  {"flood": 5},
    "Atlanta":        {"heat": 5},
    "Charlotte":      {},
    "Nashville":      {"flood": 5},
    "Memphis":        {"flood": 5},
    "Kansas City":    {},
    "Minneapolis":    {},
    "Detroit":        {},
    "Cleveland":      {},
    "Pittsburgh":     {},
    "Indianapolis":   {},
    "Columbus":       {},
    "Cincinnati":     {},
    "Louisville":     {"flood": 5},
    "St. Louis":      {"flood": 5},
    "Salt Lake City": {"wildfire": 5},
    "Raleigh":        {},
    "Richmond":       {"flood": 5},
    "Virginia Beach": {"flood": 15,  "wind": 10,  "sea_level": 20},
    "Norfolk":        {"flood": 15,  "wind": 10,  "sea_level": 20},
}

METRO_STATE = {
    "Miami": "FL",           "New Orleans": "LA",      "Houston": "TX",
    "Tampa": "FL",           "Jacksonville": "FL",     "Orlando": "FL",
    "Phoenix": "AZ",         "Las Vegas": "NV",        "Los Angeles": "CA",
    "San Diego": "CA",       "San Francisco": "CA",    "Sacramento": "CA",
    "Seattle": "WA",         "Portland": "OR",         "Denver": "CO",
    "Dallas": "TX",          "Austin": "TX",           "San Antonio": "TX",
    "Chicago": "IL",         "New York City": "NY",    "Boston": "MA",
    "Washington DC": "DC",   "Atlanta": "GA",          "Charlotte": "NC",
    "Nashville": "TN",       "Memphis": "TN",          "Kansas City": "MO",
    "Minneapolis": "MN",     "Detroit": "MI",          "Cleveland": "OH",
    "Pittsburgh": "PA",      "Indianapolis": "IN",     "Columbus": "OH",
    "Cincinnati": "OH",      "Louisville": "KY",       "St. Louis": "MO",
    "Salt Lake City": "UT",  "Raleigh": "NC",          "Richmond": "VA",
    "Virginia Beach": "VA",  "Norfolk": "VA",
}

# ── Property-type risk context (for personalization insight prompt) ────────────
PROPERTY_RISK_CONTEXT = {
    "Industrial":    "flood exposure (ground-level ops), wind damage to large roof spans, and wildfire proximity to logistics corridors",
    "Office":        "business continuity disruption, flood damage to lower floors, and extreme heat impact on HVAC costs",
    "Retail":        "flood damage to parking lots and storefronts, wind damage to signage and facades, and heat-driven foot traffic decline",
    "Multifamily":   "flood displacement risk, hurricane/wind damage to units, extreme heat driving utility costs, and insurance premium escalation",
    "Data Center":   "extreme heat driving cooling costs, flood risk to critical equipment, and power reliability during extreme weather events",
    "Healthcare":    "all-hazard business continuity risk, flood access disruption, and storm-related patient surge",
    "Hospitality":   "hurricane/wind structural damage, flood displacement of guests, and climate-driven tourism pattern shifts",
    "Self-Storage":  "flood damage to contents, moderate wind risk, and lower operational exposure vs. other property types",
    "Mixed-Use":     "multi-factor exposure combining retail flood risk, residential wind risk, and rising insurance costs in high-risk markets",
}


# ── FEMA OpenFEMA Disaster Declarations API ───────────────────────────────────
FEMA_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

# Map FEMA incident types to our 5 risk factors
INCIDENT_TO_FACTOR = {
    "Flood":               "flood",
    "Coastal Storm":       "flood",
    "Dam/Levee Break":     "flood",
    "Tsunami":             "flood",
    "Hurricane":           "wind",
    "Typhoon":             "wind",
    "Severe Storm":        "wind",
    "Tornado":             "wind",
    "High Wind":           "wind",
    "Tropical Storm":      "wind",
    "Straight-Line Winds": "wind",
    "Fire":                "wildfire",
}


def fetch_fema_declarations(start_year: int = 2015) -> list:
    """Fetch all FEMA disaster declarations since start_year. Paginated."""
    print(f"[Climate Risk] Fetching FEMA declarations since {start_year}...")
    records = []
    skip = 0
    top = 1000
    while True:
        params = {
            "$filter": f"declarationDate ge '{start_year}-01-01T00:00:00.000z'",
            "$select": "state,incidentType,declarationDate",
            "$top": top,
            "$skip": skip,
            "$format": "json",
        }
        try:
            r = requests.get(FEMA_URL, params=params, timeout=30)
            r.raise_for_status()
            batch = r.json().get("DisasterDeclarationsSummaries", [])
            if not batch:
                break
            records.extend(batch)
            print(f"  [FEMA] {len(records)} records fetched...")
            if len(batch) < top:
                break
            skip += top
            time.sleep(0.4)
        except Exception as e:
            print(f"  [FEMA] Error at skip={skip}: {e}")
            break
    print(f"[Climate Risk] FEMA total: {len(records)} records")
    return records


# ── NIFC Wildfire Data (WFIGS ArcGIS) ────────────────────────────────────────
NIFC_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
    "/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)


def fetch_wildfire_acres(start_year: int = 2019) -> dict:
    """Fetch total wildfire acres burned per state via NIFC WFIGS ArcGIS API."""
    print("[Climate Risk] Fetching NIFC wildfire acres...")
    state_acres = defaultdict(float)
    # Date filter using the polygon creation date as proxy for fire year
    start_date = f"{start_year}-01-01"
    params = {
        "where": f"poly_CreateDate >= DATE '{start_date}'",
        "outStatistics": json.dumps([{
            "statisticType": "sum",
            "onStatisticField": "poly_GISAcres",
            "outStatisticFieldName": "total_acres",
        }]),
        "groupByFieldsForStatistics": "attr_POOState",
        "returnGeometry": "false",
        "f": "json",
    }
    try:
        r = requests.get(NIFC_URL, params=params, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", [])
        for feat in features:
            attrs = feat.get("attributes", {})
            raw_state = (attrs.get("attr_POOState") or "").strip().upper()
            # NIFC returns "US-CA" format; strip the "US-" prefix
            state = raw_state.replace("US-", "") if raw_state.startswith("US-") else raw_state
            acres = attrs.get("total_acres") or 0
            if state and len(state) == 2 and acres:
                state_acres[state] += float(acres)
        print(f"[Climate Risk] NIFC: {len(state_acres)} states with wildfire data")
    except Exception as e:
        print(f"[Climate Risk] NIFC unavailable ({e}), falling back to FEMA fire declarations")
    return dict(state_acres)


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(values: dict, cap_pct: float = 95) -> dict:
    """Normalize state→float to 0–100. Caps at 95th percentile to reduce outlier distortion."""
    if not values:
        return {}
    sorted_vals = sorted(v for v in values.values() if v > 0)
    if not sorted_vals:
        return {k: 0.0 for k in values}
    cap_idx = max(0, int(len(sorted_vals) * cap_pct / 100) - 1)
    cap_val = sorted_vals[cap_idx]
    if cap_val == 0:
        return {k: 0.0 for k in values}
    return {k: round(min(100.0, v / cap_val * 100), 1) for k, v in values.items()}


# ── Core Scoring ──────────────────────────────────────────────────────────────

def compute_state_scores(fema_records: list, wildfire_acres: dict) -> dict:
    """Compute per-state factor scores and composite score."""
    flood_counts    = defaultdict(int)
    wind_counts     = defaultdict(int)
    fema_fire_counts = defaultdict(int)
    yearly_events   = defaultdict(lambda: defaultdict(int))  # state → year → count

    for rec in fema_records:
        state   = rec.get("state", "")
        incident = rec.get("incidentType", "")
        date    = rec.get("declarationDate", "")
        year    = date[:4] if date and len(date) >= 4 else ""
        factor  = INCIDENT_TO_FACTOR.get(incident)

        if factor == "flood":
            flood_counts[state] += 1
        elif factor == "wind":
            wind_counts[state] += 1
        elif factor == "wildfire":
            fema_fire_counts[state] += 1

        if factor and year:
            yearly_events[state][year] += 1

    # Wildfire: NIFC acres (primary) blended with FEMA fire declarations (fallback)
    wildfire_raw = {}
    for state in ALL_STATES:
        acres = wildfire_acres.get(state, 0)
        fema_fires = fema_fire_counts.get(state, 0)
        if acres > 0:
            wildfire_raw[state] = acres
        else:
            # Use FEMA fire declarations scaled to rough acres equivalent
            wildfire_raw[state] = fema_fires * 50_000

    flood_norm    = _normalize(flood_counts)
    wind_norm     = _normalize(wind_counts)
    wildfire_norm = _normalize(wildfire_raw)

    state_scores = {}
    for state in ALL_STATES:
        flood_s    = flood_norm.get(state, 0)
        wildfire_s = wildfire_norm.get(state, 0)
        heat_s     = _heat_score(state)
        wind_s     = wind_norm.get(state, 0)
        sea_s      = _sea_score(state)

        composite = round(
            flood_s    * WEIGHTS["flood"]    +
            wildfire_s * WEIGHTS["wildfire"] +
            heat_s     * WEIGHTS["heat"]     +
            wind_s     * WEIGHTS["wind"]     +
            sea_s      * WEIGHTS["sea_level"],
            1,
        )
        label = score_label(composite)

        state_scores[state] = {
            "state":           state,
            "composite_score": composite,
            "label":           label,
            "label_color":     label_color(label),
            "factors": {
                "flood":     round(flood_s, 1),
                "wildfire":  round(wildfire_s, 1),
                "heat":      round(heat_s, 1),
                "wind":      round(wind_s, 1),
                "sea_level": round(sea_s, 1),
            },
            "trend": _year_trend(yearly_events.get(state, {})),
        }

    return state_scores


def _year_trend(year_counts: dict) -> list:
    """Year-by-year FEMA disaster event counts for trend charts."""
    years = [str(y) for y in range(2018, datetime.now().year + 1)]
    return [{"year": y, "events": year_counts.get(y, 0)} for y in years]


def compute_metro_scores(state_scores: dict) -> list:
    """Derive metro-level scores by applying local adjustments to state scores."""
    metros = []
    for metro, state in METRO_STATE.items():
        if state not in state_scores:
            continue
        st  = state_scores[state]
        adj = METRO_ADJUSTMENTS.get(metro, {})

        factors = {}
        for factor in WEIGHTS:
            base  = st["factors"][factor]
            delta = adj.get(factor, 0)
            factors[factor] = round(min(100.0, max(0.0, base + delta)), 1)

        composite = round(
            sum(factors[f] * w for f, w in WEIGHTS.items()), 1
        )
        label = score_label(composite)

        metros.append({
            "metro":           metro,
            "state":           state,
            "composite_score": composite,
            "label":           label,
            "label_color":     label_color(label),
            "factors":         factors,
            "trend":           st["trend"],
        })

    metros.sort(key=lambda x: x["composite_score"], reverse=True)
    return metros


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_climate_risk_agent() -> dict:
    print("=" * 60)
    print("[Climate Risk] Starting run...")
    print("=" * 60)

    fema_records   = fetch_fema_declarations(start_year=2015)
    wildfire_acres = fetch_wildfire_acres(start_year=2019)

    state_scores = compute_state_scores(fema_records, wildfire_acres)
    metro_scores = compute_metro_scores(state_scores)

    sorted_states = sorted(
        state_scores.values(), key=lambda x: x["composite_score"], reverse=True
    )

    result = {
        "agent_name":         "climate_risk",
        "timestamp":          datetime.now().isoformat(),
        "weights":            WEIGHTS,
        "weight_labels":      WEIGHT_LABELS,
        "states":             state_scores,
        "metros":             metro_scores,
        "top_risk_states":    [{"state": s["state"], "score": s["composite_score"], "label": s["label"]} for s in sorted_states[:10]],
        "lowest_risk_states": [{"state": s["state"], "score": s["composite_score"], "label": s["label"]} for s in sorted_states[-10:]],
        "data_sources": [
            "OpenFEMA Disaster Declarations API (2015–present)",
            "NIFC WFIGS Wildfire Perimeters (2019–present)",
            "NOAA 1991–2020 Climate Normals (heat baseline)",
            "NOAA 2022 Sea Level Rise Technical Report (coastal exposure)",
        ],
    }

    top = result["top_risk_states"][0]
    print(f"[Climate Risk] Done. Top risk state: {top['state']} ({top['score']} — {top['label']})")
    print("=" * 60)
    return result


if __name__ == "__main__":
    result = run_climate_risk_agent()
    print("\n--- Top 5 Riskiest States ---")
    for s in result["top_risk_states"][:5]:
        print(f"  {s['state']}: {s['score']:.1f} ({s['label']})")
    print("\n--- Top 5 Safest States ---")
    for s in result["lowest_risk_states"][:5]:
        print(f"  {s['state']}: {s['score']:.1f} ({s['label']})")
