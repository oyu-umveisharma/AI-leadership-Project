"""
CRE Investment Recommendation Engine
=====================================
Synthesizes all platform data sources into a personalized investment brief.

Flow:
  1. parse_prompt(text)           → extract structured params via Groq (regex fallback)
  2. resolve_markets(params)      → candidate markets from location string
  3. gather_market_data(markets)  → pull from all caches
  4. get_weights(params)          → AI-determined factor weights (default fallback)
  5. score_markets(data, weights) → composite opportunity score per market
  6. estimate_financials(market)  → cost, NOI, profit, buildout timeline
  7. generate_narrative(...)      → Groq 3-paragraph investment rationale
  8. build_recommendation(...)    → full structured output dict
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"

# ── Cache reader ──────────────────────────────────────────────────────────────

def _read(key: str) -> dict:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("data") or d
    except Exception:
        return {}

# ── Groq client ───────────────────────────────────────────────────────────────

def _groq_client():
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("your_"):
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None

def _groq_complete(system: str, user: str, max_tokens: int = 1200) -> str:
    client = _groq_client()
    if not client:
        return ""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

# ── Canonical market list (intersection of all agents) ───────────────────────

ALL_MARKETS = [
    "Austin, TX", "Dallas, TX", "Houston, TX", "Phoenix, AZ",
    "Nashville, TN", "Charlotte, NC", "Atlanta, GA", "Denver, CO",
    "Las Vegas, NV", "Raleigh, NC", "Tampa, FL", "Orlando, FL",
    "Indianapolis, IN", "Los Angeles, CA", "Seattle, WA",
    "Chicago, IL", "New York, NY", "Miami, FL", "Boston, MA",
]

# Climate risk metro name → canonical market
_CLIMATE_METRO_MAP = {
    "Houston":        "Houston, TX",
    "Dallas":         "Dallas, TX",
    "Austin":         "Austin, TX",
    "San Antonio":    None,
    "Miami":          "Miami, FL",
    "Tampa":          "Tampa, FL",
    "Jacksonville":   None,
    "Orlando":        "Orlando, FL",
    "Phoenix":        "Phoenix, AZ",
    "Las Vegas":      "Las Vegas, NV",
    "Los Angeles":    "Los Angeles, CA",
    "San Diego":      None,
    "San Francisco":  None,
    "Sacramento":     None,
    "Seattle":        "Seattle, WA",
    "Portland":       None,
    "Denver":         "Denver, CO",
    "Chicago":        "Chicago, IL",
    "New York City":  "New York, NY",
    "Boston":         "Boston, MA",
    "Washington DC":  None,
    "Atlanta":        "Atlanta, GA",
    "Charlotte":      "Charlotte, NC",
    "Nashville":      "Nashville, TN",
    "Minneapolis":    None,
    "Detroit":        None,
    "Indianapolis":   "Indianapolis, IN",
    "Raleigh":        "Raleigh, NC",
}

# Location keyword → candidate markets
_REGION_MAP = {
    "southern texas":  ["Houston, TX"],
    "south texas":     ["Houston, TX"],
    "north texas":     ["Dallas, TX"],
    "west texas":      ["Dallas, TX"],
    "texas":           ["Dallas, TX", "Houston, TX", "Austin, TX"],
    "houston":         ["Houston, TX"],
    "dallas":          ["Dallas, TX"],
    "austin":          ["Austin, TX"],
    "florida":         ["Tampa, FL", "Orlando, FL", "Miami, FL"],
    "miami":           ["Miami, FL"],
    "tampa":           ["Tampa, FL"],
    "orlando":         ["Orlando, FL"],
    "southeast":       ["Atlanta, GA", "Charlotte, NC", "Nashville, TN", "Tampa, FL", "Orlando, FL", "Miami, FL", "Raleigh, NC"],
    "sunbelt":         ["Dallas, TX", "Houston, TX", "Austin, TX", "Phoenix, AZ", "Nashville, TN", "Atlanta, GA", "Charlotte, NC", "Tampa, FL", "Miami, FL", "Las Vegas, NV"],
    "southwest":       ["Phoenix, AZ", "Las Vegas, NV", "Denver, CO"],
    "arizona":         ["Phoenix, AZ"],
    "phoenix":         ["Phoenix, AZ"],
    "nevada":          ["Las Vegas, NV"],
    "las vegas":       ["Las Vegas, NV"],
    "tennessee":       ["Nashville, TN"],
    "nashville":       ["Nashville, TN"],
    "north carolina":  ["Charlotte, NC", "Raleigh, NC"],
    "charlotte":       ["Charlotte, NC"],
    "raleigh":         ["Raleigh, NC"],
    "georgia":         ["Atlanta, GA"],
    "atlanta":         ["Atlanta, GA"],
    "colorado":        ["Denver, CO"],
    "denver":          ["Denver, CO"],
    "california":      ["Los Angeles, CA"],
    "los angeles":     ["Los Angeles, CA"],
    "west coast":      ["Los Angeles, CA", "Seattle, WA"],
    "pacific northwest": ["Seattle, WA"],
    "washington":      ["Seattle, WA"],
    "seattle":         ["Seattle, WA"],
    "midwest":         ["Chicago, IL", "Indianapolis, IN"],
    "illinois":        ["Chicago, IL"],
    "chicago":         ["Chicago, IL"],
    "indiana":         ["Indianapolis, IN"],
    "indianapolis":    ["Indianapolis, IN"],
    "northeast":       ["New York, NY", "Boston, MA"],
    "new york":        ["New York, NY"],
    "boston":          ["Boston, MA"],
    "massachusetts":   ["Boston, MA"],
}

# ── Property type normalization ───────────────────────────────────────────────

_PROP_SYNONYMS = {
    "warehouse": "Industrial", "logistics": "Industrial", "distribution": "Industrial",
    "industrial": "Industrial", "manufacturing": "Industrial", "flex": "Industrial",
    "office": "Office", "office building": "Office", "offices": "Office",
    "retail": "Retail", "shopping": "Retail", "strip mall": "Retail",
    "restaurant": "Retail", "storefront": "Retail",
    "multifamily": "Multifamily", "apartment": "Multifamily", "apartments": "Multifamily",
    "residential": "Multifamily", "housing": "Multifamily",
    "data center": "Data Center", "datacenter": "Data Center",
    "healthcare": "Healthcare", "medical": "Healthcare", "hospital": "Healthcare",
    "mixed use": "Mixed-Use", "mixed-use": "Mixed-Use", "mixed": "Mixed-Use",
    "hospitality": "Hospitality", "hotel": "Hospitality", "motel": "Hospitality",
    "self storage": "Self-Storage", "storage": "Self-Storage",
}

def _normalize_property_type(raw: str) -> str | None:
    if not raw:
        return None
    r = raw.lower().strip()
    for k, v in _PROP_SYNONYMS.items():
        if k in r:
            return v
    return None

# ── Construction cost and timeline tables ─────────────────────────────────────

_CONSTRUCTION_COST_PSF = {         # $/sqft base (MODERATE signal)
    "Industrial":   110,
    "Office":       220,
    "Retail":       145,
    "Multifamily":  175,
    "Data Center":  550,
    "Healthcare":   420,
    "Mixed-Use":    190,
    "Hospitality":  280,
    "Self-Storage":  65,
}

_BUILDOUT_MONTHS_BASE = {
    "Industrial":  14, "Office": 26, "Retail": 10, "Multifamily": 20,
    "Data Center": 32, "Healthcare": 40, "Mixed-Use": 22,
    "Hospitality": 26, "Self-Storage": 8,
}

_SIGNAL_COST_MULT  = {"LOW": 0.88, "MODERATE": 1.0, "HIGH": 1.20}
_SIGNAL_TIME_DELTA = {"LOW": -2,   "MODERATE": 0,   "HIGH": +4}

# Rent growth cache key by property type
_RENT_KEY = {
    "Industrial": "industrial_psf", "Office": "office_psf",
    "Retail": "retail_psf", "Multifamily": "multifamily",
    "Data Center": "industrial_psf", "Healthcare": "office_psf",
    "Mixed-Use": "retail_psf", "Hospitality": "retail_psf", "Self-Storage": "multifamily",
}

# Default factor weights by property type (used when Groq unavailable)
_DEFAULT_WEIGHTS = {
    "Industrial":  {"market_fundamentals": 0.25, "rent_growth": 0.20, "cap_rate": 0.20, "migration": 0.15, "climate_risk": 0.10, "macro": 0.10},
    "Multifamily": {"market_fundamentals": 0.20, "rent_growth": 0.25, "migration": 0.20, "cap_rate": 0.15, "climate_risk": 0.10, "macro": 0.10},
    "Office":      {"market_fundamentals": 0.25, "cap_rate": 0.20, "rent_growth": 0.15, "migration": 0.15, "macro": 0.15, "climate_risk": 0.10},
    "Retail":      {"market_fundamentals": 0.25, "migration": 0.20, "rent_growth": 0.20, "cap_rate": 0.15, "climate_risk": 0.10, "macro": 0.10},
    "Data Center": {"market_fundamentals": 0.20, "macro": 0.20, "cap_rate": 0.20, "migration": 0.15, "rent_growth": 0.15, "climate_risk": 0.10},
    "Healthcare":  {"market_fundamentals": 0.25, "migration": 0.20, "macro": 0.20, "cap_rate": 0.15, "rent_growth": 0.10, "climate_risk": 0.10},
}
_DEFAULT_WEIGHTS["Mixed-Use"]    = _DEFAULT_WEIGHTS["Retail"]
_DEFAULT_WEIGHTS["Hospitality"]  = _DEFAULT_WEIGHTS["Retail"]
_DEFAULT_WEIGHTS["Self-Storage"] = _DEFAULT_WEIGHTS["Multifamily"]

_DEFAULT_RATIONALES = {
    "market_fundamentals": "Captures overall market health — migration inflow, vacancy, land availability, and employment demand.",
    "rent_growth":         "Directly drives NOI expansion and future exit valuation.",
    "cap_rate":            "Higher cap rates signal better initial yield and acquisition value.",
    "migration":           "Population and corporate inflows sustain long-term demand for space.",
    "climate_risk":        "Physical climate hazards (flood, wildfire, heat) drive insurance costs and asset value risk.",
    "macro":               "Credit conditions and GDP cycle determine financing cost and occupancy trajectory.",
}

# ── Step 1: Parse prompt ──────────────────────────────────────────────────────

def parse_prompt(text: str) -> dict:
    """Extract structured investment params from free-text. Groq first, regex fallback."""
    result = {
        "raw_prompt": text,
        "property_type": None,
        "location_raw": None,
        "budget": None,
        "sqft": None,
        "timeline_years": None,
        "risk_tolerance": None,
        "missing_fields": [],
        "parse_source": "regex",
    }

    # ── Try Groq first ────────────────────────────────────────────────────────
    groq_resp = _groq_complete(
        system=(
            "You are a commercial real estate investment analyst parsing user investment queries. "
            "Extract the following fields from the user's message and return ONLY valid JSON. "
            "Fields: property_type (string or null), location_raw (string or null), "
            "budget (number in dollars or null), sqft (number or null), "
            "timeline_years (number or null), risk_tolerance ('conservative','moderate','aggressive' or null). "
            "Return null for any field not mentioned. No explanation, just JSON."
        ),
        user=text,
        max_tokens=300,
    )

    if groq_resp:
        try:
            # Extract JSON from response
            match = re.search(r'\{.*\}', groq_resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                result["property_type"]   = _normalize_property_type(parsed.get("property_type") or "")
                result["location_raw"]    = parsed.get("location_raw")
                result["budget"]          = _parse_number(parsed.get("budget"))
                result["sqft"]            = _parse_number(parsed.get("sqft"))
                result["timeline_years"]  = _parse_number(parsed.get("timeline_years"))
                result["risk_tolerance"]  = parsed.get("risk_tolerance")
                result["parse_source"]    = "groq"
        except Exception:
            pass

    # ── Regex fallback for any still-null fields ───────────────────────────────
    t = text.lower()

    if not result["property_type"]:
        result["property_type"] = _normalize_property_type(t)

    if not result["location_raw"]:
        for region in _REGION_MAP:
            if region in t:
                result["location_raw"] = region
                break

    if not result["budget"]:
        for m in re.finditer(r'\$?([\d,.]+)\s*(m|million|b|billion|k|thousand)?', t):
            num = float(m.group(1).replace(",", ""))
            suffix = (m.group(2) or "").lower()
            if suffix in ("m", "million"):     num *= 1_000_000
            elif suffix in ("b", "billion"):   num *= 1_000_000_000
            elif suffix in ("k", "thousand"):  num *= 1_000
            if num >= 100_000:
                result["budget"] = num
                break

    if not result["sqft"]:
        for m in re.finditer(r'([\d,]+)\s*(sq\s?ft|square\s?feet|sqft|sf)', t):
            result["sqft"] = float(m.group(1).replace(",", ""))
            break

    if not result["timeline_years"]:
        for m in re.finditer(r'(\d+)[\s-]*(year|yr)', t):
            result["timeline_years"] = int(m.group(1))
            break

    if not result["risk_tolerance"]:
        if any(w in t for w in ["conservative", "safe", "low risk"]):
            result["risk_tolerance"] = "conservative"
        elif any(w in t for w in ["aggressive", "high risk", "high return", "speculative"]):
            result["risk_tolerance"] = "aggressive"
        else:
            result["risk_tolerance"] = "moderate"

    # ── Identify missing required fields ──────────────────────────────────────
    missing = []
    if not result["property_type"]:  missing.append("property_type")
    if not result["location_raw"]:   missing.append("location")
    if not result["budget"]:         missing.append("budget")
    if not result["sqft"]:           missing.append("sqft")
    if not result["timeline_years"]: missing.append("timeline_years")
    result["missing_fields"] = missing

    return result


def _parse_number(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return None

# ── Step 2: Resolve markets from location ─────────────────────────────────────

def resolve_markets(location_raw: str | None) -> list[str]:
    """Map location string to list of candidate markets."""
    if not location_raw:
        return ALL_MARKETS[:8]
    loc = location_raw.lower().strip()
    # Longest match wins
    for key in sorted(_REGION_MAP, key=len, reverse=True):
        if key in loc:
            markets = _REGION_MAP[key]
            # Always score at least 3 markets for runner-ups
            if len(markets) < 3:
                # Add top national markets as fallback alternatives
                extra = [m for m in ALL_MARKETS if m not in markets]
                markets = markets + extra[:3 - len(markets)]
            return markets
    # No match — return all markets
    return ALL_MARKETS

# ── Step 3: Gather market data ────────────────────────────────────────────────

def gather_market_data(markets: list[str], property_type: str) -> list[dict]:
    """Pull all relevant cache data for each candidate market."""
    # Load caches once
    ms_data     = _read("market_score")
    cap_data    = _read("cap_rate")
    rg_data     = _read("rent_growth")
    vac_data    = _read("vacancy")
    cr_data     = _read("climate_risk").get("data") or _read("climate_risk")
    oz_data     = _read("opportunity_zone")
    credit_data = _read("credit_data")
    gdp_data    = _read("gdp_data")
    energy_data = _read("energy_data")
    mig_data    = _read("migration")

    # Index market_score by market name
    ms_by_market = {r["market"]: r for r in ms_data.get("rankings", [])}

    # Index cap rates
    cap_by_market = cap_data.get("market_cap_rates", {})
    national_cap  = cap_data.get("national", {})

    # Index rent growth
    rg_by_market = rg_data.get("market_rent_growth", {})

    # Index vacancy
    vac_rows = vac_data.get("market_rows", [])
    vac_by_key = {}
    for row in vac_rows:
        vac_by_key[(row["market"], row["property_type"])] = row

    # Index climate risk metros
    climate_metros = {m["metro"]: m for m in cr_data.get("metros", [])}

    # Index OZ markets
    oz_markets = oz_data.get("oz_markets", {})

    # Credit / GDP / energy — global signals
    credit_signal  = (credit_data.get("signal") or {}).get("label", "NEUTRAL")
    gdp_cycle      = (gdp_data.get("cycle") or {}).get("label", "UNKNOWN")
    energy_signal  = energy_data.get("construction_cost_signal", "MODERATE")

    # Migration state scores
    mig_rows = mig_data.get("migration", [])
    mig_by_state = {r.get("state_abbr", ""): r for r in mig_rows}

    rent_key = _RENT_KEY.get(property_type, "industrial_psf")

    result = []
    for market in markets:
        state = market.split(", ")[-1] if ", " in market else ""
        city  = market.split(", ")[0]

        # Market score
        ms = ms_by_market.get(market, {})
        ms_composite = ms.get("composite", 50.0)
        ms_factors   = ms.get("factors", {})

        # Cap rate
        market_caps = cap_by_market.get(market, {})
        cap_rate = market_caps.get(property_type)
        if cap_rate is None:
            # Try matching partial name
            for k, v in cap_by_market.items():
                if city.lower() in k.lower():
                    cap_rate = v.get(property_type)
                    break
        if cap_rate is None:
            national = cap_data.get("national", {}).get(property_type, {})
            cap_rate = national.get("rate", 6.0) if isinstance(national, dict) else 6.0

        # Rent growth
        rg = rg_by_market.get(market, {})
        if not rg:
            for k, v in rg_by_market.items():
                if city.lower() in k.lower():
                    rg = v
                    break
        rent_growth = rg.get(rent_key, 0.0) if rg else 0.0

        # Vacancy
        pt_vac_key = _map_vacancy_pt(property_type)
        vac_row = vac_by_key.get((market, pt_vac_key))
        vacancy_rate = vac_row["vacancy_rate"] if vac_row else None

        # Climate risk
        climate = climate_metros.get(city, {})
        if not climate:
            for k, v in climate_metros.items():
                if city.lower() in k.lower():
                    climate = v
                    break
        climate_score  = climate.get("composite_score", 30.0)
        climate_label  = climate.get("label", "Low")
        climate_factors = climate.get("factors", {})

        # OZ
        oz = oz_markets.get(market, {})
        oz_score = oz.get("opportunity_score", 0)

        # Migration
        mig = mig_by_state.get(state, {})
        mig_score = float(mig.get("migration_score", 50)) if mig else 50.0

        result.append({
            "market":          market,
            "city":            city,
            "state":           state,
            "ms_composite":    ms_composite,
            "ms_factors":      ms_factors,
            "cap_rate":        round(cap_rate, 2),
            "rent_growth":     round(rent_growth, 2),
            "vacancy_rate":    vacancy_rate,
            "climate_score":   climate_score,
            "climate_label":   climate_label,
            "climate_factors": climate_factors,
            "climate_trend":   climate.get("trend", []),
            "oz_score":        oz_score,
            "oz_zones":        oz.get("key_zones", []),
            "mig_score":       mig_score,
            "credit_signal":   credit_signal,
            "gdp_cycle":       gdp_cycle,
            "energy_signal":   energy_signal,
        })

    return result


def _map_vacancy_pt(property_type: str) -> str:
    mapping = {
        "Industrial": "Industrial", "Multifamily": "Multifamily",
        "Office": "Office", "Retail": "Retail",
        "Data Center": "Industrial", "Healthcare": "Office",
        "Mixed-Use": "Retail", "Hospitality": "Retail", "Self-Storage": "Industrial",
    }
    return mapping.get(property_type, "Office")

# ── Step 4: Get AI-determined weights ────────────────────────────────────────

def get_weights(property_type: str, risk_tolerance: str) -> dict:
    """Ask Groq to determine factor weights. Falls back to defaults."""
    groq_resp = _groq_complete(
        system=(
            "You are a CRE investment strategist. Given a property type and risk tolerance, "
            "return ONLY valid JSON with factor weights that sum to exactly 1.0. "
            "Factors: market_fundamentals, rent_growth, cap_rate, migration, climate_risk, macro. "
            "Also include a one-sentence rationale for each weight. "
            "Format: {\"weights\": {\"factor\": {\"weight\": float, \"rationale\": \"...\"}}, \"methodology\": \"one sentence\"}"
        ),
        user=f"Property type: {property_type}. Risk tolerance: {risk_tolerance}.",
        max_tokens=500,
    )

    if groq_resp:
        try:
            match = re.search(r'\{.*\}', groq_resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                raw_weights = parsed.get("weights", {})
                if raw_weights:
                    weights = {}
                    rationales = {}
                    for factor, v in raw_weights.items():
                        if isinstance(v, dict):
                            weights[factor]    = float(v.get("weight", 0))
                            rationales[factor] = v.get("rationale", "")
                        else:
                            weights[factor] = float(v)
                            rationales[factor] = _DEFAULT_RATIONALES.get(factor, "")
                    # Normalize to sum to 1.0
                    total = sum(weights.values())
                    if total > 0:
                        weights = {k: round(v / total, 3) for k, v in weights.items()}
                    return {
                        "weights":     weights,
                        "rationales":  rationales,
                        "methodology": parsed.get("methodology", ""),
                        "source":      "groq",
                    }
        except Exception:
            pass

    # Default weights
    defaults = _DEFAULT_WEIGHTS.get(property_type, _DEFAULT_WEIGHTS["Industrial"])
    return {
        "weights":     defaults,
        "rationales":  _DEFAULT_RATIONALES,
        "methodology": f"Weights optimized for {property_type} asset class using platform-standard factor priorities.",
        "source":      "default",
    }

# ── Step 5: Score markets ─────────────────────────────────────────────────────

def score_markets(market_data: list[dict], weights: dict, property_type: str) -> list[dict]:
    """Compute composite opportunity score for each market."""
    w = weights["weights"]

    # Determine score ranges across all markets for normalization
    cap_rates   = [m["cap_rate"] for m in market_data if m["cap_rate"]]
    rents       = [m["rent_growth"] for m in market_data]
    mig_scores  = [m["mig_score"] for m in market_data]

    cap_min,  cap_max  = (min(cap_rates), max(cap_rates))   if cap_rates  else (4, 9)
    rent_min, rent_max = (min(rents), max(rents))           if rents      else (-5, 15)
    mig_min,  mig_max  = (min(mig_scores), max(mig_scores)) if mig_scores else (0, 100)

    def _norm(val, lo, hi, invert=False):
        if hi == lo: return 50.0
        s = max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))
        return (100.0 - s) if invert else s

    # Credit/GDP/energy → macro score
    credit_map = {"LOOSE": 85, "NEUTRAL": 60, "TIGHT": 30}
    gdp_map    = {"EXPANSION": 90, "SLOWDOWN": 55, "CONTRACTION": 20, "UNKNOWN": 55}

    scored = []
    for m in market_data:
        # Factor scores (0–100 each)
        factor_scores = {}

        # Market fundamentals (from market_score agent)
        factor_scores["market_fundamentals"] = m["ms_composite"]

        # Rent growth (normalized; positive = better)
        factor_scores["rent_growth"] = _norm(m["rent_growth"], rent_min, rent_max)

        # Cap rate attractiveness (higher = more attractive for most types)
        factor_scores["cap_rate"] = _norm(m["cap_rate"], cap_min, cap_max)

        # Migration (higher = better)
        factor_scores["migration"] = _norm(m["mig_score"], mig_min, mig_max)

        # Climate risk (inverted — lower risk = higher score)
        factor_scores["climate_risk"] = 100.0 - m["climate_score"]

        # Macro environment
        credit_label = m["credit_signal"].upper() if m["credit_signal"] else "NEUTRAL"
        gdp_label    = m["gdp_cycle"].upper() if m["gdp_cycle"] else "UNKNOWN"
        macro_raw    = (credit_map.get(credit_label, 60) + gdp_map.get(gdp_label, 55)) / 2
        # Opportunity zone bonus (+5 if OZ present)
        oz_bonus = min(5.0, m["oz_score"] * 0.05) if m["oz_score"] else 0
        factor_scores["macro"] = min(100.0, macro_raw + oz_bonus)

        # Composite weighted score
        composite = sum(
            factor_scores.get(factor, 50.0) * weight
            for factor, weight in w.items()
        )
        composite = round(composite, 1)

        # Weighted breakdown (for display)
        breakdown = {}
        for factor, weight in w.items():
            fs = factor_scores.get(factor, 50.0)
            breakdown[factor] = {
                "raw_score":     round(fs, 1),
                "weight":        weight,
                "weighted":      round(fs * weight, 1),
                "rationale":     weights.get("rationales", {}).get(factor, ""),
            }

        scored.append({**m, "opportunity_score": composite, "factor_scores": breakdown})

    scored.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return scored

# ── Step 6: Estimate financials ───────────────────────────────────────────────

def estimate_financials(market: dict, property_type: str, budget: float,
                         sqft: float, timeline_years: int) -> dict:
    """Estimate costs, NOI, profit, and buildout timeline."""
    energy_signal = market.get("energy_signal", "MODERATE")
    cost_mult     = _SIGNAL_COST_MULT.get(energy_signal, 1.0)
    time_delta    = _SIGNAL_TIME_DELTA.get(energy_signal, 0)

    base_psf     = _CONSTRUCTION_COST_PSF.get(property_type, 150)
    construction = sqft * base_psf * cost_mult
    land_cost    = budget * 0.18        # ~18% of budget for land (industry avg)
    soft_costs   = (construction + land_cost) * 0.15   # permits, arch, fees
    total_cost   = land_cost + construction + soft_costs

    cap_rate      = market.get("cap_rate", 6.0) / 100
    annual_noi    = total_cost * cap_rate
    rent_growth   = market.get("rent_growth", 3.0) / 100
    # Compound NOI over hold period
    total_noi     = sum(annual_noi * (1 + rent_growth) ** yr for yr in range(timeline_years))

    # Exit cap rate: slight compression (0.25pp) if market score is strong
    exit_cap_adjustment = -0.0025 if market.get("opportunity_score", 50) > 70 else 0.0
    exit_cap      = cap_rate + exit_cap_adjustment
    exit_noi      = annual_noi * (1 + rent_growth) ** timeline_years
    exit_value    = exit_noi / exit_cap

    total_profit  = total_noi + exit_value - total_cost
    roi_pct       = (total_profit / total_cost) * 100 if total_cost else 0

    # Simple IRR estimate using average annual return
    avg_annual    = ((total_cost + total_profit) / total_cost) ** (1 / timeline_years) - 1
    irr_est       = avg_annual * 100

    buildout_months = _BUILDOUT_MONTHS_BASE.get(property_type, 18) + time_delta

    return {
        "land_cost":        round(land_cost),
        "construction_cost": round(construction),
        "soft_costs":        round(soft_costs),
        "total_cost":        round(total_cost),
        "annual_noi":        round(annual_noi),
        "cap_rate_pct":      market.get("cap_rate", 6.0),
        "rent_growth_pct":   market.get("rent_growth", 3.0),
        "total_noi":         round(total_noi),
        "exit_value":        round(exit_value),
        "total_profit":      round(total_profit),
        "roi_pct":           round(roi_pct, 1),
        "irr_est":           round(irr_est, 1),
        "buildout_months":   buildout_months,
        "energy_signal":     energy_signal,
        "sqft":              sqft,
        "budget_input":      budget,
    }

# ── Step 7: Generate Groq narrative ──────────────────────────────────────────

def generate_narrative(params: dict, primary: dict, runners: list[dict],
                        financials: dict, weights: dict) -> str:
    """Generate a 3-paragraph investment rationale via Groq."""
    climate_note = (
        f" Climate risk is rated {primary['climate_label']} ({primary['climate_score']:.0f}/100)."
        if primary["climate_score"] > 40 else ""
    )
    fin = financials
    prompt = (
        f"Property type: {params['property_type']}. "
        f"Location: {primary['market']}. "
        f"Budget: ${fin['budget_input']:,.0f}. "
        f"Square footage: {fin['sqft']:,.0f} sq ft. "
        f"Hold period: {params['timeline_years']} years. "
        f"Opportunity score: {primary['opportunity_score']:.1f}/100. "
        f"Cap rate: {fin['cap_rate_pct']}%. "
        f"Rent growth: {fin['rent_growth_pct']}% YoY. "
        f"Estimated total cost: ${fin['total_cost']:,.0f}. "
        f"Estimated exit value: ${fin['exit_value']:,.0f}. "
        f"Estimated ROI: {fin['roi_pct']}% over {params['timeline_years']} years.{climate_note} "
        f"Market score: {primary['ms_composite']:.1f}/100. "
        f"GDP cycle: {primary['gdp_cycle']}. Credit: {primary['credit_signal']}. "
        f"Risk tolerance: {params.get('risk_tolerance', 'moderate')}."
    )

    narrative = _groq_complete(
        system=(
            "You are a senior CRE investment advisor writing a concise investment brief. "
            "Write exactly 3 paragraphs (no headers, no bullet points): "
            "1) Why this market is the top recommendation for this property type right now. "
            "2) Key risks to monitor and how they affect this investment. "
            "3) Strategic outlook — what macro trends support a hold over the specified period. "
            "Be specific with numbers. Keep each paragraph 3-4 sentences. Professional tone."
        ),
        user=prompt,
        max_tokens=600,
    )

    if not narrative:
        # Template fallback
        narrative = (
            f"{primary['market']} ranks as the top recommendation for {params['property_type']} "
            f"with an opportunity score of {primary['opportunity_score']:.1f}/100, driven by strong "
            f"market fundamentals and a {fin['cap_rate_pct']}% cap rate. "
            f"The market benefits from favorable credit conditions ({primary['credit_signal'].title()}) "
            f"and a {fin['rent_growth_pct']}% rent growth trajectory, "
            f"supporting NOI expansion throughout the {params['timeline_years']}-year hold period.\n\n"
            f"Key risks include {primary['climate_label'].lower()} climate exposure "
            f"and the current {primary['gdp_cycle'].title()} economic cycle, "
            f"which warrants conservative underwriting assumptions on exit cap rates. "
            f"Construction cost pressures ({primary['energy_signal'].title()} signal) "
            f"add approximately {financials['buildout_months']} months to the buildout timeline "
            f"and are factored into the total cost estimate of ${fin['total_cost']:,.0f}.\n\n"
            f"Over a {params['timeline_years']}-year hold, this investment targets an estimated "
            f"{fin['roi_pct']}% total return (≈{fin['irr_est']}% IRR), "
            f"supported by the platform's migration and labor market data showing sustained "
            f"population and employment inflows into {primary['city']}. "
            f"The {primary['credit_signal'].title()} credit environment maintains accessible "
            f"financing, while {primary['city']}'s position in the {primary['gdp_cycle'].title()} "
            f"phase supports occupancy stability."
        )

    return narrative

# ── Step 8: Build full recommendation ────────────────────────────────────────

def build_recommendation(params: dict) -> dict:
    """Main entry point. Returns full recommendation dict."""
    property_type  = params["property_type"]
    location_raw   = params.get("location_raw", "")
    budget         = params["budget"]
    sqft           = params["sqft"]
    timeline_years = params["timeline_years"]
    risk_tolerance = params.get("risk_tolerance", "moderate")

    markets    = resolve_markets(location_raw)
    mkt_data   = gather_market_data(markets, property_type)
    weights    = get_weights(property_type, risk_tolerance)
    scored     = score_markets(mkt_data, weights, property_type)

    if not scored:
        return {"error": "No market data available for the requested location."}

    primary  = scored[0]
    runners  = scored[1:3]
    financials = estimate_financials(primary, property_type, budget, sqft, timeline_years)
    narrative  = generate_narrative(params, primary, runners, financials, weights)

    return {
        "generated_at":   datetime.now().isoformat(),
        "params":         params,
        "primary":        primary,
        "runners":        runners,
        "financials":     financials,
        "weights":        weights,
        "narrative":      narrative,
        "all_scored":     scored,
    }
