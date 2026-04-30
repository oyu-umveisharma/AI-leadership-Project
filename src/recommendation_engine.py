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

# ── Property-type financial parameters ───────────────────────────────────────
_RENT_PSF = {
    "Industrial": 9.50, "Office": 30.00, "Retail": 22.00,
    "Multifamily": 18.00, "Mixed-Use": 24.00, "Healthcare": 27.00,
    "Data Center": 65.00, "Self-Storage": 13.00, "Hospitality": 20.00,
}
_OPEX_RATIO = {          # operating expenses as % of EGI
    "Industrial": 0.22, "Office": 0.40, "Retail": 0.28,
    "Multifamily": 0.45, "Mixed-Use": 0.38, "Healthcare": 0.35,
    "Data Center": 0.30, "Self-Storage": 0.35, "Hospitality": 0.50,
}
_STAB_VAC = {            # stabilised vacancy rate
    "Industrial": 0.05, "Office": 0.14, "Retail": 0.09,
    "Multifamily": 0.06, "Mixed-Use": 0.08, "Healthcare": 0.05,
    "Data Center": 0.03, "Self-Storage": 0.11, "Hospitality": 0.30,
}
_LTV_MAP = {             # max LTV by type
    "Industrial": 0.65, "Office": 0.65, "Retail": 0.65,
    "Multifamily": 0.75, "Mixed-Use": 0.70, "Healthcare": 0.65,
    "Data Center": 0.60, "Self-Storage": 0.70, "Hospitality": 0.60,
}
_AMORT_MAP = {           # amortisation period (years)
    "Industrial": 25, "Office": 25, "Retail": 25,
    "Multifamily": 30, "Mixed-Use": 25, "Healthcare": 25,
    "Data Center": 20, "Self-Storage": 25, "Hospitality": 25,
}
_DEPR_LIFE = {           # IRS depreciation life (years)
    "Industrial": 39, "Office": 39, "Retail": 39,
    "Multifamily": 27.5, "Mixed-Use": 39, "Healthcare": 39,
    "Data Center": 20, "Self-Storage": 39, "Hospitality": 39,
}

def _current_treasury_rate() -> float:
    """Read 10-yr Treasury yield from rates cache; default 4.5%."""
    try:
        d = _read("rates")
        for field in ["DGS10", "ten_year_yield", "treasury_10y", "10yr"]:
            v = d.get(field)
            if isinstance(v, (int, float)) and 0.5 < v < 20:
                return v / 100
        # Try nested fred block
        fred = d.get("fred_rates", d.get("fred", {}))
        if isinstance(fred, dict):
            for field in ["DGS10", "10Y"]:
                v = fred.get(field)
                if isinstance(v, (int, float)) and 0.5 < v < 20:
                    return v / 100
    except Exception:
        pass
    return 0.045


def estimate_financing(primary: dict, property_type: str,
                        financials: dict, timeline_years: int) -> dict:
    """Debt structure, DSCR, cash-on-cash, and leveraged IRR."""
    pt = property_type.split("/")[0].strip()
    total_cost = financials.get("total_cost", 0)
    noi = financials.get("annual_noi", 0)

    ltv    = _LTV_MAP.get(pt, 0.65)
    amort  = _AMORT_MAP.get(pt, 25)
    spread = 0.015 if pt == "Multifamily" else 0.020
    rate   = _current_treasury_rate() + spread

    loan      = total_cost * ltv
    equity    = total_cost - loan
    r_mo      = rate / 12
    n_mo      = amort * 12
    mo_pmt    = (loan * r_mo / (1 - (1 + r_mo) ** (-n_mo))) if r_mo > 0 else loan / n_mo
    ann_ds    = mo_pmt * 12
    cf_ds     = noi - ann_ds
    dscr      = noi / ann_ds if ann_ds > 0 else 0
    coc       = cf_ds / equity if equity > 0 else 0

    # Loan balance at exit
    n_paid    = timeline_years * 12
    if r_mo > 0:
        bal_exit = loan * (1 + r_mo) ** n_paid - mo_pmt * ((1 + r_mo) ** n_paid - 1) / r_mo
    else:
        bal_exit = loan - (loan / n_mo) * n_paid
    bal_exit  = max(0.0, bal_exit)

    exit_val  = financials.get("exit_value", total_cost)
    eq_exit   = exit_val - bal_exit
    total_eq_ret = eq_exit - equity + cf_ds * timeline_years

    # Leveraged IRR via bisection
    cfs = [-equity] + [cf_ds] * timeline_years
    cfs[-1] += eq_exit
    try:
        lo, hi = -0.9, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            npv = sum(c / (1 + mid) ** t for t, c in enumerate(cfs))
            if npv > 0: lo = mid
            else:       hi = mid
        lev_irr = mid
    except Exception:
        lev_irr = 0.0

    return {
        "ltv_pct":             round(ltv * 100, 1),
        "loan_amount":         round(loan),
        "equity_required":     round(equity),
        "loan_rate_pct":       round(rate * 100, 2),
        "amort_years":         amort,
        "annual_debt_service": round(ann_ds),
        "dscr":                round(dscr, 2),
        "cash_flow_after_ds":  round(cf_ds),
        "cash_on_cash_pct":    round(coc * 100, 2),
        "loan_balance_exit":   round(bal_exit),
        "equity_at_exit":      round(eq_exit),
        "leveraged_irr_pct":   round(lev_irr * 100, 2),
        "leveraged_roi_pct":   round(total_eq_ret / equity * 100, 1) if equity > 0 else 0,
    }


def estimate_proforma(property_type: str, financials: dict,
                       financing: dict, primary: dict,
                       timeline_years: int) -> list[dict]:
    """Year-by-year 10-year operating pro forma."""
    pt          = property_type.split("/")[0].strip()
    gross_yr1   = financials.get("annual_gross_revenue", financials.get("annual_noi", 0) / max(1 - _OPEX_RATIO.get(pt, 0.35), 0.01))
    opex_ratio  = _OPEX_RATIO.get(pt, 0.35)
    vac_rate    = _STAB_VAC.get(pt, 0.08)
    rent_g      = (primary.get("rent_growth", 3.0)) / 100
    opex_inf    = 0.025
    ann_ds      = financing.get("annual_debt_service", 0)
    opex_base   = gross_yr1 * opex_ratio

    rows, cum_cf = [], 0
    gross = gross_yr1
    for yr in range(1, min(int(timeline_years), 10) + 1):
        if yr > 1:
            gross *= (1 + rent_g)
        yr_vac = min(vac_rate * 1.6, 0.25) if yr == 1 else vac_rate
        vac_loss = gross * yr_vac
        egi      = gross - vac_loss
        opex     = opex_base * (1 + opex_inf) ** (yr - 1)
        noi      = egi - opex
        cf       = noi - ann_ds
        cum_cf  += cf
        rows.append({
            "year":          yr,
            "gross_revenue": round(gross),
            "vacancy_loss":  round(vac_loss),
            "egi":           round(egi),
            "opex":          round(opex),
            "noi":           round(noi),
            "debt_service":  round(ann_ds),
            "cf_after_ds":   round(cf),
            "cum_cf":        round(cum_cf),
        })
    return rows


def estimate_tax_benefits(property_type: str, financials: dict,
                            primary: dict) -> dict:
    """Depreciation schedule, cost-segregation tax shield, and OZ benefits."""
    pt           = property_type.split("/")[0].strip()
    total_cost   = financials.get("total_cost", 0)

    land_pct     = 0.10 if pt == "Data Center" else 0.20
    bldg_val     = total_cost * (1 - land_pct)
    land_val     = total_cost * land_pct

    # Cost-segregation allocation
    pp_val  = bldg_val * 0.15   # 5-yr personal property
    li_val  = bldg_val * 0.10   # 15-yr land improvements
    str_val = bldg_val * 0.75   # 39-yr (or 27.5) structure

    struct_life  = _DEPR_LIFE.get(pt, 39)
    bonus_pct    = 0.40          # 2025 bonus depreciation (40%)

    # Year-1 depreciation
    yr1_depr = (
        pp_val  * bonus_pct + (pp_val  * (1 - bonus_pct)) / 5 +
        li_val  * bonus_pct + (li_val  * (1 - bonus_pct)) / 15 +
        str_val / struct_life
    )
    # Years 2+ regular annual
    ann_reg = (
        (pp_val  * (1 - bonus_pct)) / 5 +
        (li_val  * (1 - bonus_pct)) / 15 +
        str_val / struct_life
    )

    tax_rate      = 0.37
    yr1_shield    = yr1_depr  * tax_rate
    ann_shield    = ann_reg   * tax_rate
    cum10_savings = yr1_shield + ann_shield * 9

    oz_eligible   = bool(primary.get("oz_eligible", False))
    oz_note = (
        "OZ Fund eligible: defer capital gains through 2026, "
        "exclude 10-year appreciation from federal CGT entirely."
        if oz_eligible else ""
    )

    return {
        "land_value":           round(land_val),
        "building_value":       round(bldg_val),
        "personal_prop_value":  round(pp_val),
        "land_improv_value":    round(li_val),
        "structure_value":      round(str_val),
        "struct_depr_life":     struct_life,
        "bonus_depr_pct":       int(bonus_pct * 100),
        "yr1_depreciation":     round(yr1_depr),
        "yr1_tax_shield":       round(yr1_shield),
        "annual_depr_reg":      round(ann_reg),
        "annual_tax_shield":    round(ann_shield),
        "cum10_tax_savings":    round(cum10_savings),
        "tax_rate_pct":         int(tax_rate * 100),
        "oz_eligible":          oz_eligible,
        "oz_note":              oz_note,
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
                _ty                       = _parse_number(parsed.get("timeline_years"))
                result["timeline_years"]  = int(_ty) if _ty is not None else None
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
            _raw = m.group(1).replace(",", "").strip(".")
            if not _raw or _raw == ".":
                continue
            num = float(_raw)
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

    # Gross revenue estimate (sqft × market rent PSF)
    pt_key = property_type.split("/")[0].strip()
    rent_psf = _RENT_PSF.get(pt_key, 18.0)
    annual_gross_revenue = sqft * rent_psf

    return {
        "land_cost":             round(land_cost),
        "construction_cost":     round(construction),
        "soft_costs":            round(soft_costs),
        "total_cost":            round(total_cost),
        "annual_gross_revenue":  round(annual_gross_revenue),
        "annual_noi":            round(annual_noi),
        "cap_rate_pct":          market.get("cap_rate", 6.0),
        "rent_growth_pct":       market.get("rent_growth", 3.0),
        "total_noi":             round(total_noi),
        "exit_value":            round(exit_value),
        "total_profit":          round(total_profit),
        "roi_pct":               round(roi_pct, 1),
        "irr_est":               round(irr_est, 1),
        "buildout_months":       buildout_months,
        "hold_years":            timeline_years,
        "energy_signal":         energy_signal,
        "sqft":                  sqft,
        "budget_input":          budget,
    }

# ── Step 7: Generate Groq narrative ──────────────────────────────────────────

def _forecast_context_line() -> str:
    """Pull Agent 22 FRED projections and condense into a one-line macro outlook."""
    try:
        from pathlib import Path as _Path
        import json as _json
        _fp = _Path(__file__).parent.parent / "cache" / "forecast.json"
        if not _fp.exists():
            return ""
        _proj = (_json.loads(_fp.read_text()).get("data") or {}).get("projections") or {}
        if not _proj:
            return ""
        parts = []
        for _name, _p in _proj.items():
            _cur = _p.get("current")
            _q4  = _p.get("q4_2026")
            if _cur is None or _q4 is None:
                continue
            parts.append(f"{_name}: {_cur:.2f}% → Q4 2026 {_q4:.2f}%")
        if not parts:
            return ""
        return " Macro forecast — " + "; ".join(parts) + "."
    except Exception:
        return ""


def generate_narrative(params: dict, primary: dict, runners: list[dict],
                        financials: dict, weights: dict) -> str:
    """Generate a 3-paragraph investment rationale via Groq."""
    climate_note = (
        f" Climate risk is rated {primary['climate_label']} ({primary['climate_score']:.0f}/100)."
        if primary["climate_score"] > 40 else ""
    )
    forecast_note = _forecast_context_line()
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
        f"Risk tolerance: {params.get('risk_tolerance', 'moderate')}.{forecast_note}"
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
    # timeline_years is semantically an integer count of years; some parse
    # paths (Groq JSON) return a float. Normalize once so downstream range()
    # and list-multiplication sites never crash on "5.0".
    timeline_years = int(round(params["timeline_years"]))
    params["timeline_years"] = timeline_years
    risk_tolerance = params.get("risk_tolerance", "moderate")

    markets    = resolve_markets(location_raw)
    mkt_data   = gather_market_data(markets, property_type)
    weights    = get_weights(property_type, risk_tolerance)
    scored     = score_markets(mkt_data, weights, property_type)

    if not scored:
        return {"error": "No market data available for the requested location."}

    primary    = scored[0]
    runners    = scored[1:3]
    financials = estimate_financials(primary, property_type, budget, sqft, timeline_years)
    financing  = estimate_financing(primary, property_type, financials, timeline_years)
    proforma   = estimate_proforma(property_type, financials, financing, primary, timeline_years)
    tax        = estimate_tax_benefits(property_type, financials, primary)
    narrative  = generate_narrative(params, primary, runners, financials, weights)

    return {
        "generated_at":   datetime.now().isoformat(),
        "params":         params,
        "primary":        primary,
        "runners":        runners,
        "financials":     financials,
        "financing":      financing,
        "proforma":       proforma,
        "tax_benefits":   tax,
        "weights":        weights,
        "narrative":      narrative,
        "all_scored":     scored,
    }
