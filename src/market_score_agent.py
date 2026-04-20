"""
Market Opportunity Score Agent — synthesizes signals from all other agents into
a single 0-100 composite score per CRE market.

Reads from existing caches (migration, vacancy, labor_market, rates, land_market,
cap_rate, rent_growth) and weights them into an actionable market ranking.

Factor weights:
  - Migration / population inflow     25%
  - Vacancy & absorption health       20%
  - Employment / tenant demand        15%
  - Rent growth momentum              15%
  - Cap rate attractiveness           10%
  - Land availability / pipeline      10%
  - Macro environment (rates/credit)   5%

Returns a ranked list of markets with composite scores and factor breakdowns.
"""

import os
from datetime import datetime
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

CACHE_DIR = Path(__file__).parent.parent / "cache"

# Markets tracked across all agents — the intersection of all datasets
SCORED_MARKETS = [
    "Austin, TX", "Dallas, TX", "Houston, TX", "Phoenix, AZ",
    "Nashville, TN", "Charlotte, NC", "Atlanta, GA", "Denver, CO",
    "Las Vegas, NV", "Raleigh, NC", "Tampa, FL", "Orlando, FL",
    "Indianapolis, IN", "Los Angeles, CA", "Seattle, WA",
    "Chicago, IL", "New York, NY", "Miami, FL", "Boston, MA",
]

# State abbreviation lookup for matching migration cache
_MARKET_STATE = {
    "Austin, TX": "TX", "Dallas, TX": "TX", "Houston, TX": "TX",
    "Phoenix, AZ": "AZ", "Nashville, TN": "TN", "Charlotte, NC": "NC",
    "Atlanta, GA": "GA", "Denver, CO": "CO", "Las Vegas, NV": "NV",
    "Raleigh, NC": "NC", "Tampa, FL": "FL", "Orlando, FL": "FL",
    "Indianapolis, IN": "IN", "Los Angeles, CA": "CA", "Seattle, WA": "WA",
    "Chicago, IL": "IL", "New York, NY": "NY", "Miami, FL": "FL",
    "Boston, MA": "MA",
}


def _read_cache(key: str) -> dict:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            payload = json.load(f)
        return payload.get("data") or {}
    except Exception:
        return {}


def _normalize(value: float, low: float, high: float, invert: bool = False) -> float:
    """Normalize value to 0-100 range. Clamp to bounds."""
    if high == low:
        return 50.0
    score = (value - low) / (high - low) * 100
    score = max(0.0, min(100.0, score))
    return (100.0 - score) if invert else score


def _migration_scores(mig_data: dict) -> dict:
    """Return {market: score} from migration cache keyed by state_abbr."""
    result = {}
    rows = mig_data.get("migration", [])
    for row in rows:
        abbr = row.get("state_abbr", "")
        comp = row.get("composite_score", 50)
        for mkt, st in _MARKET_STATE.items():
            if st == abbr and mkt not in result:
                result[mkt] = float(comp)
    return result


def _vacancy_scores(vac_data: dict) -> dict:
    """Return {market: score} — lower overall vacancy + positive absorption = higher score."""
    result = {}
    rows = vac_data.get("market_rows", [])
    abs_rows = vac_data.get("absorption_rows", [])

    # Average vacancy across property types per market
    vac_by_mkt: dict[str, list] = {}
    for r in rows:
        mkt = r.get("market", "")
        vac_by_mkt.setdefault(mkt, []).append(r.get("vacancy_rate", 10.0))

    # Absorption signal per market
    abs_by_mkt: dict[str, int] = {}
    for r in abs_rows:
        mkt = r.get("market", "")
        net = r.get("net_absorption_ksf", 0)
        abs_by_mkt[mkt] = abs_by_mkt.get(mkt, 0) + net

    for mkt in SCORED_MARKETS:
        vacs = vac_by_mkt.get(mkt, [10.0])
        avg_vac = sum(vacs) / len(vacs)
        # Vacancy: 4% = 100, 25% = 0 (inverted — lower vacancy is better)
        vac_score = _normalize(avg_vac, 4.0, 25.0, invert=True)

        # Absorption: positive = good
        abs_val = abs_by_mkt.get(mkt, 0)
        abs_score = _normalize(abs_val, -2000, 8000)

        result[mkt] = round((vac_score * 0.6 + abs_score * 0.4), 1)
    return result


def _rent_scores(rent_data: dict) -> dict:
    """Return {market: score} from rent growth data."""
    result = {}
    market_rg = rent_data.get("market_rent_growth", {})
    for mkt in SCORED_MARKETS:
        rg = market_rg.get(mkt, {})
        if not rg:
            result[mkt] = 50.0
            continue
        # Weight: industrial (40%), multifamily (35%), retail (15%), office (10%)
        ind  = rg.get("industrial_psf", 0)
        mf   = rg.get("multifamily", 0)
        ret  = rg.get("retail_psf", 0)
        off  = rg.get("office_psf", 0)
        weighted = ind * 0.40 + mf * 0.35 + ret * 0.15 + off * 0.10
        result[mkt] = round(_normalize(weighted, -5.0, 12.0), 1)
    return result


def _cap_rate_scores(cap_data: dict) -> dict:
    """Return {market: score} — higher spread vs treasury = more attractive."""
    result = {}
    spreads = cap_data.get("spreads", {})
    treasury = cap_data.get("treasury_10y") or 4.3  # fallback
    market_caps = cap_data.get("market_cap_rates", {})

    for mkt in SCORED_MARKETS:
        mkt_caps = market_caps.get(mkt, {})
        if not mkt_caps:
            result[mkt] = 50.0
            continue
        # Average cap rate across property types, then subtract treasury
        avg_cap = sum(mkt_caps.values()) / len(mkt_caps)
        spread = avg_cap - treasury
        # spread > 3% = 100, spread < 0.5% = 0
        result[mkt] = round(_normalize(spread, 0.5, 3.5), 1)
    return result


def _land_scores(land_data: dict) -> dict:
    """Return {market: score} — more available land + faster entitlement = better."""
    result = {}
    land_avail = land_data.get("land_availability", {})
    for mkt in SCORED_MARKETS:
        info = land_avail.get(mkt, {})
        if not info:
            result[mkt] = 50.0
            continue
        total_ac = info.get("industrial_ac", 0) + info.get("mixed_use_ac", 0) + info.get("residential_ac", 0)
        ent_mo   = info.get("entitlement_mo", 24)
        ac_score  = _normalize(total_ac, 500, 60000)
        ent_score = _normalize(ent_mo, 8, 60, invert=True)  # faster = better
        result[mkt] = round((ac_score * 0.5 + ent_score * 0.5), 1)
    return result


def _climate_penalties(climate_data: dict) -> dict:
    """Return {market: penalty_points} — higher climate risk reduces composite score.

    Penalty formula: max(0, (climate_score - 60) * 0.20), capped at 10 points.
    Applied only for markets with Severe/Extreme climate exposure (score >= 60).
    """
    result = {mkt: 0.0 for mkt in SCORED_MARKETS}
    states = climate_data.get("states", {})
    if not states:
        return result
    for mkt, abbr in _MARKET_STATE.items():
        state_info = states.get(abbr, {})
        cs = state_info.get("composite_score", 0.0)
        if cs >= 60:
            penalty = min(10.0, (cs - 60) * 0.20)
            result[mkt] = round(penalty, 1)
    return result


def _macro_score(rate_data: dict, credit_data: dict) -> float:
    """Single national macro score (same for all markets)."""
    # Rate environment: lower 10Y = better for CRE
    t10y = rate_data.get("current_10y") or 4.3
    rate_score = _normalize(t10y, 3.0, 6.5, invert=True)

    # Credit conditions signal score
    cr_score = credit_data.get("signal", {}).get("score", 50)

    return round((rate_score * 0.5 + float(cr_score) * 0.5), 1)


def _grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 70: return "B+"
    if score >= 60: return "B"
    if score >= 50: return "C+"
    if score >= 40: return "C"
    return "D"


def run_market_score_agent() -> dict:
    """
    Aggregate signals from all caches into per-market composite scores.
    Returns ranked list with factor breakdowns.
    """
    # Load all caches
    mig_data      = _read_cache("migration")
    vac_data      = _read_cache("vacancy")
    rent_data     = _read_cache("rent_growth")
    cap_data      = _read_cache("cap_rate")
    land_data     = _read_cache("vacancy")   # LAND_AVAILABILITY in vacancy cache
    rate_data     = _read_cache("rates")
    credit_data   = _read_cache("credit_data")
    climate_data  = _read_cache("climate_risk")

    # Compute per-factor scores
    mig_scores      = _migration_scores(mig_data)
    vac_scores      = _vacancy_scores(vac_data)
    rent_scores     = _rent_scores(rent_data)
    cap_scores      = _cap_rate_scores(cap_data)
    land_scores     = _land_scores(vac_data)   # land_availability lives in vacancy cache
    macro_s         = _macro_score(rate_data, credit_data)
    climate_penalty = _climate_penalties(climate_data)

    # Weights
    W = {
        "migration": 0.25,
        "vacancy":   0.20,
        "rent":      0.15,
        "cap_rate":  0.10,
        "land":      0.10,
        "macro":     0.05,
        # labor = remaining 15% — placeholder from migration proxy
    }

    rankings = []
    for mkt in SCORED_MARKETS:
        factors = {
            "migration": mig_scores.get(mkt, 50.0),
            "vacancy":   vac_scores.get(mkt, 50.0),
            "rent":      rent_scores.get(mkt, 50.0),
            "cap_rate":  cap_scores.get(mkt, 50.0),
            "land":      land_scores.get(mkt, 50.0),
            "macro":     macro_s,
        }

        raw = round(
            factors["migration"] * W["migration"] +
            factors["vacancy"]   * W["vacancy"]   +
            factors["rent"]      * W["rent"]       +
            factors["cap_rate"]  * W["cap_rate"]   +
            factors["land"]      * W["land"]       +
            factors["macro"]     * W["macro"] +
            factors["migration"] * W.get("labor", 0.15),  # labor proxy via migration
            1
        )
        penalty  = climate_penalty.get(mkt, 0.0)
        composite = round(max(0.0, raw - penalty), 1)

        rankings.append({
            "market":          mkt,
            "state":           _MARKET_STATE.get(mkt, ""),
            "composite":       composite,
            "raw_composite":   raw,
            "climate_penalty": penalty,
            "grade":           _grade(composite),
            "factors":         factors,
            "rank":            0,  # filled in below
        })

    rankings.sort(key=lambda x: x["composite"], reverse=True)
    for i, r in enumerate(rankings, 1):
        r["rank"] = i

    # National summary
    scores = [r["composite"] for r in rankings]
    avg_score = round(sum(scores) / len(scores), 1)

    top3  = [r["market"] for r in rankings[:3]]
    avoid = [r["market"] for r in rankings[-3:]]

    return {
        "rankings":      rankings,
        "top3_markets":  top3,
        "avoid_markets": avoid,
        "avg_score":     avg_score,
        "factor_weights": W,
        "fetched_at":    datetime.now().isoformat(),
        "data_as_of":    "Q1 2025",
    }
