"""
Opportunity Zone & State Tax Incentive Tracker
================================================
Static agent — no live API required. Data sourced from IRS Revenue Ruling 2018-29
OZ designations, HUD Opportunity Zone census tract counts, and state economic
development program publications current as of 2025.

Covers:
  - Federal Opportunity Zone markets (top CRE-relevant metros)
  - State-level CRE tax incentive programs (15 states)
  - Federal OZ tax benefit summary (deferral, step-up, permanent exclusion)

No cache needed — all data is hardcoded and deterministic.
"""

import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Federal Opportunity Zones — top CRE-relevant OZ markets
# tract_count = number of OZ census tracts in that metro
# ---------------------------------------------------------------------------
OZ_MARKETS = {
    "Atlanta, GA":       {"tract_count": 260, "key_zones": ["Westside Atlanta", "South DeKalb", "Airport Corridor"],          "cre_types": ["Industrial", "Multifamily", "Mixed-Use"], "opportunity_score": 82},
    "Dallas, TX":        {"tract_count": 198, "key_zones": ["West Dallas", "South Dallas", "Opportunity Dallas"],              "cre_types": ["Industrial", "Multifamily"],             "opportunity_score": 78},
    "Houston, TX":       {"tract_count": 212, "key_zones": ["Fifth Ward", "Near Northside", "East End"],                      "cre_types": ["Industrial", "Mixed-Use"],               "opportunity_score": 74},
    "Baltimore, MD":     {"tract_count": 288, "key_zones": ["East Baltimore", "Cherry Hill", "Port Covington"],                "cre_types": ["Multifamily", "Mixed-Use", "Office"],    "opportunity_score": 76},
    "Detroit, MI":       {"tract_count": 334, "key_zones": ["Midtown", "New Center", "East Detroit"],                         "cre_types": ["Multifamily", "Industrial", "Mixed-Use"],"opportunity_score": 71},
    "Chicago, IL":       {"tract_count": 291, "key_zones": ["South Side", "West Side", "Englewood"],                          "cre_types": ["Multifamily", "Industrial"],             "opportunity_score": 68},
    "Philadelphia, PA":  {"tract_count": 252, "key_zones": ["North Philly", "West Philly", "Kensington"],                     "cre_types": ["Multifamily", "Mixed-Use"],              "opportunity_score": 72},
    "Charlotte, NC":     {"tract_count": 118, "key_zones": ["West Charlotte", "University City", "Hidden Valley"],            "cre_types": ["Multifamily", "Industrial"],             "opportunity_score": 80},
    "Nashville, TN":     {"tract_count": 84,  "key_zones": ["North Nashville", "Antioch", "Bordeaux"],                        "cre_types": ["Multifamily", "Mixed-Use"],              "opportunity_score": 81},
    "Las Vegas, NV":     {"tract_count": 96,  "key_zones": ["Downtown Las Vegas", "Historic Westside", "Fremont East"],       "cre_types": ["Multifamily", "Retail", "Mixed-Use"],    "opportunity_score": 75},
    "Phoenix, AZ":       {"tract_count": 188, "key_zones": ["South Phoenix", "West Phoenix", "Mesa Downtown"],                "cre_types": ["Industrial", "Multifamily"],             "opportunity_score": 77},
    "Miami, FL":         {"tract_count": 142, "key_zones": ["Little Havana", "Opa-locka", "Liberty City"],                    "cre_types": ["Multifamily", "Mixed-Use"],              "opportunity_score": 79},
    "New York, NY":      {"tract_count": 306, "key_zones": ["South Bronx", "East New York", "Jamaica Queens"],                "cre_types": ["Multifamily", "Mixed-Use"],              "opportunity_score": 70},
    "Los Angeles, CA":   {"tract_count": 274, "key_zones": ["Boyle Heights", "Watts", "Pacoima"],                             "cre_types": ["Industrial", "Multifamily"],             "opportunity_score": 65},
    "New Orleans, LA":   {"tract_count": 148, "key_zones": ["Lower Ninth Ward", "Central City", "Mid-City"],                  "cre_types": ["Multifamily", "Mixed-Use"],              "opportunity_score": 73},
}

# ---------------------------------------------------------------------------
# State-level CRE tax incentive programs
# ---------------------------------------------------------------------------
STATE_INCENTIVES = {
    "TX": {"program": "Texas Enterprise Zone",                         "benefit": "State sales tax refunds up to $2,500/job created",                        "cre_types": ["Industrial", "Office"],                    "cap": "$25M per project",       "url_note": "governor.state.tx.us/ecodev/"},
    "FL": {"program": "Florida Brownfields Redevelopment",             "benefit": "Tax credit $2,500/job + site rehab costs",                               "cre_types": ["Industrial", "Mixed-Use"],                  "cap": "No cap",                 "url_note": "floridajobs.org"},
    "GA": {"program": "Georgia Opportunity Zone Tax Credit",           "benefit": "$3,500 tax credit per job created in OZ tracts",                         "cre_types": ["Industrial", "Retail", "Mixed-Use"],        "cap": "$5,000/job max",         "url_note": "georgia.org"},
    "NC": {"program": "NC Opportunity Zone Investment Tax Credit",     "benefit": "25% state tax credit on qualified OZ investments",                       "cre_types": ["Multifamily", "Mixed-Use", "Industrial"],   "cap": "$500K per investor",     "url_note": "nccommerce.com"},
    "TN": {"program": "Tennessee FastTrack Infrastructure Program",    "benefit": "Grants for infrastructure supporting job creation",                      "cre_types": ["Industrial", "Office"],                    "cap": "Case-by-case",           "url_note": "tnecd.gov"},
    "AZ": {"program": "Arizona Qualified Facility Tax Credit",         "benefit": "Tax credits for qualified manufacturing/industrial facilities",           "cre_types": ["Industrial"],                              "cap": "$30M total per year",    "url_note": "azcommerce.com"},
    "NV": {"program": "Nevada Transferable Tax Credits",               "benefit": "Up to 15% tax abatement for qualifying projects",                        "cre_types": ["Industrial", "Data Center"],               "cap": "Project-specific",       "url_note": "diversifynevada.com"},
    "SC": {"program": "SC Job Tax Credit",                             "benefit": "$1,500–$25,000 per job depending on county tier",                        "cre_types": ["Industrial", "Office"],                    "cap": "No cap",                 "url_note": "sccommerce.com"},
    "IN": {"program": "IEDC Industrial Recovery Site Tax Abatement",   "benefit": "Up to 10-year tax abatement on improvements",                            "cre_types": ["Industrial"],                              "cap": "County-approved",        "url_note": "iedc.in.gov"},
    "OH": {"program": "Ohio Enterprise Zone Program",                  "benefit": "Real property tax exemption 60–75% for 10 years",                        "cre_types": ["Industrial", "Office", "Mixed-Use"],        "cap": "County-negotiated",      "url_note": "development.ohio.gov"},
    "CO": {"program": "Colorado Enterprise Zone Investment Tax Credit", "benefit": "3% state income tax credit on investments in EZ",                       "cre_types": ["Industrial", "Mixed-Use"],                  "cap": "No cap",                 "url_note": "choosecolorado.com"},
    "IL": {"program": "Illinois EDGE Tax Credit",                      "benefit": "Corporate income tax credit up to $50M over 10 years",                  "cre_types": ["Office", "Industrial"],                    "cap": "$50M",                   "url_note": "illinois.gov/dceo"},
    "NY": {"program": "NY Excelsior Jobs Program",                     "benefit": "Tax credits up to 6.85% of wages for job creation",                      "cre_types": ["Office", "Industrial", "Data Center"],     "cap": "Award-based",            "url_note": "esd.ny.gov"},
    "CA": {"program": "California Competes Tax Credit",                "benefit": "Negotiated income tax credits for businesses locating/expanding in CA",  "cre_types": ["Industrial", "Office"],                    "cap": "$180M/year statewide",   "url_note": "business.ca.gov"},
    "WA": {"program": "Washington B&O Tax Credit for R&D",             "benefit": "B&O tax credit for qualified R&D spending in WA facilities",             "cre_types": ["Office", "Industrial"],                    "cap": "No cap",                 "url_note": "dor.wa.gov"},
}

# ---------------------------------------------------------------------------
# Federal OZ tax benefits summary
# ---------------------------------------------------------------------------
OZ_FEDERAL_BENEFITS = [
    {"benefit": "Temporary Gain Deferral",  "detail": "Capital gains invested in QOF deferred until Dec 31, 2026 or earlier sale"},
    {"benefit": "Basis Step-Up",            "detail": "10% basis increase if investment held 5+ years (investments made before end of 2021); step-up eliminated for post-2021 investments"},
    {"benefit": "Permanent Exclusion",      "detail": "100% exclusion of appreciation if QOF investment held 10+ years — the primary benefit"},
    {"benefit": "QOF Structure",            "detail": "Must invest via Qualified Opportunity Fund; 90% asset test; 31-month working capital safe harbor"},
]


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
def run_opportunity_zone_agent() -> dict:
    print("=" * 60)
    print("[OZAgent] Compiling Opportunity Zone & State Incentive data ...")
    print("=" * 60)

    top_markets_by_score = sorted(
        [(market, info["opportunity_score"]) for market, info in OZ_MARKETS.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"[OZAgent] {len(OZ_MARKETS)} OZ markets loaded.")
    print(f"[OZAgent] {len(STATE_INCENTIVES)} state incentive programs loaded.")
    print(f"[OZAgent] Top market: {top_markets_by_score[0][0]} (score {top_markets_by_score[0][1]})")
    print("=" * 60)

    return {
        "oz_markets":            OZ_MARKETS,
        "state_incentives":      STATE_INCENTIVES,
        "federal_benefits":      OZ_FEDERAL_BENEFITS,
        "top_markets_by_score":  top_markets_by_score,
        "fetched_at":            datetime.now().isoformat(),
        "data_as_of":            "2025 (IRS Rev. Rul. 2018-29 designations)",
    }


if __name__ == "__main__":
    result = run_opportunity_zone_agent()
    print("\nTop 5 OZ Markets by Opportunity Score:")
    for market, score in result["top_markets_by_score"][:5]:
        info = result["oz_markets"][market]
        print(f"  {market:25s}  score={score}  tracts={info['tract_count']}  types={info['cre_types']}")
    print(f"\nFederal Benefits:")
    for b in result["federal_benefits"]:
        print(f"  [{b['benefit']}] {b['detail']}")
