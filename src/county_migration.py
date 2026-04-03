"""
County-Level Migration Data
============================
Generates mock county migration data for state-level drill-down maps.
Structured for easy replacement with real Census Bureau County Population Estimates.

Usage:
    from src.county_migration import get_county_data
    df = get_county_data("TX")  # returns DataFrame with FIPS, name, scores
"""

import numpy as np
import pandas as pd

COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

# State FIPS codes (2-digit, zero-padded)
_ABBR_TO_STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

# Top counties per state with real FIPS codes and realistic data.
# Format: (county_fips_suffix, county_name, population, base_growth, base_score, driver)
_COUNTY_SEEDS = {
    "TX": [
        ("201", "Harris", 4731145, 2.3, 82, "Energy, medical center, port"),
        ("113", "Dallas", 2613539, 2.1, 80, "Corporate HQ corridor, finance"),
        ("439", "Tarrant", 2110640, 1.9, 76, "Defense, manufacturing"),
        ("029", "Bexar", 2009324, 1.7, 72, "Military, cybersecurity hub"),
        ("453", "Travis", 1290188, 3.1, 90, "Tech capital, university"),
        ("085", "Collin", 1064465, 3.8, 94, "Corporate relocations, suburbs"),
        ("121", "Denton", 906874, 3.5, 88, "University, suburban growth"),
        ("491", "Williamson", 609017, 4.2, 96, "Tesla, Samsung, tech overflow"),
        ("157", "Fort Bend", 822779, 2.8, 84, "Energy suburbs, diverse economy"),
        ("141", "El Paso", 839238, 0.8, 48, "Border trade, military"),
        ("303", "Lubbock", 310569, 0.5, 40, "Agriculture, university"),
        ("375", "Potter", 117415, -0.2, 28, "Rural decline"),
    ],
    "FL": [
        ("086", "Miami-Dade", 2701767, 1.8, 78, "Finance migration, Latin America gateway"),
        ("011", "Broward", 1944375, 1.6, 74, "Fort Lauderdale, tech startups"),
        ("099", "Palm Beach", 1492191, 2.1, 82, "Wealth migration, finance"),
        ("057", "Hillsborough", 1459762, 1.9, 76, "Tampa Bay tech corridor"),
        ("095", "Orange", 1393452, 2.0, 80, "Orlando, tourism, simulation tech"),
        ("031", "Duval", 995567, 1.4, 68, "Jacksonville, logistics, military"),
        ("103", "Pinellas", 959107, 0.8, 52, "St. Petersburg, aging population"),
        ("071", "Lee", 760822, 2.5, 86, "Cape Coral, retirement migration"),
        ("115", "Sarasota", 434006, 2.3, 84, "Retirement, healthcare"),
        ("109", "St. Johns", 276047, 4.5, 98, "Fastest-growing FL county"),
    ],
    "AZ": [
        ("013", "Maricopa", 4420568, 2.2, 84, "Phoenix metro, semiconductor fabs"),
        ("019", "Pima", 1043433, 1.0, 56, "Tucson, aerospace, mining"),
        ("021", "Pinal", 425264, 3.2, 90, "Exurban growth, logistics"),
        ("015", "Mohave", 215927, 1.5, 62, "Lake Havasu, retirement"),
        ("027", "Yuma", 203881, 0.6, 38, "Agriculture, border"),
        ("007", "Gila", 53272, -0.3, 24, "Rural decline, mining legacy"),
    ],
    "CA": [
        ("037", "Los Angeles", 9829544, -0.4, 22, "Net outflow, high costs"),
        ("073", "San Diego", 3286069, 0.3, 42, "Biotech, military, moderate growth"),
        ("059", "Orange", 3186989, 0.1, 38, "Suburban, tech satellites"),
        ("065", "Riverside", 2418185, 1.2, 60, "Inland Empire logistics boom"),
        ("071", "San Bernardino", 2181654, 0.8, 50, "Warehousing, distribution"),
        ("085", "Santa Clara", 1936259, -0.2, 30, "Silicon Valley outflow"),
        ("001", "Alameda", 1682353, -0.3, 26, "Oakland exodus"),
        ("075", "San Francisco", 808437, -1.1, 12, "Tech remote work exodus"),
        ("067", "Sacramento", 1585055, 0.9, 54, "State capital, CA refugee destination"),
        ("029", "Kern", 909235, 0.4, 36, "Agriculture, oil"),
    ],
    "NY": [
        ("061", "New York (Manhattan)", 1694251, -0.8, 18, "Urban outflow, remote work"),
        ("047", "Kings (Brooklyn)", 2736074, -0.3, 28, "Moderate outflow"),
        ("081", "Queens", 2405464, -0.2, 30, "Slight outflow"),
        ("005", "Bronx", 1472654, -0.4, 24, "Outflow, affordability"),
        ("085", "Richmond (Staten Is)", 495747, 0.1, 36, "Stable"),
        ("103", "Suffolk", 1525920, -0.1, 34, "Long Island suburban"),
        ("059", "Nassau", 1395774, -0.2, 32, "Long Island suburban decline"),
        ("119", "Westchester", 1004457, 0.0, 35, "NYC suburb"),
        ("055", "Monroe (Rochester)", 759443, -0.3, 26, "Rust belt legacy"),
        ("029", "Erie (Buffalo)", 954236, -0.2, 28, "Modest decline"),
    ],
    "IL": [
        ("031", "Cook", 5275541, -0.5, 20, "Chicago outflow, fiscal issues"),
        ("043", "DuPage", 928589, 0.1, 38, "Suburban stability"),
        ("089", "Kane", 516522, 0.4, 44, "Western suburbs growth"),
        ("097", "Lake", 714342, -0.1, 34, "North Shore, stable"),
        ("197", "Will", 690743, 0.6, 48, "Southern suburbs, logistics"),
        ("163", "St. Clair", 259686, -0.3, 26, "East St. Louis decline"),
    ],
    "GA": [
        ("121", "Fulton", 1066710, 1.6, 72, "Atlanta core, corporate HQs"),
        ("089", "DeKalb", 764382, 1.2, 64, "Atlanta east, diverse economy"),
        ("067", "Cobb", 766149, 1.4, 68, "Marietta, defense, suburban"),
        ("135", "Gwinnett", 957062, 2.0, 80, "Fastest-growing Atlanta suburb"),
        ("063", "Clayton", 297595, 1.0, 54, "Airport corridor, logistics"),
        ("151", "Henry", 240712, 2.5, 86, "Rivian plant, south Atlanta"),
    ],
    "NC": [
        ("119", "Mecklenburg", 1115482, 2.0, 82, "Charlotte, banking capital"),
        ("183", "Wake", 1129410, 2.4, 88, "Raleigh, Research Triangle"),
        ("063", "Durham", 324833, 1.8, 76, "Duke, biotech corridor"),
        ("081", "Guilford", 537174, 0.8, 50, "Greensboro, manufacturing"),
        ("067", "Forsyth", 382590, 0.9, 52, "Winston-Salem, medical"),
        ("051", "Cumberland", 334728, 0.4, 38, "Fort Bragg, military"),
    ],
    "TN": [
        ("037", "Davidson", 715884, 1.8, 78, "Nashville, entertainment, healthcare"),
        ("157", "Shelby", 929744, 0.3, 36, "Memphis, logistics hub"),
        ("093", "Knox", 478971, 1.2, 62, "Knoxville, university, energy"),
        ("065", "Hamilton", 366207, 1.5, 70, "Chattanooga, VW plant, tech"),
        ("149", "Rutherford", 341486, 2.2, 84, "Nashville suburb, fastest growth"),
        ("187", "Williamson", 247726, 2.8, 92, "Wealthiest TN county, corporate"),
    ],
    "WA": [
        ("033", "King", 2269675, 0.8, 56, "Seattle, tech (MS, Amazon, Boeing)"),
        ("053", "Pierce", 921130, 1.2, 64, "Tacoma, military, logistics"),
        ("061", "Snohomish", 827957, 1.0, 58, "Boeing, suburban Seattle"),
        ("063", "Spokane", 539339, 0.9, 52, "Eastern WA hub"),
        ("011", "Clark", 503311, 2.0, 80, "Portland overflow, no income tax"),
    ],
    "CO": [
        ("031", "Denver", 713252, 0.9, 56, "State capital, tech, outdoor economy"),
        ("005", "Arapahoe", 655070, 1.0, 58, "Denver suburb, diverse economy"),
        ("035", "Douglas", 357978, 1.8, 76, "Wealthy suburb, corporate parks"),
        ("041", "El Paso", 730395, 1.4, 66, "Colorado Springs, military, space"),
        ("069", "Larimer", 359066, 1.2, 62, "Fort Collins, university, tech"),
        ("013", "Boulder", 326196, 0.6, 46, "University, climate tech"),
    ],
}


def get_county_data(state_abbr: str) -> pd.DataFrame:
    """
    Returns county-level migration data for a single state.
    Uses seeded mock data with realistic FIPS codes and scores.
    """
    state_fips = _ABBR_TO_STATE_FIPS.get(state_abbr)
    if not state_fips:
        return pd.DataFrame()

    seeds = _COUNTY_SEEDS.get(state_abbr)
    if seeds:
        rows = []
        for suffix, name, pop, growth, score, driver in seeds:
            rows.append({
                "fips": state_fips + suffix,
                "name": name,
                "state": state_abbr,
                "population": pop,
                "pop_growth_pct": growth,
                "migration_score": score,
                "top_driver": driver,
            })
        return pd.DataFrame(rows)

    # Generate synthetic counties for states without seed data
    rng = np.random.RandomState(hash(state_abbr) % 2**31)
    n_counties = rng.randint(8, 18)
    rows = []
    for i in range(n_counties):
        suffix = f"{rng.randint(1, 200):03d}"
        pop = int(rng.lognormal(11, 1.2))
        growth = round(rng.normal(0.5, 1.0), 1)
        score = max(5, min(95, int(50 + growth * 15 + rng.normal(0, 8))))
        rows.append({
            "fips": state_fips + suffix,
            "name": f"{state_abbr} County {i+1}",
            "state": state_abbr,
            "population": pop,
            "pop_growth_pct": growth,
            "migration_score": score,
            "top_driver": "",
        })
    return pd.DataFrame(rows)
