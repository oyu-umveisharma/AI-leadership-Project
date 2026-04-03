"""
ZIP / Neighborhood-Level Migration Data
=========================================
Generates mock zip-code-level migration data for metro-area drill-down maps.
Structured for easy replacement with real Census Bureau ACS 5-Year data.

Usage:
    from src.zip_migration import get_zip_data, get_available_metros
    df = get_zip_data("Chicago")  # returns DataFrame with zip, lat, lon, scores
"""

import numpy as np
import pandas as pd

# Metro definitions: (center_lat, center_lon, state_abbr, neighborhoods)
# Each neighborhood: (name, lat_offset, lon_offset, base_score, type)
_METRO_DATA = {
    "New York City": {
        "center": (40.7128, -74.0060), "state": "NY", "zones": [
            ("Midtown Manhattan", 0.008, -0.002, 55, "Urban Core"),
            ("Lower Manhattan / FiDi", -0.006, -0.005, 50, "Urban Core"),
            ("Upper East Side", 0.018, 0.002, 48, "Urban Core"),
            ("Upper West Side", 0.016, -0.008, 46, "Urban Core"),
            ("Harlem", 0.025, -0.004, 58, "Urban Core"),
            ("Brooklyn Heights", -0.008, 0.005, 52, "Urban Core"),
            ("Williamsburg", -0.002, 0.012, 60, "Urban Core"),
            ("Park Slope", -0.015, 0.003, 54, "Urban Core"),
            ("Bushwick", -0.005, 0.022, 62, "Urban Core"),
            ("Long Island City", 0.005, 0.010, 65, "Urban Core"),
            ("Astoria", 0.010, 0.015, 58, "Suburban"),
            ("Flushing", 0.012, 0.035, 56, "Suburban"),
            ("Jersey City", 0.000, -0.020, 68, "Suburban"),
            ("Hoboken", 0.004, -0.018, 64, "Suburban"),
            ("Fort Lee", 0.020, -0.015, 50, "Suburban"),
            ("Yonkers", 0.040, -0.010, 42, "Suburban"),
            ("White Plains", 0.055, 0.005, 44, "Exurban"),
            ("Stamford", 0.065, 0.020, 52, "Exurban"),
        ],
    },
    "Los Angeles": {
        "center": (34.0522, -118.2437), "state": "CA", "zones": [
            ("Downtown LA", 0.000, 0.000, 28, "Urban Core"),
            ("Hollywood", 0.015, -0.015, 24, "Urban Core"),
            ("Santa Monica", -0.005, -0.050, 30, "Urban Core"),
            ("Beverly Hills", 0.010, -0.030, 22, "Urban Core"),
            ("West LA / Westwood", 0.005, -0.040, 26, "Urban Core"),
            ("Silver Lake", 0.012, 0.005, 35, "Urban Core"),
            ("Koreatown", 0.003, -0.008, 32, "Urban Core"),
            ("South LA", -0.015, 0.005, 20, "Urban Core"),
            ("Pasadena", 0.030, 0.015, 38, "Suburban"),
            ("Glendale", 0.025, 0.005, 36, "Suburban"),
            ("Burbank", 0.028, -0.010, 40, "Suburban"),
            ("Long Beach", -0.040, 0.045, 42, "Suburban"),
            ("Torrance", -0.030, 0.025, 44, "Suburban"),
            ("Irvine", -0.060, 0.080, 55, "Exurban"),
            ("Ontario / Inland Empire", 0.010, 0.100, 62, "Exurban"),
            ("Riverside", 0.000, 0.140, 60, "Exurban"),
        ],
    },
    "Chicago": {
        "center": (41.8781, -87.6298), "state": "IL", "zones": [
            ("The Loop", 0.000, 0.000, 35, "Urban Core"),
            ("River North", 0.005, -0.002, 40, "Urban Core"),
            ("Lincoln Park", 0.015, -0.005, 42, "Urban Core"),
            ("Wicker Park", 0.010, -0.012, 45, "Urban Core"),
            ("West Loop", 0.002, -0.008, 48, "Urban Core"),
            ("South Loop", -0.005, 0.003, 38, "Urban Core"),
            ("Hyde Park", -0.020, 0.010, 36, "Urban Core"),
            ("Pilsen", -0.008, -0.005, 44, "Urban Core"),
            ("Bronzeville", -0.012, 0.005, 32, "Urban Core"),
            ("South Side", -0.025, 0.008, 18, "Urban Core"),
            ("Evanston", 0.035, 0.002, 46, "Suburban"),
            ("Oak Park", 0.005, -0.020, 50, "Suburban"),
            ("Naperville", -0.020, -0.080, 68, "Exurban"),
            ("Schaumburg", 0.025, -0.055, 58, "Exurban"),
            ("Aurora", -0.010, -0.095, 54, "Exurban"),
        ],
    },
    "Houston": {
        "center": (29.7604, -95.3698), "state": "TX", "zones": [
            ("Downtown", 0.000, 0.000, 72, "Urban Core"),
            ("Midtown", -0.005, -0.003, 74, "Urban Core"),
            ("Montrose", -0.002, -0.012, 70, "Urban Core"),
            ("Heights", 0.010, -0.008, 76, "Urban Core"),
            ("Medical Center / NRG", -0.015, -0.005, 68, "Urban Core"),
            ("Galleria / Uptown", 0.005, -0.020, 72, "Urban Core"),
            ("EaDo", -0.003, 0.005, 78, "Urban Core"),
            ("Katy", 0.010, -0.080, 82, "Exurban"),
            ("Sugar Land", -0.025, -0.055, 80, "Exurban"),
            ("The Woodlands", 0.055, -0.020, 86, "Exurban"),
            ("Pearland", -0.035, -0.010, 78, "Suburban"),
            ("Cypress", 0.030, -0.060, 84, "Exurban"),
        ],
    },
    "Phoenix": {
        "center": (33.4484, -112.0740), "state": "AZ", "zones": [
            ("Downtown", 0.000, 0.000, 75, "Urban Core"),
            ("Scottsdale", 0.020, 0.015, 82, "Suburban"),
            ("Tempe", -0.015, 0.010, 78, "Suburban"),
            ("Mesa", -0.010, 0.030, 72, "Suburban"),
            ("Chandler", -0.025, 0.025, 80, "Suburban"),
            ("Gilbert", -0.030, 0.040, 86, "Exurban"),
            ("Glendale", 0.010, -0.020, 68, "Suburban"),
            ("Peoria", 0.020, -0.030, 70, "Exurban"),
            ("Surprise", 0.025, -0.050, 74, "Exurban"),
            ("Buckeye", 0.000, -0.075, 88, "Exurban"),
            ("Queen Creek", -0.035, 0.055, 90, "Exurban"),
        ],
    },
    "Dallas": {
        "center": (32.7767, -96.7970), "state": "TX", "zones": [
            ("Downtown", 0.000, 0.000, 75, "Urban Core"),
            ("Uptown", 0.005, -0.005, 78, "Urban Core"),
            ("Deep Ellum", -0.002, 0.008, 72, "Urban Core"),
            ("Bishop Arts", -0.008, -0.005, 70, "Urban Core"),
            ("Plano", 0.040, 0.005, 85, "Suburban"),
            ("Frisco", 0.055, 0.005, 92, "Exurban"),
            ("McKinney", 0.065, 0.015, 90, "Exurban"),
            ("Richardson", 0.030, 0.008, 76, "Suburban"),
            ("Arlington", -0.015, -0.035, 68, "Suburban"),
            ("Irving", 0.008, -0.025, 74, "Suburban"),
            ("Fort Worth", -0.005, -0.080, 72, "Suburban"),
            ("Allen", 0.050, 0.015, 88, "Exurban"),
        ],
    },
    "Austin": {
        "center": (30.2672, -97.7431), "state": "TX", "zones": [
            ("Downtown", 0.000, 0.000, 88, "Urban Core"),
            ("South Congress", -0.008, -0.002, 86, "Urban Core"),
            ("East Austin", 0.002, 0.012, 82, "Urban Core"),
            ("Hyde Park", 0.012, -0.002, 80, "Urban Core"),
            ("Domain / North Austin", 0.025, -0.005, 90, "Suburban"),
            ("Mueller", 0.010, 0.008, 84, "Suburban"),
            ("Cedar Park", 0.035, -0.015, 92, "Exurban"),
            ("Round Rock", 0.042, 0.005, 94, "Exurban"),
            ("Georgetown", 0.060, 0.005, 88, "Exurban"),
            ("Pflugerville", 0.025, 0.015, 90, "Exurban"),
            ("Bee Cave / Lakeway", -0.005, -0.035, 86, "Exurban"),
        ],
    },
    "Miami": {
        "center": (25.7617, -80.1918), "state": "FL", "zones": [
            ("Downtown / Brickell", 0.000, 0.000, 76, "Urban Core"),
            ("Wynwood / Midtown", 0.008, -0.002, 78, "Urban Core"),
            ("South Beach", -0.008, 0.012, 68, "Urban Core"),
            ("Coconut Grove", -0.010, -0.005, 72, "Urban Core"),
            ("Coral Gables", -0.015, -0.010, 70, "Suburban"),
            ("Doral", 0.015, -0.020, 80, "Suburban"),
            ("Aventura", 0.025, 0.005, 74, "Suburban"),
            ("Homestead", -0.040, -0.015, 66, "Exurban"),
            ("Fort Lauderdale", 0.050, 0.000, 72, "Suburban"),
            ("Hollywood", 0.035, -0.005, 70, "Suburban"),
            ("Pembroke Pines", 0.030, -0.015, 68, "Suburban"),
        ],
    },
    "Seattle": {
        "center": (47.6062, -122.3321), "state": "WA", "zones": [
            ("Downtown", 0.000, 0.000, 52, "Urban Core"),
            ("Capitol Hill", 0.005, 0.005, 54, "Urban Core"),
            ("South Lake Union", 0.008, -0.003, 58, "Urban Core"),
            ("Ballard", 0.015, -0.010, 56, "Urban Core"),
            ("Fremont", 0.012, -0.005, 55, "Urban Core"),
            ("Beacon Hill", -0.008, 0.003, 48, "Urban Core"),
            ("West Seattle", -0.010, -0.012, 46, "Urban Core"),
            ("Bellevue", 0.000, 0.025, 62, "Suburban"),
            ("Redmond", 0.010, 0.040, 66, "Suburban"),
            ("Kirkland", 0.015, 0.025, 60, "Suburban"),
            ("Tacoma", -0.050, 0.005, 58, "Suburban"),
            ("Everett", 0.045, -0.010, 54, "Suburban"),
        ],
    },
    "Denver": {
        "center": (39.7392, -104.9903), "state": "CO", "zones": [
            ("Downtown / LoDo", 0.000, 0.000, 58, "Urban Core"),
            ("RiNo / Five Points", 0.005, 0.003, 62, "Urban Core"),
            ("Capitol Hill", -0.003, 0.005, 56, "Urban Core"),
            ("Cherry Creek", -0.008, 0.008, 54, "Urban Core"),
            ("Highlands", 0.008, -0.005, 60, "Urban Core"),
            ("Aurora", 0.000, 0.030, 52, "Suburban"),
            ("Lakewood", -0.005, -0.025, 50, "Suburban"),
            ("Arvada", 0.015, -0.020, 56, "Suburban"),
            ("Boulder", 0.050, -0.030, 48, "Exurban"),
            ("Castle Rock", -0.040, 0.010, 72, "Exurban"),
            ("Parker", -0.025, 0.025, 68, "Exurban"),
        ],
    },
    "Atlanta": {
        "center": (33.7490, -84.3880), "state": "GA", "zones": [
            ("Downtown", 0.000, 0.000, 65, "Urban Core"),
            ("Midtown", 0.008, -0.002, 70, "Urban Core"),
            ("Buckhead", 0.015, -0.005, 68, "Urban Core"),
            ("Old Fourth Ward", 0.005, 0.005, 72, "Urban Core"),
            ("West Midtown", 0.008, -0.010, 74, "Urban Core"),
            ("Decatur", 0.005, 0.020, 66, "Suburban"),
            ("Sandy Springs", 0.020, -0.010, 64, "Suburban"),
            ("Marietta", 0.030, -0.025, 62, "Suburban"),
            ("Roswell", 0.035, -0.015, 68, "Suburban"),
            ("Alpharetta", 0.045, -0.010, 76, "Exurban"),
            ("Suwanee / Gwinnett", 0.035, 0.015, 80, "Exurban"),
        ],
    },
    "Boston": {
        "center": (42.3601, -71.0589), "state": "MA", "zones": [
            ("Downtown / Financial", 0.000, 0.000, 46, "Urban Core"),
            ("Back Bay", -0.003, -0.008, 44, "Urban Core"),
            ("South Boston / Seaport", -0.005, 0.005, 52, "Urban Core"),
            ("Cambridge", 0.008, -0.005, 48, "Urban Core"),
            ("Somerville", 0.012, -0.002, 50, "Urban Core"),
            ("Charlestown", 0.005, 0.002, 46, "Urban Core"),
            ("Jamaica Plain", -0.010, -0.010, 44, "Urban Core"),
            ("Brookline", -0.005, -0.015, 42, "Suburban"),
            ("Newton", -0.002, -0.030, 40, "Suburban"),
            ("Quincy", -0.015, 0.010, 48, "Suburban"),
        ],
    },
    "Nashville": {
        "center": (36.1627, -86.7816), "state": "TN", "zones": [
            ("Downtown / SoBro", 0.000, 0.000, 80, "Urban Core"),
            ("The Gulch", -0.003, -0.003, 82, "Urban Core"),
            ("Germantown", 0.005, -0.002, 78, "Urban Core"),
            ("East Nashville", 0.005, 0.010, 84, "Urban Core"),
            ("12 South", -0.008, -0.008, 76, "Urban Core"),
            ("Sylvan Park", 0.002, -0.015, 74, "Urban Core"),
            ("Brentwood", -0.025, -0.008, 72, "Suburban"),
            ("Franklin", -0.040, -0.010, 88, "Exurban"),
            ("Murfreesboro", -0.060, 0.005, 82, "Exurban"),
            ("Hendersonville", 0.025, 0.005, 78, "Suburban"),
            ("Mt. Juliet", 0.015, 0.025, 86, "Exurban"),
        ],
    },
    "Charlotte": {
        "center": (35.2271, -80.8431), "state": "NC", "zones": [
            ("Uptown", 0.000, 0.000, 78, "Urban Core"),
            ("South End", -0.005, -0.003, 82, "Urban Core"),
            ("NoDa", 0.008, 0.005, 76, "Urban Core"),
            ("Plaza Midwood", 0.005, 0.010, 74, "Urban Core"),
            ("Ballantyne", -0.025, -0.008, 80, "Suburban"),
            ("Huntersville", 0.025, -0.005, 84, "Exurban"),
            ("Mooresville", 0.040, -0.010, 78, "Exurban"),
            ("Fort Mill (SC)", -0.030, -0.002, 86, "Exurban"),
            ("Indian Trail", -0.015, 0.020, 82, "Exurban"),
        ],
    },
    "Raleigh": {
        "center": (35.7796, -78.6382), "state": "NC", "zones": [
            ("Downtown", 0.000, 0.000, 84, "Urban Core"),
            ("North Hills", 0.008, -0.005, 82, "Urban Core"),
            ("Glenwood South", 0.003, -0.005, 80, "Urban Core"),
            ("Durham", 0.010, -0.030, 78, "Urban Core"),
            ("Chapel Hill", 0.000, -0.055, 72, "Suburban"),
            ("Cary", -0.010, -0.012, 88, "Suburban"),
            ("Apex", -0.020, -0.020, 90, "Exurban"),
            ("Holly Springs", -0.025, -0.015, 86, "Exurban"),
            ("Wake Forest", 0.020, 0.005, 84, "Exurban"),
            ("Fuquay-Varina", -0.030, -0.008, 82, "Exurban"),
        ],
    },
    "San Francisco": {
        "center": (37.7749, -122.4194), "state": "CA", "zones": [
            ("Financial District", 0.000, 0.000, 18, "Urban Core"),
            ("SoMa", -0.003, 0.005, 22, "Urban Core"),
            ("Mission", -0.008, 0.002, 20, "Urban Core"),
            ("Castro", -0.005, -0.008, 16, "Urban Core"),
            ("Marina", 0.008, -0.010, 24, "Urban Core"),
            ("Hayes Valley", 0.002, -0.005, 26, "Urban Core"),
            ("Sunset", -0.005, -0.030, 28, "Suburban"),
            ("Oakland", -0.005, 0.030, 30, "Urban Core"),
            ("Berkeley", 0.010, 0.035, 32, "Suburban"),
            ("San Mateo", -0.030, 0.015, 36, "Suburban"),
            ("Palo Alto", -0.050, 0.025, 34, "Suburban"),
            ("Fremont", -0.060, 0.055, 42, "Exurban"),
        ],
    },
    "Minneapolis": {
        "center": (44.9778, -93.2650), "state": "MN", "zones": [
            ("Downtown", 0.000, 0.000, 42, "Urban Core"),
            ("North Loop", 0.005, -0.005, 48, "Urban Core"),
            ("Uptown", -0.008, -0.008, 44, "Urban Core"),
            ("Northeast", 0.008, 0.005, 50, "Urban Core"),
            ("St. Paul Downtown", 0.000, 0.020, 40, "Urban Core"),
            ("Edina", -0.015, -0.012, 46, "Suburban"),
            ("Bloomington", -0.020, -0.005, 44, "Suburban"),
            ("Plymouth", 0.015, -0.020, 52, "Suburban"),
            ("Maple Grove", 0.025, -0.025, 56, "Exurban"),
            ("Woodbury", -0.005, 0.035, 54, "Exurban"),
        ],
    },
    "Detroit": {
        "center": (42.3314, -83.0458), "state": "MI", "zones": [
            ("Downtown", 0.000, 0.000, 38, "Urban Core"),
            ("Midtown", 0.005, -0.003, 42, "Urban Core"),
            ("Corktown", 0.002, -0.008, 44, "Urban Core"),
            ("Rivertown", -0.003, 0.005, 36, "Urban Core"),
            ("East Side", 0.005, 0.015, 18, "Urban Core"),
            ("West Side", 0.003, -0.015, 22, "Urban Core"),
            ("Dearborn", -0.005, -0.020, 40, "Suburban"),
            ("Royal Oak", 0.020, -0.010, 52, "Suburban"),
            ("Troy", 0.025, 0.005, 56, "Suburban"),
            ("Ann Arbor", 0.010, -0.080, 50, "Exurban"),
            ("Novi", 0.020, -0.035, 58, "Exurban"),
        ],
    },
}


def get_available_metros() -> list[str]:
    """Returns sorted list of metro names that have zip-level data."""
    return sorted(_METRO_DATA.keys())


def get_zip_data(metro: str) -> pd.DataFrame:
    """
    Returns zip/neighborhood-level migration data for a metro area.
    Uses seeded mock data with realistic lat/lon offsets from city center.
    """
    info = _METRO_DATA.get(metro)
    if not info:
        # Try case-insensitive match
        for name, data in _METRO_DATA.items():
            if name.lower() == metro.lower():
                info = data
                metro = name
                break
    if not info:
        return pd.DataFrame()

    center_lat, center_lon = info["center"]
    state = info["state"]
    rng = np.random.RandomState(hash(metro) % 2**31)

    rows = []
    for name, lat_off, lon_off, base_score, zone_type in info["zones"]:
        # Add small random noise for natural look
        lat = center_lat + lat_off + rng.normal(0, 0.002)
        lon = center_lon + lon_off + rng.normal(0, 0.002)
        score = max(5, min(98, base_score + rng.randint(-5, 6)))
        growth = round((score - 50) / 18 + rng.normal(0, 0.3), 1)
        rent_growth = round(growth * 1.2 + rng.normal(0, 0.5), 1)
        zip_code = f"{rng.randint(10000, 99999)}"

        rows.append({
            "zip": zip_code,
            "name": name,
            "metro": metro,
            "state": state,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "pop_growth_pct": growth,
            "migration_score": score,
            "median_rent_growth_pct": rent_growth,
            "neighborhood_type": zone_type,
        })

    return pd.DataFrame(rows)


def get_metro_center(metro: str) -> tuple:
    """Returns (lat, lon) for a metro center, or None."""
    info = _METRO_DATA.get(metro)
    if not info:
        for name, data in _METRO_DATA.items():
            if name.lower() == metro.lower():
                return data["center"]
        return None
    return info["center"]
