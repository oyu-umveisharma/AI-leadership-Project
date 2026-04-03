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

# Metro definitions: center coords, state, and neighborhoods with REAL lat/lon
# Each neighborhood: (name, lat, lon, base_score, type)
_METRO_DATA = {
    "New York City": {
        "center": (40.7128, -74.0060), "state": "NY", "zones": [
            ("Midtown Manhattan", 40.7549, -73.9840, 55, "Urban Core"),
            ("Lower Manhattan / FiDi", 40.7075, -74.0113, 50, "Urban Core"),
            ("Upper East Side", 40.7736, -73.9566, 48, "Urban Core"),
            ("Upper West Side", 40.7870, -73.9754, 46, "Urban Core"),
            ("Harlem", 40.8116, -73.9465, 58, "Urban Core"),
            ("Brooklyn Heights", 40.6960, -73.9936, 52, "Urban Core"),
            ("Williamsburg", 40.7081, -73.9571, 60, "Urban Core"),
            ("Park Slope", 40.6710, -73.9777, 54, "Urban Core"),
            ("Bushwick", 40.6944, -73.9213, 62, "Urban Core"),
            ("Long Island City", 40.7425, -73.9561, 65, "Urban Core"),
            ("Astoria", 40.7723, -73.9301, 58, "Suburban"),
            ("Flushing", 40.7580, -73.8330, 56, "Suburban"),
            ("Jersey City", 40.7178, -74.0431, 68, "Suburban"),
            ("Hoboken", 40.7440, -74.0324, 64, "Suburban"),
            ("Fort Lee", 40.8509, -73.9712, 50, "Suburban"),
            ("Yonkers", 40.9312, -73.8987, 42, "Suburban"),
            ("White Plains", 41.0340, -73.7629, 44, "Exurban"),
            ("Stamford", 41.0534, -73.5387, 52, "Exurban"),
        ],
    },
    "Los Angeles": {
        "center": (34.0522, -118.2437), "state": "CA", "zones": [
            ("Downtown LA", 34.0407, -118.2468, 28, "Urban Core"),
            ("Hollywood", 34.0928, -118.3287, 24, "Urban Core"),
            ("Santa Monica", 34.0195, -118.4912, 30, "Urban Core"),
            ("Beverly Hills", 34.0736, -118.4004, 22, "Urban Core"),
            ("West LA / Westwood", 34.0585, -118.4441, 26, "Urban Core"),
            ("Silver Lake", 34.0869, -118.2702, 35, "Urban Core"),
            ("Koreatown", 34.0578, -118.3015, 32, "Urban Core"),
            ("South LA", 33.9425, -118.2551, 20, "Urban Core"),
            ("Pasadena", 34.1478, -118.1445, 38, "Suburban"),
            ("Glendale", 34.1425, -118.2551, 36, "Suburban"),
            ("Burbank", 34.1808, -118.3090, 40, "Suburban"),
            ("Long Beach", 33.7701, -118.1937, 42, "Suburban"),
            ("Torrance", 33.8358, -118.3406, 44, "Suburban"),
            ("Irvine", 33.6846, -117.8265, 55, "Exurban"),
            ("Ontario / Inland Empire", 34.0633, -117.6509, 62, "Exurban"),
            ("Riverside", 33.9533, -117.3962, 60, "Exurban"),
        ],
    },
    "Chicago": {
        "center": (41.8781, -87.6298), "state": "IL", "zones": [
            ("The Loop", 41.8819, -87.6278, 35, "Urban Core"),
            ("River North", 41.8926, -87.6341, 40, "Urban Core"),
            ("Lincoln Park", 41.9214, -87.6513, 42, "Urban Core"),
            ("Wicker Park", 41.9088, -87.6796, 45, "Urban Core"),
            ("West Loop", 41.8825, -87.6548, 48, "Urban Core"),
            ("South Loop", 41.8569, -87.6247, 38, "Urban Core"),
            ("Hyde Park", 41.7943, -87.5907, 36, "Urban Core"),
            ("Pilsen", 41.8566, -87.6564, 44, "Urban Core"),
            ("Bronzeville", 41.8236, -87.6150, 32, "Urban Core"),
            ("South Side", 41.7508, -87.6242, 18, "Urban Core"),
            ("Evanston", 42.0451, -87.6877, 46, "Suburban"),
            ("Oak Park", 41.8850, -87.7845, 50, "Suburban"),
            ("Naperville", 41.7508, -88.1535, 68, "Exurban"),
            ("Schaumburg", 42.0334, -88.0834, 58, "Exurban"),
            ("Aurora", 41.7606, -88.3201, 54, "Exurban"),
            ("Joliet", 41.5250, -88.0817, 52, "Exurban"),
        ],
    },
    "Houston": {
        "center": (29.7604, -95.3698), "state": "TX", "zones": [
            ("Downtown", 29.7604, -95.3698, 72, "Urban Core"),
            ("Midtown", 29.7381, -95.3861, 74, "Urban Core"),
            ("Montrose", 29.7460, -95.3952, 70, "Urban Core"),
            ("Heights", 29.7905, -95.3982, 76, "Urban Core"),
            ("Medical Center / NRG", 29.7079, -95.3982, 68, "Urban Core"),
            ("Galleria / Uptown", 29.7358, -95.4613, 72, "Urban Core"),
            ("EaDo", 29.7513, -95.3463, 78, "Urban Core"),
            ("Katy", 29.7858, -95.8245, 82, "Exurban"),
            ("Sugar Land", 29.6197, -95.6349, 80, "Exurban"),
            ("The Woodlands", 30.1658, -95.4613, 86, "Exurban"),
            ("Pearland", 29.5636, -95.2860, 78, "Suburban"),
            ("Cypress", 29.9691, -95.6970, 84, "Exurban"),
        ],
    },
    "Phoenix": {
        "center": (33.4484, -112.0740), "state": "AZ", "zones": [
            ("Downtown", 33.4484, -112.0740, 75, "Urban Core"),
            ("Scottsdale", 33.4942, -111.9261, 82, "Suburban"),
            ("Tempe", 33.4255, -111.9400, 78, "Suburban"),
            ("Mesa", 33.4152, -111.8315, 72, "Suburban"),
            ("Chandler", 33.3062, -111.8413, 80, "Suburban"),
            ("Gilbert", 33.3528, -111.7890, 86, "Exurban"),
            ("Glendale", 33.5387, -112.1860, 68, "Suburban"),
            ("Peoria", 33.5806, -112.2374, 70, "Exurban"),
            ("Surprise", 33.6292, -112.3680, 74, "Exurban"),
            ("Buckeye", 33.3703, -112.5838, 88, "Exurban"),
            ("Queen Creek", 33.2487, -111.6343, 90, "Exurban"),
        ],
    },
    "Dallas": {
        "center": (32.7767, -96.7970), "state": "TX", "zones": [
            ("Downtown", 32.7767, -96.7970, 75, "Urban Core"),
            ("Uptown", 32.8003, -96.8014, 78, "Urban Core"),
            ("Deep Ellum", 32.7835, -96.7835, 72, "Urban Core"),
            ("Bishop Arts", 32.7448, -96.8258, 70, "Urban Core"),
            ("Plano", 33.0198, -96.6989, 85, "Suburban"),
            ("Frisco", 33.1507, -96.8236, 92, "Exurban"),
            ("McKinney", 33.1972, -96.6397, 90, "Exurban"),
            ("Richardson", 32.9483, -96.7299, 76, "Suburban"),
            ("Arlington", 32.7357, -97.1081, 68, "Suburban"),
            ("Irving", 32.8140, -96.9489, 74, "Suburban"),
            ("Fort Worth", 32.7555, -97.3308, 72, "Suburban"),
            ("Allen", 33.1032, -96.6714, 88, "Exurban"),
        ],
    },
    "Austin": {
        "center": (30.2672, -97.7431), "state": "TX", "zones": [
            ("Downtown", 30.2672, -97.7431, 88, "Urban Core"),
            ("South Congress", 30.2466, -97.7494, 86, "Urban Core"),
            ("East Austin", 30.2649, -97.7186, 82, "Urban Core"),
            ("Hyde Park", 30.3043, -97.7277, 80, "Urban Core"),
            ("Domain / North Austin", 30.4021, -97.7253, 90, "Suburban"),
            ("Mueller", 30.2988, -97.7025, 84, "Suburban"),
            ("Cedar Park", 30.5052, -97.8203, 92, "Exurban"),
            ("Round Rock", 30.5083, -97.6789, 94, "Exurban"),
            ("Georgetown", 30.6333, -97.6778, 88, "Exurban"),
            ("Pflugerville", 30.4394, -97.6200, 90, "Exurban"),
            ("Bee Cave / Lakeway", 30.3085, -97.9433, 86, "Exurban"),
        ],
    },
    "Miami": {
        "center": (25.7617, -80.1918), "state": "FL", "zones": [
            ("Downtown / Brickell", 25.7617, -80.1918, 76, "Urban Core"),
            ("Wynwood / Midtown", 25.8029, -80.1988, 78, "Urban Core"),
            ("South Beach", 25.7826, -80.1341, 68, "Urban Core"),
            ("Coconut Grove", 25.7270, -80.2564, 72, "Urban Core"),
            ("Coral Gables", 25.7215, -80.2684, 70, "Suburban"),
            ("Doral", 25.8195, -80.3553, 80, "Suburban"),
            ("Aventura", 25.9564, -80.1392, 74, "Suburban"),
            ("Homestead", 25.4687, -80.4776, 66, "Exurban"),
            ("Fort Lauderdale", 26.1224, -80.1373, 72, "Suburban"),
            ("Hollywood", 26.0112, -80.1495, 70, "Suburban"),
            ("Pembroke Pines", 26.0131, -80.2894, 68, "Suburban"),
        ],
    },
    "Seattle": {
        "center": (47.6062, -122.3321), "state": "WA", "zones": [
            ("Downtown", 47.6062, -122.3321, 52, "Urban Core"),
            ("Capitol Hill", 47.6253, -122.3222, 54, "Urban Core"),
            ("South Lake Union", 47.6256, -122.3368, 58, "Urban Core"),
            ("Ballard", 47.6677, -122.3846, 56, "Urban Core"),
            ("Fremont", 47.6511, -122.3502, 55, "Urban Core"),
            ("Beacon Hill", 47.5680, -122.3088, 48, "Urban Core"),
            ("West Seattle", 47.5714, -122.3867, 46, "Urban Core"),
            ("Bellevue", 47.6101, -122.2015, 62, "Suburban"),
            ("Redmond", 47.6740, -122.1215, 66, "Suburban"),
            ("Kirkland", 47.6815, -122.2087, 60, "Suburban"),
            ("Tacoma", 47.2529, -122.4443, 58, "Suburban"),
            ("Everett", 47.9790, -122.2021, 54, "Suburban"),
        ],
    },
    "Denver": {
        "center": (39.7392, -104.9903), "state": "CO", "zones": [
            ("Downtown / LoDo", 39.7509, -105.0002, 58, "Urban Core"),
            ("RiNo / Five Points", 39.7668, -104.9801, 62, "Urban Core"),
            ("Capitol Hill", 39.7312, -104.9792, 56, "Urban Core"),
            ("Cherry Creek", 39.7168, -104.9534, 54, "Urban Core"),
            ("Highlands", 39.7614, -105.0116, 60, "Urban Core"),
            ("Aurora", 39.7294, -104.8319, 52, "Suburban"),
            ("Lakewood", 39.7047, -105.0814, 50, "Suburban"),
            ("Arvada", 39.8028, -105.0875, 56, "Suburban"),
            ("Boulder", 40.0150, -105.2705, 48, "Exurban"),
            ("Castle Rock", 39.3722, -104.8561, 72, "Exurban"),
            ("Parker", 39.5186, -104.7614, 68, "Exurban"),
        ],
    },
    "Atlanta": {
        "center": (33.7490, -84.3880), "state": "GA", "zones": [
            ("Downtown", 33.7490, -84.3880, 65, "Urban Core"),
            ("Midtown", 33.7844, -84.3833, 70, "Urban Core"),
            ("Buckhead", 33.8388, -84.3796, 68, "Urban Core"),
            ("Old Fourth Ward", 33.7692, -84.3639, 72, "Urban Core"),
            ("West Midtown", 33.7884, -84.4150, 74, "Urban Core"),
            ("Decatur", 33.7748, -84.2963, 66, "Suburban"),
            ("Sandy Springs", 33.9304, -84.3733, 64, "Suburban"),
            ("Marietta", 33.9526, -84.5499, 62, "Suburban"),
            ("Roswell", 34.0234, -84.3616, 68, "Suburban"),
            ("Alpharetta", 34.0754, -84.2941, 76, "Exurban"),
            ("Suwanee / Gwinnett", 34.0515, -84.0713, 80, "Exurban"),
        ],
    },
    "Boston": {
        "center": (42.3601, -71.0589), "state": "MA", "zones": [
            ("Downtown / Financial", 42.3554, -71.0605, 46, "Urban Core"),
            ("Back Bay", 42.3503, -71.0810, 44, "Urban Core"),
            ("South Boston / Seaport", 42.3382, -71.0455, 52, "Urban Core"),
            ("Cambridge", 42.3736, -71.1097, 48, "Urban Core"),
            ("Somerville", 42.3876, -71.0995, 50, "Urban Core"),
            ("Charlestown", 42.3782, -71.0602, 46, "Urban Core"),
            ("Jamaica Plain", 42.3097, -71.1151, 44, "Urban Core"),
            ("Brookline", 42.3318, -71.1212, 42, "Suburban"),
            ("Newton", 42.3370, -71.2092, 40, "Suburban"),
            ("Quincy", 42.2529, -71.0023, 48, "Suburban"),
        ],
    },
    "Nashville": {
        "center": (36.1627, -86.7816), "state": "TN", "zones": [
            ("Downtown / SoBro", 36.1627, -86.7762, 80, "Urban Core"),
            ("The Gulch", 36.1524, -86.7896, 82, "Urban Core"),
            ("Germantown", 36.1770, -86.7875, 78, "Urban Core"),
            ("East Nashville", 36.1783, -86.7552, 84, "Urban Core"),
            ("12 South", 36.1265, -86.7925, 76, "Urban Core"),
            ("Sylvan Park", 36.1567, -86.8233, 74, "Urban Core"),
            ("Brentwood", 36.0331, -86.7828, 72, "Suburban"),
            ("Franklin", 35.9251, -86.8689, 88, "Exurban"),
            ("Murfreesboro", 35.8456, -86.3903, 82, "Exurban"),
            ("Hendersonville", 36.3048, -86.6200, 78, "Suburban"),
            ("Mt. Juliet", 36.2001, -86.5186, 86, "Exurban"),
        ],
    },
    "Charlotte": {
        "center": (35.2271, -80.8431), "state": "NC", "zones": [
            ("Uptown", 35.2271, -80.8431, 78, "Urban Core"),
            ("South End", 35.2112, -80.8556, 82, "Urban Core"),
            ("NoDa", 35.2480, -80.8190, 76, "Urban Core"),
            ("Plaza Midwood", 35.2241, -80.8094, 74, "Urban Core"),
            ("Ballantyne", 35.0582, -80.8447, 80, "Suburban"),
            ("Huntersville", 35.4107, -80.8428, 84, "Exurban"),
            ("Mooresville", 35.5849, -80.8101, 78, "Exurban"),
            ("Fort Mill (SC)", 35.0074, -80.9451, 86, "Exurban"),
            ("Indian Trail", 35.0765, -80.6692, 82, "Exurban"),
        ],
    },
    "Raleigh": {
        "center": (35.7796, -78.6382), "state": "NC", "zones": [
            ("Downtown", 35.7796, -78.6382, 84, "Urban Core"),
            ("North Hills", 35.8378, -78.6432, 82, "Urban Core"),
            ("Glenwood South", 35.7903, -78.6500, 80, "Urban Core"),
            ("Durham", 35.9940, -78.8986, 78, "Urban Core"),
            ("Chapel Hill", 35.9132, -79.0558, 72, "Suburban"),
            ("Cary", 35.7915, -78.7811, 88, "Suburban"),
            ("Apex", 35.7327, -78.8503, 90, "Exurban"),
            ("Holly Springs", 35.6512, -78.8336, 86, "Exurban"),
            ("Wake Forest", 35.9799, -78.5097, 84, "Exurban"),
            ("Fuquay-Varina", 35.5843, -78.8000, 82, "Exurban"),
        ],
    },
    "San Francisco": {
        "center": (37.7749, -122.4194), "state": "CA", "zones": [
            ("Financial District", 37.7946, -122.3999, 18, "Urban Core"),
            ("SoMa", 37.7785, -122.3950, 22, "Urban Core"),
            ("Mission", 37.7599, -122.4148, 20, "Urban Core"),
            ("Castro", 37.7609, -122.4350, 16, "Urban Core"),
            ("Marina", 37.8015, -122.4368, 24, "Urban Core"),
            ("Hayes Valley", 37.7759, -122.4245, 26, "Urban Core"),
            ("Sunset", 37.7527, -122.4946, 28, "Suburban"),
            ("Oakland", 37.8044, -122.2712, 30, "Urban Core"),
            ("Berkeley", 37.8716, -122.2727, 32, "Suburban"),
            ("San Mateo", 37.5630, -122.3255, 36, "Suburban"),
            ("Palo Alto", 37.4419, -122.1430, 34, "Suburban"),
            ("Fremont", 37.5485, -121.9886, 42, "Exurban"),
        ],
    },
    "Minneapolis": {
        "center": (44.9778, -93.2650), "state": "MN", "zones": [
            ("Downtown", 44.9778, -93.2650, 42, "Urban Core"),
            ("North Loop", 44.9878, -93.2793, 48, "Urban Core"),
            ("Uptown", 44.9488, -93.2990, 44, "Urban Core"),
            ("Northeast", 44.9998, -93.2471, 50, "Urban Core"),
            ("St. Paul Downtown", 44.9537, -93.0900, 40, "Urban Core"),
            ("Edina", 44.8897, -93.3499, 46, "Suburban"),
            ("Bloomington", 44.8408, -93.2983, 44, "Suburban"),
            ("Plymouth", 45.0105, -93.4555, 52, "Suburban"),
            ("Maple Grove", 45.0725, -93.4558, 56, "Exurban"),
            ("Woodbury", 44.9239, -92.9594, 54, "Exurban"),
        ],
    },
    "Detroit": {
        "center": (42.3314, -83.0458), "state": "MI", "zones": [
            ("Downtown", 42.3314, -83.0458, 38, "Urban Core"),
            ("Midtown", 42.3541, -83.0644, 42, "Urban Core"),
            ("Corktown", 42.3360, -83.0713, 44, "Urban Core"),
            ("Rivertown", 42.3365, -83.0182, 36, "Urban Core"),
            ("East Side", 42.3726, -82.9953, 18, "Urban Core"),
            ("West Side", 42.3541, -83.1096, 22, "Urban Core"),
            ("Dearborn", 42.3223, -83.1763, 40, "Suburban"),
            ("Royal Oak", 42.4895, -83.1446, 52, "Suburban"),
            ("Troy", 42.6064, -83.1498, 56, "Suburban"),
            ("Ann Arbor", 42.2808, -83.7430, 50, "Exurban"),
            ("Novi", 42.4801, -83.4755, 58, "Exurban"),
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

    state = info["state"]
    rng = np.random.RandomState(hash(metro) % 2**31)

    rows = []
    for name, abs_lat, abs_lon, base_score, zone_type in info["zones"]:
        # Add small random noise for natural look
        lat = abs_lat + rng.normal(0, 0.001)
        lon = abs_lon + rng.normal(0, 0.001)
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
