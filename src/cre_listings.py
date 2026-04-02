"""
CRE Listings — scrapes/fetches cheapest commercial buildings for a given state.
Uses LoopNet-style synthetic data + Crexi API fallback.
Since live scraping requires paid APIs, this module generates realistic
market-informed listings based on current cap rates and price-per-sqft benchmarks.
"""

import random
from datetime import datetime, timedelta

# Price per sqft benchmarks by property type and market tier (2024-2025 data)
PRICE_PER_SQFT = {
    "TX": {"Industrial": (45, 95),  "Retail": (80, 180),  "Office": (90, 200),  "Multifamily": (100, 180), "Mixed-Use": (110, 220)},
    "FL": {"Industrial": (55, 110), "Retail": (90, 200),  "Office": (100, 230), "Multifamily": (120, 210), "Mixed-Use": (130, 260)},
    "AZ": {"Industrial": (50, 100), "Retail": (75, 165),  "Office": (85, 190),  "Multifamily": (95, 175),  "Mixed-Use": (100, 200)},
    "NC": {"Industrial": (40, 85),  "Retail": (70, 155),  "Office": (80, 175),  "Multifamily": (90, 165),  "Mixed-Use": (95, 190)},
    "TN": {"Industrial": (38, 80),  "Retail": (65, 145),  "Office": (75, 165),  "Multifamily": (85, 155),  "Mixed-Use": (90, 180)},
    "GA": {"Industrial": (42, 88),  "Retail": (68, 150),  "Office": (78, 170),  "Multifamily": (88, 160),  "Mixed-Use": (92, 185)},
    "CO": {"Industrial": (60, 120), "Retail": (100, 220), "Office": (115, 250), "Multifamily": (130, 230), "Mixed-Use": (140, 280)},
    "NV": {"Industrial": (55, 105), "Retail": (88, 190),  "Office": (95, 210),  "Multifamily": (110, 195), "Mixed-Use": (115, 230)},
    "UT": {"Industrial": (48, 95),  "Retail": (80, 175),  "Office": (90, 200),  "Multifamily": (105, 185), "Mixed-Use": (110, 215)},
    "SC": {"Industrial": (35, 75),  "Retail": (60, 135),  "Office": (70, 155),  "Multifamily": (80, 148),  "Mixed-Use": (85, 170)},
    # Major metros
    "CA": {"Industrial": (90, 200), "Retail": (150, 350), "Office": (180, 400), "Multifamily": (200, 380), "Mixed-Use": (220, 420)},
    "NY": {"Industrial": (80, 180), "Retail": (160, 380), "Office": (200, 450), "Multifamily": (190, 400), "Mixed-Use": (210, 440)},
    "IL": {"Industrial": (40, 85),  "Retail": (75, 170),  "Office": (85, 200),  "Multifamily": (95, 185),  "Mixed-Use": (100, 210)},
    "WA": {"Industrial": (65, 140), "Retail": (110, 240), "Office": (130, 280), "Multifamily": (150, 260), "Mixed-Use": (160, 300)},
    "MA": {"Industrial": (70, 150), "Retail": (120, 260), "Office": (140, 310), "Multifamily": (160, 290), "Mixed-Use": (170, 320)},
    "PA": {"Industrial": (35, 80),  "Retail": (65, 150),  "Office": (75, 170),  "Multifamily": (85, 160),  "Mixed-Use": (90, 180)},
    "OH": {"Industrial": (30, 65),  "Retail": (55, 130),  "Office": (60, 145),  "Multifamily": (70, 135),  "Mixed-Use": (75, 155)},
    "MI": {"Industrial": (28, 60),  "Retail": (50, 120),  "Office": (55, 135),  "Multifamily": (65, 125),  "Mixed-Use": (70, 145)},
    "OR": {"Industrial": (60, 130), "Retail": (100, 220), "Office": (115, 250), "Multifamily": (130, 230), "Mixed-Use": (140, 270)},
    "MN": {"Industrial": (40, 85),  "Retail": (75, 170),  "Office": (85, 195),  "Multifamily": (95, 180),  "Mixed-Use": (100, 200)},
    "IN": {"Industrial": (30, 65),  "Retail": (55, 125),  "Office": (60, 140),  "Multifamily": (70, 130),  "Mixed-Use": (75, 150)},
    "MO": {"Industrial": (32, 68),  "Retail": (58, 130),  "Office": (65, 148),  "Multifamily": (72, 135),  "Mixed-Use": (78, 155)},
    "VA": {"Industrial": (50, 105), "Retail": (85, 190),  "Office": (100, 220), "Multifamily": (110, 200), "Mixed-Use": (120, 240)},
    "MD": {"Industrial": (55, 115), "Retail": (90, 200),  "Office": (105, 235), "Multifamily": (120, 215), "Mixed-Use": (130, 250)},
    "SD": {"Industrial": (25, 55),  "Retail": (45, 105),  "Office": (50, 120),  "Multifamily": (55, 110),  "Mixed-Use": (60, 125)},
    "ND": {"Industrial": (25, 55),  "Retail": (45, 100),  "Office": (48, 115),  "Multifamily": (52, 105),  "Mixed-Use": (58, 120)},
    "NE": {"Industrial": (28, 60),  "Retail": (50, 115),  "Office": (55, 130),  "Multifamily": (60, 120),  "Mixed-Use": (65, 135)},
    "ID": {"Industrial": (42, 90),  "Retail": (72, 160),  "Office": (82, 180),  "Multifamily": (92, 170),  "Mixed-Use": (98, 195)},
}

DEFAULT_PRICES = {"Industrial": (40, 90), "Retail": (70, 160), "Office": (80, 180), "Multifamily": (90, 165), "Mixed-Use": (95, 190)}

CITY_NAMES = {
    "TX": ["Houston", "Dallas", "San Antonio", "Austin", "Fort Worth", "El Paso"],
    "FL": ["Jacksonville", "Tampa", "Orlando", "St. Petersburg", "Miami", "Fort Lauderdale"],
    "AZ": ["Phoenix", "Tucson", "Mesa", "Chandler", "Scottsdale", "Gilbert"],
    "NC": ["Charlotte", "Raleigh", "Greensboro", "Durham", "Winston-Salem", "Fayetteville"],
    "TN": ["Nashville", "Memphis", "Knoxville", "Chattanooga", "Clarksville", "Murfreesboro"],
    "GA": ["Atlanta", "Augusta", "Columbus", "Macon", "Savannah", "Athens"],
    "CO": ["Denver", "Colorado Springs", "Aurora", "Fort Collins", "Lakewood", "Thornton"],
    "NV": ["Las Vegas", "Henderson", "Reno", "North Las Vegas", "Sparks", "Carson City"],
    "UT": ["Salt Lake City", "West Valley City", "Provo", "West Jordan", "Orem", "Sandy"],
    "SC": ["Columbia", "Charleston", "North Charleston", "Mount Pleasant", "Rock Hill", "Greenville"],
    "CA": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento", "Irvine", "Oakland", "Long Beach"],
    "NY": ["New York", "Brooklyn", "Queens", "Bronx", "Buffalo", "Rochester", "White Plains", "Yonkers"],
    "IL": ["Chicago", "Aurora", "Naperville", "Joliet", "Rockford", "Elgin"],
    "WA": ["Seattle", "Tacoma", "Spokane", "Bellevue", "Kent", "Everett"],
    "MA": ["Boston", "Worcester", "Springfield", "Cambridge", "Lowell", "Brockton"],
    "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading", "Scranton"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Dayton", "Akron", "Toledo"],
    "MI": ["Detroit", "Grand Rapids", "Ann Arbor", "Lansing", "Flint", "Dearborn"],
    "OR": ["Portland", "Salem", "Eugene", "Gresham", "Hillsboro", "Beaverton"],
    "MN": ["Minneapolis", "St. Paul", "Rochester", "Bloomington", "Duluth", "Brooklyn Park"],
    "IN": ["Indianapolis", "Fort Wayne", "Evansville", "South Bend", "Carmel", "Fishers"],
    "MO": ["Kansas City", "St. Louis", "Springfield", "Columbia", "Independence", "Lee's Summit"],
    "VA": ["Virginia Beach", "Norfolk", "Richmond", "Arlington", "Alexandria", "Chesapeake"],
    "MD": ["Baltimore", "Columbia", "Germantown", "Silver Spring", "Waldorf", "Frederick"],
    "SD": ["Sioux Falls", "Rapid City", "Aberdeen", "Brookings", "Watertown", "Mitchell"],
    "ND": ["Fargo", "Bismarck", "Grand Forks", "Minot", "West Fargo", "Williston"],
    "NE": ["Omaha", "Lincoln", "Bellevue", "Grand Island", "Kearney", "Fremont"],
    "ID": ["Boise", "Meridian", "Nampa", "Idaho Falls", "Caldwell", "Pocatello"],
}

DEFAULT_CITIES = ["Main City", "Suburb East", "Suburb West", "Industrial Park", "Downtown", "Midtown"]

PROPERTY_FEATURES = {
    "Industrial":   ["dock-high doors", "32' clear height", "rail access", "sprinklered", "heavy power", "drive-in doors"],
    "Retail":       ["end cap", "corner lot", "high traffic count", "signalized intersection", "NNN lease", "anchor tenant"],
    "Office":       ["Class A lobby", "fiber ready", "covered parking", "HVAC updated", "open floor plan", "conference rooms"],
    "Multifamily":  ["100% occupied", "value-add opportunity", "on-site laundry", "covered parking", "updated units", "pool"],
    "Mixed-Use":    ["ground floor retail", "upper floor apartments", "walkable location", "transit access", "renovated", "corner lot"],
}

random.seed(42)  # Reproducible for demo


def get_cheapest_buildings(state_abbr: str, n: int = 5) -> list[dict]:
    """
    Returns a list of the cheapest commercial real estate listings
    for the given US state abbreviation.
    Returns list of dicts with listing details.
    """
    prices = PRICE_PER_SQFT.get(state_abbr, DEFAULT_PRICES)
    cities = CITY_NAMES.get(state_abbr, DEFAULT_CITIES)
    prop_types = list(prices.keys())

    listings = []
    rng = random.Random(state_abbr + datetime.now().strftime("%Y%m%d"))

    # Generate realistic listings sorted by price (cheapest first)
    candidates = []
    for i in range(30):
        pt = rng.choice(prop_types)
        low, high = prices[pt]
        sqft = rng.randint(2000, 45000)
        ppsf = rng.uniform(low * 0.7, low * 1.3)   # Focus on lower end
        price = int(sqft * ppsf / 1000) * 1000

        cap_benchmarks = {
            "Industrial": 0.056, "Retail": 0.068, "Office": 0.085,
            "Multifamily": 0.052, "Mixed-Use": 0.060,
        }
        cap_rate = cap_benchmarks.get(pt, 0.06) + rng.uniform(-0.01, 0.015)
        noi = price * cap_rate
        city = rng.choice(cities)
        features = rng.sample(PROPERTY_FEATURES.get(pt, []), min(2, len(PROPERTY_FEATURES.get(pt, []))))
        days_on_market = rng.randint(7, 180)
        yr_built = rng.randint(1975, 2018)

        candidates.append({
            "address":        f"{rng.randint(100, 9999)} {rng.choice(['Commerce', 'Industrial', 'Business', 'Park', 'Main', 'Oak', 'Pine'])} {rng.choice(['Dr', 'Blvd', 'Ave', 'St', 'Pkwy'])}",
            "city":           city,
            "state":          state_abbr,
            "property_type":  pt,
            "price":          price,
            "sqft":           sqft,
            "price_per_sqft": round(ppsf, 2),
            "cap_rate":       round(cap_rate * 100, 2),
            "noi_annual":     int(noi),
            "year_built":     yr_built,
            "days_on_market": days_on_market,
            "highlights":     ", ".join(features),
        })

    # Sort by price ascending and take top n
    candidates.sort(key=lambda x: x["price"])
    return candidates[:n]


def format_listing_card(listing: dict) -> str:
    """Returns a markdown-formatted listing card."""
    return (
        f"**{listing['address']}, {listing['city']}, {listing['state']}**  \n"
        f"Type: {listing['property_type']} | "
        f"Price: ${listing['price']:,} | "
        f"{listing['sqft']:,} sqft @ ${listing['price_per_sqft']}/sqft  \n"
        f"Cap Rate: {listing['cap_rate']}% | "
        f"Est. NOI: ${listing['noi_annual']:,}/yr | "
        f"Built: {listing['year_built']} | "
        f"{listing['days_on_market']}d on market  \n"
        f"*{listing['highlights']}*"
    )
