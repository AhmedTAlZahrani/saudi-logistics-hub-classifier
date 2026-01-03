import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic


# Saudi Arabia geographic bounds
LAT_MIN = 16.0
LAT_MAX = 33.0
LON_MIN = 34.0
LON_MAX = 56.0

# Number of candidate locations to generate
NUM_LOCATIONS = 300

# Anchor cities with real coordinates and characteristics
ANCHOR_CITIES = {
    "Jubail": {"lat": 27.01, "lon": 49.66, "region": "Eastern", "type": "petrochemical"},
    "Jeddah": {"lat": 21.54, "lon": 39.17, "region": "Western", "type": "port"},
    "Riyadh": {"lat": 24.71, "lon": 46.68, "region": "Central", "type": "capital"},
    "Dammam": {"lat": 26.43, "lon": 50.10, "region": "Eastern", "type": "port"},
    "King Abdullah Economic City": {"lat": 22.45, "lon": 39.13, "region": "Western", "type": "economic"},
    "Ras Al-Khair": {"lat": 27.49, "lon": 49.24, "region": "Eastern", "type": "industrial"},
    "NEOM": {"lat": 28.00, "lon": 35.00, "region": "Northern", "type": "megaproject"},
    "Yanbu": {"lat": 24.09, "lon": 38.06, "region": "Western", "type": "port"},
    "Tabuk": {"lat": 28.39, "lon": 36.57, "region": "Northern", "type": "logistics"},
    "Abha": {"lat": 18.22, "lon": 42.50, "region": "Southern", "type": "regional"},
    "Jizan": {"lat": 16.89, "lon": 42.57, "region": "Southern", "type": "port"},
    "Hail": {"lat": 27.52, "lon": 41.69, "region": "Northern", "type": "agriculture"},
    "Sudair Industrial City": {"lat": 25.59, "lon": 45.58, "region": "Central", "type": "industrial"},
    "Rabigh": {"lat": 22.80, "lon": 39.03, "region": "Western", "type": "refinery"},
    "Al Kharj": {"lat": 24.15, "lon": 47.31, "region": "Central", "type": "military"},
}

# Region definitions for classification
REGIONS = ["Eastern", "Central", "Western", "Northern", "Southern"]

# Suitability labels
SUITABILITY_LABELS = ["optimal", "good", "marginal", "unsuitable"]


def _assign_region(lat, lon):
    if lon > 47.0:
        return "Eastern"
    elif lon < 40.0 and lat > 26.0:
        return "Northern"
    elif lon < 40.5:
        return "Western"
    elif lat < 20.0:
        return "Southern"
    else:
        return "Central"


def _compute_suitability(row):
    score = 0.0

    # Distance factors (lower is better)
    if row["distance_to_rail_km"] < 30:
        score += 25
    elif row["distance_to_rail_km"] < 80:
        score += 15
    elif row["distance_to_rail_km"] < 150:
        score += 5

    if row["distance_to_highway_km"] < 20:
        score += 15
    elif row["distance_to_highway_km"] < 60:
        score += 8

    if row["distance_to_port_km"] < 100:
        score += 10
    elif row["distance_to_port_km"] < 250:
        score += 5

    # Population and commercial factors
    if row["population_50km_radius"] > 500000:
        score += 15
    elif row["population_50km_radius"] > 200000:
        score += 8

    score += row["commercial_activity_index"] * 2
    score += row["industrial_zones_nearby"] * 3

    # Infrastructure factors
    if row["power_grid_capacity_mw"] > 200:
        score += 10
    elif row["power_grid_capacity_mw"] > 80:
        score += 5

    score += row["water_availability_score"]
    score -= row["land_cost_index"] * 0.5

    # Labor and freight
    if row["labor_market_size"] > 20000:
        score += 5
    if row["rail_freight_volume_nearby"] > 500000:
        score += 10
    elif row["rail_freight_volume_nearby"] > 100000:
        score += 5

    # Classify
    if score >= 70:
        return "optimal"
    elif score >= 50:
        return "good"
    elif score >= 30:
        return "marginal"
    else:
        return "unsuitable"


def generate_locations(n=NUM_LOCATIONS, seed=42):
    """Generate synthetic logistics hub candidate locations across Saudi Arabia.

    Creates candidate locations clustered around real Saudi cities with
    varying quality attributes. Each location receives a suitability
    label based on composite scoring of infrastructure, accessibility,
    and market factors.

    Args:
        n: Number of candidate locations to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with location features and suitability labels.
    """
    rng = np.random.RandomState(seed)
    records = []
    anchor_list = list(ANCHOR_CITIES.items())

    for i in range(n):
        # Pick an anchor city to cluster around
        anchor_name, anchor = anchor_list[i % len(anchor_list)]
        jitter_lat = rng.normal(0, 0.8)
        jitter_lon = rng.normal(0, 0.8)

        lat = np.clip(anchor["lat"] + jitter_lat, LAT_MIN, LAT_MAX)
        lon = np.clip(anchor["lon"] + jitter_lon, LON_MIN, LON_MAX)
        region = _assign_region(lat, lon)

        # Generate a descriptive name
        suffix = rng.choice(["North", "South", "East", "West", "Central",
                             "Industrial", "Port", "Junction", "Terminal", "Gateway"])
        name = f"{anchor_name}-{suffix}-{i:03d}"

        # Distance features (km)
        distance_to_rail = rng.exponential(50) + rng.uniform(0, 20)
        distance_to_highway = rng.exponential(25) + rng.uniform(0, 10)
        distance_to_port = geodesic((lat, lon), (26.43, 50.10)).km * rng.uniform(0.7, 1.3)

        # Adjust distances for favorable anchor types
        if anchor["type"] in ("port", "petrochemical"):
            distance_to_port *= 0.3
            distance_to_rail *= 0.5
        elif anchor["type"] in ("capital", "economic"):
            distance_to_highway *= 0.4
            distance_to_rail *= 0.6

        # Population and economic features
        base_pop = rng.lognormal(11, 1.5)
        if anchor["type"] in ("capital", "port"):
            base_pop *= 3
        population_50km = int(np.clip(base_pop, 5000, 4000000))

        industrial_zones = int(rng.poisson(3 if anchor["type"] == "industrial" else 1))
        commercial_index = round(rng.uniform(1, 10) * (1.5 if anchor["type"] in ("capital", "port", "economic") else 1.0), 1)
        commercial_index = min(commercial_index, 10.0)

        # Infrastructure features
        land_cost = round(rng.uniform(1, 10), 1)
        water_score = round(rng.uniform(1, 10), 1)
        power_capacity = round(rng.lognormal(4.5, 1.0), 1)
        labor_size = int(rng.lognormal(9, 1.2))
        existing_warehouse = int(rng.lognormal(9, 1.5))
        avg_temp = round(rng.uniform(25, 48) if lat < 25 else rng.uniform(18, 42), 1)
        rail_freight = int(rng.lognormal(11, 1.8))

        records.append({
            "name": name,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "region": region,
            "distance_to_rail_km": round(distance_to_rail, 1),
            "distance_to_highway_km": round(distance_to_highway, 1),
            "distance_to_port_km": round(distance_to_port, 1),
            "population_50km_radius": population_50km,
            "industrial_zones_nearby": industrial_zones,
            "commercial_activity_index": commercial_index,
            "land_cost_index": land_cost,
            "water_availability_score": water_score,
            "power_grid_capacity_mw": power_capacity,
            "labor_market_size": labor_size,
            "existing_warehouse_sqm": existing_warehouse,
            "avg_temperature": avg_temp,
            "rail_freight_volume_nearby": rail_freight,
        })

    df = pd.DataFrame(records)
    df["suitability"] = df.apply(_compute_suitability, axis=1)

    print(f"Generated {len(df)} candidate locations across {df['region'].nunique()} regions")
    print(f"Suitability distribution:\n{df['suitability'].value_counts().to_string()}")

    return df


def save_locations(df, path="data/hub_locations.csv"):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} locations to {output}")


if __name__ == "__main__":
    locations = generate_locations()
    save_locations(locations)
