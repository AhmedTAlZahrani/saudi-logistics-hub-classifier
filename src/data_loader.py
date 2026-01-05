import pandas as pd
import numpy as np
from pathlib import Path


# Saudi Arabia geographic bounds for validation
LAT_BOUNDS = (16.0, 33.0)
LON_BOUNDS = (34.0, 56.0)

# Feature columns (excluding name and target)
FEATURE_COLS = [
    "lat", "lon", "distance_to_rail_km", "distance_to_highway_km",
    "distance_to_port_km", "population_50km_radius", "industrial_zones_nearby",
    "commercial_activity_index", "land_cost_index", "water_availability_score",
    "power_grid_capacity_mw", "labor_market_size", "existing_warehouse_sqm",
    "avg_temperature", "rail_freight_volume_nearby", "region",
]

TARGET_COL = "suitability"


def load_location_data(path="data/hub_locations.csv"):
    """Load and validate logistics hub location data.

    Reads the CSV file, validates that coordinates fall within Saudi
    Arabia's geographic bounds, and separates features from target.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (features DataFrame, target Series, metadata DataFrame).
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")

    # Validate required columns
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate geographic coordinates
    df = _validate_coordinates(df)

    # Handle missing values
    df = _handle_missing(df)

    # Separate metadata, features, and target
    metadata = df[["name", "lat", "lon", "region"]].copy()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print(f"Features shape: {X.shape} | Target classes: {y.nunique()}")
    print(f"Class distribution:\n{y.value_counts().to_string()}")

    return X, y, metadata


def _validate_coordinates(df):
    """Validate that all coordinates fall within Saudi Arabia bounds.

    Args:
        df: DataFrame with lat and lon columns.

    Returns:
        Filtered DataFrame with only valid coordinates.
    """
    initial_count = len(df)

    valid_lat = df["lat"].between(LAT_BOUNDS[0], LAT_BOUNDS[1])
    valid_lon = df["lon"].between(LON_BOUNDS[0], LON_BOUNDS[1])
    valid = valid_lat & valid_lon

    df = df[valid].reset_index(drop=True)
    dropped = initial_count - len(df)

    if dropped > 0:
        print(f"Warning: Dropped {dropped} locations outside Saudi bounds")

    return df


def _handle_missing(df):
    """Handle missing values in the dataset.

    Numeric columns are filled with median, categorical with mode.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median={median_val:.2f}")

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode={mode_val}")

    return df


def get_feature_types(X):
    """Identify numeric and categorical columns.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (numeric column names, categorical column names).
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def get_regions():
    """Return the list of Saudi regions used in the dataset.

    Returns:
        List of region name strings.
    """
    return ["Eastern", "Central", "Western", "Northern", "Southern"]


def get_suitability_labels():
    """Return ordered suitability labels.

    Returns:
        List of suitability label strings from best to worst.
    """
    return ["optimal", "good", "marginal", "unsuitable"]
