import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from geopy.distance import geodesic

from .data_loader import get_feature_types


# Weights for composite accessibility score
RAIL_WEIGHT = 0.5
HIGHWAY_WEIGHT = 0.3
PORT_WEIGHT = 0.2

# Maximum expected distances for normalization
MAX_RAIL_DIST = 500.0
MAX_HIGHWAY_DIST = 300.0
MAX_PORT_DIST = 1500.0


class HubFeatureBuilder(BaseEstimator):
    """Feature engineering pipeline for logistics hub classification.

    Creates derived features including composite accessibility scores,
    market potential indices, and infrastructure readiness metrics.
    """

    def __init__(self):
        self._preprocessor = None
        self._numeric_cols = None
        self._categorical_cols = None
        self._label_encoder = None

    def fit(self, X: pd.DataFrame, y=None) -> "HubFeatureBuilder":
        """Fit the preprocessing pipeline.

        Args:
            X: Feature DataFrame.
            y: Ignored (API compatibility).

        Returns:
            self
        """
        X = self._add_derived_features(X)
        self._numeric_cols, self._categorical_cols = get_feature_types(X)

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self._numeric_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False,
                                      handle_unknown="ignore"),
                 self._categorical_cols),
            ],
            remainder="drop",
        )
        self._preprocessor.fit(X)
        print(f"HubFeatureBuilder fitted | {len(self._numeric_cols)} numeric, "
              f"{len(self._categorical_cols)} categorical features")
        return self

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Transformed DataFrame with engineered features.
        """
        X = self._add_derived_features(X)
        transformed = self._preprocessor.transform(X)
        feature_names = self.get_feature_names()
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    def fit_process(self, X, y=None):
        return self.fit(X, y).process(X)

    def get_feature_names(self):
        """Return the feature names after transformation."""
        return self._preprocessor.get_feature_names_out().tolist()

    def encode_target(self, y: pd.Series) -> tuple:
        """Encode suitability labels to numeric values.

        Args:
            y: Series of suitability labels.

        Returns:
            Encoded numpy array and the fitted LabelEncoder.
        """
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        print(f"Target classes: {list(self._label_encoder.classes_)}")
        return y_encoded, self._label_encoder

    def decode_target(self, y_encoded):
        if self._label_encoder is None:
            raise ValueError("Label encoder not fitted. Call encode_target first.")
        return self._label_encoder.inverse_transform(y_encoded)

    @staticmethod
    def _add_derived_features(X):
        """Create composite and interaction features."""
        X = X.copy()

        # Composite accessibility score (weighted inverse distance)
        if {"distance_to_rail_km", "distance_to_highway_km", "distance_to_port_km"}.issubset(X.columns):
            rail_norm = 1.0 - np.clip(X["distance_to_rail_km"] / MAX_RAIL_DIST, 0, 1)
            highway_norm = 1.0 - np.clip(X["distance_to_highway_km"] / MAX_HIGHWAY_DIST, 0, 1)
            port_norm = 1.0 - np.clip(X["distance_to_port_km"] / MAX_PORT_DIST, 0, 1)
            X["accessibility_score"] = (
                RAIL_WEIGHT * rail_norm +
                HIGHWAY_WEIGHT * highway_norm +
                PORT_WEIGHT * port_norm
            )

        # Market potential index
        if {"population_50km_radius", "commercial_activity_index"}.issubset(X.columns):
            X["market_potential_index"] = (
                np.log1p(X["population_50km_radius"]) *
                X["commercial_activity_index"] / 10.0
            )

        # Infrastructure readiness score
        if {"power_grid_capacity_mw", "water_availability_score", "existing_warehouse_sqm"}.issubset(X.columns):
            power_norm = np.clip(X["power_grid_capacity_mw"] / 500.0, 0, 1)
            water_norm = X["water_availability_score"] / 10.0
            warehouse_norm = np.clip(np.log1p(X["existing_warehouse_sqm"]) / 15.0, 0, 1)
            X["infrastructure_readiness"] = (
                0.4 * power_norm + 0.3 * water_norm + 0.3 * warehouse_norm
            )

        # Log-transformed freight volume
        if "rail_freight_volume_nearby" in X.columns:
            X["log_freight_volume"] = np.log1p(X["rail_freight_volume_nearby"])

        # Labor efficiency metric
        if {"labor_market_size", "existing_warehouse_sqm"}.issubset(X.columns):
            X["labor_per_warehouse_sqm"] = (
                X["labor_market_size"] / np.maximum(X["existing_warehouse_sqm"], 1)
            )

        # Temperature penalty (extreme heat reduces suitability)
        if "avg_temperature" in X.columns:
            # XXX: hardcoded for now, should come from config
            X["temp_penalty"] = np.where(X["avg_temperature"] > 40, 1, 0)

        # Geographic clustering: distance to nearest major hub anchor
        if {"lat", "lon"}.issubset(X.columns):
            hub_coords = [
                (27.01, 49.66),  # Jubail
                (21.54, 39.17),  # Jeddah
                (24.71, 46.68),  # Riyadh
                (26.43, 50.10),  # Dammam
            ]
            min_dists = []
            for _, row in X[["lat", "lon"]].iterrows():
                dists = [
                    geodesic((row["lat"], row["lon"]), (h[0], h[1])).km
                    for h in hub_coords
                ]
                min_dists.append(min(dists))
            X["distance_to_nearest_hub_km"] = min_dists

        return X
