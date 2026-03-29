"""Tests for hub feature computations and classification thresholds."""

import numpy as np
import pandas as pd
import pytest

from src.hub_features import (
    HubFeatureBuilder,
    MAX_HIGHWAY_DIST,
    MAX_PORT_DIST,
    MAX_RAIL_DIST,
    RAIL_WEIGHT,
    HIGHWAY_WEIGHT,
    PORT_WEIGHT,
)
from src.classify import ClassificationBenchmark, TOPSIS_WEIGHTS, TOPSIS_BENEFIT
from src.generate_sites import _compute_suitability, _assign_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_site(**overrides):
    """Build a single-row DataFrame with default site values."""
    defaults = dict(
        lat=24.71, lon=46.68, region="Central",
        distance_to_rail_km=50.0, distance_to_highway_km=30.0,
        distance_to_port_km=400.0, population_50km_radius=300_000,
        industrial_zones_nearby=2, commercial_activity_index=6.0,
        land_cost_index=5.0, water_availability_score=6.0,
        power_grid_capacity_mw=150.0, labor_market_size=15_000,
        existing_warehouse_sqm=10_000, avg_temperature=35.0,
        rail_freight_volume_nearby=200_000,
    )
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _make_sites(n=20, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        rows.append(dict(
            lat=rng.uniform(17, 32), lon=rng.uniform(35, 55),
            region=rng.choice(["Eastern", "Central", "Western"]),
            distance_to_rail_km=rng.exponential(60),
            distance_to_highway_km=rng.exponential(30),
            distance_to_port_km=rng.exponential(300),
            population_50km_radius=int(rng.lognormal(11, 1.5)),
            industrial_zones_nearby=int(rng.poisson(2)),
            commercial_activity_index=round(rng.uniform(1, 10), 1),
            land_cost_index=round(rng.uniform(1, 10), 1),
            water_availability_score=round(rng.uniform(1, 10), 1),
            power_grid_capacity_mw=round(rng.lognormal(4.5, 1), 1),
            labor_market_size=int(rng.lognormal(9, 1.2)),
            existing_warehouse_sqm=int(rng.lognormal(9, 1.5)),
            avg_temperature=round(rng.uniform(20, 45), 1),
            rail_freight_volume_nearby=int(rng.lognormal(11, 1.8)),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _add_derived_features tests
# ---------------------------------------------------------------------------

class TestAccessibilityScore:
    """Verify the composite accessibility score formula."""

    @pytest.mark.parametrize("rail,hwy,port,expected_min,expected_max", [
        (0, 0, 0, 0.99, 1.01),          # all zero distance => max accessibility
        (MAX_RAIL_DIST, MAX_HIGHWAY_DIST, MAX_PORT_DIST, -0.01, 0.01),  # max dist => 0
        (250, 150, 750, 0.0, 1.0),       # mid-range
    ], ids=["all-close", "all-far", "mid-range"])
    def test_score_bounds(self, rail, hwy, port, expected_min, expected_max):
        df = _make_site(distance_to_rail_km=rail,
                        distance_to_highway_km=hwy,
                        distance_to_port_km=port)
        result = HubFeatureBuilder._add_derived_features(df)
        score = result["accessibility_score"].iloc[0]
        assert expected_min <= score <= expected_max

    def test_weight_sum_equals_one(self):
        assert abs(RAIL_WEIGHT + HIGHWAY_WEIGHT + PORT_WEIGHT - 1.0) < 1e-9

    def test_clipping_beyond_max(self):
        """Distances beyond the max should still yield score >= 0."""
        df = _make_site(distance_to_rail_km=9999,
                        distance_to_highway_km=9999,
                        distance_to_port_km=9999)
        result = HubFeatureBuilder._add_derived_features(df)
        assert result["accessibility_score"].iloc[0] == pytest.approx(0.0, abs=1e-9)


class TestMarketPotentialIndex:

    @pytest.mark.parametrize("pop,commercial,should_be_positive", [
        (1_000_000, 8.0, True),
        (0, 0.0, False),
        (500_000, 5.0, True),
    ], ids=["high-market", "zero-market", "medium-market"])
    def test_market_potential(self, pop, commercial, should_be_positive):
        df = _make_site(population_50km_radius=pop,
                        commercial_activity_index=commercial)
        result = HubFeatureBuilder._add_derived_features(df)
        val = result["market_potential_index"].iloc[0]
        if should_be_positive:
            assert val > 0
        else:
            assert val == pytest.approx(0.0, abs=1e-9)


class TestInfrastructureReadiness:

    def test_all_max_infrastructure(self):
        df = _make_site(power_grid_capacity_mw=500.0,
                        water_availability_score=10.0,
                        existing_warehouse_sqm=3_000_000)
        result = HubFeatureBuilder._add_derived_features(df)
        score = result["infrastructure_readiness"].iloc[0]
        # power_norm=1, water_norm=1, warehouse high => close to 1
        assert score > 0.85

    def test_all_min_infrastructure(self):
        df = _make_site(power_grid_capacity_mw=0.0,
                        water_availability_score=0.0,
                        existing_warehouse_sqm=0)
        result = HubFeatureBuilder._add_derived_features(df)
        score = result["infrastructure_readiness"].iloc[0]
        assert score < 0.15


class TestDerivedColumns:
    """Check that all expected derived columns are created."""

    def test_all_derived_columns_present(self):
        df = _make_site()
        result = HubFeatureBuilder._add_derived_features(df)
        for col in ("accessibility_score", "market_potential_index",
                     "infrastructure_readiness", "log_freight_volume",
                     "labor_per_warehouse_sqm", "temp_penalty",
                     "distance_to_nearest_hub_km"):
            assert col in result.columns, f"Missing column: {col}"

    def test_temp_penalty_above_40(self):
        df = _make_site(avg_temperature=45.0)
        result = HubFeatureBuilder._add_derived_features(df)
        assert result["temp_penalty"].iloc[0] == 1

    def test_temp_penalty_below_40(self):
        df = _make_site(avg_temperature=35.0)
        result = HubFeatureBuilder._add_derived_features(df)
        assert result["temp_penalty"].iloc[0] == 0

    def test_log_freight_volume(self):
        df = _make_site(rail_freight_volume_nearby=100_000)
        result = HubFeatureBuilder._add_derived_features(df)
        assert result["log_freight_volume"].iloc[0] == pytest.approx(
            np.log1p(100_000), rel=1e-6
        )

    def test_labor_per_warehouse_sqm_no_division_by_zero(self):
        df = _make_site(labor_market_size=5000, existing_warehouse_sqm=0)
        result = HubFeatureBuilder._add_derived_features(df)
        # np.maximum(0, 1) = 1 => 5000/1
        assert result["labor_per_warehouse_sqm"].iloc[0] == pytest.approx(5000.0)

    def test_original_columns_unchanged(self):
        df = _make_site()
        original_rail = df["distance_to_rail_km"].iloc[0]
        result = HubFeatureBuilder._add_derived_features(df)
        assert result["distance_to_rail_km"].iloc[0] == original_rail


# ---------------------------------------------------------------------------
# HubFeatureBuilder fit / process
# ---------------------------------------------------------------------------

class TestHubFeatureBuilderPipeline:

    def test_fit_process_shape(self):
        df = _make_sites(30)
        builder = HubFeatureBuilder()
        transformed = builder.fit_process(df)
        assert len(transformed) == 30
        assert transformed.shape[1] > 0

    def test_process_returns_dataframe(self):
        df = _make_sites(20)
        builder = HubFeatureBuilder()
        builder.fit(df)
        result = builder.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_encode_decode_roundtrip(self):
        labels = pd.Series(["optimal", "good", "marginal", "unsuitable", "good"])
        builder = HubFeatureBuilder()
        encoded, le = builder.encode_target(labels)
        decoded = builder.decode_target(encoded)
        assert list(decoded) == list(labels)

    def test_decode_before_encode_raises(self):
        builder = HubFeatureBuilder()
        with pytest.raises(ValueError, match="Label encoder not fitted"):
            builder.decode_target(np.array([0, 1]))

    def test_get_feature_names_after_fit(self):
        df = _make_sites(15)
        builder = HubFeatureBuilder()
        builder.fit(df)
        names = builder.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0


# ---------------------------------------------------------------------------
# _compute_suitability thresholds
# ---------------------------------------------------------------------------

class TestSuitabilityThresholds:

    @pytest.mark.parametrize("overrides,expected_label", [
        # Optimal: close to rail/highway, large pop, high commercial, good infra
        (dict(distance_to_rail_km=10, distance_to_highway_km=5,
              distance_to_port_km=50, population_50km_radius=800_000,
              commercial_activity_index=9.0, industrial_zones_nearby=5,
              power_grid_capacity_mw=300, water_availability_score=9.0,
              land_cost_index=2.0, labor_market_size=30_000,
              rail_freight_volume_nearby=1_000_000), "optimal"),
        # Unsuitable: far from everything, low pop, low infra
        (dict(distance_to_rail_km=400, distance_to_highway_km=200,
              distance_to_port_km=1200, population_50km_radius=8_000,
              commercial_activity_index=1.0, industrial_zones_nearby=0,
              power_grid_capacity_mw=10, water_availability_score=1.0,
              land_cost_index=9.0, labor_market_size=500,
              rail_freight_volume_nearby=5_000), "unsuitable"),
    ], ids=["optimal-site", "unsuitable-site"])
    def test_extreme_sites(self, overrides, expected_label):
        row = _make_site(**overrides).iloc[0]
        assert _compute_suitability(row) == expected_label

    def test_suitability_labels_are_valid(self):
        from src.generate_sites import SUITABILITY_LABELS
        row = _make_site().iloc[0]
        label = _compute_suitability(row)
        assert label in SUITABILITY_LABELS


# ---------------------------------------------------------------------------
# _assign_region
# ---------------------------------------------------------------------------

class TestAssignRegion:

    @pytest.mark.parametrize("lat,lon,expected", [
        (26.43, 50.10, "Eastern"),
        (24.71, 46.68, "Central"),
        (21.54, 39.17, "Western"),
        (28.00, 36.00, "Northern"),
        (17.00, 42.50, "Southern"),
    ], ids=["eastern", "central", "western", "northern", "southern"])
    def test_region_assignment(self, lat, lon, expected):
        assert _assign_region(lat, lon) == expected


# ---------------------------------------------------------------------------
# TOPSIS classification thresholds
# ---------------------------------------------------------------------------

class TestTopsisClassify:

    def test_default_thresholds_produce_four_classes(self):
        """TOPSIS classify with varied data should produce multiple classes."""
        bench = ClassificationBenchmark(output_dir="models/_test_tmp")
        df = _make_sites(50, seed=123)
        builder = HubFeatureBuilder()
        enriched = HubFeatureBuilder._add_derived_features(df)
        labels = bench.topsis_classify(enriched)
        unique = set(labels)
        # At least 2 classes with diverse enough data
        assert len(unique) >= 2

    def test_custom_thresholds(self):
        bench = ClassificationBenchmark(output_dir="models/_test_tmp")
        df = _make_sites(30, seed=99)
        enriched = HubFeatureBuilder._add_derived_features(df)
        labels = bench.topsis_classify(
            enriched, thresholds={"optimal": 0.95, "good": 0.80, "marginal": 0.50}
        )
        assert all(l in ("optimal", "good", "marginal", "unsuitable") for l in labels)

    def test_topsis_scores_between_zero_and_one(self):
        bench = ClassificationBenchmark(output_dir="models/_test_tmp")
        df = _make_sites(20)
        enriched = HubFeatureBuilder._add_derived_features(df)
        scores = bench.topsis_rank(enriched)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_topsis_no_criteria_returns_zeros(self):
        bench = ClassificationBenchmark(output_dir="models/_test_tmp")
        df = pd.DataFrame({"irrelevant_col": [1, 2, 3]})
        scores = bench.topsis_rank(df)
        assert (scores == 0).all()

    def test_topsis_weights_sum(self):
        total = sum(TOPSIS_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_topsis_benefit_keys_match_weights(self):
        assert set(TOPSIS_WEIGHTS.keys()) == set(TOPSIS_BENEFIT.keys())
