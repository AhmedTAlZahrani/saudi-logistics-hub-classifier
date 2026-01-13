import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic


# Catchment area radii in kilometers
CATCHMENT_RADII = [50, 100, 200]

# Corridor definitions (region groupings along rail lines)
CORRIDORS = {
    "Eastern": {
        "regions": ["Eastern"],
        "description": "Dammam-Jubail-Ras Al-Khair industrial corridor",
        "anchor_lat": 26.43,
        "anchor_lon": 50.10,
    },
    "Western": {
        "regions": ["Western"],
        "description": "Jeddah-Yanbu-KAEC port corridor",
        "anchor_lat": 21.54,
        "anchor_lon": 39.17,
    },
    "Central": {
        "regions": ["Central"],
        "description": "Riyadh-Sudair inland logistics corridor",
        "anchor_lat": 24.71,
        "anchor_lon": 46.68,
    },
    "North-South Link": {
        "regions": ["Northern", "Central"],
        "description": "NEOM-Tabuk-Hail northern development corridor",
        "anchor_lat": 28.00,
        "anchor_lon": 36.00,
    },
    "Southern": {
        "regions": ["Southern"],
        "description": "Abha-Jizan southern gateway corridor",
        "anchor_lat": 18.22,
        "anchor_lon": 42.50,
    },
}

class SpatialAnalyzer:
    """Geospatial analysis for logistics hub candidate locations.

    Performs proximity calculations, catchment area analysis,
    corridor-level scoring, and hub density analysis using
    coordinate-based distance computations.
    """

    def __init__(self, df, metadata):
        self.df = df.copy()
        self.metadata = metadata.copy()
        self._merge_coords()

    def _merge_coords(self):
        if "lat" not in self.df.columns and "lat" in self.metadata.columns:
            self.df["lat"] = self.metadata["lat"].values
            self.df["lon"] = self.metadata["lon"].values
        if "region" not in self.df.columns and "region" in self.metadata.columns:
            self.df["region"] = self.metadata["region"].values

    @staticmethod
    def _distance_km(lat1, lon1, lat2, lon2):
        """Calculate distance between two points using geopy geodesic."""
        return geodesic((lat1, lon1), (lat2, lon2)).km

    def compute_distance_matrix(self):
        """Compute pairwise distance matrix between all locations."""
        n = len(self.metadata)
        lats = self.metadata["lat"].values
        lons = self.metadata["lon"].values

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance_km(lats[i], lons[i], lats[j], lons[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        names = self.metadata["name"].values
        print(f"Computed {n}x{n} distance matrix")
        return pd.DataFrame(dist_matrix, index=names, columns=names)

    def catchment_analysis(self, radii=None):
        """Analyze catchment areas at specified radii for each location.

        Counts the number of other candidate locations within each
        radius, providing a density measure for hub clustering.

        Args:
            radii: List of radii in km to analyze.

        Returns:
            DataFrame with catchment counts at each radius.
        """
        radii = radii or CATCHMENT_RADII
        lats = self.metadata["lat"].values
        lons = self.metadata["lon"].values
        n = len(lats)

        results = {"name": self.metadata["name"].values}
        for radius in radii:
            counts = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d = self._distance_km(lats[i], lons[i], lats[j], lons[j])
                        if d <= radius:
                            counts[i] += 1
            results[f"locations_within_{radius}km"] = counts

        df_result = pd.DataFrame(results)
        print(f"Catchment analysis complete for radii: {radii}")
        return df_result

    def corridor_scoring(self, suitability=None):
        """Score each corridor based on location attributes and suitability.

        Args:
            suitability: Optional Series of suitability labels.

        Returns:
            DataFrame with corridor-level aggregate scores.
        """
        rows = []
        for corridor_name, corridor_info in CORRIDORS.items():
            mask = self.metadata["region"].isin(corridor_info["regions"])
            subset = self.df[mask]

            if len(subset) == 0:
                continue

            row = {
                "corridor": corridor_name,
                "description": corridor_info["description"],
                "num_locations": len(subset),
            }

            # Aggregate numeric features
            numeric_cols = subset.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                row[f"avg_{col}"] = round(subset[col].mean(), 2)

            # Suitability breakdown
            if suitability is not None:
                suit_subset = suitability[mask]
                for label in ["optimal", "good", "marginal", "unsuitable"]:
                    row[f"pct_{label}"] = round(
                        (suit_subset == label).mean() * 100, 1
                    )

            rows.append(row)

        result = pd.DataFrame(rows)
        print(f"Corridor scoring complete for {len(result)} corridors")
        return result

    def hub_density_analysis(self, grid_size_deg=1.0):
        """Analyze hub density across a geographic grid.

        Divides Saudi Arabia into grid cells and counts candidate
        locations per cell to identify concentration areas.

        Args:
            grid_size_deg: Grid cell size in degrees.

        Returns:
            DataFrame with grid cell coordinates and location counts.
        """
        lat_bins = np.arange(16, 34, grid_size_deg)
        lon_bins = np.arange(34, 57, grid_size_deg)

        lats = self.metadata["lat"].values
        lons = self.metadata["lon"].values

        grid_counts = []
        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                mask = (
                    (lats >= lat_bins[i]) & (lats < lat_bins[i + 1]) &
                    (lons >= lon_bins[j]) & (lons < lon_bins[j + 1])
                )
                count = mask.sum()
                if count > 0:
                    grid_counts.append({
                        "lat_center": round((lat_bins[i] + lat_bins[i + 1]) / 2, 2),
                        "lon_center": round((lon_bins[j] + lon_bins[j + 1]) / 2, 2),
                        "count": int(count),
                        "lat_min": lat_bins[i],
                        "lat_max": lat_bins[i + 1],
                        "lon_min": lon_bins[j],
                        "lon_max": lon_bins[j + 1],
                    })

        result = pd.DataFrame(grid_counts)
        print(f"Hub density analysis: {len(result)} non-empty grid cells")
        return result

    def nearest_neighbor_distances(self, k=5):
        """Find k-nearest neighbor distances for each location.

        Args:
            k: Number of nearest neighbors to find.

        Returns:
            DataFrame with mean, min, max neighbor distances per location.
        """
        lats = self.metadata["lat"].values
        lons = self.metadata["lon"].values
        n = len(lats)

        nn_stats = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i != j:
                    d = self._distance_km(lats[i], lons[i], lats[j], lons[j])
                    dists.append(d)
            dists.sort()
            k_dists = dists[:k]
            nn_stats.append({
                "name": self.metadata["name"].values[i],
                "nn_mean_km": round(np.mean(k_dists), 1),
                "nn_min_km": round(min(k_dists), 1),
                "nn_max_km": round(max(k_dists), 1),
            })

        result = pd.DataFrame(nn_stats)
        print(f"Nearest neighbor analysis (k={k}) complete")
        return result

    def proximity_to_corridors(self):
        """Calculate distance from each location to corridor anchors.

        Returns:
            DataFrame with distance to each corridor anchor point.
        """
        lats = self.metadata["lat"].values
        lons = self.metadata["lon"].values

        results = {"name": self.metadata["name"].values}
        for corridor_name, info in CORRIDORS.items():
            dists = [
                self._distance_km(lats[i], lons[i], info["anchor_lat"], info["anchor_lon"])
                for i in range(len(lats))
            ]
            results[f"dist_to_{corridor_name}_km"] = np.round(dists, 1)

        return pd.DataFrame(results)

    def save_analysis(self, analysis_name, df, output_dir="output"):
        """Save analysis results to CSV.

        Args:
            analysis_name: Name for the output file.
            df: DataFrame to save.
            output_dir: Output directory path.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{analysis_name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {analysis_name} to {path}")
