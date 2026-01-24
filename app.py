import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.generate_sites import generate_locations, save_locations
from src.data_loader import load_location_data, get_suitability_labels
from src.hub_features import HubFeatureBuilder
from src.classify import ClassificationBenchmark, MODELS
from src.evaluation import ModelEvaluator
from src.spatial_analysis import SpatialAnalyzer, CORRIDORS


# Page configuration
st.set_page_config(
    page_title="Saudi Logistics Hub Classifier",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color mapping for suitability levels
SUITABILITY_COLORS = {
    "optimal": "#2ecc71",
    "good": "#3498db",
    "marginal": "#f39c12",
    "unsuitable": "#e74c3c",
}

SUITABILITY_ICONS = {
    "optimal": "star",
    "good": "ok-sign",
    "marginal": "warning-sign",
    "unsuitable": "remove-sign",
}


@st.cache_data
def load_data():
    """Load or generate location data.

    Returns:
        Tuple of (features, target, metadata) DataFrames.
    """
    data_path = Path("data/hub_locations.csv")
    if not data_path.exists():
        df = generate_locations()
        save_locations(df)
    return load_location_data(str(data_path))


@st.cache_resource
def train_models(X_processed, y_encoded):
    """Train all models and return benchmark instance.

    Args:
        X_processed: Processed feature matrix.
        y_encoded: Encoded target vector.

    Returns:
        Tuple of (ClassificationBenchmark, comparison DataFrame).
    """
    benchmark = ClassificationBenchmark()
    comparison = benchmark.compare_models(X_processed, y_encoded)
    return benchmark, comparison


def render_sidebar():
    st.sidebar.title("Controls")
    st.sidebar.markdown("---")

    model_choice = st.sidebar.selectbox(
        "Select Model", list(MODELS.keys()), index=1
    )

    st.sidebar.markdown("### TOPSIS Weights")
    rail_w = st.sidebar.slider("Rail Access", 0.0, 1.0, 0.25, 0.05)
    market_w = st.sidebar.slider("Market Potential", 0.0, 1.0, 0.20, 0.05)
    infra_w = st.sidebar.slider("Infrastructure", 0.0, 1.0, 0.20, 0.05)

    st.sidebar.markdown("### Map Settings")
    map_style = st.sidebar.selectbox(
        "Map Tiles", ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"]
    )

    return {
        "model": model_choice,
        "rail_weight": rail_w,
        "market_weight": market_w,
        "infra_weight": infra_w,
        "map_style": map_style,
    }


def render_location_explorer(metadata, y, map_style):
    st.header("Location Explorer")
    st.markdown("Interactive map of 300 candidate logistics hub locations "
                "across Saudi Arabia, colored by predicted suitability.")

    col1, col2 = st.columns([3, 1])

    with col1:
        center_lat = metadata["lat"].mean()
        center_lon = metadata["lon"].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=map_style,
        )

        for idx, row in metadata.iterrows():
            label = y.iloc[idx] if idx < len(y) else "unknown"
            color = SUITABILITY_COLORS.get(label, "gray")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['name']}</b><br>"
                    f"Region: {row['region']}<br>"
                    f"Suitability: {label}<br>"
                    f"Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}",
                    max_width=250,
                ),
            ).add_to(m)

        st_folium(m, width=900, height=600)

    with col2:
        st.markdown("### Legend")
        for label, color in SUITABILITY_COLORS.items():
            st.markdown(
                f'<span style="color:{color}; font-size:20px;">&#9679;</span> '
                f'**{label.capitalize()}**',
                unsafe_allow_html=True,
            )

        st.markdown("### Distribution")
        counts = y.value_counts()
        for label in get_suitability_labels():
            if label in counts.index:
                st.metric(label.capitalize(), counts[label])

        st.markdown("### By Region")
        region_counts = metadata["region"].value_counts()
        fig = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title="Locations by Region",
        )
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(comparison_df, benchmark, X_processed, y_encoded, X_raw, fe):
    st.header("Model Comparison")

    evaluator = ModelEvaluator()

    # Model comparison chart
    fig = evaluator.plot_model_comparison(comparison_df)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("Cross-Validation Results")
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=comparison_df.columns[1:]),
                 use_container_width=True)

    # Best model confusion matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Confusion Matrix ({benchmark.best_model_name})")
        y_pred = benchmark.best_model.predict(X_processed)
        fig_cm = evaluator.plot_confusion_matrix(y_encoded, y_pred)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("ML vs TOPSIS")
        topsis_preds = benchmark.topsis_classify(X_raw)
        comparison = evaluator.compare_ml_vs_topsis(
            y_encoded, y_pred, topsis_preds, fe._label_encoder
        )
        st.dataframe(comparison, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    report_df = evaluator.get_classification_report(y_encoded, y_pred)
    st.dataframe(report_df, use_container_width=True)


def render_feature_analysis(benchmark, X_processed, feature_names):
    st.header("Feature Analysis")

    evaluator = ModelEvaluator()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        fig_imp = evaluator.plot_feature_importance(
            benchmark.best_model, feature_names, top_n=15
        )
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        st.subheader("Importance Summary")
        if hasattr(benchmark.best_model, "feature_importances_"):
            importances = benchmark.best_model.feature_importances_
            top_n = min(15, len(feature_names))
            top_idx = np.argsort(importances)[-top_n:]

            fig_imp2 = go.Figure(go.Bar(
                x=importances[top_idx],
                y=[feature_names[i] for i in top_idx],
                orientation="h",
                marker_color="coral",
            ))
            fig_imp2.update_layout(
                title="Feature Importances (model)",
                xaxis_title="Importance",
                template="plotly_dark",
                height=500,
            )
            st.plotly_chart(fig_imp2, use_container_width=True)

    # Partial dependence for top features
    st.subheader("Feature Distributions by Suitability")
    if hasattr(benchmark.best_model, "feature_importances_"):
        top_features_idx = np.argsort(benchmark.best_model.feature_importances_)[-4:]
        cols = st.columns(2)
        for i, feat_idx in enumerate(top_features_idx):
            with cols[i % 2]:
                feat_name = feature_names[feat_idx]
                fig = px.histogram(
                    x=X_processed.iloc[:, feat_idx],
                    nbins=30,
                    title=f"Distribution: {feat_name}",
                    labels={"x": feat_name, "y": "Count"},
                )
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)


def render_corridor_analysis(metadata, X, y):
    st.header("Corridor Analysis")
    st.markdown("Regional corridor-level assessment of logistics hub "
                "potential along Saudi Arabia's rail network.")

    analyzer = SpatialAnalyzer(X, metadata)

    # Corridor scoring
    corridor_df = analyzer.corridor_scoring(suitability=y)
    st.subheader("Corridor Summary")
    display_cols = ["corridor", "description", "num_locations"]
    pct_cols = [c for c in corridor_df.columns if c.startswith("pct_")]
    st.dataframe(corridor_df[display_cols + pct_cols], use_container_width=True)

    # Corridor map
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Corridor Map")
        m = folium.Map(location=[24.5, 44.0], zoom_start=5, tiles="CartoDB positron")
        for name, info in CORRIDORS.items():
            folium.Marker(
                location=[info["anchor_lat"], info["anchor_lon"]],
                popup=f"<b>{name}</b><br>{info['description']}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
            folium.Circle(
                location=[info["anchor_lat"], info["anchor_lon"]],
                radius=150000,
                color="steelblue",
                fill=True,
                fill_opacity=0.1,
            ).add_to(m)
        st_folium(m, width=700, height=500)

    with col2:
        st.subheader("Hub Density")
        density_df = analyzer.hub_density_analysis()
        fig = px.scatter(
            density_df,
            x="lon_center", y="lat_center",
            size="count", color="count",
            color_continuous_scale="YlOrRd",
            labels={"lon_center": "Longitude", "lat_center": "Latitude"},
            title="Location Density Grid",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Catchment analysis
    st.subheader("Catchment Area Analysis")
    catchment_df = analyzer.catchment_analysis(radii=[50, 100, 200])
    catchment_summary = catchment_df.describe().round(1)
    st.dataframe(catchment_summary, use_container_width=True)


def main():
    """Main application entry point."""
    st.title("Saudi Logistics Hub Classifier")
    st.markdown("**ML-Powered Location Intelligence for Saudi Arabia's "
                "Rail Corridor Logistics Network**")
    st.markdown("---")

    # Sidebar
    settings = render_sidebar()

    # Load and process data
    X, y, metadata = load_data()
    fe = HubFeatureBuilder()
    X_processed = fe.fit_process(X)
    y_encoded, label_encoder = fe.encode_target(y)

    # Train models
    benchmark, comparison = train_models(X_processed, y_encoded)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Location Explorer",
        "Model Comparison",
        "Feature Analysis",
        "Corridor Analysis",
    ])

    with tab1:
        render_location_explorer(metadata, y, settings["map_style"])

    with tab2:
        render_model_comparison(comparison, benchmark, X_processed, y_encoded, X, fe)

    with tab3:
        feature_names = fe.get_feature_names()
        render_feature_analysis(benchmark, X_processed, feature_names)

    with tab4:
        render_corridor_analysis(metadata, X, y)


if __name__ == "__main__":
    main()
