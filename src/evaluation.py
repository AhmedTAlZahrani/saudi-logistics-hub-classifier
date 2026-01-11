import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


SUITABILITY_LABELS = ["optimal", "good", "marginal", "unsuitable"]


class ModelEvaluator:
    """Evaluate classification model performance with visual reports.

    Generates classification reports, confusion matrices, feature
    importance rankings, and TOPSIS vs ML comparisons for the
    logistics hub suitability classifier.
    """

    @staticmethod
    def get_classification_report(y_true, y_pred, label_names=None):
        """Return classification report as a DataFrame.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            label_names: List of class label names.

        Returns:
            DataFrame with precision, recall, f1-score per class.
        """
        labels = label_names or SUITABILITY_LABELS
        report = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
        return pd.DataFrame(report).transpose().round(3)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        """Generate an annotated confusion matrix heatmap.

        Args:
            y_true: True labels (encoded).
            y_pred: Predicted labels (encoded).
            labels: List of class label names.

        Returns:
            Plotly Figure with the confusion matrix.
        """
        labels = labels or SUITABILITY_LABELS
        cm = confusion_matrix(y_true, y_pred)

        fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix - Hub Suitability Classification",
        )
        fig.update_layout(template="plotly_dark", height=500)
        return fig

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """Plot feature importance rankings from a trained model.

        Args:
            model: Trained model with feature_importances_ attribute.
            feature_names: List of feature names.
            top_n: Number of top features to display.

        Returns:
            Plotly Figure with horizontal bar chart.
        """
        if not hasattr(model, "feature_importances_"):
            print("Model does not support feature_importances_")
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_dark",
            height=500,
        )
        return fig

    @staticmethod
    def compare_ml_vs_topsis(y_true, ml_preds, topsis_preds, label_encoder=None):
        """Compare ML model predictions with TOPSIS rankings.

        Args:
            y_true: True suitability labels.
            ml_preds: ML model predicted labels (encoded).
            topsis_preds: TOPSIS predicted labels (string).
            label_encoder: Fitted LabelEncoder to decode ML predictions.

        Returns:
            DataFrame with comparison metrics.
        """
        if label_encoder is not None:
            ml_decoded = label_encoder.inverse_transform(ml_preds)
            y_true_decoded = label_encoder.inverse_transform(y_true)
        else:
            ml_decoded = ml_preds
            y_true_decoded = y_true

        ml_acc = accuracy_score(y_true_decoded, ml_decoded)
        topsis_acc = accuracy_score(y_true_decoded, topsis_preds)

        ml_report = classification_report(
            y_true_decoded, ml_decoded, output_dict=True, zero_division=0
        )
        topsis_report = classification_report(
            y_true_decoded, topsis_preds, output_dict=True, zero_division=0
        )

        comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
            "ML Model": [
                round(ml_acc, 4),
                round(ml_report["weighted avg"]["precision"], 4),
                round(ml_report["weighted avg"]["recall"], 4),
                round(ml_report["weighted avg"]["f1-score"], 4),
            ],
            "TOPSIS": [
                round(topsis_acc, 4),
                round(topsis_report["weighted avg"]["precision"], 4),
                round(topsis_report["weighted avg"]["recall"], 4),
                round(topsis_report["weighted avg"]["f1-score"], 4),
            ],
        })

        print("ML vs TOPSIS Comparison:")
        print(comparison.to_string(index=False))
        return comparison

    @staticmethod
    def plot_model_comparison(results_df):
        """Plot comparison of multiple ML models.

        Args:
            results_df: DataFrame from ModelTrainer.compare_models().

        Returns:
            Plotly Figure with grouped bar chart.
        """
        metrics = [c for c in results_df.columns if c != "Model"]
        fig = go.Figure()

        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=results_df["Model"],
                y=results_df[metric],
                text=results_df[metric].apply(lambda v: f"{v:.3f}"),
                textposition="auto",
            ))

        fig.update_layout(
            barmode="group",
            title="Model Comparison - Cross-Validation Metrics",
            xaxis_title="Model",
            yaxis_title="Score",
            template="plotly_dark",
            height=500,
        )
        return fig

    @staticmethod
    def plot_class_distribution(y, title="Suitability Class Distribution"):
        """Plot distribution of suitability classes.

        Args:
            y: Series or array of suitability labels.
            title: Chart title.

        Returns:
            Plotly Figure with bar chart.
        """
        counts = pd.Series(y).value_counts()
        fig = px.bar(
            x=counts.index, y=counts.values,
            color=counts.index,
            color_discrete_map={
                "optimal": "#2ecc71", "good": "#3498db",
                "marginal": "#f39c12", "unsuitable": "#e74c3c",
            },
            labels={"x": "Suitability", "y": "Count"},
            title=title,
        )
        fig.update_layout(template="plotly_dark", height=400, showlegend=False)
        return fig

    @staticmethod
    def save_metrics(metrics, path="output/metrics.json"):
        """Save evaluation metrics to a JSON file.

        Args:
            metrics: Dictionary of metrics to save.
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {output}")
