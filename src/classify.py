import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

_log_dir = Path("logs")
_log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("hub_classifier")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(_log_dir / "classify.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


# Model definitions
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        class_weight="balanced", random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42,
    ),
    "SVM": SVC(
        kernel="rbf", C=10.0, gamma="scale",
        class_weight="balanced", probability=True,
        random_state=42,
    ),
}

# TOPSIS criteria weights (for multi-criteria decision analysis)
TOPSIS_WEIGHTS = {
    "accessibility_score": 0.25,
    "market_potential_index": 0.20,
    "infrastructure_readiness": 0.20,
    "log_freight_volume": 0.15,
    "labor_per_warehouse_sqm": 0.05,
    "distance_to_nearest_hub_km": 0.10,
    "land_cost_index": 0.05,
}

# Criteria direction: True = benefit (higher is better), False = cost
TOPSIS_BENEFIT = {
    "accessibility_score": True,
    "market_potential_index": True,
    "infrastructure_readiness": True,
    "log_freight_volume": True,
    "labor_per_warehouse_sqm": True,
    "distance_to_nearest_hub_km": False,
    "land_cost_index": False,
}


class ClassificationBenchmark:
    """Train and compare classification models for hub suitability.

    Supports Random Forest, XGBoost, and SVM classifiers with
    stratified k-fold cross-validation. Includes TOPSIS multi-criteria
    decision analysis as a non-ML baseline.
    """

    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None
        self.best_model_name = None

    def train(self, X: np.ndarray, y: np.ndarray, model_name: str = "XGBoost"):
        """Train a single model.

        Args:
            X: Feature matrix.
            y: Target vector (encoded).
            model_name: Name of the model to train.

        Returns:
            Fitted model instance.
        """
        model = MODELS[model_name]
        model.fit(X, y)
        logger.info("Trained %s on %d samples", model_name, X.shape[0])
        print(f"Trained {model_name} on {X.shape[0]} samples")
        return model

    def cross_validate_model(self, X, y, model_name="XGBoost", n_folds=5):
        model = MODELS[model_name]
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

        results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=False
        )

        return {
            metric: {
                "mean": round(results[f"test_{metric}"].mean(), 4),
                "std": round(results[f"test_{metric}"].std(), 4),
            }
            for metric in scoring
        }

    def compare_models(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> pd.DataFrame:
        """Compare all ML models using cross-validation.

        Args:
            X: Feature matrix.
            y: Target vector (encoded).
            n_folds: Number of CV folds.

        Returns:
            DataFrame with comparison results.
        """
        rows = []
        best_f1 = -1

        for name in MODELS:
            print(f"  Evaluating {name}...")
            cv_results = self.cross_validate_model(X, y, name, n_folds)
            row = {"Model": name}
            for metric, values in cv_results.items():
                row[metric] = values["mean"]
            rows.append(row)

            if cv_results["f1_weighted"]["mean"] > best_f1:
                best_f1 = cv_results["f1_weighted"]["mean"]
                self.best_model_name = name

        # Train best model on full data
        self.best_model = self.train(X, y, self.best_model_name)
        logger.info("Best model: %s (F1=%.4f)", self.best_model_name, best_f1)
        print(f"\nBest model: {self.best_model_name} (F1={best_f1:.4f})")

        return pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)

    def topsis_rank(self, X_raw: pd.DataFrame) -> pd.Series:
        """Rank locations using TOPSIS multi-criteria decision analysis.

        Args:
            X_raw: Raw feature DataFrame (before sklearn preprocessing).

        Returns:
            Series of TOPSIS scores (0-1, higher is better).
        """
        available = [c for c in TOPSIS_WEIGHTS if c in X_raw.columns]
        if not available:
            print("Warning: No TOPSIS criteria columns found in data")
            return pd.Series(np.zeros(len(X_raw)), index=X_raw.index)

        matrix = X_raw[available].copy().values.astype(float)

        # Normalize the decision matrix
        norms = np.sqrt((matrix ** 2).sum(axis=0))
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        # Apply weights
        weights = np.array([TOPSIS_WEIGHTS[c] for c in available])
        weighted = normalized * weights

        # Determine ideal best and worst
        ideal_best = np.zeros(len(available))
        ideal_worst = np.zeros(len(available))
        for i, col in enumerate(available):
            if TOPSIS_BENEFIT.get(col, True):
                ideal_best[i] = weighted[:, i].max()
                ideal_worst[i] = weighted[:, i].min()
            else:
                ideal_best[i] = weighted[:, i].min()
                ideal_worst[i] = weighted[:, i].max()

        # Calculate distances to ideal solutions
        dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

        # TOPSIS score
        denom = dist_best + dist_worst
        denom[denom == 0] = 1.0
        scores = dist_worst / denom

        print(f"TOPSIS ranking computed for {len(scores)} locations")
        return pd.Series(scores, index=X_raw.index, name="topsis_score")

    def topsis_classify(self, X_raw, thresholds=None):
        if thresholds is None:
            thresholds = {"optimal": 0.70, "good": 0.50, "marginal": 0.30}

        scores = self.topsis_rank(X_raw)
        labels = np.where(
            scores >= thresholds["optimal"], "optimal",
            np.where(
                scores >= thresholds["good"], "good",
                np.where(
                    scores >= thresholds["marginal"], "marginal",
                    "unsuitable"
                )
            )
        )
        return labels

    def save_model(self, model=None, name="best_model"):
        model = model or self.best_model
        path = self.output_dir / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"Model saved to {path}")

    def load_model(self, name="best_model"):
        path = self.output_dir / f"{name}.pkl"
        return joblib.load(path)
