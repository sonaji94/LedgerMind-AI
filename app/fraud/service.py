"""
Machine Learning Fraud Inference Service with SHAP Explainability.
Serves real-time risk predictions with localized feature attribution.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings

settings = get_settings()

DEFAULT_MODEL_PATH = Path("ml/models/fraud_model.joblib")
DEFAULT_PREPROCESSOR_PATH = Path("ml/models/preprocessor.joblib")
EXPECTED_PCA = [f"V{i}" for i in range(1, 29)]


class FraudInferenceService:
    """Singleton-style inference service caching trained models & SHAP explainer."""

    _model: Any = None
    _preprocessor: Any = None
    _threshold: float = 0.50
    _model_type: str = "Unknown"
    _explainer: Any = None

    @classmethod
    def load(cls) -> None:
        """Load model, preprocessor, and initialize SHAP TreeExplainer."""
        if DEFAULT_PREPROCESSOR_PATH.is_file():
            cls._preprocessor = joblib.load(DEFAULT_PREPROCESSOR_PATH)

        if DEFAULT_MODEL_PATH.is_file():
            bundle = joblib.load(DEFAULT_MODEL_PATH)
            if isinstance(bundle, dict):
                cls._model = bundle.get("model")
                cls._threshold = bundle.get("optimal_threshold", 0.50)
                cls._model_type = bundle.get("model_type", "RandomForestClassifier")
            else:
                cls._model = bundle
                cls._threshold = 0.50
                cls._model_type = type(bundle).__name__

            # Initialize SHAP explainer if Tree-based
            if cls._model and hasattr(cls._model, "estimators_"):
                try:
                    import shap
                    cls._explainer = shap.TreeExplainer(cls._model)
                except Exception:
                    cls._explainer = None

    @classmethod
    def is_ready(cls) -> bool:
        return cls._model is not None and cls._preprocessor is not None

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        return {
            "model_type": cls._model_type,
            "optimal_threshold": cls._threshold,
            "features_count": 31,
            "preprocessor_type": type(cls._preprocessor).__name__ if cls._preprocessor else "None",
            "explainability_enabled": cls._explainer is not None,
            "status": "ready" if cls.is_ready() else "not_loaded",
        }

    @classmethod
    def predict(
        cls,
        features: dict[str, float],
        explain: bool = False,
    ) -> tuple[float, bool, float, dict[str, float] | None]:
        """
        Run inference on raw transaction features and optionally compute SHAP attributions.

        Returns
        -------
        tuple[risk_score, is_suspicious, latency_ms, top_shap_contributions]
        """
        start_t = time.perf_counter()

        record: dict[str, float] = {
            "Time": float(features.get("Time", 0.0)),
            "Amount": float(features.get("Amount", 0.0)),
        }
        for pca in EXPECTED_PCA:
            record[pca] = float(features.get(pca, 0.0))

        df = pd.DataFrame([record])
        X_trans = cls._preprocessor.transform(df) if cls._preprocessor else df.to_numpy()

        if cls._model and hasattr(cls._model, "predict_proba"):
            proba = float(cls._model.predict_proba(X_trans)[0, 1])
        elif cls._model and hasattr(cls._model, "decision_function"):
            score = -float(cls._model.decision_function(X_trans)[0])
            proba = float(1.0 / (1.0 + np.exp(-score)))
        else:
            proba = 0.05 if record["Amount"] < 500.0 else 0.85

        is_suspicious = proba >= cls._threshold

        # Compute SHAP explanation if requested and explainer is initialized
        top_contributions: dict[str, float] | None = None
        if explain and cls._explainer is not None:
            try:
                shap_vals = cls._explainer.shap_values(X_trans)
                # shap_values for binary classifier returns array for class 1
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    vals = np.array(shap_vals[1][0])
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = np.array(shap_vals)[0]

                feature_names = list(record.keys())
                # Top 5 most influential features by absolute attribution
                abs_indices = np.argsort(np.abs(vals))[::-1][:5]
                top_contributions = {
                    feature_names[i] if i < len(feature_names) else f"feature_{i}": float(vals[i])
                    for i in abs_indices
                }
            except Exception:
                top_contributions = None

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return proba, is_suspicious, latency_ms, top_contributions
