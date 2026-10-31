"""
Feature Engineering & Transformation Pipeline for LedgerMind AI.

Transforms raw financial transaction features into model-ready arrays:
- Cyclical transformation of Time into 24-hour sine/cosine features
- Outlier-resistant scaling (RobustScaler) of transaction Amount
- Passthrough of anonymized PCA components (V1 - V28)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

if TYPE_CHECKING:
    from collections.abc import Sequence

PCA_FEATURE_NAMES: list[str] = [f"V{i}" for i in range(1, 29)]


class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms elapsed seconds (Time) into cyclical 24-hour representations.
    Produces two features: hour_sin and hour_cos.
    """

    def __init__(self, time_column: str = "Time") -> None:
        self.time_column = time_column

    def fit(self, X: pd.DataFrame, y: Sequence[int] | None = None) -> CyclicalTimeTransformer:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            time_series = X[self.time_column].to_numpy()
        else:
            time_series = np.asarray(X)[:, 0]

        # Convert seconds to hour of day (0 to 23.999...)
        hours = (time_series / 3600.0) % 24.0

        sin_time = np.sin(2.0 * np.pi * hours / 24.0)
        cos_time = np.cos(2.0 * np.pi * hours / 24.0)

        return np.column_stack([sin_time, cos_time])

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        return ["time_sin", "time_cos"]


def create_feature_pipeline() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that processes:
    1. Time -> Cyclical (sin, cos) features
    2. Amount -> RobustScaler (median-centered, IQR-scaled)
    3. V1..V28 -> Passthrough
    """
    time_pipe = Pipeline([
        ("cyclical_time", CyclicalTimeTransformer(time_column="Time")),
    ])

    amount_pipe = Pipeline([
        ("robust_scale", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("time", time_pipe, ["Time"]),
            ("amount", amount_pipe, ["Amount"]),
            ("pca", "passthrough", PCA_FEATURE_NAMES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
