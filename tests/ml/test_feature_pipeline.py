"""
Unit tests for time-aware splitting and feature transformation pipeline.
"""

from pathlib import Path

import joblib
import numpy as np
import pytest

from ml.preprocessing.pipeline import (
    CyclicalTimeTransformer,
    create_feature_pipeline,
)
from ml.preprocessing.time_split import split_by_time
from ml.preprocessing.validator import generate_synthetic_benchmark


@pytest.mark.ml
def test_split_by_time_chronological_ordering() -> None:
    """Verify that temporal splits are strictly non-overlapping and ordered by Time."""
    df = generate_synthetic_benchmark(n_samples=1000, random_state=42)
    split = split_by_time(df, train_ratio=0.70, val_ratio=0.15)

    assert len(split.train_df) == 700
    assert len(split.val_df) == 150
    assert len(split.test_df) == 150

    # Strict chronological boundary check
    assert split.train_time_range[1] <= split.val_time_range[0]
    assert split.val_time_range[1] <= split.test_time_range[0]


@pytest.mark.ml
def test_split_by_time_invalid_ratios() -> None:
    """Ratios exceeding 1.0 must raise ValueError."""
    df = generate_synthetic_benchmark(n_samples=100)
    with pytest.raises(ValueError, match="must be < 1.0"):
        split_by_time(df, train_ratio=0.80, val_ratio=0.25)


@pytest.mark.ml
def test_cyclical_time_transformer() -> None:
    """Verify sine and cosine outputs are bounded in [-1.0, 1.0]."""
    df = generate_synthetic_benchmark(n_samples=500, random_state=42)
    transformer = CyclicalTimeTransformer(time_column="Time")

    out = transformer.transform(df)

    assert out.shape == (500, 2)
    assert np.all(out[:, 0] >= -1.0) and np.all(out[:, 0] <= 1.0)
    assert np.all(out[:, 1] >= -1.0) and np.all(out[:, 1] <= 1.0)


@pytest.mark.ml
def test_feature_pipeline_fit_transform_and_serialization(tmp_path: Path) -> None:
    """Verify ColumnTransformer fits, transforms to 31 features, and serializes cleanly."""
    df = generate_synthetic_benchmark(n_samples=200, random_state=42)
    X = df.drop(columns=["Class"])

    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(X)

    # 2 (Time sin/cos) + 1 (Amount) + 28 (PCA) = 31 features
    assert transformed.shape == (200, 31)

    # Verify serialization
    model_file = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, model_file)
    assert model_file.is_file()

    loaded_pipe = joblib.load(model_file)
    transformed_reloaded = loaded_pipe.transform(X)
    np.testing.assert_array_almost_equal(transformed, transformed_reloaded)


@pytest.mark.ml
def test_cyclical_time_transformer_midnight_continuity() -> None:
    import numpy as np
    import pandas as pd

    from ml.preprocessing.pipeline import CyclicalTimeTransformer

    # One second before midnight and exactly at midnight
    df = pd.DataFrame({"Time": [86399.0, 0.0]})

    transformer = CyclicalTimeTransformer(time_column="Time")
    transformed = transformer.fit_transform(df)

    # Output columns: [time_sin, time_cos]
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (2, 2)

    distance = np.linalg.norm(transformed[0] - transformed[1])

    # Both timestamps should be close on the unit circle.
    assert distance < 0.01
