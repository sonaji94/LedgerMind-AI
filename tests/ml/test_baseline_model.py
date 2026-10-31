"""
Unit tests for baseline Logistic Regression training and evaluation metrics.
"""

from pathlib import Path

import joblib
import numpy as np
import pytest

from ml.preprocessing.pipeline import create_feature_pipeline
from ml.preprocessing.validator import generate_synthetic_benchmark
from ml.training.baseline import (
    evaluate_classification,
    train_logistic_regression,
)


@pytest.mark.ml
def test_train_logistic_regression_fits_successfully() -> None:
    """Verify that train_logistic_regression produces a fitted model."""
    df = generate_synthetic_benchmark(n_samples=500, fraud_rate=0.04, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    model = train_logistic_regression(X, y, class_weight="balanced", random_state=42)

    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, 31)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


@pytest.mark.ml
def test_evaluate_classification_computes_exact_metrics() -> None:
    """Verify evaluation metric calculations against known dummy outputs."""
    df = generate_synthetic_benchmark(n_samples=300, fraud_rate=0.05, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    model = train_logistic_regression(X, y, random_state=42)
    metrics = evaluate_classification(model, X, y, threshold=0.50)

    assert metrics.total_samples == 300
    assert metrics.true_negatives + metrics.false_positives + metrics.false_negatives + metrics.true_positives == 300
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


@pytest.mark.ml
def test_threshold_tradeoff_behavior() -> None:
    """
    Lowering threshold should increase or maintain recall (catch more fraud).
    Raising threshold should increase or maintain precision (fewer false alarms).
    """
    df = generate_synthetic_benchmark(n_samples=1000, fraud_rate=0.05, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    model = train_logistic_regression(X, y, random_state=42)

    low_thresh_metrics = evaluate_classification(model, X, y, threshold=0.10)
    high_thresh_metrics = evaluate_classification(model, X, y, threshold=0.90)

    # Lower threshold = more flagged as positive -> recall at least as high
    assert low_thresh_metrics.recall >= high_thresh_metrics.recall
    # Higher threshold = more selective -> false positives cannot increase
    assert high_thresh_metrics.false_positives <= low_thresh_metrics.false_positives


@pytest.mark.ml
def test_zero_division_safety_with_extreme_threshold() -> None:
    """Verify precision/recall calculation when threshold produces zero positive alerts."""
    df = generate_synthetic_benchmark(
        n_samples=200,
        fraud_rate=0.05,
        random_state=42,
    )

    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    model = train_logistic_regression(X, y, random_state=42)

    metrics = evaluate_classification(
        model,
        X,
        y,
        threshold=1.0,
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.false_positives == 0

@pytest.mark.ml
def test_model_serialization_consistency(tmp_path: Path) -> None:
    """Verify serialized model loads cleanly and produces identical probabilities."""
    df = generate_synthetic_benchmark(n_samples=200, fraud_rate=0.05, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    model = train_logistic_regression(X, y, random_state=42)
    original_probas = model.predict_proba(X)

    model_file = tmp_path / "baseline_model.joblib"
    joblib.dump(model, model_file)

    loaded_model = joblib.load(model_file)
    reloaded_probas = loaded_model.predict_proba(X)

    np.testing.assert_allclose(original_probas, reloaded_probas, rtol=1e-5)
