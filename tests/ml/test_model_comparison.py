"""
Unit tests for model comparison, Random Forest, Isolation Forest, and threshold calibration.
"""

import numpy as np
import pytest

from ml.preprocessing.validator import generate_synthetic_benchmark

try:
    from ml.preprocessing.pipeline import create_feature_pipeline
    from ml.training.comparison import (
        evaluate_isolation_forest,
        find_optimal_threshold,
        train_isolation_forest,
        train_random_forest,
    )
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not HAS_SKLEARN, reason="scipy/sklearn C-extension blocked by OS policy"),
]
def test_train_random_forest() -> None:
    """Verify that Random Forest trains and outputs probability distributions."""
    df = generate_synthetic_benchmark(n_samples=300, fraud_rate=0.05, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    rf = train_random_forest(X, y, n_estimators=20, max_depth=5, random_state=42)

    assert hasattr(rf, "estimators_")
    assert len(rf.estimators_) == 20
    probas = rf.predict_proba(X)
    assert probas.shape == (300, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


@pytest.mark.ml
def test_train_and_evaluate_isolation_forest() -> None:
    """Verify unsupervised Isolation Forest anomaly detection pipeline."""
    df = generate_synthetic_benchmark(n_samples=250, fraud_rate=0.04, random_state=42)
    X_raw = df.drop(columns=["Class"])
    y = df["Class"].to_numpy()

    pipe = create_feature_pipeline()
    X = pipe.fit_transform(X_raw)

    iso = train_isolation_forest(X, contamination=0.05, random_state=42)
    metrics = evaluate_isolation_forest(iso, X, y)

    assert metrics.total_samples == 250
    assert metrics.true_positives + metrics.false_positives > 0
    assert 0.0 <= metrics.f1 <= 1.0


@pytest.mark.ml
def test_find_optimal_threshold() -> None:
    """Verify cost-sensitive threshold search finds minimum loss."""
    y_true = np.array([0] * 90 + [1] * 10)
    # Synthetic probabilities where fraud has higher probability on average
    rng = np.random.default_rng(42)
    y_proba = np.concatenate([
        rng.uniform(0.01, 0.30, size=90),
        rng.uniform(0.60, 0.99, size=10),
    ])

    best_cal = find_optimal_threshold(y_true, y_proba, cost_fp=5.0, cost_fn=100.0)

    assert 0.0 < best_cal.threshold < 1.0
    assert best_cal.total_financial_loss >= 0.0
    assert best_cal.recall > 0.80  # Should catch almost all high-cost fraud
