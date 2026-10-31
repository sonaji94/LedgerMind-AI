"""
Unit tests for EDA statistical profiling and chart generation logic.
"""

from pathlib import Path

import pandas as pd
import pytest

from ml.preprocessing.eda import (
    compute_amount_statistics,
    generate_eda_figures,
    profile_dataset,
)
from ml.preprocessing.validator import generate_synthetic_benchmark


@pytest.mark.ml
def test_compute_amount_statistics() -> None:
    """Verify amount statistics calculation accuracy."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 100.0])
    stats = compute_amount_statistics(series)
    assert stats.min_amount == 10.0
    assert stats.max_amount == 100.0
    assert stats.median == 30.0
    assert stats.mean == 40.0
    assert stats.p25 == 20.0
    assert stats.p75 == 40.0
    assert stats.std > 0.0
    assert stats.p95 > stats.p75

@pytest.mark.ml
def test_profile_dataset_empty_check():
    df = generate_synthetic_benchmark(
        n_samples=10,
        fraud_rate=0.0,
    )

    profile = profile_dataset(df)

    assert profile.fraud_count == 0
    assert profile.fraud_ratio == 0.0


@pytest.mark.ml
def test_profile_dataset() -> None:
    """Verify dataset profiling accurately identifies class splits and correlations."""
    df = generate_synthetic_benchmark(n_samples=500, fraud_rate=0.02, random_state=42)
    profile = profile_dataset(df, top_n_correlations=3)

    assert profile.total_records == 500
    assert profile.fraud_count > 0
    assert profile.legitimate_count + profile.fraud_count == 500
    assert 0.0 < profile.fraud_ratio < 0.1
    assert len(profile.top_negative_correlations) <= 3
    assert len(profile.top_positive_correlations) <= 3


@pytest.mark.ml
def test_generate_eda_figures(tmp_path: Path) -> None:
    """Verify EDA figures are rendered cleanly without GUI dependencies."""
    df = generate_synthetic_benchmark(n_samples=200, fraud_rate=0.05, random_state=42)
    figures = generate_eda_figures(df, output_dir=tmp_path)

    assert len(figures) == 3
    for fig_path in figures:
        assert fig_path.is_file()
        assert fig_path.stat().st_size > 0
