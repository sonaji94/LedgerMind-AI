"""
Unit tests for ML dataset validation logic and schema invariants.
"""

import pytest

from ml.preprocessing.validator import (
    EXPECTED_ALL_COLUMNS,
    generate_synthetic_benchmark,
    validate_dataframe,
)


@pytest.mark.ml
def test_valid_synthetic_benchmark_passes_validation() -> None:
    """A valid synthetic dataset matching the schema must pass all validation checks."""
    df = generate_synthetic_benchmark(n_samples=500, fraud_rate=0.01, random_state=42)
    summary = validate_dataframe(df)

    assert summary.is_valid is True
    assert summary.total_rows == 500
    assert summary.total_columns == len(EXPECTED_ALL_COLUMNS)
    assert summary.missing_values_count == 0
    assert summary.fraud_count > 0
    assert len(summary.validation_errors) == 0


@pytest.mark.ml
def test_validation_fails_on_missing_column() -> None:
    """Validator must reject a dataset missing any required column."""
    df = generate_synthetic_benchmark(n_samples=100)
    df_missing = df.drop(columns=["Amount"])

    summary = validate_dataframe(df_missing)

    assert summary.is_valid is False
    assert any("Missing required columns" in err for err in summary.validation_errors)


@pytest.mark.ml
def test_validation_fails_on_null_values() -> None:
    """Validator must reject a dataset containing null or NaN entries."""
    df = generate_synthetic_benchmark(n_samples=100)
    df.loc[5, "V1"] = None

    summary = validate_dataframe(df)

    assert summary.is_valid is False
    assert summary.missing_values_count == 1
    assert any("null/NaN values" in err for err in summary.validation_errors)


@pytest.mark.ml
def test_validation_fails_on_negative_amounts() -> None:
    """Monetary transaction amounts cannot be negative."""
    df = generate_synthetic_benchmark(n_samples=100)
    df.loc[10, "Amount"] = -25.50

    summary = validate_dataframe(df)

    assert summary.is_valid is False
    assert any("negative Amount values" in err for err in summary.validation_errors)


@pytest.mark.ml
def test_validation_fails_on_invalid_target_class() -> None:
    """Target class must only be binary 0 or 1."""
    df = generate_synthetic_benchmark(n_samples=100)
    df.loc[12, "Class"] = 99

    summary = validate_dataframe(df)

    assert summary.is_valid is False
    assert any("Invalid target values in 'Class'" in err for err in summary.validation_errors)


@pytest.mark.ml
def test_validation_fails_on_negative_time() -> None:
    df = generate_synthetic_benchmark(n_samples=100)
    df.loc[0, "Time"] = -10.0

    summary = validate_dataframe(df)

    assert summary.is_valid is False
    assert any(
        "negative Time values" in err
        for err in summary.validation_errors
    )

