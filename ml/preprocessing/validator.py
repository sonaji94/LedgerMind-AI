"""
Dataset Validator & Schema Profiler for LedgerMind AI.

Validates the ULB Credit Card Fraud dataset against required financial
and machine learning invariants before data reaches preprocessing or model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

EXPECTED_PCA_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
EXPECTED_ALL_COLUMNS: tuple[str, ...] = ("Time", *EXPECTED_PCA_COLUMNS, "Amount", "Class")


@dataclass(frozen=True)
class DataValidationSummary:
    """Summary of dataset profiling and schema validation checks."""

    total_rows: int
    total_columns: int
    fraud_count: int
    legitimate_count: int
    fraud_percentage: float
    missing_values_count: int
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        """Output a clean human-readable validation summary."""
        print("=" * 60)
        print("📊 LEDGERMIND AI — DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"Total Transactions : {self.total_rows:,}")
        print(f"Total Columns      : {self.total_columns}")
        print(f"Legitimate (Class 0): {self.legitimate_count:,}")
        print(f"Fraudulent (Class 1): {self.fraud_count:,}")
        print(f"Fraud Prevalence   : {self.fraud_percentage:.4f}%")
        print(f"Missing Values     : {self.missing_values_count}")
        print(f"Validation Status  : {'✅ PASSED' if self.is_valid else '❌ FAILED'}")
        if self.validation_errors:
            print("\nValidation Errors:")
            for err in self.validation_errors:
                print(f"  • {err}")
        print("=" * 60)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str] = EXPECTED_ALL_COLUMNS,
) -> DataValidationSummary:
    """
    Validate a pandas DataFrame against expected schema and integrity invariants.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded transactions dataset.
    required_columns : Sequence[str]
        Expected column names in the dataset.

    Returns
    -------
    DataValidationSummary
        Structured report detailing validity and statistics.
    """
    errors: list[str] = []

    # 1. Column presence check
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Null/NaN check
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        errors.append(f"Dataset contains {null_count} null/NaN values across columns.")

    # 3. Target column invariants
    if "Class" in df.columns:
        unique_classes = set(df["Class"].unique())
        if not unique_classes.issubset({0, 1}):
            errors.append(f"Invalid target values in 'Class': {unique_classes}. Expected only 0 or 1.")
        fraud_count = int((df["Class"] == 1).sum())
        legitimate_count = int((df["Class"] == 0).sum())
    else:
        fraud_count = 0
        legitimate_count = 0
        errors.append("Target column 'Class' is missing.")

    # 4. Monetary Amount invariants (amounts cannot be negative)
    if "Amount" in df.columns:
        negative_amounts = int((df["Amount"] < 0).sum())
        if negative_amounts > 0:
            errors.append(f"Found {negative_amounts} records with negative Amount values.")

    # 5. Temporal invariant (Time must be non-negative)
    if "Time" in df.columns:
        negative_times = int((df["Time"] < 0).sum())
        if negative_times > 0:
            errors.append(f"Found {negative_times} records with negative Time values.")

    total_rows = len(df)
    fraud_pct = (fraud_count / total_rows * 100.0) if total_rows > 0 else 0.0

    return DataValidationSummary(
        total_rows=total_rows,
        total_columns=len(df.columns),
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        fraud_percentage=fraud_pct,
        missing_values_count=null_count,
        is_valid=(len(errors) == 0),
        validation_errors=errors,
    )


def load_dataset(file_path: Path | str) -> pd.DataFrame:
    """
    Load transactions dataset from CSV or Parquet format.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .parquet")


def generate_synthetic_benchmark(
    n_samples: int = 2000,
    fraud_rate: float = 0.005,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a statistically representative synthetic dataset matching the ULB schema.
    Used for unit testing, CI pipelines, and offline development.
    """
    rng = np.random.default_rng(random_state)

    times = np.sort(rng.uniform(0, 172800, size=n_samples))  # 48 hours in seconds
    pca_data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 28))
    amounts = np.round(rng.exponential(scale=88.0, size=n_samples), 2)

    n_fraud = int(n_samples * fraud_rate)
    classes = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, size=n_fraud, replace=False)
    classes[fraud_indices] = 1

    # Induce subtle anomalies in fraud amounts
    amounts[fraud_indices] = np.round(rng.uniform(10.0, 2000.0, size=n_fraud), 2)

    data_dict: dict[str, np.ndarray] = {"Time": times}
    for i, col in enumerate(EXPECTED_PCA_COLUMNS):
        data_dict[col] = pca_data[:, i]
    data_dict["Amount"] = amounts
    data_dict["Class"] = classes

    return pd.DataFrame(data_dict)
