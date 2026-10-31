"""
Time-Aware Dataset Splitting for Financial Transactions.

Enforces strict chronological partitioning (Train -> Validation -> Test)
to eliminate lookahead data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class TemporalDatasetSplit:
    """Chronologically partitioned transaction subsets."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_time_range: tuple[float, float]
    val_time_range: tuple[float, float]
    test_time_range: tuple[float, float]


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    time_column: str = "Time",
) -> TemporalDatasetSplit:
    """
    Split a DataFrame chronologically based on the time_column.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction records sorted or unsorted by time.
    train_ratio : float
        Proportion of earliest records for training (e.g., 0.70).
    val_ratio : float
        Proportion of middle records for validation (e.g., 0.15).
    time_column : str
        Column containing elapsed seconds or timestamps.

    Returns
    -------
    TemporalDatasetSplit
        Container with train, val, and test dataframes and their time bounds.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")

    # Ensure chronological order
    sorted_df = df.sort_values(by=time_column).reset_index(drop=True)
    n = len(sorted_df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    return TemporalDatasetSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_time_range=(float(train_df[time_column].min()), float(train_df[time_column].max())),
        val_time_range=(float(val_df[time_column].min()), float(val_df[time_column].max())),
        test_time_range=(float(test_df[time_column].min()), float(test_df[time_column].max())),
    )
