"""
Exploratory Data Analysis (EDA) and Statistical Profiler for LedgerMind AI.

Computes financial distributions, class imbalance metrics, correlation rankings,
and generates publication-quality figures headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated scripts/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class AmountStatistics:
    """Summary statistics for transaction monetary amounts."""

    mean: float
    median: float
    std: float
    min_amount: float
    max_amount: float
    p25: float
    p75: float
    p95: float
    p99: float


@dataclass(frozen=True)
class EDAProfile:
    """Complete statistical summary of the transaction dataset."""

    total_records: int
    fraud_count: int
    legitimate_count: int
    fraud_ratio: float
    legit_amount_stats: AmountStatistics
    fraud_amount_stats: AmountStatistics
    top_negative_correlations: list[tuple[str, float]]
    top_positive_correlations: list[tuple[str, float]]

    def print_summary(self) -> None:
        """Print a structured analytical executive summary."""
        print("=" * 65)
        print("📈 LEDGERMIND AI — FINANCIAL & STATISTICAL PROFILE")
        print("=" * 65)
        print(f"Total Transactions     : {self.total_records:,}")
        print(f"Legitimate Transactions: {self.legitimate_count:,} ({100 - self.fraud_ratio * 100:.3f}%)")
        print(f"Fraudulent Transactions: {self.fraud_count:,} ({self.fraud_ratio * 100:.3f}%)")
        print("-" * 65)
        print("💰 TRANSACTION AMOUNT COMPARISON:")
        print(f"  • Legit Median Amount: ${self.legit_amount_stats.median:.2f} (Mean: ${self.legit_amount_stats.mean:.2f})")
        print(f"  • Fraud Median Amount: ${self.fraud_amount_stats.median:.2f} (Mean: ${self.fraud_amount_stats.mean:.2f})")
        print(f"  • Legit 99th %-ile   : ${self.legit_amount_stats.p99:.2f}")
        print(f"  • Fraud Max Amount   : ${self.fraud_amount_stats.max_amount:.2f}")
        print("-" * 65)
        print("🔍 TOP CORRELATED FEATURES WITH FRAUD:")
        print("  Positive Correlation (Increases fraud likelihood):")
        for feat, corr in self.top_positive_correlations[:3]:
            print(f"    - {feat:5s} : +{corr:.4f}")
        print("  Negative Correlation (Decreases fraud likelihood):")
        for feat, corr in self.top_negative_correlations[:3]:
            print(f"    - {feat:5s} : {corr:.4f}")
        print("=" * 65)


def compute_amount_statistics(series: pd.Series) -> AmountStatistics:
    """Calculate granular percentile and spread statistics for amounts."""
    return AmountStatistics(
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std()),
        min_amount=float(series.min()),
        max_amount=float(series.max()),
        p25=float(series.quantile(0.25)),
        p75=float(series.quantile(0.75)),
        p95=float(series.quantile(0.95)),
        p99=float(series.quantile(0.99)),
    )


def profile_dataset(df: pd.DataFrame, top_n_correlations: int = 5) -> EDAProfile:
    """Generate statistical profile and feature correlation rankings."""
    total_records = len(df)
    fraud_mask = df["Class"] == 1
    fraud_count = int(fraud_mask.sum())
    legit_count = total_records - fraud_count
    fraud_ratio = fraud_count / total_records if total_records > 0 else 0.0

    legit_stats = compute_amount_statistics(df.loc[~fraud_mask, "Amount"])
    fraud_stats = compute_amount_statistics(df.loc[fraud_mask, "Amount"])

    # Compute correlation of all features against Class
    correlations = df.corr()["Class"].drop("Class")

    # Sorted negative correlations (strongest negative first)
    neg_corrs = correlations[correlations < 0].sort_values()
    top_neg = [(str(idx), float(val)) for idx, val in neg_corrs.head(top_n_correlations).items()]

    # Sorted positive correlations (strongest positive first)
    pos_corrs = correlations[correlations > 0].sort_values(ascending=False)
    top_pos = [(str(idx), float(val)) for idx, val in pos_corrs.head(top_n_correlations).items()]

    return EDAProfile(
        total_records=total_records,
        fraud_count=fraud_count,
        legitimate_count=legit_count,
        fraud_ratio=fraud_ratio,
        legit_amount_stats=legit_stats,
        fraud_amount_stats=fraud_stats,
        top_negative_correlations=top_neg,
        top_positive_correlations=top_pos,
    )


def generate_eda_figures(df: pd.DataFrame, output_dir: Path | str = "reports/figures") -> list[Path]:
    """
    Produce visual EDA charts:
    1. Class distribution (linear and log scale).
    2. Transaction amount distribution (Legit vs Fraud).
    3. Temporal hourly transaction frequency.
    4. Correlation ranking bar chart.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    sns.set_theme(style="whitegrid", palette="muted")

    # ── Figure 1: Class Imbalance ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["Class"].value_counts()
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    axes[0].bar(labels, counts.values, color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Transaction Class Counts (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")

    axes[1].bar(labels, counts.values, color=["#2ecc71", "#e74c3c"])
    axes[1].set_yscale("log")
    axes[1].set_title("Transaction Class Counts (Logarithmic Scale)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count (log)")

    fig.tight_layout()
    f1 = out_path / "01_class_imbalance.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    generated_files.append(f1)

    # ── Figure 2: Amount Distribution by Class ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    legit_amounts = df.loc[df["Class"] == 0, "Amount"]
    fraud_amounts = df.loc[df["Class"] == 1, "Amount"]

    # Log1p transform for visual interpretability of skewed amounts
    sns.kdeplot(np.log1p(legit_amounts), label="Legitimate", color="#2ecc71", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(np.log1p(fraud_amounts), label="Fraudulent", color="#e74c3c", fill=True, alpha=0.4, ax=ax)
    ax.set_title("Log-Transformed Transaction Amount Distribution: log(1 + Amount)", fontsize=12, fontweight="bold")
    ax.set_xlabel("log(1 + Amount in €)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    f2 = out_path / "02_amount_distribution.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    generated_files.append(f2)

    # ── Figure 3: Correlation with Fraud ────────────────────────────────
    profile = profile_dataset(df)
    top_features = profile.top_positive_correlations[:5] + profile.top_negative_correlations[:5]
    feats, corrs = zip(*top_features, strict=False)

    colors = ["#e74c3c" if c > 0 else "#3498db" for c in corrs]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, corrs, color=colors, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson Correlation Coefficient with Fraud (Class)")
    ax.set_title("Top 10 Feature Correlations with Fraud", fontsize=12, fontweight="bold")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    f3 = out_path / "03_feature_correlations.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    generated_files.append(f3)

    return generated_files
