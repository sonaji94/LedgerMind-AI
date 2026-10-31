"""
Baseline Supervised Fraud Detection Model (Class-Weighted Logistic Regression).

Provides model training, threshold evaluation, and metric computation
specifically designed for severely imbalanced financial transaction data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ModelEvaluationMetrics:
    """Evaluation metrics tailored for imbalanced fraud classification."""

    total_samples: int
    threshold: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float

    def print_report(self, split_name: str = "Evaluation Split") -> None:
        """Output structured metric performance."""
        print("=" * 65)
        print(f"📊 PERFORMANCE REPORT: {split_name.upper()} (Threshold = {self.threshold:.2f})")
        print("=" * 65)
        print(f"Total Transactions : {self.total_samples:,}")
        print("\nConfusion Matrix:")
        print(f"  • True Negatives  (Legit  -> Legit) : {self.true_negatives:,}")
        print(f"  • False Positives (Legit  -> Fraud) : {self.false_positives:,} (False Alarms)")
        print(f"  • False Negatives (Fraud  -> Legit) : {self.false_negatives:,} (Missed Fraud)")
        print(f"  • True Positives  (Fraud  -> Fraud) : {self.true_positives:,} (Caught Fraud)")
        print("\nPrimary Imbalanced Metrics:")
        print(f"  • Precision (Quality of alert) : {self.precision:.4f} ({self.precision * 100:.2f}%)")
        print(f"  • Recall    (Fraud caught)     : {self.recall:.4f} ({self.recall * 100:.2f}%)")
        print(f"  • F1-Score  (Harmonic mean)    : {self.f1:.4f}")
        print(f"  • PR-AUC    (Avg Precision)    : {self.pr_auc:.4f}")
        print(f"  • ROC-AUC   (Discrimination)   : {self.roc_auc:.4f}")
        print("=" * 65)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: Sequence[int] | np.ndarray,
    class_weight: str | dict[int, float] = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a class-weighted Logistic Regression classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Transformed feature matrix.
    y_train : Sequence[int] | np.ndarray
        Binary labels (0 for legitimate, 1 for fraud).
    class_weight : str | dict
        Weighting strategy ('balanced' dynamically adjusts weights).
    max_iter : int
        Maximum iterations for solver convergence.
    random_state : int
        Deterministic random seed.

    Returns
    -------
    LogisticRegression
        Fitted scikit-learn model.
    """
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(
    model: LogisticRegression,
    X: np.ndarray,
    y: Sequence[int] | np.ndarray,
    threshold: float = 0.50,
) -> ModelEvaluationMetrics:
    """
    Evaluate model predictions against ground truth labels at a given threshold.

    Parameters
    ----------
    model : LogisticRegression
        Fitted model providing predict_proba().
    X : np.ndarray
        Feature matrix.
    y : Sequence[int] | np.ndarray
        Ground-truth binary labels.
    threshold : float
        Decision boundary threshold for positive class (default 0.50).
    """
    y_true = np.asarray(y)

    # Predict continuous probabilities for class 1 (fraud)
    y_proba = model.predict_proba(X)[:, 1]

    # Apply decision threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Zero-division safe precision and recall
    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    # Curve AUCs
    # If the evaluation set has only 1 class, AUC is undefined
    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_proba))
        roc_auc = float(roc_auc_score(y_true, y_proba))
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    return ModelEvaluationMetrics(
        total_samples=len(y_true),
        threshold=threshold,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
    )
