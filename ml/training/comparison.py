"""
Model Comparison, Anomaly Detection & Threshold Calibration for LedgerMind AI.

Compares:
1. Baseline Logistic Regression
2. Random Forest Classifier (Nonlinear tree ensemble)
3. Isolation Forest (Unsupervised anomaly detection)

Calibrates optimal classification thresholds based on cost-sensitive loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.training.baseline import ModelEvaluationMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class CostSensitiveEvaluation:
    """Evaluation incorporating financial business costs."""

    threshold: float
    precision: float
    recall: float
    f1: float
    cost_false_positive: float
    cost_false_negative: float
    total_financial_loss: float
    false_positives: int
    false_negatives: int


def train_random_forest(
    X_train: np.ndarray,
    y_train: Sequence[int] | np.ndarray,
    n_estimators: int = 50,
    max_depth: int | None = 10,
    class_weight: str | dict[int, float] = "balanced",
    random_state: int = 42,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    """Train a class-weighted Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, y_train)
    return rf


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.005,
    random_state: int = 42,
    n_jobs: int = 1,
) -> IsolationForest:
    """
    Train an unsupervised Isolation Forest for zero-day anomaly detection.
    Does not use fraud labels during training.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    iso.fit(X_train)
    return iso


def evaluate_isolation_forest(
    model: IsolationForest,
    X: np.ndarray,
    y: Sequence[int] | np.ndarray,
) -> ModelEvaluationMetrics:
    """
    Evaluate Isolation Forest predictions.
    IsolationForest outputs -1 for outliers (anomalies) and +1 for inliers.
    We map: -1 -> 1 (fraud/anomaly), +1 -> 0 (legitimate).
    """
    raw_preds = model.predict(X)
    y_pred = (raw_preds == -1).astype(int)
    # Decision function gives anomaly score (lower is more anomalous)
    scores = -model.decision_function(X)
    # Min-max scale scores to [0, 1] pseudo-probability
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        y_proba = (scores - score_min) / (score_max - score_min)
    else:
        y_proba = np.zeros_like(scores)

    y_true = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_proba))
        roc_auc = float(roc_auc_score(y_true, y_proba))
    else:
        pr_auc = 0.0
        roc_auc = 0.0

    return ModelEvaluationMetrics(
        total_samples=len(y_true),
        threshold=0.50,
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


def find_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 250.0,
    steps: int = 100,
) -> CostSensitiveEvaluation:
    """
    Calibrate the decision threshold that minimizes total financial loss:
    Loss = (False Positives * cost_fp) + (False Negatives * cost_fn)
    """
    y_arr = np.asarray(y_true)
    best_loss = float("inf")
    best_eval: CostSensitiveEvaluation | None = None

    thresholds = np.linspace(0.01, 0.99, steps)
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()

        total_loss = (float(fp) * cost_fp) + (float(fn) * cost_fn)
        precision = float(precision_score(y_arr, y_pred, zero_division=0.0))
        recall = float(recall_score(y_arr, y_pred, zero_division=0.0))
        f1 = float(f1_score(y_arr, y_pred, zero_division=0.0))

        if total_loss < best_loss:
            best_loss = total_loss
            best_eval = CostSensitiveEvaluation(
                threshold=float(thresh),
                precision=precision,
                recall=recall,
                f1=f1,
                cost_false_positive=cost_fp,
                cost_false_negative=cost_fn,
                total_financial_loss=total_loss,
                false_positives=int(fp),
                false_negatives=int(fn),
            )

    assert best_eval is not None
    return best_eval
