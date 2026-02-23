"""
Precision, Recall, F1 and related classification metrics for lip-sync evaluation.

Usage::

    from app.utils.metrics import compute_metrics, compute_metrics_at_threshold

    # From binary predicted labels
    m = compute_metrics(y_true=[1, 0, 1, 0], y_pred=[1, 1, 0, 0])
    print(m["precision"], m["recall"], m["f1"])

    # From raw P(REAL) confidence scores + a threshold
    m = compute_metrics_at_threshold(
        y_true=[1, 0, 1, 0],
        scores=[0.3, 0.7, 0.4, 0.8],
        threshold=0.5,
    )
"""

from typing import List

import numpy as np


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    positive_label: int = 1,
) -> dict:
    """
    Compute precision, recall, F1 and accuracy from binary labels.

    Args:
        y_true:         Ground truth labels. Convention: 1 = fake, 0 = real.
        y_pred:         Predicted labels.
        positive_label: Which class is treated as "positive" for P/R.
                        Default 1 (fake detection as the positive class).

    Returns:
        Dict with keys: precision, recall, f1, accuracy, tp, tn, fp, fn, total.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true length ({len(y_true_arr)}) must equal y_pred length ({len(y_pred_arr)})"
        )
    if len(y_true_arr) == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0,
        }

    neg = 1 - positive_label
    tp = int(np.sum((y_true_arr == positive_label) & (y_pred_arr == positive_label)))
    tn = int(np.sum((y_true_arr == neg) & (y_pred_arr == neg)))
    fp = int(np.sum((y_true_arr == neg) & (y_pred_arr == positive_label)))
    fn = int(np.sum((y_true_arr == positive_label) & (y_pred_arr == neg)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true_arr)

    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "accuracy": round(float(accuracy), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": int(len(y_true_arr)),
    }


def compute_metrics_at_threshold(
    y_true: List[int],
    scores: List[float],
    threshold: float = 0.5,
    positive_is_fake: bool = True,
) -> dict:
    """
    Compute classification metrics at a specific P(REAL) confidence threshold.

    Args:
        y_true:           Ground truth labels (1 = fake, 0 = real).
        scores:           Model P(REAL) confidence scores. Higher = more likely real.
        threshold:        confidence >= threshold → predicted real (0), else fake (1).
        positive_is_fake: If True, fake (1) is the positive class for P/R.

    Returns:
        Metrics dict plus the applied threshold.
    """
    y_pred = [0 if s >= threshold else 1 for s in scores]
    metrics = compute_metrics(y_true, y_pred, positive_label=1 if positive_is_fake else 0)
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_threshold(
    y_true: List[int],
    scores: List[float],
    metric: str = "f1",
    n_thresholds: int = 100,
) -> dict:
    """
    Search for the P(REAL) threshold that maximises the given metric.

    Args:
        y_true:        Ground truth labels (1 = fake, 0 = real).
        scores:        P(REAL) confidence scores.
        metric:        One of 'f1', 'precision', 'recall', 'accuracy'.
        n_thresholds:  Number of candidate thresholds to evaluate.

    Returns:
        Dict with best_threshold, optimized_for, and full metrics at that threshold.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    best_threshold = 0.5
    best_value = -1.0
    best_metrics: dict = {}

    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, scores, threshold=float(t))
        val = float(m.get(metric, 0.0))
        if val > best_value:
            best_value = val
            best_threshold = float(t)
            best_metrics = m

    best_metrics["best_threshold"] = best_threshold
    best_metrics["optimized_for"] = metric
    return best_metrics
