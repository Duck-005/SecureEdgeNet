from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (scores >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
    }


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, scores, 0.5)
    best_value = float(best_metrics.get(metric, 0.0))
    for threshold in np.linspace(min_threshold, max_threshold, steps):
        current = classification_metrics(y_true, scores, float(threshold))
        value = float(current.get(metric, 0.0))
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics
