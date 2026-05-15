from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_roc_curve(y_true: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_pr_curve(y_true: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, scores: np.ndarray, threshold: float, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=["legit", "fraud"]).plot(values_format="d")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(rows: list[dict], path: str | Path, top_k: int = 20) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    top = rows[:top_k]
    labels = [str(row["feature"]) for row in top][::-1]
    values = [float(row["importance"]) for row in top][::-1]
    plt.figure(figsize=(9, max(4, len(labels) * 0.28)))
    plt.barh(labels, values)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
