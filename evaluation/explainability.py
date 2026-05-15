from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.checkpoints import save_json


def save_xgboost_shap_summary(model, X_sample: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    """Save mean absolute SHAP values for an XGBoost model when shap is installed."""
    try:
        import shap
    except ImportError:
        return

    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(X_sample)
    mean_abs = np.abs(values).mean(axis=0)
    rows = sorted(
        (
            {"feature": feature, "mean_abs_shap": float(score)}
            for feature, score in zip(feature_names, mean_abs)
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )
    save_json({"features": rows}, output_path)
