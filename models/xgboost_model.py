from __future__ import annotations

from pathlib import Path

import numpy as np

from training.local_train import positive_class_weight


class LocalXGBoostModel:
    """Client-side XGBoost classifier. Models are ensembled without sharing raw data."""

    def __init__(self, params: dict, seed: int = 42) -> None:
        from xgboost import XGBClassifier

        self.params = dict(params)
        self.seed = seed
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
            **self.params,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        self.model.set_params(scale_pos_weight=positive_class_weight(y))
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> list[dict[str, float | str]]:
        importances = getattr(self.model, "feature_importances_", np.zeros(len(feature_names)))
        rows = [
            {"feature": name, "importance": float(score)}
            for name, score in zip(feature_names, importances)
        ]
        return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        self.model.load_model(str(path))
