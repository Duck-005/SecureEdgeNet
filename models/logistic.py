from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from training.local_train import positive_class_weight


class FederatedLogisticRegression:
    """Logistic regression trained with local SGD and aggregated via FedAvg."""

    def __init__(self, input_dim: int, alpha: float = 0.0001, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            random_state=seed,
            learning_rate="optimal",
        )
        self._initialized = False
        self.initialize()

    def initialize(self) -> None:
        dummy_X = np.zeros((2, self.input_dim), dtype=np.float64)
        dummy_y = np.array([0, 1], dtype=np.int64)
        self.model.partial_fit(dummy_X, dummy_y, classes=np.array([0, 1]))
        self.model.coef_ = np.zeros((1, self.input_dim), dtype=np.float64)
        self.model.intercept_ = np.zeros(1, dtype=np.float64)
        self.model.classes_ = np.array([0, 1], dtype=np.int64)
        self.model.n_features_in_ = self.input_dim
        self.model.t_ = 1.0
        self._initialized = True

    def get_parameters(self) -> list[np.ndarray]:
        return [
            self.model.coef_.astype(np.float32).copy(),
            self.model.intercept_.astype(np.float32).copy(),
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.model.coef_ = np.asarray(parameters[0], dtype=np.float64).reshape(1, self.input_dim)
        self.model.intercept_ = np.asarray(parameters[1], dtype=np.float64).reshape(1)
        self.model.classes_ = np.array([0, 1], dtype=np.int64)
        self.model.n_features_in_ = self.input_dim
        if not hasattr(self.model, "t_"):
            self.model.t_ = 1.0
        self._initialized = True

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        if not self._initialized:
            self.initialize()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        pos_weight = positive_class_weight(y)
        sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float64)
        for _ in range(max(1, epochs)):
            self.model.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self.input_dim = int(self.model.coef_.shape[1])
        self._initialized = True
