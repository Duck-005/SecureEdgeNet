from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleScores:
    logistic: np.ndarray | None = None
    mlp: np.ndarray | None = None
    xgboost: np.ndarray | None = None

    def available(self) -> dict[str, np.ndarray]:
        return {
            name: scores
            for name, scores in {
                "logistic": self.logistic,
                "mlp": self.mlp,
                "xgboost": self.xgboost,
            }.items()
            if scores is not None
        }


class WeightedFraudEnsemble:
    def __init__(
        self,
        weights: dict[str, float],
        method: str = "weighted_average",
        threshold: float = 0.5,
    ) -> None:
        self.weights = weights
        self.method = method
        self.threshold = threshold

    def predict_scores(self, scores: EnsembleScores) -> np.ndarray:
        available = scores.available()
        if not available:
            raise ValueError("No model scores were provided to the ensemble.")

        if self.method == "majority_vote":
            votes = [(values >= self.threshold).astype(float) for values in available.values()]
            return np.mean(np.vstack(votes), axis=0)

        total = sum(float(self.weights.get(name, 0.0)) for name in available)
        if total <= 0:
            total = float(len(available))
            normalized = {name: 1.0 / total for name in available}
        else:
            normalized = {name: float(self.weights.get(name, 0.0)) / total for name in available}

        result = np.zeros(len(next(iter(available.values()))), dtype=np.float64)
        for name, values in available.items():
            result += normalized[name] * values
        return result

    def predict(self, scores: EnsembleScores) -> np.ndarray:
        return (self.predict_scores(scores) >= self.threshold).astype(int)

    def contribution(self, scores: EnsembleScores, row_index: int) -> dict[str, float]:
        available = scores.available()
        total = sum(float(self.weights.get(name, 0.0)) for name in available) or 1.0
        return {
            name: float(self.weights.get(name, 0.0) / total * values[row_index])
            for name, values in available.items()
        }
