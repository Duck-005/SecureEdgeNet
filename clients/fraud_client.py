from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evaluation.metrics import classification_metrics
from models.factory import create_logistic, create_mlp
from models.xgboost_model import LocalXGBoostModel

LOGGER = logging.getLogger(__name__)


def split_model_parameters(parameters: list[np.ndarray], mlp_parameter_count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return parameters[:2], parameters[2 : 2 + mlp_parameter_count]


class FraudFlowerClient:
    """Flower NumPyClient that keeps client data local and shares only model weights."""

    def __init__(
        self,
        cid: str,
        partition: dict[str, np.ndarray],
        config,
        input_dim: int,
        checkpoint_dir: str | Path,
    ) -> None:
        self.cid = cid
        self.partition = partition
        self.config = config
        self.input_dim = input_dim
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logistic = create_logistic(config, input_dim)
        self.mlp = create_mlp(config, input_dim)
        self.mlp_parameter_count = len(self.mlp.get_parameters())

    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return self.logistic.get_parameters() + self.mlp.get_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        lr_params, mlp_params = split_model_parameters(parameters, self.mlp_parameter_count)
        self.logistic.set_parameters(lr_params)
        self.mlp.set_parameters(mlp_params)

    def fit(self, parameters: list[np.ndarray], config: dict | None = None) -> tuple[list[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        server_config = config or {}
        epochs = int(server_config.get("local_epochs", self.config.federated.local_epochs))
        batch_size = int(server_config.get("batch_size", self.config.federated.batch_size))

        X = self.partition["x_train"]
        y = self.partition["y_train"]
        self.logistic.fit(X, y, epochs=epochs)
        losses = self.mlp.fit(X, y, epochs=epochs, batch_size=batch_size)

        metrics = self._evaluate_models(self.partition["x_val"], self.partition["y_val"])
        metrics["mlp_loss"] = float(losses[-1]) if losses else 0.0
        LOGGER.info("Client %s finished fit: %s", self.cid, metrics)
        return self.get_parameters(), len(y), metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict | None = None) -> tuple[float, int, dict]:
        self.set_parameters(parameters)
        metrics = self._evaluate_models(self.partition["x_val"], self.partition["y_val"])
        loss = 1.0 - metrics["ensemble_f1"]
        return float(loss), len(self.partition["y_val"]), metrics

    def train_local_xgboost(self, feature_names: list[str]) -> tuple[LocalXGBoostModel, list[dict]]:
        model = LocalXGBoostModel(dict(self.config.models.xgboost), seed=int(self.config.project.seed))
        model.fit(
            self.partition["x_train"],
            self.partition["y_train"],
            self.partition["x_val"],
            self.partition["y_val"],
        )
        path = self.checkpoint_dir / f"xgboost_client_{self.cid}.json"
        model.save(path)
        return model, model.feature_importance(feature_names)

    def _evaluate_models(self, X: np.ndarray, y: np.ndarray) -> dict:
        lr_scores = self.logistic.predict_proba(X)
        mlp_scores = self.mlp.predict_proba(X)
        mean_scores = 0.5 * lr_scores + 0.5 * mlp_scores
        lr = classification_metrics(y, lr_scores)
        mlp = classification_metrics(y, mlp_scores)
        ensemble = classification_metrics(y, mean_scores)
        return {
            "logistic_f1": lr["f1"],
            "logistic_recall": lr["recall"],
            "mlp_f1": mlp["f1"],
            "mlp_recall": mlp["recall"],
            "ensemble_f1": ensemble["f1"],
            "ensemble_recall": ensemble["recall"],
        }


def make_numpy_client(
    cid: str,
    partition: dict[str, np.ndarray],
    config,
    input_dim: int,
    checkpoint_dir: str | Path,
):
    import flwr as fl

    class _Client(FraudFlowerClient, fl.client.NumPyClient):
        pass

    return _Client(cid, partition, config, input_dim, checkpoint_dir)
