from __future__ import annotations

from models.logistic import FederatedLogisticRegression
from models.mlp import FederatedMLP


def create_logistic(config, input_dim: int) -> FederatedLogisticRegression:
    return FederatedLogisticRegression(
        input_dim=input_dim,
        alpha=float(config.models.logistic.alpha),
        seed=int(config.project.seed),
    )


def create_mlp(config, input_dim: int) -> FederatedMLP:
    return FederatedMLP(
        input_dim=input_dim,
        hidden_layers=list(config.models.mlp.hidden_layers),
        dropout=float(config.models.mlp.dropout),
        batch_norm=bool(config.models.mlp.batch_norm),
        learning_rate=float(config.models.mlp.learning_rate),
        seed=int(config.project.seed),
    )
