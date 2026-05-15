from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.checkpoints import save_json


def weighted_average(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    keys = set().union(*(metric.keys() for _, metric in metrics))
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [
            num_examples * float(metric[key])
            for num_examples, metric in metrics
            if key in metric and isinstance(metric[key], (int, float))
        ]
        if values:
            aggregated[key] = sum(values) / total
    return aggregated


class SavingFedAvg:
    """Factory for a Flower FedAvg strategy that stores the latest aggregated weights."""

    def __new__(cls, config, initial_parameters: list[np.ndarray], checkpoint_dir: str | Path):
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        class _SavingFedAvg(fl.server.strategy.FedAvg):
            latest_parameters: list[np.ndarray] = initial_parameters

            def aggregate_fit(self, server_round: int, results: Any, failures: Any):
                aggregated = super().aggregate_fit(server_round, results, failures)
                parameters, metrics = aggregated
                if parameters is not None:
                    self.latest_parameters = parameters_to_ndarrays(parameters)
                    save_json(
                        {"round": server_round, "metrics": metrics or {}},
                        Path(checkpoint_dir) / f"round_{server_round}_metrics.json",
                    )
                return aggregated

        min_clients = min(int(config.federated.min_available_clients), int(config.federated.num_clients))
        return _SavingFedAvg(
            fraction_fit=float(config.federated.fraction_fit),
            fraction_evaluate=float(config.federated.fraction_evaluate),
            min_fit_clients=min(int(config.federated.min_fit_clients), int(config.federated.num_clients)),
            min_evaluate_clients=min(int(config.federated.min_evaluate_clients), int(config.federated.num_clients)),
            min_available_clients=min_clients,
            initial_parameters=ndarrays_to_parameters(initial_parameters),
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=lambda rnd: {
                "server_round": rnd,
                "local_epochs": int(config.federated.local_epochs),
                "batch_size": int(config.federated.batch_size),
            },
        )
