from __future__ import annotations

from pathlib import Path

import numpy as np

from clients.fraud_client import FraudFlowerClient, make_numpy_client
from models.factory import create_logistic, create_mlp
from server.strategy import SavingFedAvg


def initial_global_parameters(config, input_dim: int) -> list[np.ndarray]:
    logistic = create_logistic(config, input_dim)
    mlp = create_mlp(config, input_dim)
    return logistic.get_parameters() + mlp.get_parameters()


def run_flower_simulation(
    config,
    client_partitions: list[dict[str, np.ndarray]],
    input_dim: int,
    checkpoint_dir: str | Path,
) -> list[np.ndarray]:
    import flwr as fl

    initial_parameters = initial_global_parameters(config, input_dim)
    strategy = SavingFedAvg(config, initial_parameters, checkpoint_dir)

    def client_fn(context):
        node_config = getattr(context, "node_config", {})
        cid = str(node_config.get("partition-id", context))
        if not cid.isdigit():
            cid = str(len(cid) % len(client_partitions))
        partition_id = int(cid) % len(client_partitions)
        client = make_numpy_client(
            cid=str(partition_id),
            partition=client_partitions[partition_id],
            config=config,
            input_dim=input_dim,
            checkpoint_dir=checkpoint_dir,
        )
        return client.to_client() if hasattr(client, "to_client") else client

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_partitions),
        config=fl.server.ServerConfig(num_rounds=int(config.federated.num_rounds)),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    return strategy.latest_parameters


def run_local_fedavg(
    config,
    client_partitions: list[dict[str, np.ndarray]],
    input_dim: int,
    checkpoint_dir: str | Path,
) -> list[np.ndarray]:
    """Notebook-safe fallback with the same client contract when Flower is unavailable."""
    parameters = initial_global_parameters(config, input_dim)
    clients = [
        FraudFlowerClient(str(i), partition, config, input_dim, checkpoint_dir)
        for i, partition in enumerate(client_partitions)
    ]
    for round_id in range(int(config.federated.num_rounds)):
        weighted: list[tuple[list[np.ndarray], int]] = []
        for client in clients:
            new_params, num_examples, _ = client.fit(parameters, {})
            weighted.append((new_params, num_examples))
        total = sum(num for _, num in weighted)
        parameters = [
            sum(params[layer] * (num / total) for params, num in weighted)
            for layer in range(len(parameters))
        ]
    return parameters
