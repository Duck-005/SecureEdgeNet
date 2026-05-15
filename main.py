from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from clients.fraud_client import FraudFlowerClient, split_model_parameters
from ensemble.aggregator import EnsembleScores, WeightedFraudEnsemble
from evaluation.explainability import save_xgboost_shap_summary
from evaluation.metrics import classification_metrics, tune_threshold
from evaluation.plots import (
    save_confusion_matrix,
    save_feature_importance,
    save_pr_curve,
    save_roc_curve,
)
from models.factory import create_logistic, create_mlp
from server.flower_server import run_flower_simulation, run_local_fedavg
from training.data_loader import prepare_data
from utils.checkpoints import save_joblib, save_json
from utils.config import ensure_dirs, load_config
from utils.logging import configure_logging
from utils.random import set_seed

LOGGER = logging.getLogger("secureedgenet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated ensemble fraud detection training")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--no-flower", action="store_true", help="Use the in-process FedAvg fallback")
    return parser.parse_args()


def restore_global_models(config, input_dim: int, parameters: list[np.ndarray]):
    logistic = create_logistic(config, input_dim)
    mlp = create_mlp(config, input_dim)
    mlp_param_count = len(mlp.get_parameters())
    lr_params, mlp_params = split_model_parameters(parameters, mlp_param_count)
    logistic.set_parameters(lr_params)
    mlp.set_parameters(mlp_params)
    return logistic, mlp


def train_client_xgboost_models(config, client_partitions, input_dim: int, checkpoint_dir: Path, feature_names: list[str]):
    xgb_models = []
    importance_rows: list[dict] = []
    for cid, partition in enumerate(client_partitions):
        client = FraudFlowerClient(str(cid), partition, config, input_dim, checkpoint_dir)
        model, rows = client.train_local_xgboost(feature_names)
        xgb_models.append(model)
        for row in rows:
            importance_rows.append({"client": cid, **row})
    return xgb_models, importance_rows


def average_xgb_scores(models, X: np.ndarray) -> np.ndarray | None:
    if not models:
        return None
    scores = [model.predict_proba(X) for model in models]
    return np.mean(np.vstack(scores), axis=0)


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_seed(int(config.project.seed))

    checkpoint_dir = Path(config.paths.checkpoints_dir)
    reports_dir = Path(config.paths.reports_dir)

    client_partitions, test, preprocessor, feature_names, dataset_path = prepare_data(config)
    input_dim = len(feature_names)
    LOGGER.info("Loaded %s with %d features across %d clients", dataset_path, input_dim, len(client_partitions))

    use_flower = bool(config.federated.use_flower) and not args.no_flower
    if use_flower:
        try:
            parameters = run_flower_simulation(config, client_partitions, input_dim, checkpoint_dir)
        except ImportError:
            LOGGER.warning("Flower is not installed; falling back to in-process FedAvg.")
            parameters = run_local_fedavg(config, client_partitions, input_dim, checkpoint_dir)
    else:
        parameters = run_local_fedavg(config, client_partitions, input_dim, checkpoint_dir)

    logistic, mlp = restore_global_models(config, input_dim, parameters)
    xgb_models, importance_rows = train_client_xgboost_models(
        config, client_partitions, input_dim, checkpoint_dir, feature_names
    )

    lr_scores = logistic.predict_proba(test["x"])
    mlp_scores = mlp.predict_proba(test["x"])
    xgb_scores = average_xgb_scores(xgb_models, test["x"])

    ensemble = WeightedFraudEnsemble(
        weights=dict(config.ensemble.weights),
        method=str(config.ensemble.method),
        threshold=float(config.ensemble.threshold),
    )
    final_scores = ensemble.predict_scores(
        EnsembleScores(logistic=lr_scores, mlp=mlp_scores, xgboost=xgb_scores)
    )

    if bool(config.ensemble.tune_threshold):
        threshold, metrics = tune_threshold(
            test["y"], final_scores, metric=str(config.ensemble.threshold_metric)
        )
        ensemble.threshold = threshold
    else:
        metrics = classification_metrics(test["y"], final_scores, ensemble.threshold)

    logistic.save(checkpoint_dir / "logistic_regression.joblib")
    mlp.save(checkpoint_dir / "mlp.pth")
    save_joblib(preprocessor, checkpoint_dir / "preprocessor.joblib")
    save_json(
        {
            "input_dim": input_dim,
            "feature_names": feature_names,
            "label_column": preprocessor.label_column,
            "mlp": {
                "hidden_layers": list(config.models.mlp.hidden_layers),
                "dropout": float(config.models.mlp.dropout),
                "batch_norm": bool(config.models.mlp.batch_norm),
                "learning_rate": float(config.models.mlp.learning_rate),
            },
            "xgboost_models": [f"xgboost_client_{i}.json" for i in range(len(xgb_models))],
        },
        checkpoint_dir / "model_metadata.json",
    )
    save_json(
        {
            "method": ensemble.method,
            "weights": dict(config.ensemble.weights),
            "threshold": ensemble.threshold,
        },
        checkpoint_dir / "ensemble_config.json",
    )
    save_json(metrics, checkpoint_dir / "metrics.json")
    save_json({"feature_importance": importance_rows}, checkpoint_dir / "xgboost_feature_importance.json")

    save_roc_curve(test["y"], final_scores, reports_dir / "roc_curve.png")
    save_pr_curve(test["y"], final_scores, reports_dir / "pr_curve.png")
    save_confusion_matrix(test["y"], final_scores, ensemble.threshold, reports_dir / "confusion_matrix.png")
    if importance_rows:
        averaged_importance = {}
        for row in importance_rows:
            averaged_importance.setdefault(row["feature"], []).append(float(row["importance"]))
        rows = sorted(
            (
                {"feature": feature, "importance": float(np.mean(values))}
                for feature, values in averaged_importance.items()
            ),
            key=lambda row: row["importance"],
            reverse=True,
        )
        save_feature_importance(rows, reports_dir / "xgboost_feature_importance.png")

    if xgb_models:
        save_xgboost_shap_summary(
            xgb_models[0],
            test["x"][: min(1000, len(test["x"]))],
            feature_names,
            checkpoint_dir / "xgboost_shap_summary.json",
        )

    LOGGER.info("Training complete. Metrics: %s", metrics)
    LOGGER.info("Checkpoints written to %s", checkpoint_dir)


if __name__ == "__main__":
    main()
