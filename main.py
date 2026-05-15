from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

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
from training.data_loader import load_dataset, prepare_data, prepare_frames
from utils.checkpoints import save_joblib, save_json
from utils.config import ensure_dirs, load_config
from utils.logging import configure_logging
from utils.random import set_seed

LOGGER = logging.getLogger("secureedgenet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated ensemble fraud detection training")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--no-flower", action="store_true", help="Use the in-process FedAvg fallback")
    parser.add_argument("--cross-validate", action="store_true", help="Run stratified k-fold cross-validation")
    parser.add_argument("--folds", type=int, default=None, help="Override cross_validation.n_splits")
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


def validation_arrays(client_partitions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "x": np.concatenate([partition["x_val"] for partition in client_partitions], axis=0),
        "y": np.concatenate([partition["y_val"] for partition in client_partitions], axis=0),
    }


def ensemble_scores(logistic, mlp, xgb_models, X: np.ndarray) -> EnsembleScores:
    return EnsembleScores(
        logistic=logistic.predict_proba(X),
        mlp=mlp.predict_proba(X),
        xgboost=average_xgb_scores(xgb_models, X),
    )


def averaged_feature_importance(importance_rows: list[dict]) -> list[dict]:
    averaged: dict[str, list[float]] = {}
    for row in importance_rows:
        averaged.setdefault(row["feature"], []).append(float(row["importance"]))
    return sorted(
        (
            {"feature": feature, "importance": float(np.mean(values))}
            for feature, values in averaged.items()
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )


def save_training_artifacts(
    config,
    checkpoint_dir: Path,
    reports_dir: Path,
    preprocessor,
    feature_names: list[str],
    logistic,
    mlp,
    xgb_models,
    importance_rows: list[dict],
    ensemble: WeightedFraudEnsemble,
    metrics: dict,
    test: dict[str, np.ndarray],
    final_scores: np.ndarray,
) -> None:
    logistic.save(checkpoint_dir / "logistic_regression.joblib")
    mlp.save(checkpoint_dir / "mlp.pth")
    save_joblib(preprocessor, checkpoint_dir / "preprocessor.joblib")
    save_json(
        {
            "input_dim": len(feature_names),
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
        save_feature_importance(
            averaged_feature_importance(importance_rows),
            reports_dir / "xgboost_feature_importance.png",
        )

    if xgb_models:
        save_xgboost_shap_summary(
            xgb_models[0],
            test["x"][: min(1000, len(test["x"]))],
            feature_names,
            checkpoint_dir / "xgboost_shap_summary.json",
        )


def train_and_evaluate_split(
    config,
    client_partitions: list[dict[str, np.ndarray]],
    test: dict[str, np.ndarray],
    preprocessor,
    feature_names: list[str],
    checkpoint_dir: Path,
    reports_dir: Path,
    use_flower: bool,
    save_artifacts: bool = True,
) -> dict:
    input_dim = len(feature_names)

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

    ensemble = WeightedFraudEnsemble(
        weights=dict(config.ensemble.weights),
        method=str(config.ensemble.method),
        threshold=float(config.ensemble.threshold),
    )

    if bool(config.ensemble.tune_threshold):
        validation = validation_arrays(client_partitions)
        validation_scores = ensemble.predict_scores(
            ensemble_scores(logistic, mlp, xgb_models, validation["x"])
        )
        threshold, metrics = tune_threshold(
            validation["y"], validation_scores, metric=str(config.ensemble.threshold_metric)
        )
        ensemble.threshold = threshold

    final_scores = ensemble.predict_scores(
        ensemble_scores(logistic, mlp, xgb_models, test["x"])
    )
    metrics = classification_metrics(test["y"], final_scores, ensemble.threshold)
    metrics["threshold_source"] = "client_validation" if bool(config.ensemble.tune_threshold) else "config"

    if save_artifacts:
        save_training_artifacts(
            config,
            checkpoint_dir,
            reports_dir,
            preprocessor,
            feature_names,
            logistic,
            mlp,
            xgb_models,
            importance_rows,
            ensemble,
            metrics,
            test,
            final_scores,
        )

    return metrics


def summarize_cv_metrics(fold_metrics: list[dict]) -> dict:
    ignored = {"fold", "confusion_matrix", "threshold_source"}
    numeric_keys = sorted(
        {
            key
            for metrics in fold_metrics
            for key, value in metrics.items()
            if key not in ignored and isinstance(value, (int, float))
        }
    )
    return {
        key: {
            "mean": float(np.mean([metrics[key] for metrics in fold_metrics])),
            "std": float(np.std([metrics[key] for metrics in fold_metrics])),
        }
        for key in numeric_keys
    }


def run_cross_validation(config, args: argparse.Namespace, checkpoint_dir: Path, reports_dir: Path, use_flower: bool) -> None:
    frame, label_column, dataset_path = load_dataset(
        config.data.data_dir,
        config.data.label_column,
        config.data.max_rows,
    )
    n_splits = int(args.folds or config.cross_validation.n_splits)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(config.cross_validation.shuffle),
        random_state=int(config.project.seed),
    )
    fold_metrics: list[dict] = []
    labels = frame[label_column].astype(int)
    LOGGER.info("Running %d-fold cross-validation on %s", n_splits, dataset_path)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame, labels), start=1):
        LOGGER.info("Starting fold %d/%d", fold_id, n_splits)
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        client_partitions, test, preprocessor, feature_names = prepare_frames(
            config,
            train_frame,
            test_frame,
            label_column,
            seed=int(config.project.seed) + fold_id,
        )
        fold_checkpoint_dir = checkpoint_dir / "folds" / f"fold_{fold_id}"
        fold_reports_dir = reports_dir / "folds" / f"fold_{fold_id}"
        metrics = train_and_evaluate_split(
            config,
            client_partitions,
            test,
            preprocessor,
            feature_names,
            fold_checkpoint_dir,
            fold_reports_dir,
            use_flower=use_flower,
            save_artifacts=bool(config.cross_validation.save_fold_artifacts),
        )
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)
        LOGGER.info("Fold %d metrics: %s", fold_id, metrics)

    summary = summarize_cv_metrics(fold_metrics)
    save_json({"folds": fold_metrics, "summary": summary}, checkpoint_dir / "cross_validation_metrics.json")
    LOGGER.info("Cross-validation summary: %s", summary)


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    ensure_dirs(config)
    set_seed(int(config.project.seed))

    checkpoint_dir = Path(config.paths.checkpoints_dir)
    reports_dir = Path(config.paths.reports_dir)
    use_flower = bool(config.federated.use_flower) and not args.no_flower

    if args.cross_validate or bool(config.cross_validation.enabled):
        run_cross_validation(config, args, checkpoint_dir, reports_dir, use_flower)
        LOGGER.info("Cross-validation metrics written to %s", checkpoint_dir / "cross_validation_metrics.json")
        return

    client_partitions, test, preprocessor, feature_names, dataset_path = prepare_data(config)
    LOGGER.info("Loaded %s with %d features across %d clients", dataset_path, len(feature_names), len(client_partitions))
    metrics = train_and_evaluate_split(
        config,
        client_partitions,
        test,
        preprocessor,
        feature_names,
        checkpoint_dir,
        reports_dir,
        use_flower=use_flower,
        save_artifacts=True,
    )
    LOGGER.info("Training complete. Metrics: %s", metrics)
    LOGGER.info("Checkpoints written to %s", checkpoint_dir)


if __name__ == "__main__":
    main()
