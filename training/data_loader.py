from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.preprocessing import FraudPreprocessor, detect_label_column


def find_csv_dataset(data_dir: str | Path) -> Path:
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV dataset was found in {data_dir}.")
    return candidates[0]


def load_dataset(data_dir: str | Path, label_column: str = "auto", max_rows: int | None = None) -> tuple[pd.DataFrame, str, Path]:
    dataset_path = find_csv_dataset(data_dir)
    frame = pd.read_csv(dataset_path, nrows=max_rows)
    label = detect_label_column(frame.columns, label_column)
    return frame, label, dataset_path


def split_dataset(
    frame: pd.DataFrame,
    label_column: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[label_column].astype(int),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def partition_client_arrays(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    validation_size: float,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        for client_id, chunk in enumerate(np.array_split(label_indices, num_clients)):
            client_indices[client_id].extend(chunk.tolist())

    partitions: list[dict[str, np.ndarray]] = []
    for idx in client_indices:
        idx = np.asarray(idx, dtype=np.int64)
        rng.shuffle(idx)
        X_client, y_client = X[idx], y[idx]
        unique, counts = np.unique(y_client, return_counts=True)
        stratify = y_client if len(unique) == 2 and counts.min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_client,
            y_client,
            test_size=validation_size,
            random_state=seed,
            stratify=stratify,
        )
        partitions.append(
            {
                "x_train": X_train.astype(np.float32),
                "y_train": y_train.astype(np.int64),
                "x_val": X_val.astype(np.float32),
                "y_val": y_val.astype(np.int64),
            }
        )
    return partitions


def prepare_frames(
    config,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_column: str,
    seed: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], FraudPreprocessor, list[str]]:
    preprocessor = FraudPreprocessor(
        label_column=label_column,
        missing_numeric=config.preprocessing.missing_numeric,
        missing_categorical=config.preprocessing.missing_categorical,
        scale_numeric=bool(config.preprocessing.scale_numeric),
    )
    train_prepared = preprocessor.fit_transform(train_frame)
    test_prepared = preprocessor.transform(test_frame)

    clients = partition_client_arrays(
        train_prepared.X,
        train_prepared.y,
        num_clients=int(config.federated.num_clients),
        validation_size=float(config.data.validation_size),
        seed=seed,
    )
    test = {"x": test_prepared.X, "y": test_prepared.y}
    return clients, test, preprocessor, train_prepared.feature_names


def prepare_data(config) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], FraudPreprocessor, list[str], Path]:
    frame, label_column, dataset_path = load_dataset(
        config.data.data_dir,
        config.data.label_column,
        config.data.max_rows,
    )
    train_frame, test_frame = split_dataset(
        frame,
        label_column,
        test_size=float(config.data.test_size),
        seed=int(config.project.seed),
    )

    clients, test, preprocessor, feature_names = prepare_frames(
        config,
        train_frame,
        test_frame,
        label_column,
        seed=int(config.project.seed),
    )
    return clients, test, preprocessor, feature_names, dataset_path
