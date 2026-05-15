from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COMMON_LABEL_COLUMNS = ("Class", "class", "isFraud", "is_fraud", "fraud", "label", "target")


def detect_label_column(columns: Iterable[str], configured: str = "auto") -> str:
    if configured and configured != "auto":
        if configured not in columns:
            raise ValueError(f"Configured label column '{configured}' was not found in the dataset.")
        return configured

    for candidate in COMMON_LABEL_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not infer the fraud label column. Set data.label_column in configs/default.yaml."
    )


@dataclass
class PreparedArrays:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


class FraudPreprocessor:
    """Schema validation and reusable preprocessing for tabular fraud data."""

    def __init__(
        self,
        label_column: str,
        missing_numeric: str = "median",
        missing_categorical: str = "most_frequent",
        scale_numeric: bool = True,
    ) -> None:
        self.label_column = label_column
        self.missing_numeric = missing_numeric
        self.missing_categorical = missing_categorical
        self.scale_numeric = scale_numeric
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def validate(self, frame: pd.DataFrame) -> None:
        if self.label_column not in frame.columns:
            raise ValueError(f"Dataset is missing label column '{self.label_column}'.")
        if frame.empty:
            raise ValueError("Dataset is empty.")
        labels = set(frame[self.label_column].dropna().unique().tolist())
        if not labels.issubset({0, 1, False, True}):
            raise ValueError(
                f"Label column '{self.label_column}' must contain binary fraud labels; got {labels}."
            )

    def fit(self, frame: pd.DataFrame) -> "FraudPreprocessor":
        self.validate(frame)
        features = frame.drop(columns=[self.label_column])
        self.feature_columns = list(features.columns)
        self.numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.categorical_columns = [
            col for col in self.feature_columns if col not in self.numeric_columns
        ]

        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=self.missing_numeric))
        ]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        categorical_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=self.missing_categorical))
        ]
        categorical_steps.append(("encoder", self._make_onehot_encoder()))

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", Pipeline(numeric_steps), self.numeric_columns),
                ("cat", Pipeline(categorical_steps), self.categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer.fit(features)
        self.feature_names_ = self._feature_names()
        return self

    def transform(self, frame: pd.DataFrame) -> PreparedArrays:
        if self.transformer is None:
            raise RuntimeError("FraudPreprocessor must be fitted before transform().")
        self.validate(frame)
        features = frame.drop(columns=[self.label_column])
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Dataset is missing feature columns: {sorted(missing)}")
        transformed = self.transformer.transform(features[self.feature_columns])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        X = np.asarray(transformed, dtype=np.float32)
        y = frame[self.label_column].astype(int).to_numpy(dtype=np.int64)
        return PreparedArrays(X=X, y=y, feature_names=self.feature_names_)

    def fit_transform(self, frame: pd.DataFrame) -> PreparedArrays:
        self.fit(frame)
        return self.transform(frame)

    def transform_records(self, records: list[dict[str, object]]) -> np.ndarray:
        frame = pd.DataFrame.from_records(records)
        frame[self.label_column] = 0
        return self.transform(frame).X

    def _feature_names(self) -> list[str]:
        if self.transformer is None:
            return []
        try:
            return self.transformer.get_feature_names_out().tolist()
        except Exception:
            return self.numeric_columns + self.categorical_columns

    @staticmethod
    def _make_onehot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


def maybe_apply_smote(X: np.ndarray, y: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return X, y
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("SMOTE requires imbalanced-learn. Install requirements.txt.") from exc
    sampler = SMOTE(random_state=42)
    return sampler.fit_resample(X, y)
