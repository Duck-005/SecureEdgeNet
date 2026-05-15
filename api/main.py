from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ensemble.aggregator import EnsembleScores, WeightedFraudEnsemble
from models.logistic import FederatedLogisticRegression
from models.mlp import FederatedMLP
from models.xgboost_model import LocalXGBoostModel
from utils.checkpoints import load_joblib, load_json

CHECKPOINT_DIR = Path("checkpoints")


class PredictRequest(BaseModel):
    features: dict[str, Any] | None = Field(default=None)

    class Config:
        extra = "allow"


class ModelBundle:
    def __init__(self, checkpoint_dir: Path) -> None:
        metadata = load_json(checkpoint_dir / "model_metadata.json")
        ensemble_config = load_json(checkpoint_dir / "ensemble_config.json")
        self.preprocessor = load_joblib(checkpoint_dir / "preprocessor.joblib")
        self.logistic = FederatedLogisticRegression(input_dim=int(metadata["input_dim"]))
        self.logistic.load(checkpoint_dir / "logistic_regression.joblib")
        mlp_cfg = metadata["mlp"]
        self.mlp = FederatedMLP(
            input_dim=int(metadata["input_dim"]),
            hidden_layers=list(mlp_cfg["hidden_layers"]),
            dropout=float(mlp_cfg["dropout"]),
            batch_norm=bool(mlp_cfg["batch_norm"]),
            learning_rate=float(mlp_cfg["learning_rate"]),
        )
        self.mlp.load(checkpoint_dir / "mlp.pth")
        self.xgb_models = []
        for filename in metadata.get("xgboost_models", []):
            path = checkpoint_dir / filename
            if path.exists():
                model = LocalXGBoostModel({})
                model.load(path)
                self.xgb_models.append(model)
        self.ensemble = WeightedFraudEnsemble(
            weights=dict(ensemble_config["weights"]),
            method=str(ensemble_config["method"]),
            threshold=float(ensemble_config["threshold"]),
        )

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        X = self.preprocessor.transform_records([features])
        lr_score = self.logistic.predict_proba(X)
        mlp_score = self.mlp.predict_proba(X)
        xgb_score = None
        if self.xgb_models:
            xgb_score = np.mean(np.vstack([model.predict_proba(X) for model in self.xgb_models]), axis=0)
        scores = EnsembleScores(logistic=lr_score, mlp=mlp_score, xgboost=xgb_score)
        fraud_probability = float(self.ensemble.predict_scores(scores)[0])
        decision = "fraud" if fraud_probability >= self.ensemble.threshold else "legitimate"
        return {
            "fraud_probability": fraud_probability,
            "fraud_decision": decision,
            "threshold": self.ensemble.threshold,
            "model_confidence": float(max(fraud_probability, 1.0 - fraud_probability)),
            "contributions": self.ensemble.contribution(scores, 0),
        }


app = FastAPI(title="SecureEdgeNet Fraud Inference API", version="1.0.0")
bundle: ModelBundle | None = None


@app.on_event("startup")
def load_models() -> None:
    global bundle
    if CHECKPOINT_DIR.exists() and (CHECKPOINT_DIR / "model_metadata.json").exists():
        bundle = ModelBundle(CHECKPOINT_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "models": "loaded" if bundle is not None else "missing"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded. Run training first.")
    payload = request.features or request.dict(exclude={"features"}, exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="Provide transaction features.")
    return bundle.predict_one(payload)
