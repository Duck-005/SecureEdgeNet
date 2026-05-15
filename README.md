# SecureEdgeNet

SecureEdgeNet is a notebook-friendly Federated Learning fraud detection system built with Flower, PyTorch, scikit-learn, and XGBoost. It simulates multiple financial clients in one Python or Google Colab runtime, trains local fraud models on private partitions, shares only model parameters for federated models, and evaluates a configurable ensemble.

## What It Builds

The training pipeline implements:

- Federated Averaging with Flower across simulated clients.
- Federated Logistic Regression using local SGD parameter updates.
- Federated PyTorch MLP for nonlinear fraud detection.
- Client-local XGBoost models for high-performance tabular scoring and feature importance.
- Weighted or majority-vote ensemble prediction.
- Fraud-focused metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix.
- Threshold tuning for imbalanced fraud detection.
- Checkpoint storage and a lightweight FastAPI inference endpoint.

The project intentionally avoids Docker, Kubernetes, Redis, RabbitMQ, Kafka, and cloud orchestration so it can run in a single Colab runtime.

## Project Structure

```text
SecureEdgeNet/
├── api/                 # FastAPI inference service
├── checkpoints/         # Trained models, metrics, plots, metadata
├── clients/             # Flower client implementation
├── configs/             # YAML training configuration
├── data/                # Local CSV datasets
├── docs/                # Architecture and usage guides
├── ensemble/            # Weighted averaging and majority voting
├── evaluation/          # Metrics, plots, SHAP helpers
├── models/              # Logistic regression, MLP, XGBoost wrappers
├── notebooks/           # Colab notebook workspace
├── server/              # Flower FedAvg strategy and simulation runner
├── training/            # Data loading, preprocessing, partitioning
├── utils/               # Config, checkpoint, logging utilities
├── main.py              # End-to-end training entrypoint
└── requirements.txt
```

## Dataset

Place a CSV fraud dataset in `./data/`. The loader automatically uses the first CSV it finds.

Supported label names include:

- `Class`
- `isFraud`
- `is_fraud`
- `fraud`
- `label`
- `target`

You can also set the label explicitly in `configs/default.yaml`:

```yaml
data:
  label_column: Class
```

The included pipeline supports numeric transaction features, categorical features, missing values, scaling, one-hot encoding, stratified train/test splits, and stratified client partitioning.

## Setup

```bash
python -m pip install -r requirements.txt
```

For Google Colab:

```python
!pip install -r requirements.txt
```

## Train

```bash
python main.py --config configs/default.yaml
```

If you want a pure in-process FedAvg fallback without Flower simulation:

```bash
python main.py --no-flower
```

## K-Fold Cross-Validation

Run stratified k-fold cross-validation with:

```bash
python main.py --cross-validate --no-flower
```

Override the number of folds:

```bash
python main.py --cross-validate --folds 3 --no-flower
```

Cross-validation fits a fresh preprocessor inside every fold, trains federated clients on the fold training split, tunes the ensemble threshold on client validation partitions, and evaluates on that fold's held-out test split. Results are saved to:

```text
checkpoints/cross_validation_metrics.json
checkpoints/folds/fold_*/
checkpoints/reports/folds/fold_*/
```

You can also enable it from `configs/default.yaml`:

```yaml
cross_validation:
  enabled: true
  n_splits: 5
```

Outputs are written to `./checkpoints/`:

- `logistic_regression.joblib`
- `mlp.pth`
- `xgboost_client_*.json`
- `preprocessor.joblib`
- `ensemble_config.json`
- `model_metadata.json`
- `metrics.json`
- ROC, PR, confusion matrix, and feature-importance plots under `checkpoints/reports/`

## Inference API

After training:

```bash
uvicorn api.main:app --reload
```

Request:

```http
POST /predict
Content-Type: application/json
```

```json
{
  "features": {
    "Time": 0,
    "V1": -1.3598,
    "V2": -0.0727,
    "Amount": 149.62
  }
}
```

The payload must include the feature columns expected by the fitted preprocessor.

Response:

```json
{
  "fraud_probability": 0.82,
  "fraud_decision": "fraud",
  "threshold": 0.47,
  "model_confidence": 0.82,
  "contributions": {
    "logistic": 0.19,
    "mlp": 0.21,
    "xgboost": 0.42
  }
}
```

## Federated Workflow

```text
Load CSV dataset
    ↓
Validate schema and preprocess features
    ↓
Partition rows into simulated clients
    ↓
Flower server initializes global LR + MLP parameters
    ↓
Clients train locally on private partitions
    ↓
FedAvg aggregates client parameters weighted by local examples
    ↓
Global LR + MLP are restored
    ↓
Each client trains a local XGBoost model
    ↓
Weighted ensemble evaluates held-out test data
    ↓
Threshold, metrics, plots, and checkpoints are saved
```

## Configuration

Most behavior is controlled by `configs/default.yaml`:

- `federated.num_clients`
- `federated.num_rounds`
- `federated.local_epochs`
- `federated.fraction_fit`
- `models.mlp.hidden_layers`
- `models.xgboost.*`
- `ensemble.weights`
- `ensemble.threshold`
- `ensemble.tune_threshold`
- `cross_validation.enabled`
- `cross_validation.n_splits`

Default ensemble:

```text
final_score = 0.3 * logistic + 0.3 * mlp + 0.4 * xgboost
```

## Privacy Notes

In this simulation, raw transaction rows are partitioned locally and the Flower clients only return Logistic Regression and MLP parameters. XGBoost is trained client-side and saved as separate client model artifacts for ensemble scoring. This is appropriate for educational FL experiments, portfolio work, and Colab demos. For regulated production environments, add secure aggregation, transport security, client authentication, and formal privacy accounting.

## More Docs

- [Architecture](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Inference Guide](docs/inference_guide.md)
