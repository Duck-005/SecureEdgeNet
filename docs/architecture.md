# Architecture

SecureEdgeNet uses a lightweight single-runtime simulation of a federated fraud detection platform.

## Components

- `training/`: loads CSV datasets, validates labels, fits a reusable preprocessing pipeline, and partitions examples into clients.
- `clients/`: defines the Flower client contract. Each client owns one data partition and trains local model updates.
- `server/`: configures Flower FedAvg and stores the latest aggregated global parameters.
- `models/`: contains wrappers for Logistic Regression, PyTorch MLP, and XGBoost.
- `ensemble/`: combines probability outputs with weighted averaging or majority voting.
- `evaluation/`: computes fraud metrics, threshold tuning, plots, feature importance, and optional SHAP summaries.
- `api/`: loads checkpoints and serves `/predict`.

## Federated Models

Logistic Regression and MLP expose their trainable weights as NumPy arrays. Flower FedAvg aggregates those arrays weighted by the number of local training examples.

XGBoost does not naturally support parameter FedAvg in the same way as neural or linear models. SecureEdgeNet trains XGBoost locally per client and ensembles the resulting client boosters at prediction time. This keeps the raw data on the client partitions while still using XGBoost for strong tabular performance.

## Data Flow

```text
CSV in ./data
  -> schema validation
  -> imputation, scaling, encoding
  -> stratified test split
  -> stratified client partitions
  -> Flower FedAvg rounds for LR and MLP
  -> local XGBoost training
  -> ensemble evaluation
  -> checkpoints and reports
```

## Checkpoints

`checkpoints/` contains all artifacts needed for inference:

- fitted preprocessor
- global Logistic Regression model
- global MLP weights
- client XGBoost models
- ensemble weights and threshold
- model metadata
- metrics and plots
