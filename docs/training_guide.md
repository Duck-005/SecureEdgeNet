# Training Guide

## 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 2. Add Data

Put a CSV file in `data/`. For the Kaggle credit card dataset, the expected label column is `Class`.

## 3. Configure

Edit `configs/default.yaml`.

Useful quick-run settings:

```yaml
data:
  max_rows: 50000

federated:
  num_clients: 3
  num_rounds: 2
  local_epochs: 1
```

For a fuller experiment, increase `num_rounds`, `local_epochs`, and XGBoost estimators.

## 4. Run

```bash
python main.py
```

For environments where Flower simulation dependencies are unavailable:

```bash
python main.py --no-flower
```

## 5. Inspect Results

Open:

- `checkpoints/metrics.json`
- `checkpoints/reports/roc_curve.png`
- `checkpoints/reports/pr_curve.png`
- `checkpoints/reports/confusion_matrix.png`
- `checkpoints/reports/xgboost_feature_importance.png`

The main fraud objective is usually high recall because false negatives are expensive.
