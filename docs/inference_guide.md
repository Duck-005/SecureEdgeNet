# Inference Guide

Train first:

```bash
python main.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"Time":0,"V1":-1.3598,"V2":-0.0727,"Amount":149.62}}'
```

The request must include the raw feature columns that were present when the preprocessor was trained. The API returns fraud probability, decision, confidence, threshold, and weighted model contributions.
