# MLOps & Deployment

**Goal:** Put models in production reliably.

## Pipelines
- Deterministic preprocessing + model; `sklearn.pipeline.Pipeline`.
- Versioning data & models; experiment tracking (IDs, metrics, params).

## Serving a Model with FastAPI
```python
# save as serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Item(BaseModel):
    features: list

@app.post("/predict")
def predict(item: Item):
    import numpy as np
    X = np.array(item.features).reshape(1, -1)
    y = model.predict(X).tolist()
    return {"prediction": y}
```

Run:
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

## Docker (conceptual)
- Dockerfile → build image → run container; environment parity.

## Monitoring
- Data drift, concept drift; shadow deployments; A/B tests.

## Exercises
1. Wrap a trained scikit‑learn model and query via `curl`.
2. Sketch a CI/CD pipeline for retraining weekly.
3. Define KPIs and alerts for a fraud model.
