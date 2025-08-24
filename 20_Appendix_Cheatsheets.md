# Appendix: Cheatsheets & Checklists

### Common Losses
- Regression: MSE, MAE, Huber
- Classification: Cross‑Entropy, Focal Loss (imbalanced)

### Activation Functions
- ReLU, LeakyReLU, GELU, Sigmoid, Tanh

### Optimizers
- SGD(+momentum), Adam, AdamW; typical LR: 1e‑3 (Adam), 1e‑2 (SGD baseline)

### Regularization
- L1/L2, dropout, early stopping, data augmentation

### Metrics by Task
- Binary: ROC‑AUC, PR‑AUC, F1
- Multi‑class: macro/micro F1, accuracy
- Regression: MAE, RMSE, R²

### Train/Val/Test Splits
- 60/20/20 or 70/15/15; time‑ordered split for time series

### Quick Pipeline Template (sklearn)
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), ["age","income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"])
])

pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
```

### Debugging Checklist
- Sanity check shapes & NaNs
- Shuffle where needed; stratify for classification
- Compare against naive baseline
- Inspect errors; adjust thresholds/metrics
- Re‑evaluate for leakage

### Reading List (Optional, if you want more)
- Not required—this course is self‑contained. Use only if you enjoy extra depth.
