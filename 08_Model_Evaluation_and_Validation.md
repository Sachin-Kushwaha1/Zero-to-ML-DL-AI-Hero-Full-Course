# Model Evaluation & Validation

**Goal:** Measure true generalization and select models reliably.

## Data Splits
- Train/validation/test; cross‑validation (K‑Fold/Stratified).

## Metrics
- Regression: MAE/MSE/RMSE/R².
- Classification: Accuracy, Precision/Recall, F1, ROC‑AUC, PR‑AUC, LogLoss.
- Calibration.

```python
from sklearn.metrics import roc_auc_score, roc_curve
# Fit any classifier with predict_proba to compute ROC‑AUC and plot ROC.
```

## Hyperparameter Tuning
- GridSearchCV, RandomizedSearchCV.
- Nested CV for small datasets.

## Exercises
1. Build a full pipeline + GridSearchCV for a tabular dataset.
2. Plot ROC and PR curves; interpret thresholds.
3. Show why accuracy can be misleading on imbalanced data.
