# Supervised Learning — Regression

**Goal:** Predict continuous values; build baselines and improve.

## Linear Regression
- Hypothesis ŷ = Xw + b; OLS solution; assumptions (multicollinearity etc.).

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

X, y = fetch_california_housing(return_X_y=True)
model = make_pipeline(StandardScaler(), LinearRegression())
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(Xtr,ytr)
pred = model.predict(Xte)
print("MAE:", mean_absolute_error(yte,pred), "R2:", r2_score(yte,pred))
```

## Regularization
- **Ridge (L2)** shrinks weights, handles multicollinearity; **Lasso (L1)** → sparsity.
- ElasticNet = L1 + L2.

## Non‑linear Regression
- kNN Regressor; polynomial features; tree‑based methods (see Chapter 7).

## Exercises
1. Compare LinearRegression vs Ridge vs Lasso on same split.
2. Add PolynomialFeatures and observe bias–variance trade‑off.
3. Explain in one paragraph when you'd prefer MAE over MSE.
