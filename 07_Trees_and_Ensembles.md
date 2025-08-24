# Trees & Ensembles (Decision Trees, Random Forests, Gradient Boosting)

**Goal:** Learn powerful non‑linear models and ensembles.

## Decision Trees
- Splitting by impurity (Gini/Entropy); depth/pruning; interpretability.

## Random Forests
- Bagging; feature randomness; robust baselines.

## Gradient Boosting (GBM concept)
- Sequential learners fixing residuals; learning rate; trees depth.
- scikit‑learn `GradientBoosting*`, `HistGradientBoosting*` as fast baselines.

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42)
gb = HistGradientBoostingClassifier(random_state=42)
for name, model in [("RF", rf), ("HGB", gb)]:
    model.fit(Xtr, ytr)
    print(name, accuracy_score(yte, model.predict(Xte)))
```

## Exercises
1. Compare feature importances between RF and GBM; discuss differences.
2. Tune `max_depth`/`n_estimators`; plot validation curves.
3. Explain overfitting symptoms for trees & ensembles.
