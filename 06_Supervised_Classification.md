# Supervised Learning — Classification

**Goal:** Predict categorical labels; understand decision boundaries.

## Logistic Regression
- Sigmoid; log‑odds; cross‑entropy loss; regularization.

## kNN, Naive Bayes, SVM (intuition & usage)
- kNN: simple, lazy; sensitive to scale.
- NB: fast baseline for text.
- SVM: margin maximization; kernels (RBF).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42)
clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
clf.fit(Xtr,ytr)
pred = clf.predict(Xte)
print(classification_report(yte, pred))
```

## Exercises
1. Train LogisticRegression and SVC; compare ROC‑AUC (see Chapter 8 for metrics).
2. Try kNN with different k; visualize accuracy vs k.
3. When would you prefer F1 over accuracy? Explain.
