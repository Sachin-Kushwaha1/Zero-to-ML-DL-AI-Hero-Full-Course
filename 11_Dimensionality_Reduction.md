# Dimensionality Reduction

**Goal:** Compress features while preserving structure.

## PCA (principal components)
- Variance maximization; eigen decomposition; whitening; explained variance.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
pca = PCA(n_components=0.95)
Xr = pca.fit_transform(X)
print("Reduced dims:", Xr.shape[1])
```

## Non‑linear Manifolds (intuition)
- t‑SNE/UMAP for visualization; not for downstream modeling without care.

## Exercises
1. Use PCA before a classifier; compare speed and accuracy.
2. Visualize digits with 2D PCA and color by label.
3. Discuss when DR helps vs harms.
