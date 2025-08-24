# Unsupervised Learning

**Goal:** Discover structure without labels.

## Clustering
- k‑Means (centroid‑based), choosing k (elbow/silhouette).
- DBSCAN (density‑based) for arbitrary shapes & outliers.
- Hierarchical clustering (linkages).

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=1000, centers=3, random_state=42, cluster_std=1.2)
for k in [2,3,4,5]:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    print(k, silhouette_score(X, km.labels_))
db = DBSCAN(eps=0.6, min_samples=5).fit(X)
print("DBSCAN clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
```

## Exercises
1. Compare k‑Means vs DBSCAN on noisy data.
2. Try hierarchical clustering and plot a dendrogram.
3. Explain when clustering is meaningful for downstream tasks.
