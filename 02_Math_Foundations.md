# Math Foundations (Linear Algebra, Calculus, Probability & Stats)

**Goal:** Build the exact math you need for ML—not more, not less.

## Linear Algebra (just enough)
- Vectors, matrices, tensors; shapes and broadcasting.
- Matrix multiplication and geometric view (projections).
- Norms: L1, L2; why regularization uses them.
- Eigenvalues/eigenvectors; SVD (intuition) → PCA.

**Practice (NumPy):**
```python
import numpy as np
A = np.array([[1,2],[3,4]])
v = np.array([1,0])
print("Av =", A @ v)
print("Eigenvalues:", np.linalg.eigvals(A))
U, S, Vt = np.linalg.svd(A)
print("SVD singular values:", S)
```

## Calculus (optimization‑oriented)
- Derivatives, gradients, gradient descent.
- Chain rule → backprop.
- Loss landscapes; local minima, saddle points.
- Learning rate intuition.

**Practice:**
```python
import numpy as np
# Minimize f(w)= (w-3)^2
w = 0.0
lr = 0.1
for step in range(30):
    grad = 2*(w-3)
    w -= lr*grad
print("w≈", w)
```

## Probability & Statistics
- Random variables, distributions (Gaussian, Bernoulli).
- Expectation, variance; Law of Large Numbers.
- Bayes' rule (intuition) → Naive Bayes.
- Sampling, confidence intervals, p‑values (practical view).

**Practice:**
```python
import numpy as np
samples = np.random.normal(loc=0, scale=1, size=10000)
print(samples.mean(), samples.std())
```

## Exercises
1. Derive gradient for MSE: L = (1/n) Σ (y - ŷ)^2 for linear regression.
2. Show L1 vs L2 penalty difference in a simple line fit (code).
3. Compute PCA on a 2D dataset and project to 1D; visualize with matplotlib.
