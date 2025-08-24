# Deep Learning Foundations (Perceptron, MLP, Backprop from Scratch)

**Goal:** Build intuition and a tiny neural net from scratch (NumPy).

## Perceptron & MLP
- Neurons, activations (ReLU/Sigmoid/Tanh), layers, parameters.
- Loss & optimization; initialization; vanishing/exploding gradients.

```python
import numpy as np

def relu(x): return np.maximum(0, x)

# Tiny 2-layer MLP for regression
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
w1 = rng.normal(scale=0.1, size=(3, 8))
b1 = np.zeros(8)
w2 = rng.normal(scale=0.1, size=(8, 1))
b2 = np.zeros(1)
y = (X @ np.array([1.0, -2.0, 0.5]) + 0.1*rng.normal(size=200)).reshape(-1,1)

lr = 0.05
for step in range(1000):
    h = relu(X @ w1 + b1)
    yhat = h @ w2 + b2
    loss = ((yhat - y)**2).mean()

    # Backprop
    d_yhat = 2*(yhat - y)/len(X)
    d_w2 = h.T @ d_yhat
    d_b2 = d_yhat.sum(axis=0)
    d_h = d_yhat @ w2.T
    d_h[h<=0] = 0
    d_w1 = X.T @ d_h
    d_b1 = d_h.sum(axis=0)

    w1 -= lr*d_w1; b1 -= lr*d_b1
    w2 -= lr*d_w2; b2 -= lr*d_b2

print("Final loss:", float(loss))
```

## Exercises
1. Swap ReLU with Tanh; observe training.
2. Add L2 regularization to the above and report loss.
3. Classify XOR with a 2‑layer MLP.
