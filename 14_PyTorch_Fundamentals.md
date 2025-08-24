# PyTorch Fundamentals

**Goal:** Learn tensors, autograd, Dataset/DataLoader, training loop.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(500, 4)
true_w = torch.tensor([[1.0], [-2.0], [0.5], [0.0]])
y = X @ true_w + 0.1*torch.randn(500,1)

ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(4,16), nn.ReLU(), nn.Linear(16,1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(20):
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
print("Example loss:", loss.item())
```

## Saving/Loading, Inference
```python
torch.save(model.state_dict(), "regressor.pt")
m2 = nn.Sequential(nn.Linear(4,16), nn.ReLU(), nn.Linear(16,1))
m2.load_state_dict(torch.load("regressor.pt", map_location="cpu"))
m2.eval()
```

## Exercises
1. Implement a classification MLP; report accuracy.
2. Add early stopping on validation loss.
3. Explain what `model.train()` vs `model.eval()` do.
