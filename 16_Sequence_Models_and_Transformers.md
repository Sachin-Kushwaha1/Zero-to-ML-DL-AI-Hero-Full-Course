# Sequence Models & Transformers

**Goal:** Understand RNNs/LSTMs/Attention and the Transformer intuition.

## Recurrent Nets
- RNN → vanishing gradients → LSTM/GRU mitigate.

## Attention
- Weighted sum over sequence; query/key/value idea.

## Transformer (encoder–decoder intuition)
- Stacked self‑attention + FFNs; positional encodings; parallelism.

```python
import torch
from torch import nn

# Minimal toy Transformer encoder block
class TinyEncoder(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        h,_ = self.attn(x,x,x)
        x = self.ln1(x + h)
        h = self.ff(x)
        return self.ln2(x + h)
```

## Exercises
1. Build a character‑level next‑token predictor on a tiny corpus.
2. Compare LSTM vs tiny Transformer on the same toy task.
3. Explain why attention helps with long‑range dependencies.
