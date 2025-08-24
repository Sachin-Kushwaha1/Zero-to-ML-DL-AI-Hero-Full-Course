# CNNs for Vision

**Goal:** Build image classifiers effectively.

## Convolutions & Pooling
- Local receptive fields; stride/padding; feature maps.
- Pooling reduces spatial size → invariance.

## Practical Classifier (CIFAR‑10‑like shape)
```python
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)
```

## Transfer Learning (quick)
- Load a pretrained backbone; replace head; fine‑tune last layers.

## Exercises
1. Implement data augmentation; compare accuracy.
2. Try transfer learning with a pretrained model; unfreeze gradually.
3. Explain when CNNs beat classical ML for images.
