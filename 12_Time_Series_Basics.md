# Time Series Basics

**Goal:** Forecasting and sequence modeling for classic ML.

## Essentials
- Stationarity; trends/seasonality; train/test splits by time.
- Features: lags, rolling stats, calendar features.
- Baselines: naive, moving average.
- Classical models: ARIMA (intuition), ETS.

```python
import pandas as pd
import numpy as np
# Create a simple synthetic series and compute lag features
s = pd.Series(np.sin(np.linspace(0, 20, 200)) + 0.2*np.random.randn(200))
df = pd.DataFrame({"y": s})
for lag in [1,2,3]:
    df[f"lag{lag}"] = df["y"].shift(lag)
df = df.dropna()
print(df.head())
```

## Exercises
1. Implement a rolling forecast origin evaluation.
2. Compare naive vs lag‑based regression baseline.
3. Write pros/cons of ARIMA vs ML with engineered features.
