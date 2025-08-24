# Python + Numpy + Pandas + Matplotlib

**Goal:** Get productive with the core data stack.

## Numpy
- Arrays, shapes, dtype, broadcasting, vectorization.

```python
import numpy as np
a = np.arange(12).reshape(3,4)
print(a.mean(axis=0))
print((a - a.mean())/a.std())
```

## Pandas
- Series & DataFrame; indexing; joins; groupby; missing values.

```python
import pandas as pd
df = pd.DataFrame({
    "city": ["A","B","A","C"],
    "sales": [10, 15, 7, None]
})
print(df.isna().sum())
df["sales"] = df["sales"].fillna(df["sales"].median())
print(df.groupby("city")["sales"].mean())
```

## Matplotlib (quick intro)
- Line/Scatter/Bar; labels; legends; saving plots.

```python
import matplotlib.pyplot as plt
xs = [1,2,3]; ys = [1,4,9]
plt.plot(xs, ys)
plt.xlabel("x"); plt.ylabel("y")
plt.title("y = x^2")
plt.show()
```

## Exercises
1. Load a CSV (any public dataset); compute summary stats and a bar plot.
2. Perform a left join between two small DataFrames you design.
3. Vectorize a loop using numpy broadcasting.
