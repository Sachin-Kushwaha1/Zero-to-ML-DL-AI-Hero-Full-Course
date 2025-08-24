# Data Preprocessing & EDA

**Goal:** Clean, explore, and understand data before modeling.

## Checklist
- Understand columns (types, units), target leakage risks.
- Handle missing values (drop, impute), outliers (cap/transform).
- Encode categoricals (one‑hot/ordinal/target encoding).
- Scale features (Standard/MinMax/Robust).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "age":[25,32,None,40,29],
    "city":["A","B","A","C","B"],
    "income":[30_000, 55_000, 41_000, None, 60_000],
    "bought":[0,1,0,1,1]
})

num = ["age","income"]
cat = ["city"]

pre = ColumnTransformer([
    ("num",  # impute + scale
     make := (SimpleImputer(strategy="median"), StandardScaler()),
     num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
], remainder="drop")

# note: 'make' is a tuple; scikit-learn Pipeline is often preferred—see next chapters.
```

## Data Leakage Watchlist
- Temporal leakage (using future info).
- Target leakage (using variables derived from target).

## Exercises
1. Identify potential leakage in a credit risk dataset scenario.
2. Try different imputations and scaling; compare results on a small model.
3. Plot feature distributions before/after scaling.
