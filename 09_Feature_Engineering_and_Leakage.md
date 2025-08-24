# Feature Engineering & Leakage

**Goal:** Improve signal and avoid hidden pitfalls.

## Techniques
- Encodings: one‑hot, ordinal, target encoding (with CV to avoid leakage).
- Interactions: polynomial/crossed features.
- Date/time feature extraction; text TF‑IDF for classical ML.
- Feature selection: filter (mutual info), wrapper (RFE), embedded (L1).

## Leakage Patterns
- Using future info; post‑event features; target‑derived columns.

## Exercises
1. Implement target encoding with K‑Fold averaging.
2. Detect leakage in a sample dataset description you write yourself.
3. Compare performance with/without engineered features.
