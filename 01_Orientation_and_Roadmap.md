# Orientation & Roadmap

**Goal:** Understand what ML/DL/AI are, how they relate, and how this course gets you job‑ready.

## What is ML, DL, AI?
- **AI**: Systems that perform tasks requiring human‑like intelligence.
- **Machine Learning (ML)**: Algorithms that learn patterns from data.
- **Deep Learning (DL)**: ML using deep neural networks; excels in vision, language, audio.

## Typical ML Workflow
1. Problem framing → business & data understanding
2. Data collection → labeling → ethics & consent
3. EDA & preprocessing
4. Modeling (baseline → improved models)
5. Evaluation & validation
6. Deployment (serving, monitoring)
7. Iteration (MLOps, feedback loops)

## Supervised vs Unsupervised vs Reinforcement
- **Supervised**: Train with input–output pairs (regression/classification).
- **Unsupervised**: Discover structure without labels (clustering, dimensionality reduction).
- **Reinforcement**: Learn via trial/error with rewards.

## Your First Baseline (Hands‑on)
We'll use scikit‑learn to train a tiny model. Install environment first:

```bash
# Create environment (either conda or venv)
conda create -n mlcourse python=3.10 -y && conda activate mlcourse
# OR using venv
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install essentials
pip install -r requirements.txt
```


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

## Exercises
1. Change the random_state and test_size; observe accuracy.
2. Replace LogisticRegression with KNeighborsClassifier. Compare.
3. Write down in 2–3 lines how you'd frame a churn prediction problem.
