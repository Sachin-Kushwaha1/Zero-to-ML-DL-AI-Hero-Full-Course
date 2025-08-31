# Zero-to-ML-DL-AI-Hero-Full-Course

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/Sachin-Kushwaha1/Zero-to-ML-DL-AI-Hero-Full-Course)

A structured, end-to-end curriculum to go from zero to productive in Machine Learning, Deep Learning, and AI. The course is organized into concise modules, each with clear objectives and references.

## Who is this for?
- Beginners looking for a guided path into ML/DL/AI.
- Practitioners seeking a refresher or a structured roadmap.

## Prerequisites
- Basic Python familiarity is helpful (covered early in the course).
- Ability to install Python and common packages.

## Setup

Option A — pip:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional for contributors:
pip install -r requirements-dev.txt
pre-commit install
```

Option B — conda (optional):
```bash
conda create -n z2ai python=3.11 -y
conda activate z2ai
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## How to use this repository
- Start at 01 and proceed sequentially, or jump to any topic you need.
- Each module aims to include objectives, key terms, references, and (where applicable) exercises.
- Hands-on notebooks and datasets are optional add-ons; larger datasets are linked instead of stored in the repo.

## Table of Contents

1. [Orientation and Roadmap](./01_Orientation_and_Roadmap.md)
2. [Math Foundations](./02_Math_Foundations.md)
3. [Python, NumPy, Pandas, Matplotlib](./03_Python_Numpy_Pandas_Matplotlib.md)
4. [Data Preprocessing and EDA](./04_Data_Preprocessing_and_EDA.md)
5. [Supervised Learning — Regression](./05_Supervised_Regression.md)
6. [Supervised Learning — Classification](./06_Supervised_Classification.md)
7. [Trees and Ensembles](./07_Trees_and_Ensembles.md)
8. [Model Evaluation and Validation](./08_Model_Evaluation_and_Validation.md)
9. [Feature Engineering and Leakage](./09_Feature_Engineering_and_Leakage.md)
10. [Unsupervised Learning](./10_Unsupervised_Learning.md)
11. [Dimensionality Reduction](./11_Dimensionality_Reduction.md)
12. [Time Series Basics](./12_Time_Series_Basics.md)
13. [Deep Learning Foundations](./13_Deep_Learning_Foundations.md)
14. [PyTorch Fundamentals](./14_PyTorch_Fundamentals.md)
15. [CNNs for Vision](./15_CNNs_for_Vision.md)
16. [Sequence Models and Transformers](./16_Sequence_Models_and_Transformers.md)
17. [MLOps and Deployment](./17_MLOps_and_Deployment.md)
18. [Responsible AI](./18_Responsible_AI.md)
19. [Capstone Projects and Rubrics](./19_Capstone_Projects_and_Rubrics.md)
20. [Appendix and Cheatsheets](./20_Appendix_Cheatsheets.md)

## Contributing

We welcome improvements—typos, better examples, exercises, or notebooks.
- See [CONTRIBUTING.md](./CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
- Use our issue templates for typos, broken links, content suggestions, and new exercises.
- PRs run automatic link checks.

## Roadmap (next steps)
- Add optional Jupyter notebooks for hands-on practice.
- Add small sample datasets or durable links.
- Consider publishing as a docs site (e.g., MkDocs + GitHub Pages).

---
Licensed under the [MIT License](./LICENSE).
