
# Clinical Text Classification (Simple Prototype)

A small, honest project that demonstrates:
- Proficiency in **Python** and **machine learning** programming
- Prior **ML/data science** experience (cleaning, features, metrics)
- A foundation in **linear algebra** via TF‑IDF vectors and logistic regression
- Light **NLP** familiarity (tokenization and bag‑of‑words/TF‑IDF)

The task: classify short, synthetic clinical‑like notes as **diabetes‑risk** (1) vs **not** (0).  
This is a prototype for learning only — no real patient data.

## Quick Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# Train and save model
python train.py

# Evaluate on held‑out test set, save confusion matrix plot
python evaluate.py
```
Outputs include accuracy, precision/recall/F1, and `results/confusion_matrix.png`.

## Files
- `train.py` — trains TF‑IDF + Logistic Regression and saves `models/model.joblib`
- `evaluate.py` — loads the model and test set, prints metrics, saves a plot
- `utils.py` — small helpers for loading data and plotting
- `data/` — small synthetic dataset (`train.csv`, `val.csv`, `test.csv`)
- `requirements.txt` — dependencies

## Notes
- Keeping the model linear and features vector‑space based makes the math transparent.
- This repo is intentionally minimal so it doesn't over‑sell expertise.
