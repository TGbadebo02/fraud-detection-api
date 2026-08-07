"""
Sweep the decision threshold and see how precision/recall trade off.

Why this works without retraining: the model's `predict_proba` already
gives a fraud probability per transaction. `.predict()` just applies a
0.5 cutoff to that probability. We're free to pick a different cutoff.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("models/fraud_model.pkl")

# --- Step 1: recreate the exact same test split used in train_model.py ---
# Same random_state=42 and test_size=0.25 means this is the identical
# held-out set the model was scored on originally. That's important:
# we want an apples-to-apples comparison against the 0.5-threshold metrics.
data = pd.read_csv(DATA_PATH)
X = data.drop(columns=["transaction_id", "is_fraud"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- Step 2: load the already-trained pipeline and get probabilities ---
pipeline = joblib.load(MODEL_PATH)
probabilities = pipeline.predict_proba(X_test)[:, 1]

# --- Step 3: try a range of thresholds instead of the default 0.5 ---
print(f"{'threshold':>9} | {'precision':>9} | {'recall':>7} | {'f1':>6} | {'flagged':>7}")
print("-" * 55)

for threshold in np.arange(0.05, 0.95, 0.05):
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    flagged = predictions.sum()
    print(f"{threshold:9.2f} | {precision:9.3f} | {recall:7.3f} | {f1:6.3f} | {flagged:7d}")
