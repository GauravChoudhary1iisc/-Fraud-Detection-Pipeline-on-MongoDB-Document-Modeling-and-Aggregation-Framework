"""Shared config: Mongo connection, column lists, and the model training routine."""
import json
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "creditcard.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "fraud"
TXN = "transactions"
FEATURES = "features"

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time"] + V_COLS + ["Amount"]
SEED = 42

# The user's previously verified numbers (fraud class, from the original project).
ORIGINAL = {
    "DecisionTree": {"precision": 0.83, "recall": 0.81, "f1": 0.82},
    "RandomForest": {"precision": 0.95, "recall": 0.76, "f1": 0.85},
}


def get_db():
    return MongoClient(MONGO_URI)[DB_NAME]


def save_json(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=str))


def train_and_score(X: pd.DataFrame, y: pd.Series) -> dict:
    """80/20 stratified split, default DT and 100-tree RF, metrics for the fraud class."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        out[name] = {
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        print(f"  {name:13s} P={out[name]['precision']:.4f} R={out[name]['recall']:.4f} F1={out[name]['f1']:.4f}")
    return out
