"""Steps 4-6: aggregation-based feature engineering, Mongo -> pandas, retrain, compare."""
import json

import numpy as np
import pandas as pd

from common import (FEATURES, ORIGINAL, RAW_FEATURES, RESULTS, TXN, get_db, save_json,
                    train_and_score)
from pipelines import (amount_bin_fraud_rate, class_feature_stats, hourly_window_counts,
                       per_transaction_features)

db = get_db()
coll = db[TXN]

# --- Step 4: analytical pipelines (these replace Pandas groupby code) -------------------
analytics = {
    "amount_bin_fraud_rate": amount_bin_fraud_rate(),
    "hourly_window_counts": hourly_window_counts(),
    "class_feature_stats": class_feature_stats(),
}
outputs = {name: list(coll.aggregate(p, allowDiskUse=True)) for name, p in analytics.items()}
save_json("pipelines.json", {**analytics, "per_transaction_features": per_transaction_features(FEATURES)})
save_json("aggregation_outputs.json", outputs)

print("Fraud rate by Amount bin:")
print(pd.DataFrame(outputs["amount_bin_fraud_rate"]).to_string(index=False))

# Cross-check the Mongo aggregations against the same computation in Pandas.
csv = pd.read_csv(RESULTS.parent / "data" / "creditcard.csv")
pd_stats = csv.groupby("Class")[["Amount", "V1", "V14"]].agg(["mean", "std"])
mongo_stats = pd.DataFrame(outputs["class_feature_stats"]).set_index("_id")
checks = {}
for col in ["Amount", "V1", "V14"]:
    for stat in ["mean", "std"]:
        checks[f"{col}_{stat}"] = bool(np.allclose(
            pd_stats[(col, stat)].values, mongo_stats[f"{col}_{stat}"].values, rtol=1e-9))
hourly_pd = csv.groupby(csv.Time // 3600).size().values
hourly_mongo = pd.DataFrame(outputs["hourly_window_counts"]).sort_values("hour_index")["n"].values
checks["hourly_counts"] = bool((hourly_pd == hourly_mongo).all())

# --- Per-transaction feature table, written to its own collection ---------------------
coll.aggregate(per_transaction_features(FEATURES), allowDiskUse=True)
ENGINEERED = ["tx_count_prev_1h", "tx_count_prev_10m", "amount_mean_prev_1h",
              "hour_of_day", "log_amount", "amount_vs_1h_mean", "amount_bin"]

# --- Step 5: pull back into pandas -----------------------------------------------------
proj = {c: 1 for c in RAW_FEATURES + ENGINEERED + ["Class"]}
feat = pd.DataFrame(db[FEATURES].find({}, proj, sort=[("_id", 1)])).set_index("_id")
feat.index.name = None

# Cross-check the rolling 1h count against a Pandas implementation.
t = csv["Time"].values
order = np.sort(t)
pd_roll = np.searchsorted(order, t, side="right") - np.searchsorted(order, t - 3600, side="left")
checks["rolling_1h_count"] = bool((pd_roll == feat["tx_count_prev_1h"].values).all())
print("Mongo vs Pandas cross-checks:", checks)

# --- Step 6: retrain -------------------------------------------------------------------
print("Mongo-sourced, raw columns (same features as baseline):")
mongo_raw = train_and_score(feat[RAW_FEATURES], feat["Class"])
print("Mongo-sourced, raw + aggregation-engineered features:")
mongo_eng = train_and_score(feat[RAW_FEATURES + ENGINEERED], feat["Class"])

baseline = json.loads((RESULTS / "baseline_pandas.json").read_text())
rows = []
for model in ["DecisionTree", "RandomForest"]:
    for source, res in [("Original (user-reported)", ORIGINAL), ("Pandas/CSV rebuild", baseline),
                        ("Mongo raw columns", mongo_raw), ("Mongo + engineered", mongo_eng)]:
        r = res[model]
        rows.append({"model": model, "source": source,
                     "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})
table = pd.DataFrame(rows)
print(table.to_string(index=False))

save_json("model_results.json", {
    "cross_checks": checks,
    "engineered_features": ENGINEERED,
    "mongo_raw": mongo_raw,
    "mongo_engineered": mongo_eng,
    "pandas_baseline": baseline,
    "original": ORIGINAL,
    "raw_predictions_identical_to_baseline": mongo_raw == baseline,
})
(RESULTS / "comparison_table.md").write_text(table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string(index=False))
