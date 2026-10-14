# Fraud Detection on MongoDB

This project moves the Kaggle credit-card fraud dataset (284,807 transactions, 492 of them fraud) into MongoDB 7. It rebuilds the usual Pandas feature engineering with aggregation pipelines, measures how indexes change query plans, and retrains the Decision Tree and Random Forest models on data pulled back out of Mongo.

## Reproduce

```bash
docker run -d --name mongo-fraud -p 27017:27017 mongo:7
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh        # downloads creditcard.csv if missing, writes everything to results/
```

| Step | File | Output in `results/` |
|---|---|---|
| Baseline: train on the CSV directly | `01_baseline_pandas.py` | `baseline_pandas.json` |
| 2. Load documents + integrity check | `02_load_mongo.py` | `load_report.json` |
| 3, 7. Indexes, `explain()`, timings | `03_explain_benchmark.py` | `explain_before_index.json`, `explain_after_index.json`, `benchmark_summary.json` |
| 4. Aggregation pipelines | `pipelines.py` | `pipelines.json`, `aggregation_outputs.json` |
| 5, 6. Mongo → pandas, retrain, compare | `04_features_and_models.py` | `model_results.json`, `comparison_table.md` |

Full console output from the last run: `results/run_log.txt`.

## 1. Document model

Each transaction is stored as one document in `fraud.transactions`:

```json
{
  "_id": 541,            // row number in the original CSV: a stable key that keeps row order
  "Time": 406.0,         // seconds since the first transaction (double)
  "Amount": 0.0,
  "Class": 1,            // int32, 0 = legit, 1 = fraud
  "V1": -2.31, "...": "...", "V28": -0.14
}
```

- **Types are enforced.** A `$jsonSchema` validator checks types and ranges on write. `Class` must be an int that is 0 or 1, and `Amount` and `Time` must be doubles of at least 0.
- **`_id` is the CSV row number.** The CSV contains 1,081 rows that are exact duplicates, so a key derived from the content would merge them. A row-number key keeps them, so the dataset stays identical to the original.
- **V1–V28 stay as top-level fields.** I considered nesting them as `{pca: {V1: …}}`. I kept them flat to match the original column names, and because nesting buys nothing when every document has every field.

**Does a document model fit this data? Honestly, not really.** The dataset is flat and dense: 31 numeric columns, no missing values, no nested or variable-length structure, and no relationships between records. That is exactly the shape a relational table or a columnar format like Parquet handles best. MongoDB's main strengths here (flexible schema, embedded sub-documents, documents shaped like application objects) go unused. The data also gets bigger: about 410 bytes per document, roughly 117 MB uncompressed in BSON, versus a 150 MB CSV that would compress far better in a columnar store. Every document repeats all 31 field names. The point of this project is to show MongoDB skills (validation, indexing, query planning, aggregation), not to argue that MongoDB is the best tool for this data. A document store would start to make sense for raw fraud data that *is* nested and varies by record, such as device fingerprints, merchant metadata, or a variable-length list of prior events per card.

**Integrity check after loading:** I read all 284,807 documents back and compared them cell by cell with the CSV using `DataFrame.equals`. They are identical (`roundtrip_identical_to_csv: true`).

## 2. Indexes and query plans (steps 3 and 7)

The indexes are `{Amount: 1, Class: 1}` (compound) and `{Time: 1}`. Each workload was explained with `executionStats` before and after the indexes were created. Wall time is the median of 7 warm runs on a local Docker instance.

| Workload | Plan before | Plan after | Docs examined (before → after) | Median ms (before → after) |
|---|---|---|---|---|
| `find` Amount in [1000, 2000) | COLLSCAN | FETCH ← IXSCAN | 284,807 → 2,378 | 139.5 → 6.4 |
| `find` Time in 10th hour | COLLSCAN | FETCH ← IXSCAN | 284,807 → 8,288 | 66.9 → 19.9 |
| **Agg:** `$match` Amount ≥ 500 → `$bucket` fraud rate | COLLSCAN | **PROJECTION_COVERED** ← IXSCAN | 284,807 → **0** | 95.5 → 6.3 |
| **Agg:** `$match` Time < 6h → hourly `$group` + rolling window | COLLSCAN | FETCH ← IXSCAN | 284,807 → 12,340 | 52.8 → 10.4 |
| **Agg:** `$match` Amount > 1000 → `$group` mean/std by Class | COLLSCAN | **PROJECTION_COVERED** ← IXSCAN | 284,807 → **0** | 81.1 → 1.4 |
| Agg: `$bucket` over the full collection (no `$match`) | COLLSCAN | COLLSCAN | 284,807 → 284,807 | 171.7 → 180.6 |

What the plans show:
- **Indexes only help a `$match` or `$sort` at the start of the pipeline.** The optimizer pushes these into the query layer, where they can use an index. The `$group`, `$bucket` and `$setWindowFields` stages themselves never use an index.
- **The compound `{Amount, Class}` index produces covered queries.** The amount-bin pipeline and the class-stats pipeline only need `Amount` and `Class`, so MongoDB answers them from the index alone, with `totalDocsExamined: 0`. That is the biggest speedup in the table (81 → 1.4 ms). It is also why `Class` is in the index even though nothing filters on it.
- **The Time pipeline still has to FETCH documents** because it reads `Amount`, which isn't in the `{Time}` index. Adding `{Time: 1, Amount: 1, Class: 1}` would make it covered as well.
- **The full-collection pipelines see no benefit.** The `$bucket` fraud-rate pipeline over all rows, and the feature-building pipeline, read every document no matter what. That is expected: an index helps when a query is selective, not when it scans everything.

## 3. Aggregation pipelines (step 4)

All pipelines are in `pipelines.py`, and their JSON form is in `results/pipelines.json`.

| Pipeline | Stages | Pandas equivalent |
|---|---|---|
| `amount_bin_fraud_rate` | `$bucket` on Amount (`$sum` Class) → `$addFields` fraud_rate | `df.groupby(pd.cut(df.Amount, bins)).Class.agg(['count','sum','mean'])` |
| `hourly_window_counts` | `$group` on `floor(Time/3600)` → `$setWindowFields` 3-window rolling sum | `df.groupby(df.Time//3600).size().rolling(3).sum()` |
| `class_feature_stats` | (`$match`) → `$group` by Class with `$avg` / `$stdDevSamp` on Amount, V1–V28 | `df.groupby('Class')[cols].agg(['mean','std'])` |
| `per_transaction_features` | `$setWindowFields` with a **range** window on Time → `$addFields` → `$out` | `df.rolling('1h', on='Time')` and similar |

The per-transaction pipeline writes these features to the `fraud.features` collection: `tx_count_prev_1h`, `tx_count_prev_10m`, `amount_mean_prev_1h`, `amount_vs_1h_mean`, `hour_of_day`, `log_amount` and `amount_bin`.

**Every result was cross-checked against Pandas, and all checks pass.** Per-class mean and std for Amount, V1 and V14 match to a relative tolerance of 1e-9. The hourly counts match exactly, and so does the trailing-1h rolling count for all 284,807 rows.

Fraud rate by Amount bin (from `aggregation_outputs.json`):

| Amount bin | n | frauds | fraud rate |
|---|---|---|---|
| [0, 1) | 16,804 | 68 | 0.40% |
| [1, 10) | 80,510 | 181 | 0.22% |
| [10, 50) | 92,390 | 56 | 0.06% |
| [50, 100) | 37,718 | 57 | 0.15% |
| [100, 250) | 34,482 | 56 | 0.16% |
| [250, 500) | 13,411 | 39 | 0.29% |
| [500, 1000) | 6,423 | 26 | 0.40% |
| [1000, 2500) | 2,620 | 9 | 0.34% |
| [2500, 5000) | 394 | 0 | 0% |
| 5000+ | 55 | 0 | 0% |

The fraud rate is U-shaped: highest for tiny amounts under $1 (consistent with card-testing) and for $500–$2,500. Note that this per-bin fraud rate is **not** used as a model feature. It is computed from labels across the whole dataset, so feeding it to the model would leak test labels into training. The model gets the bin index (`amount_bin`) instead.

## 4. Model comparison (steps 5 and 6)

Setup: 80/20 stratified split with `random_state=42`, giving 98 fraud cases in the test set. Models are a default `DecisionTreeClassifier` and a `RandomForestClassifier` with 100 trees. Precision, recall and F1 are for the fraud class.

| Model | Data source | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | Original project (reported) | 0.83 | 0.81 | 0.82 |
| Decision Tree | Pandas / CSV (rebuilt) | 0.7526 | 0.7449 | 0.7487 |
| Decision Tree | **MongoDB, raw columns** | **0.7526** | **0.7449** | **0.7487** |
| Decision Tree | MongoDB + aggregation features | 0.6881 | 0.7653 | 0.7246 |
| Random Forest | Original project (reported) | 0.95 | 0.76 | 0.85 |
| Random Forest | Pandas / CSV (rebuilt) | 0.9412 | 0.8163 | 0.8743 |
| Random Forest | **MongoDB, raw columns** | **0.9412** | **0.8163** | **0.8743** |
| Random Forest | MongoDB + aggregation features | 0.9398 | 0.7959 | 0.8619 |

### Findings

1. **The migration kept the data intact.** Models trained on data pulled from MongoDB give exactly the same results as models trained on the CSV directly: the same metrics and the same confusion matrices (`raw_predictions_identical_to_baseline: true`). Together with the cell-by-cell round-trip check, this shows that CSV → BSON → pandas loses or changes nothing.

2. **The gap from the original reported numbers comes from the training setup, not from MongoDB.** The original training code wasn't available, so I rebuilt the baseline with a standard setup. Because the rebuilt Pandas baseline gives the same results as the MongoDB run, the gap from the original numbers (DT F1 0.82 vs 0.75, RF recall 0.76 vs 0.82) must come from a different split, seed, test size or preprocessing (for example scaling or resampling). It cannot come from the database. The test set is also small: with only 98 fraud cases, one fraud classified differently moves recall by about 1 point, so a different random split alone can shift these metrics by several points. RF F1 (0.87 vs 0.85) is close. The Decision Tree is the least stable model and moves the most.

3. **The time-based features didn't help, and that is a real result.** Adding the rolling-window and hour-of-day features slightly lowered F1 for both models (DT 0.749 → 0.725, RF 0.874 → 0.862). The likely reasons:
   - The data covers only two days, and `Time` is an offset from the first transaction, so time-of-day patterns can't generalize much.
   - Transaction volume across the whole dataset is not a per-card signal. Real velocity features need a card or customer ID, which this anonymized dataset doesn't have.
   - V1–V28 already carry most of the signal.

   The pipelines are still worth having. On data that includes a card ID, the same `$setWindowFields` pattern with `partitionBy: "$card_id"` produces the per-card velocity features that do matter for fraud.

## Environment

MongoDB 7 (Docker image `mongo:7`), Python 3.12, pymongo, pandas and scikit-learn (versions are pinned in `requirements.txt`). Dataset: `creditcard.csv` from the public TensorFlow mirror (284,807 rows, matching the Kaggle original).
