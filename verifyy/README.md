# Verification Package: Fraud Detection on MongoDB

This folder holds copies of the real code and outputs from the project, organized by the five items requested. [EVIDENCE.md](EVIDENCE.md) puts all of them into one readable document.

## 1. Data loading: CSV to documents, schema and fields → [1_data_loading/](1_data_loading/)
- `02_load_mongo.py` is the loader. It stores one document per transaction with fields `_id` (CSV row number), `Time`, `Amount`, `V1`–`V28` and `Class`, and a `$jsonSchema` validator enforces the types.
- `load_report.json` holds the load results: 284,807 documents, 492 frauds, `roundtrip_identical_to_csv: true`, the validator, and a sample stored document.

## 2. Aggregation pipeline code → [2_aggregation_pipelines/](2_aggregation_pipelines/)
- `pipelines.py` has the actual `$match` / `$bucket` / `$group` / `$setWindowFields` / `$out` stages in Python.
- `pipelines.json` has the same pipelines as raw MongoDB JSON, which can be pasted into `mongosh`.
- `aggregation_outputs.json` holds what each pipeline returned (fraud rate per Amount bin, hourly and rolling counts, per-class mean/std).

## 3. Indexes and explain("executionStats"), with and without the index → [3_index_explain/](3_index_explain/)
- `explain_before_index.json` and `explain_after_index.json` are the full, unedited explain output for 6 workloads.
- `benchmark_summary.json` summarizes them: plan, docs examined, keys examined, explain ms and median wall ms.
- `03_explain_benchmark.py` is the code that produced these files.

Indexes: `{Amount: 1, Class: 1}` named `amount_class`, and `{Time: 1}` named `time`.

| Aggregation | Plan: no index → index | Docs examined | Keys examined | Execution ms |
|---|---|---|---|---|
| `$match` Amount ≥ 500 → `$bucket` fraud rate | COLLSCAN → IXSCAN (covered) | 284,807 → **0** | 0 → 9,492 | 98 → 6 |
| `$match` Time < 6h → `$group` hourly + rolling | COLLSCAN → IXSCAN + FETCH | 284,807 → 12,340 | 0 → 12,340 | 51 → 11 |
| `$match` Amount > 1000 → `$group` mean/std by Class | COLLSCAN → IXSCAN (covered) | 284,807 → **0** | 0 → 2,940 | 92 → 1 |
| `$bucket` over the full collection (no `$match`) | COLLSCAN → COLLSCAN | 284,807 → 284,807 | 0 → 0 | 174 → 182 |

"Covered" means MongoDB answered the query from the index alone, so it read no documents. The last row is included on purpose: an index cannot help a pipeline that reads every document.

## 4. Model comparison → [4_model_comparison/](4_model_comparison/)
- `comparison_table.md` is the table below. `model_results.json` adds confusion matrices and cross-checks.
- The code is in `04_features_and_models.py` (Mongo → pandas → retrain), `01_baseline_pandas.py` (training on the CSV directly) and `common.py` (split and model settings).

Precision, recall and F1 are for the fraud class. The test set has 56,962 transactions, 98 of them fraud.

| Model | Source | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | Original project (reported) | 0.83 | 0.81 | 0.82 |
| Decision Tree | Pandas / CSV (rebuilt) | 0.7526 | 0.7449 | 0.7487 |
| Decision Tree | **MongoDB, raw columns** | **0.7526** | **0.7449** | **0.7487** |
| Decision Tree | MongoDB + aggregation features | 0.6881 | 0.7653 | 0.7246 |
| Random Forest | Original project (reported) | 0.95 | 0.76 | 0.85 |
| Random Forest | Pandas / CSV (rebuilt) | 0.9412 | 0.8163 | 0.8743 |
| Random Forest | **MongoDB, raw columns** | **0.9412** | **0.8163** | **0.8743** |
| Random Forest | MongoDB + aggregation features | 0.9398 | 0.7959 | 0.8619 |

**Note on the original numbers.** The training code behind the original numbers was not available, so the baseline was rebuilt with an 80/20 stratified split, `random_state=42`, a default Decision Tree, and a Random Forest with 100 trees. Training on data read from MongoDB gives the **exact same** metrics and confusion matrices as training on the CSV. So the move to MongoDB preserved the data exactly, and the gap from the original numbers comes from the different training setup, not from the migration.

## 5. Terminal output and logs → [5_terminal_logs/](5_terminal_logs/)
- `run_log.txt` is the full console output of `./run_all.sh`: rows loaded, round-trip check, explain results before and after indexing, aggregation cross-checks against Pandas, and the training output.
- `mongosh_live_check.txt` comes from a separate check run directly against the live database in `mongosh`. It shows document counts, the indexes, and each of the 3 aggregations explained with the index and with a forced collection scan (`hint: {$natural: 1}`). Its numbers agree with item 3.

## How to reproduce
From the project root:
```bash
docker run -d --name mongo-fraud -p 27017:27017 mongo:7
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh
```
Execution times vary by machine. Document counts, docs/keys examined and model metrics come out the same on every run.
