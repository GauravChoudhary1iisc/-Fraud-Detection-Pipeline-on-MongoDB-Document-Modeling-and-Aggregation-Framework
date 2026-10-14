# "Tell me about this project" — Interview Answer

## 30-second version

> I took a credit-card fraud detection project that was originally built on Pandas and CSV files, and rebuilt the data layer on MongoDB. I loaded 284,807 transactions with schema validation, rewrote the feature engineering as MongoDB aggregation pipelines, used `explain()` to tune indexes (one query went from 81 ms to 1.4 ms), and retrained Decision Tree and Random Forest models on data pulled back out of Mongo. The models gave exactly the same results as the CSV baseline, which showed the migration lost nothing. The Random Forest reached an F1 of 0.87 on the fraud class.

## 2-minute version (main answer)

> The project uses the Kaggle credit-card fraud dataset: about 285,000 transactions, of which only 492 are fraud, so it's very imbalanced.
>
> **Data modelling.** I stored each transaction as one document in MongoDB 7, with a `$jsonSchema` validator that enforces types and ranges, for example `Class` must be 0 or 1 and `Amount` must be a non-negative double. I used the CSV row number as `_id`, because the CSV has 1,081 exact duplicate rows and a content-based key would have merged them. After loading, I read everything back and compared it cell by cell with the original CSV. It was identical.
>
> **Aggregation pipelines.** I rewrote the Pandas feature engineering as aggregation pipelines: `$bucket` for fraud rate by amount range, `$group` with `$avg` and `$stdDevSamp` for per-class statistics, and `$setWindowFields` with a time-range window for rolling features like "transactions in the previous hour". I checked every result against Pandas, to a tolerance of 1e-9.
>
> **Indexing and query plans.** I ran `explain("executionStats")` before and after adding indexes. A compound index on `{Amount, Class}` turned two pipelines into *covered queries*, answered from the index alone with zero documents examined, which took one query from 81 ms to 1.4 ms. I also showed where indexes *don't* help: a full-collection `$group` still scans everything, because indexes only help a selective `$match` or `$sort` at the start of the pipeline.
>
> **Models.** I pulled the data back into Pandas and trained a Decision Tree and a Random Forest. Results matched the CSV baseline exactly, including the confusion matrices. Random Forest got 0.94 precision, 0.82 recall and 0.87 F1 on the fraud class.
>
> **An honest finding.** The time-based features I built in Mongo slightly *lowered* F1. That's because the dataset only covers two days and has no card ID, so global transaction volume isn't a useful per-customer signal. With a card ID, the same `$setWindowFields` pipeline using `partitionBy: card_id` would produce real velocity features.
>
> I was also clear in the write-up that this flat, dense numeric data isn't the ideal fit for a document database. A columnar format would store it better. The goal was to show MongoDB skills (validation, indexing, query planning, aggregation) on real data.

## Likely follow-up questions

**Why MongoDB for this data?**
Honestly, it isn't the best fit. The data is flat, has 31 numeric columns and no nesting, so a relational table or Parquet would suit it better. MongoDB makes sense for raw fraud data that *is* nested: device fingerprints, merchant metadata, or variable-length event histories per card. I chose it to practise schema validation, indexing and aggregation on a real dataset.

**What is a covered query?**
A query that MongoDB answers entirely from an index, without fetching the documents. In `explain()` it shows as `PROJECTION_COVERED` with `totalDocsExamined: 0`. I put `Class` in the `{Amount, Class}` index for this reason, even though nothing filters on it.

**How did you verify the migration was correct?**
In two ways. First, a cell-by-cell `DataFrame.equals` comparison between the CSV and the data read back from Mongo. Second, the models trained on Mongo data produced identical predictions to the CSV baseline.

**How did you handle class imbalance?**
I used a stratified 80/20 split so the test set keeps the same fraud ratio, and I reported precision, recall and F1 for the fraud class instead of accuracy. Accuracy would be about 99.8% even for a model that predicts "not fraud" every time.

**Did you avoid data leakage?**
Yes. The fraud rate per amount bin is calculated from labels across the whole dataset, so I didn't use it as a feature. The model only gets the bin index.

**Why are your numbers different from the original project?**
The original training code wasn't available. My Pandas baseline and my MongoDB run match each other exactly, so the gap comes from the training setup (split, seed, preprocessing), not from the database. With only 98 fraud cases in the test set, one misclassified case moves recall by about 1 point.

**What would you improve?**
- Add a `{Time, Amount, Class}` index so the time pipeline is covered too.
- Use data with a card or customer ID to build per-card velocity features.
- Try class weighting or SMOTE, and tune the decision threshold.

## Key numbers to remember

| Fact | Value |
|---|---|
| Transactions / fraud cases | 284,807 / 492 |
| Duplicate rows kept | 1,081 |
| Best query speedup (covered) | 81 ms → 1.4 ms, 0 docs examined |
| Random Forest (fraud class) | P 0.94 · R 0.82 · F1 0.87 |
| Decision Tree (fraud class) | P 0.75 · R 0.74 · F1 0.75 |
| Stack | MongoDB 7, Python 3.12, pymongo, pandas, scikit-learn |
