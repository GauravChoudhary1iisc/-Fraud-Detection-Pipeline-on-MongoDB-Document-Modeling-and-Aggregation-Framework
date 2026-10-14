"""Step 0: rebuild the Pandas baseline straight from the CSV (the original model code was not available)."""
import pandas as pd

from common import CSV_PATH, RAW_FEATURES, save_json, train_and_score

df = pd.read_csv(CSV_PATH)
print(f"CSV: {df.shape[0]} rows, fraud={int(df.Class.sum())}")
print("Pandas/CSV baseline:")
save_json("baseline_pandas.json", train_and_score(df[RAW_FEATURES], df["Class"]))
