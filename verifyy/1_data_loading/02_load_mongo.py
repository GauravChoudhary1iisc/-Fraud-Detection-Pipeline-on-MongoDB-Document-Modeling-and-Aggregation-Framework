"""Step 2: load creditcard.csv into MongoDB, one document per transaction.

Document shape (see README for the reasoning):
    {
      _id:    <int>    row number in the original CSV (stable key, preserves order),
      Time:   <double> seconds since first transaction,
      Amount: <double>,
      Class:  <int32>  0 = legit, 1 = fraud,
      V1..V28: <double> PCA components
    }
A $jsonSchema validator enforces types so bad rows are rejected at write time.
"""
import pandas as pd
from pymongo import InsertOne

from common import CSV_PATH, RAW_FEATURES, TXN, V_COLS, get_db, save_json

VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["_id", "Time", "Amount", "Class"] + V_COLS,
        "properties": {
            "_id": {"bsonType": "int"},
            "Time": {"bsonType": "double", "minimum": 0},
            "Amount": {"bsonType": "double", "minimum": 0},
            "Class": {"bsonType": "int", "enum": [0, 1]},
            **{v: {"bsonType": "double"} for v in V_COLS},
        },
    }
}

db = get_db()
db.drop_collection(TXN)
db.create_collection(TXN, validator=VALIDATOR)

df = pd.read_csv(CSV_PATH)
df["Time"] = df["Time"].astype(float)
docs = [
    {"_id": i, **{c: float(r[c]) for c in RAW_FEATURES}, "Class": int(r["Class"])}
    for i, r in enumerate(df.to_dict("records"))
]
BATCH = 10_000
for s in range(0, len(docs), BATCH):
    db[TXN].bulk_write([InsertOne(d) for d in docs[s:s + BATCH]], ordered=False)

# Integrity check: read everything back and compare cell-by-cell with the CSV.
back = pd.DataFrame(db[TXN].find({}, sort=[("_id", 1)])).set_index("_id")[RAW_FEATURES + ["Class"]]
identical = back.reset_index(drop=True).equals(df[RAW_FEATURES + ["Class"]])
stats = db.command("collStats", TXN)
report = {
    "csv_rows": len(df),
    "mongo_docs": db[TXN].count_documents({}),
    "fraud_docs": db[TXN].count_documents({"Class": 1}),
    "roundtrip_identical_to_csv": identical,
    "data_size_bytes": stats["size"],
    "avg_doc_size_bytes": stats["avgObjSize"],
    "sample_document": db[TXN].find_one({"Class": 1}),
    "validator": VALIDATOR,
}
save_json("load_report.json", report)
print({k: v for k, v in report.items() if k not in ("sample_document", "validator")})
