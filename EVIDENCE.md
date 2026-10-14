# Evidence: Fraud Detection on MongoDB
Everything below comes straight from the project's source files and from `results/`, which `./run_all.sh` regenerates. Nothing has been retyped by hand.
## How to verify this yourself

**Option A: rerun everything from scratch (about 3 minutes).** This downloads the dataset, reloads MongoDB, rebuilds the indexes, reruns explain and retrains both models. The output should match Sections 1–5 below. Timings will vary by machine; document counts, docs/keys examined and model metrics are deterministic.

```bash
docker run -d --name mongo-fraud -p 27017:27017 mongo:7
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh
```

**Option B: spot-check the live database in `mongosh`.**

```javascript
// mongosh fraud    (or: docker exec -it mongo-fraud mongosh fraud)
db.transactions.countDocuments({})            // 284807
db.transactions.countDocuments({Class: 1})    // 492
db.transactions.getIndexes()                  // _id_, amount_class, time

// Same aggregation, with the index vs. a forced collection scan
const p = [{$match: {Amount: {$gt: 1000}}},
           {$group: {_id: "$Class", n: {$sum: 1}, mean: {$avg: "$Amount"}}}];
db.transactions.explain("executionStats").aggregate(p)
// -> IXSCAN on amount_class, totalDocsExamined: 0, totalKeysExamined: 2940
db.transactions.explain("executionStats").aggregate(p, {hint: {$natural: 1}})
// -> COLLSCAN, totalDocsExamined: 284807, totalKeysExamined: 0
```

I ran exactly these commands on the live database: the indexed run reads 0 documents and examines 2,940 keys in 1 ms, and the forced scan reads 284,807 documents in 93 ms.

**What cannot be verified here:** the original numbers (DT 0.83/0.81/0.82, RF 0.95/0.76/0.85) come from the earlier project, whose training code was not available. The models in Section 4 were retrained with a rebuilt setup. That is why they differ from the originals while matching the CSV baseline exactly.

## 1. Data loading: document schema and loader code
Sample document as stored (a fraud transaction), from `results/load_report.json`:
```json
{
  "_id": 541,
  "Time": 406.0,
  "V1": -2.3122265423263,
  "V2": 1.95199201064158,
  "V3": -1.60985073229769,
  "V4": 3.9979055875468,
  "V5": -0.522187864667764,
  "V6": -1.42654531920595,
  "V7": -2.53738730624579,
  "V8": 1.39165724829804,
  "V9": -2.77008927719433,
  "V10": -2.77227214465915,
  "V11": 3.20203320709635,
  "V12": -2.89990738849473,
  "V13": -0.595221881324605,
  "V14": -4.28925378244217,
  "V15": 0.389724120274487,
  "V16": -1.14074717980657,
  "V17": -2.83005567450437,
  "V18": -0.0168224681808257,
  "V19": 0.416955705037907,
  "V20": 0.126910559061474,
  "V21": 0.517232370861764,
  "V22": -0.0350493686052974,
  "V23": -0.465211076182388,
  "V24": 0.320198198514526,
  "V25": 0.0445191674731724,
  "V26": 0.177839798284401,
  "V27": 0.261145002567677,
  "V28": -0.143275874698919,
  "Amount": 0.0,
  "Class": 1
}
```
Load verification:
```json
{
  "csv_rows": 284807,
  "mongo_docs": 284807,
  "fraud_docs": 492,
  "roundtrip_identical_to_csv": true,
  "data_size_bytes": 116770870,
  "avg_doc_size_bytes": 410
}
```
Loader (`02_load_mongo.py`), including the `$jsonSchema` validator:
```python
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
```
## 2. Aggregation pipeline code
`pipelines.py` (the actual `$match` / `$bucket` / `$group` / `$setWindowFields` stages):
```python
"""Aggregation pipelines that replace the usual Pandas feature-engineering code.

Each function returns a plain pipeline list so it can be run, explained, and dumped to JSON.
"""
from common import V_COLS

AMOUNT_BOUNDARIES = [0, 1, 10, 50, 100, 250, 500, 1000, 2500, 5000]


def amount_bin_fraud_rate(match=None):
    """Pandas equivalent: df.groupby(pd.cut(df.Amount, bins)).Class.agg(['count', 'sum', 'mean'])"""
    return ([{"$match": match}] if match else []) + [
        {"$bucket": {
            "groupBy": "$Amount",
            "boundaries": AMOUNT_BOUNDARIES,
            "default": "5000+",
            "output": {
                "n": {"$sum": 1},
                "frauds": {"$sum": "$Class"},
                "avg_amount": {"$avg": "$Amount"},
            },
        }},
        {"$addFields": {"fraud_rate": {"$divide": ["$frauds", "$n"]}}},
    ]


def hourly_window_counts(match=None):
    """Pandas equivalent: df.groupby(df.Time // 3600).agg(...) followed by .rolling(3).sum()

    Groups transactions into 1-hour windows, then a $setWindowFields stage adds a
    trailing 3-hour rolling count across those windows.
    """
    return ([{"$match": match}] if match else []) + [
        {"$group": {
            "_id": {"$floor": {"$divide": ["$Time", 3600]}},
            "n": {"$sum": 1},
            "frauds": {"$sum": "$Class"},
            "total_amount": {"$sum": "$Amount"},
        }},
        {"$setWindowFields": {
            "sortBy": {"_id": 1},
            "output": {
                "rolling_3h_n": {"$sum": "$n", "window": {"documents": [-2, 0]}},
                "rolling_3h_frauds": {"$sum": "$frauds", "window": {"documents": [-2, 0]}},
            },
        }},
        {"$project": {
            "_id": 0, "hour_index": "$_id", "hour_of_day": {"$mod": ["$_id", 24]},
            "n": 1, "frauds": 1, "total_amount": {"$round": ["$total_amount", 2]},
            "rolling_3h_n": 1, "rolling_3h_frauds": 1,
        }},
    ]


def class_feature_stats(match=None, fields=None):
    """Pandas equivalent: df[df.<filter>].groupby('Class')[cols].agg(['mean', 'std'])"""
    fields = fields or (["Amount"] + V_COLS)
    group = {"_id": "$Class", "n": {"$sum": 1}}
    for f in fields:
        group[f"{f}_mean"] = {"$avg": f"${f}"}
        group[f"{f}_std"] = {"$stdDevSamp": f"${f}"}
    return ([{"$match": match}] if match else []) + [{"$group": group}, {"$sort": {"_id": 1}}]


def per_transaction_features(out_collection):
    """Builds the model feature table, one doc per transaction, written with $out.

    Time-based rolling features use $setWindowFields with a *range* window on Time
    (seconds), i.e. "transactions in the trailing hour", which is the Mongo version of
    df.rolling('1h', on=...) in Pandas. The current row is included in each window.
    """
    return [
        {"$setWindowFields": {
            "sortBy": {"Time": 1},
            "output": {
                "tx_count_prev_1h": {"$count": {}, "window": {"range": [-3600, 0]}},
                "tx_count_prev_10m": {"$count": {}, "window": {"range": [-600, 0]}},
                "amount_mean_prev_1h": {"$avg": "$Amount", "window": {"range": [-3600, 0]}},
            },
        }},
        {"$addFields": {
            "hour_of_day": {"$mod": [{"$floor": {"$divide": ["$Time", 3600]}}, 24]},
            "log_amount": {"$ln": {"$add": ["$Amount", 1]}},
            "amount_vs_1h_mean": {"$divide": [
                "$Amount", {"$max": ["$amount_mean_prev_1h", 0.01]}]},
            "amount_bin": {"$size": {"$filter": {
                "input": AMOUNT_BOUNDARIES[1:], "cond": {"$gte": ["$Amount", "$$this"]}}}},
        }},
        {"$out": out_collection},
    ]
```
The same pipelines as raw MQL JSON (`results/pipelines.json`), so you can paste them into `mongosh`:
```json
{
  "amount_bin_fraud_rate": [
    {
      "$bucket": {
        "groupBy": "$Amount",
        "boundaries": [
          0,
          1,
          10,
          50,
          100,
          250,
          500,
          1000,
          2500,
          5000
        ],
        "default": "5000+",
        "output": {
          "n": {
            "$sum": 1
          },
          "frauds": {
            "$sum": "$Class"
          },
          "avg_amount": {
            "$avg": "$Amount"
          }
        }
      }
    },
    {
      "$addFields": {
        "fraud_rate": {
          "$divide": [
            "$frauds",
            "$n"
          ]
        }
      }
    }
  ],
  "hourly_window_counts": [
    {
      "$group": {
        "_id": {
          "$floor": {
            "$divide": [
              "$Time",
              3600
            ]
          }
        },
        "n": {
          "$sum": 1
        },
        "frauds": {
          "$sum": "$Class"
        },
        "total_amount": {
          "$sum": "$Amount"
        }
      }
    },
    {
      "$setWindowFields": {
        "sortBy": {
          "_id": 1
        },
        "output": {
          "rolling_3h_n": {
            "$sum": "$n",
            "window": {
              "documents": [
                -2,
                0
              ]
            }
          },
          "rolling_3h_frauds": {
            "$sum": "$frauds",
            "window": {
              "documents": [
                -2,
                0
              ]
            }
          }
        }
      }
    },
    {
      "$project": {
        "_id": 0,
        "hour_index": "$_id",
        "hour_of_day": {
          "$mod": [
            "$_id",
            24
          ]
        },
        "n": 1,
        "frauds": 1,
        "total_amount": {
          "$round": [
            "$total_amount",
            2
          ]
        },
        "rolling_3h_n": 1,
        "rolling_3h_frauds": 1
      }
    }
  ],
  "class_feature_stats": [
    {
      "$group": {
        "_id": "$Class",
        "n": {
          "$sum": 1
        },
        "Amount_mean": {
          "$avg": "$Amount"
        },
        "Amount_std": {
          "$stdDevSamp": "$Amount"
        },
        "V1_mean": {
          "$avg": "$V1"
        },
        "V1_std": {
          "$stdDevSamp": "$V1"
        },
        "V2_mean": {
          "$avg": "$V2"
        },
        "V2_std": {
          "$stdDevSamp": "$V2"
        },
        "V3_mean": {
          "$avg": "$V3"
        },
        "V3_std": {
          "$stdDevSamp": "$V3"
        },
        "V4_mean": {
          "$avg": "$V4"
        },
        "V4_std": {
          "$stdDevSamp": "$V4"
        },
        "V5_mean": {
          "$avg": "$V5"
        },
        "V5_std": {
          "$stdDevSamp": "$V5"
        },
        "V6_mean": {
          "$avg": "$V6"
        },
        "V6_std": {
          "$stdDevSamp": "$V6"
        },
        "V7_mean": {
          "$avg": "$V7"
        },
        "V7_std": {
          "$stdDevSamp": "$V7"
        },
        "V8_mean": {
          "$avg": "$V8"
        },
        "V8_std": {
          "$stdDevSamp": "$V8"
        },
        "V9_mean": {
          "$avg": "$V9"
        },
        "V9_std": {
          "$stdDevSamp": "$V9"
        },
        "V10_mean": {
          "$avg": "$V10"
        },
        "V10_std": {
          "$stdDevSamp": "$V10"
        },
        "V11_mean": {
          "$avg": "$V11"
        },
        "V11_std": {
          "$stdDevSamp": "$V11"
        },
        "V12_mean": {
          "$avg": "$V12"
        },
        "V12_std": {
          "$stdDevSamp": "$V12"
        },
        "V13_mean": {
          "$avg": "$V13"
        },
        "V13_std": {
          "$stdDevSamp": "$V13"
        },
        "V14_mean": {
          "$avg": "$V14"
        },
        "V14_std": {
          "$stdDevSamp": "$V14"
        },
        "V15_mean": {
          "$avg": "$V15"
        },
        "V15_std": {
          "$stdDevSamp": "$V15"
        },
        "V16_mean": {
          "$avg": "$V16"
        },
        "V16_std": {
          "$stdDevSamp": "$V16"
        },
        "V17_mean": {
          "$avg": "$V17"
        },
        "V17_std": {
          "$stdDevSamp": "$V17"
        },
        "V18_mean": {
          "$avg": "$V18"
        },
        "V18_std": {
          "$stdDevSamp": "$V18"
        },
        "V19_mean": {
          "$avg": "$V19"
        },
        "V19_std": {
          "$stdDevSamp": "$V19"
        },
        "V20_mean": {
          "$avg": "$V20"
        },
        "V20_std": {
          "$stdDevSamp": "$V20"
        },
        "V21_mean": {
          "$avg": "$V21"
        },
        "V21_std": {
          "$stdDevSamp": "$V21"
        },
        "V22_mean": {
          "$avg": "$V22"
        },
        "V22_std": {
          "$stdDevSamp": "$V22"
        },
        "V23_mean": {
          "$avg": "$V23"
        },
        "V23_std": {
          "$stdDevSamp": "$V23"
        },
        "V24_mean": {
          "$avg": "$V24"
        },
        "V24_std": {
          "$stdDevSamp": "$V24"
        },
        "V25_mean": {
          "$avg": "$V25"
        },
        "V25_std": {
          "$stdDevSamp": "$V25"
        },
        "V26_mean": {
          "$avg": "$V26"
        },
        "V26_std": {
          "$stdDevSamp": "$V26"
        },
        "V27_mean": {
          "$avg": "$V27"
        },
        "V27_std": {
          "$stdDevSamp": "$V27"
        },
        "V28_mean": {
          "$avg": "$V28"
        },
        "V28_std": {
          "$stdDevSamp": "$V28"
        }
      }
    },
    {
      "$sort": {
        "_id": 1
      }
    }
  ],
  "per_transaction_features": [
    {
      "$setWindowFields": {
        "sortBy": {
          "Time": 1
        },
        "output": {
          "tx_count_prev_1h": {
            "$count": {},
            "window": {
              "range": [
                -3600,
                0
              ]
            }
          },
          "tx_count_prev_10m": {
            "$count": {},
            "window": {
              "range": [
                -600,
                0
              ]
            }
          },
          "amount_mean_prev_1h": {
            "$avg": "$Amount",
            "window": {
              "range": [
                -3600,
                0
              ]
            }
          }
        }
      }
    },
    {
      "$addFields": {
        "hour_of_day": {
          "$mod": [
            {
              "$floor": {
                "$divide": [
                  "$Time",
                  3600
                ]
              }
            },
            24
          ]
        },
        "log_amount": {
          "$ln": {
            "$add": [
              "$Amount",
              1
            ]
          }
        },
        "amount_vs_1h_mean": {
          "$divide": [
            "$Amount",
            {
              "$max": [
                "$amount_mean_prev_1h",
                0.01
              ]
            }
          ]
        },
        "amount_bin": {
          "$size": {
            "$filter": {
              "input": [
                1,
                10,
                50,
                100,
                250,
                500,
                1000,
                2500,
                5000
              ],
              "cond": {
                "$gte": [
                  "$Amount",
                  "$$this"
                ]
              }
            }
          }
        }
      }
    },
    {
      "$out": "features"
    }
  ]
}
```
## 3. Indexes and explain("executionStats")
Indexes: `{Amount: 1, Class: 1}` named `amount_class`, and `{Time: 1}` named `time`. Benchmark code is in `03_explain_benchmark.py`. The full explain documents are in `results/explain_before_index.json` and `results/explain_after_index.json`.
### Fraud rate by Amount bin, Amount ≥ 500 ($match → $bucket)
**Pipeline:**
```json
[
  {
    "$match": {
      "Amount": {
        "$gte": 500
      }
    }
  },
  {
    "$bucket": {
      "groupBy": "$Amount",
      "boundaries": [
        0,
        1,
        10,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000
      ],
      "default": "5000+",
      "output": {
        "n": {
          "$sum": 1
        },
        "frauds": {
          "$sum": "$Class"
        },
        "avg_amount": {
          "$avg": "$Amount"
        }
      }
    }
  },
  {
    "$addFields": {
      "fraud_rate": {
        "$divide": [
          "$frauds",
          "$n"
        ]
      }
    }
  }
]
```
**WITHOUT index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_SIMPLE",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "COLLSCAN",
      "filter": {
        "Amount": {
          "$gte": 500
        }
      },
      "direction": "forward"
    }
  },
  "executionStats": {
    "nReturned": 9492,
    "executionTimeMillis": 98,
    "totalKeysExamined": 0,
    "totalDocsExamined": 284807
  }
}
```
**WITH index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_COVERED",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "IXSCAN",
      "keyPattern": {
        "Amount": 1,
        "Class": 1
      },
      "indexName": "amount_class",
      "isMultiKey": false,
      "multiKeyPaths": {
        "Amount": [],
        "Class": []
      },
      "isUnique": false,
      "isSparse": false,
      "isPartial": false,
      "indexVersion": 2,
      "direction": "forward",
      "indexBounds": {
        "Amount": [
          "[500, inf.0]"
        ],
        "Class": [
          "[MinKey, MaxKey]"
        ]
      }
    }
  },
  "executionStats": {
    "nReturned": 9492,
    "executionTimeMillis": 6,
    "totalKeysExamined": 9492,
    "totalDocsExamined": 0
  }
}
```
### Hourly window counts, first 6h ($match → $group → $setWindowFields)
**Pipeline:**
```json
[
  {
    "$match": {
      "Time": {
        "$lt": 21600
      }
    }
  },
  {
    "$group": {
      "_id": {
        "$floor": {
          "$divide": [
            "$Time",
            3600
          ]
        }
      },
      "n": {
        "$sum": 1
      },
      "frauds": {
        "$sum": "$Class"
      },
      "total_amount": {
        "$sum": "$Amount"
      }
    }
  },
  {
    "$setWindowFields": {
      "sortBy": {
        "_id": 1
      },
      "output": {
        "rolling_3h_n": {
          "$sum": "$n",
          "window": {
            "documents": [
              -2,
              0
            ]
          }
        },
        "rolling_3h_frauds": {
          "$sum": "$frauds",
          "window": {
            "documents": [
              -2,
              0
            ]
          }
        }
      }
    }
  },
  {
    "$project": {
      "_id": 0,
      "hour_index": "$_id",
      "hour_of_day": {
        "$mod": [
          "$_id",
          24
        ]
      },
      "n": 1,
      "frauds": 1,
      "total_amount": {
        "$round": [
          "$total_amount",
          2
        ]
      },
      "rolling_3h_n": 1,
      "rolling_3h_frauds": 1
    }
  }
]
```
**WITHOUT index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_SIMPLE",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "Time": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "COLLSCAN",
      "filter": {
        "Time": {
          "$lt": 21600
        }
      },
      "direction": "forward"
    }
  },
  "executionStats": {
    "nReturned": 12340,
    "executionTimeMillis": 51,
    "totalKeysExamined": 0,
    "totalDocsExamined": 284807
  }
}
```
**WITH index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_SIMPLE",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "Time": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "FETCH",
      "inputStage": {
        "stage": "IXSCAN",
        "keyPattern": {
          "Time": 1
        },
        "indexName": "time",
        "isMultiKey": false,
        "multiKeyPaths": {
          "Time": []
        },
        "isUnique": false,
        "isSparse": false,
        "isPartial": false,
        "indexVersion": 2,
        "direction": "forward",
        "indexBounds": {
          "Time": [
            "[-inf.0, 21600)"
          ]
        }
      }
    }
  },
  "executionStats": {
    "nReturned": 12340,
    "executionTimeMillis": 11,
    "totalKeysExamined": 12340,
    "totalDocsExamined": 12340
  }
}
```
### Mean/std of Amount by Class, Amount > 1000 ($match → $group)
**Pipeline:**
```json
[
  {
    "$match": {
      "Amount": {
        "$gt": 1000
      }
    }
  },
  {
    "$group": {
      "_id": "$Class",
      "n": {
        "$sum": 1
      },
      "Amount_mean": {
        "$avg": "$Amount"
      },
      "Amount_std": {
        "$stdDevSamp": "$Amount"
      }
    }
  },
  {
    "$sort": {
      "_id": 1
    }
  }
]
```
**WITHOUT index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_SIMPLE",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "COLLSCAN",
      "filter": {
        "Amount": {
          "$gt": 1000
        }
      },
      "direction": "forward"
    }
  },
  "executionStats": {
    "nReturned": 2940,
    "executionTimeMillis": 92,
    "totalKeysExamined": 0,
    "totalDocsExamined": 284807
  }
}
```
**WITH index** (excerpt from explain):
```json
{
  "winningPlan": {
    "stage": "PROJECTION_COVERED",
    "transformBy": {
      "Amount": 1,
      "Class": 1,
      "_id": 0
    },
    "inputStage": {
      "stage": "IXSCAN",
      "keyPattern": {
        "Amount": 1,
        "Class": 1
      },
      "indexName": "amount_class",
      "isMultiKey": false,
      "multiKeyPaths": {
        "Amount": [],
        "Class": []
      },
      "isUnique": false,
      "isSparse": false,
      "isPartial": false,
      "indexVersion": 2,
      "direction": "forward",
      "indexBounds": {
        "Amount": [
          "(1000, inf.0]"
        ],
        "Class": [
          "[MinKey, MaxKey]"
        ]
      }
    }
  },
  "executionStats": {
    "nReturned": 2940,
    "executionTimeMillis": 1,
    "totalKeysExamined": 2940,
    "totalDocsExamined": 0
  }
}
```
### Summary of all workloads (explain stats + median wall time over 7 runs)
| Workload | Plan before | Plan after | Docs examined before | Docs examined after | Keys examined after | Explain ms before → after | Median wall ms before → after |
|---|---|---|---|---|---|---|---|
| `find_amount_range` | COLLSCAN | FETCH <- IXSCAN | 284,807 | 2,378 | 2,378 | 137 → 3 | 139.46 → 6.38 |
| `find_time_window` | COLLSCAN | FETCH <- IXSCAN | 284,807 | 8,288 | 8,288 | 49 → 4 | 66.88 → 19.89 |
| `agg_amount_bins_high_value` | PROJECTION_SIMPLE <- COLLSCAN | PROJECTION_COVERED <- IXSCAN | 284,807 | 0 | 9,492 | 98 → 6 | 95.48 → 6.27 |
| `agg_hourly_counts_first_6h` | PROJECTION_SIMPLE <- COLLSCAN | PROJECTION_SIMPLE <- FETCH <- IXSCAN | 284,807 | 12,340 | 12,340 | 51 → 11 | 52.8 → 10.44 |
| `agg_class_amount_stats_gt_1000` | PROJECTION_SIMPLE <- COLLSCAN | PROJECTION_COVERED <- IXSCAN | 284,807 | 0 | 2,940 | 92 → 1 | 81.09 → 1.39 |
| `agg_amount_bins_full_collection` | PROJECTION_SIMPLE <- COLLSCAN | PROJECTION_SIMPLE <- COLLSCAN | 284,807 | 284,807 | 0 | 174 → 182 | 171.7 → 180.58 |

The last row has no `$match`, so it scans the whole collection whether or not the indexes exist. It is included on purpose: an index only helps a query that selects a subset of the documents.
## 4. Model comparison (fraud class, test set of 56,962 rows with 98 frauds)
| model        | source                   |   precision |   recall |     f1 |
|:-------------|:-------------------------|------------:|---------:|-------:|
| DecisionTree | Original (user-reported) |      0.83   |   0.81   | 0.82   |
| DecisionTree | Pandas/CSV rebuild       |      0.7526 |   0.7449 | 0.7487 |
| DecisionTree | Mongo raw columns        |      0.7526 |   0.7449 | 0.7487 |
| DecisionTree | Mongo + engineered       |      0.6881 |   0.7653 | 0.7246 |
| RandomForest | Original (user-reported) |      0.95   |   0.76   | 0.85   |
| RandomForest | Pandas/CSV rebuild       |      0.9412 |   0.8163 | 0.8743 |
| RandomForest | Mongo raw columns        |      0.9412 |   0.8163 | 0.8743 |
| RandomForest | Mongo + engineered       |      0.9398 |   0.7959 | 0.8619 |

- **Pandas/CSV rebuild:** the original training code wasn't available, so the baseline was rebuilt with an 80/20 stratified split, `random_state=42`, a default DecisionTree, and a RandomForest with `n_estimators=100`.
- **Mongo raw columns:** the same features, but read from MongoDB. The metrics *and* confusion matrices are identical to the CSV run (`raw_predictions_identical_to_baseline: true`), which shows the migration preserved the data exactly.
- **Gap from the original numbers:** the Mongo run matches the CSV run exactly, so the difference from 0.83/0.81/0.82 and 0.95/0.76/0.85 comes from the training setup (split, seed, preprocessing), not from MongoDB. With only 98 fraud cases in the test set, one misclassification moves recall by about 1 point.

Confusion matrices `[[TN, FP], [FN, TP]]`:
```json
{
  "pandas_baseline": {
    "DecisionTree": [
      [
        56840,
        24
      ],
      [
        25,
        73
      ]
    ],
    "RandomForest": [
      [
        56859,
        5
      ],
      [
        18,
        80
      ]
    ]
  },
  "mongo_raw": {
    "DecisionTree": [
      [
        56840,
        24
      ],
      [
        25,
        73
      ]
    ],
    "RandomForest": [
      [
        56859,
        5
      ],
      [
        18,
        80
      ]
    ]
  },
  "mongo_engineered": {
    "DecisionTree": [
      [
        56830,
        34
      ],
      [
        23,
        75
      ]
    ],
    "RandomForest": [
      [
        56859,
        5
      ],
      [
        20,
        78
      ]
    ]
  }
}
```
Mongo vs Pandas cross-checks on the aggregation outputs:
```json
{
  "Amount_mean": true,
  "Amount_std": true,
  "V1_mean": true,
  "V1_std": true,
  "V14_mean": true,
  "V14_std": true,
  "hourly_counts": true,
  "rolling_1h_count": true
}
```
## 5. Terminal output from the full run (`results/run_log.txt`)
```text
== 01_baseline_pandas.py
CSV: 284807 rows, fraud=492
Pandas/CSV baseline:
  DecisionTree  P=0.7526 R=0.7449 F1=0.7487
  RandomForest  P=0.9412 R=0.8163 F1=0.8743
== 02_load_mongo.py
{'csv_rows': 284807, 'mongo_docs': 284807, 'fraud_docs': 492, 'roundtrip_identical_to_csv': True, 'data_size_bytes': 116770870, 'avg_doc_size_bytes': 410}
== 03_explain_benchmark.py
Without secondary indexes:
  [before] find_amount_range                  {'plan': 'COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 2378, 'explain_ms': 137, 'median_wall_ms': 139.46}
  [before] find_time_window                   {'plan': 'COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 8288, 'explain_ms': 49, 'median_wall_ms': 66.88}
  [before] agg_amount_bins_high_value         {'plan': 'PROJECTION_SIMPLE <- COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 9492, 'explain_ms': 98, 'median_wall_ms': 95.48}
  [before] agg_hourly_counts_first_6h         {'plan': 'PROJECTION_SIMPLE <- COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 12340, 'explain_ms': 51, 'median_wall_ms': 52.8}
  [before] agg_class_amount_stats_gt_1000     {'plan': 'PROJECTION_SIMPLE <- COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 2940, 'explain_ms': 92, 'median_wall_ms': 81.09}
  [before] agg_amount_bins_full_collection    {'plan': 'PROJECTION_SIMPLE <- COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 284807, 'explain_ms': 174, 'median_wall_ms': 171.7}
Indexes: ['_id_', 'amount_class', 'time']
With indexes:
  [after] find_amount_range                  {'plan': 'FETCH <- IXSCAN', 'docs_examined': 2378, 'keys_examined': 2378, 'n_returned': 2378, 'explain_ms': 3, 'median_wall_ms': 6.38}
  [after] find_time_window                   {'plan': 'FETCH <- IXSCAN', 'docs_examined': 8288, 'keys_examined': 8288, 'n_returned': 8288, 'explain_ms': 4, 'median_wall_ms': 19.89}
  [after] agg_amount_bins_high_value         {'plan': 'PROJECTION_COVERED <- IXSCAN', 'docs_examined': 0, 'keys_examined': 9492, 'n_returned': 9492, 'explain_ms': 6, 'median_wall_ms': 6.27}
  [after] agg_hourly_counts_first_6h         {'plan': 'PROJECTION_SIMPLE <- FETCH <- IXSCAN', 'docs_examined': 12340, 'keys_examined': 12340, 'n_returned': 12340, 'explain_ms': 11, 'median_wall_ms': 10.44}
  [after] agg_class_amount_stats_gt_1000     {'plan': 'PROJECTION_COVERED <- IXSCAN', 'docs_examined': 0, 'keys_examined': 2940, 'n_returned': 2940, 'explain_ms': 1, 'median_wall_ms': 1.39}
  [after] agg_amount_bins_full_collection    {'plan': 'PROJECTION_SIMPLE <- COLLSCAN', 'docs_examined': 284807, 'keys_examined': 0, 'n_returned': 284807, 'explain_ms': 182, 'median_wall_ms': 180.58}
== 04_features_and_models.py
Fraud rate by Amount bin:
  _id     n  frauds  avg_amount  fraud_rate
    0 16804      68    0.690182    0.004047
    1 80510     181    4.332109    0.002248
   10 92390      56   25.100793    0.000606
   50 37718      57   71.847921    0.001511
  100 34482      56  155.942089    0.001624
  250 13411      39  343.399691    0.002908
  500  6423      26  676.685905    0.004048
 1000  2620       9 1419.066427    0.003435
 2500   394       0 3305.267411    0.000000
5000+    55       0 7710.556000    0.000000
Mongo vs Pandas cross-checks: {'Amount_mean': True, 'Amount_std': True, 'V1_mean': True, 'V1_std': True, 'V14_mean': True, 'V14_std': True, 'hourly_counts': True, 'rolling_1h_count': True}
Mongo-sourced, raw columns (same features as baseline):
  DecisionTree  P=0.7526 R=0.7449 F1=0.7487
  RandomForest  P=0.9412 R=0.8163 F1=0.8743
Mongo-sourced, raw + aggregation-engineered features:
  DecisionTree  P=0.6881 R=0.7653 F1=0.7246
  RandomForest  P=0.9398 R=0.7959 F1=0.8619
       model                   source  precision  recall     f1
DecisionTree Original (user-reported)     0.8300  0.8100 0.8200
DecisionTree       Pandas/CSV rebuild     0.7526  0.7449 0.7487
DecisionTree        Mongo raw columns     0.7526  0.7449 0.7487
DecisionTree       Mongo + engineered     0.6881  0.7653 0.7246
RandomForest Original (user-reported)     0.9500  0.7600 0.8500
RandomForest       Pandas/CSV rebuild     0.9412  0.8163 0.8743
RandomForest        Mongo raw columns     0.9412  0.8163 0.8743
RandomForest       Mongo + engineered     0.9398  0.7959 0.8619
```
