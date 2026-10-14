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
