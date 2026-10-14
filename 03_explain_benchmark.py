"""Steps 3 & 7: query plans and execution time with and without indexes.

Runs each query/pipeline with no secondary indexes, creates the indexes, then runs
them again. Saves the full explain("executionStats") output plus a summary table.
"""
import statistics
import time

from common import TXN, get_db, save_json
from pipelines import amount_bin_fraud_rate, class_feature_stats, hourly_window_counts

db = get_db()
coll = db[TXN]

INDEXES = [
    ([("Amount", 1), ("Class", 1)], "amount_class"),
    ([("Time", 1)], "time"),
]

# Filtered workloads (an index can only help if something filters or sorts on it),
# plus one unfiltered pipeline to show where indexes do *not* help.
WORKLOADS = {
    "find_amount_range": ("find", {"Amount": {"$gte": 1000, "$lt": 2000}}),
    "find_time_window": ("find", {"Time": {"$gte": 36000, "$lt": 39600}}),
    "agg_amount_bins_high_value": ("agg", amount_bin_fraud_rate({"Amount": {"$gte": 500}})),
    "agg_hourly_counts_first_6h": ("agg", hourly_window_counts({"Time": {"$lt": 6 * 3600}})),
    "agg_class_amount_stats_gt_1000": (
        "agg", class_feature_stats({"Amount": {"$gt": 1000}}, fields=["Amount"])),
    "agg_amount_bins_full_collection": ("agg", amount_bin_fraud_rate()),
}


def explain(kind, spec):
    if kind == "find":
        cmd = {"find": TXN, "filter": spec}
    else:
        cmd = {"aggregate": TXN, "pipeline": spec, "cursor": {}}
    return db.command({"explain": cmd, "verbosity": "executionStats"})


def run(kind, spec):
    return list(coll.find(spec) if kind == "find" else coll.aggregate(spec))


def stages(plan):
    """Flatten a winning plan tree into stage names, e.g. FETCH <- IXSCAN."""
    names = []
    while plan:
        names.append(plan.get("stage"))
        plan = plan.get("inputStage") or (plan.get("inputStages") or [None])[0]
    return " <- ".join(n for n in names if n)


def summarize(exp):
    # find explains have top-level stats; pipelines nest them under the $cursor stage
    # (or under queryPlanner/executionStats directly when the whole pipeline is pushed down).
    if "executionStats" in exp:
        es, qp = exp["executionStats"], exp["queryPlanner"]
    else:
        cursor = exp["stages"][0]["$cursor"]
        es, qp = cursor["executionStats"], cursor["queryPlanner"]
    wp = qp["winningPlan"]
    wp = wp.get("queryPlan", wp)  # SBE plans wrap the classic tree
    return {
        "plan": stages(wp),
        "docs_examined": es["totalDocsExamined"],
        "keys_examined": es["totalKeysExamined"],
        "n_returned": es["nReturned"],
        "explain_ms": es["executionTimeMillis"],
    }


def wall_ms(kind, spec, reps=7):
    run(kind, spec)  # warm cache
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        run(kind, spec)
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 2)


def measure(label):
    raw, summary = {}, {}
    for name, (kind, spec) in WORKLOADS.items():
        exp = explain(kind, spec)
        raw[name] = exp
        summary[name] = {**summarize(exp), "median_wall_ms": wall_ms(kind, spec)}
        print(f"  [{label}] {name:34s} {summary[name]}")
    return raw, summary


coll.drop_indexes()
print("Without secondary indexes:")
raw_before, before = measure("before")

for keys, name in INDEXES:
    coll.create_index(keys, name=name)
print("Indexes:", list(coll.index_information()))
print("With indexes:")
raw_after, after = measure("after")

save_json("explain_before_index.json", raw_before)
save_json("explain_after_index.json", raw_after)
save_json("benchmark_summary.json", {
    "indexes": [{"name": n, "keys": k} for k, n in INDEXES],
    "workloads": {n: {"kind": k, "spec": s} for n, (k, s) in WORKLOADS.items()},
    "before": before,
    "after": after,
})
