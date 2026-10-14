#!/usr/bin/env bash
# Reproduces every result in results/. Requires: docker run -d --name mongo-fraud -p 27017:27017 mongo:7
set -euo pipefail
cd "$(dirname "$0")"
[ -f data/creditcard.csv ] || curl -sSL -o data/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
for s in 01_baseline_pandas.py 02_load_mongo.py 03_explain_benchmark.py 04_features_and_models.py; do
  echo "== $s"; .venv/bin/python "$s"
done
