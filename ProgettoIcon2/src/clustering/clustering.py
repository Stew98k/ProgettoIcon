#!/usr/bin/env python3
"""Clustering "potenziato" dei POI (fix definitivo mismatch lunghezze)."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import hdbscan
except ImportError:
    hdbscan = None

parser = argparse.ArgumentParser()
parser.add_argument("city")
parser.add_argument("--alg", choices=["kmeans", "hdbscan"], default="kmeans")
parser.add_argument("--k-min", type=int, default=4)
parser.add_argument("--k-max", type=int, default=12)
args = parser.parse_args()
city = args.city.lower()

PREPFILE = Path(f"data/poi_{city}_prep.csv")
RAWFILE  = Path(f"data/poi_{city}.csv")
OUTFILE  = Path(f"data/poi_{city}_cluster.csv")

if not PREPFILE.exists():
    raise FileNotFoundError("Preprocessed CSV mancante: esegui preprocess.py")

df_num = pd.read_csv(PREPFILE, header=None)
X = df_num.values

# ---------------- Clustering ----------------
if args.alg == "kmeans":
    best_k, best_model, best_score = None, None, -1
    for k in range(args.k_min, args.k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=1024)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(10000, X.shape[0]))
        print(f"k={k:<2} → silhouette={score:.3f}")
        if score > best_score:
            best_k, best_model, best_score = k, km, score
    final_labels = best_model.labels_
    print(f"✔︎ k scelto: {best_k}  (silhouette={best_score:.3f})")
else:
    if hdbscan is None:
        raise SystemExit("Install hdbscan or use kmeans")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
    final_labels = clusterer.fit_predict(X)

# --------------- Output DF ---------------
labels_len = len(final_labels)
try:
    df_raw = pd.read_csv(RAWFILE)
except FileNotFoundError:
    df_raw = pd.DataFrame()

common_n = min(labels_len, len(df_raw))

if common_n > 0:
    df_out = df_raw.iloc[:common_n].copy()
    df_out["cluster"] = final_labels[:common_n]
else:
    df_out = pd.DataFrame({"cluster": final_labels})

df_out.to_csv(OUTFILE, index=False)
print(f"Saved {OUTFILE} with {len(df_out)} rows")