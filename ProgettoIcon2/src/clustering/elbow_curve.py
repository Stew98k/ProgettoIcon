#!/usr/bin/env python3
"""
Disegna la curva del gomito (oppure silhouette) sui dati *già* preprocessati
(poi_<city>_prep.csv).  Usa Mini-Batch K-Means come nel progetto finale.

Esempi:
    python elbow_curve.py Rome
    python elbow_curve.py Florence --metric silhouette
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ─────────────── CLI ──────────────────────────────────────────
par = argparse.ArgumentParser()
par.add_argument("city", help="Nome città (Rome, Florence …)")
par.add_argument("--metric", choices=["inertia", "silhouette"],
                 default="inertia", help="Metri­ca da plottare")
par.add_argument("--k-min", type=int, default=2)
par.add_argument("--k-max", type=int, default=15)
args = par.parse_args()

CITY   = args.city.lower()
METRIC = args.metric
K_MIN, K_MAX = args.k_min, args.k_max

DATA = Path("data")
prep_file = DATA / f"poi_{CITY}_prep.csv"
if not prep_file.exists():
    raise SystemExit(f"💥  File non trovato: {prep_file} (esegui preprocess.py)")

# ─────────────── lettura -------------------------------
X = pd.read_csv(prep_file).values          # già numerico, nessuna pipeline

# ─────────────── calcolo metrica ------------------------
ks   = range(K_MIN, K_MAX + 1)
vals = []
for k in ks:
    kmeans = MiniBatchKMeans(
        n_clusters=k, random_state=0, batch_size=256, n_init=10
    )
    labels = kmeans.fit_predict(X)
    if METRIC == "inertia":
        vals.append(kmeans.inertia_)
    else:                                  # silhouette
        vals.append(silhouette_score(X, labels))

# ─────────────── grafico --------------------------------
plt.figure(figsize=(6, 4))
plt.plot(list(ks), vals, marker="o")
plt.xlabel("k (numero cluster)")
plt.ylabel("Inertia" if METRIC == "inertia" else "Silhouette")
title = "Elbow Curve" if METRIC == "inertia" else "Silhouette vs k"
plt.title(f"{title} – {CITY.capitalize()}")
plt.tight_layout()
plt.show()
