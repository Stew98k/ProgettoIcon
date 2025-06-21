#!/usr/bin/env python3
"""RF12 – Valutazione e benchmark.

Calcola e stampa in tabella gli indicatori "Tempo (min)" vs "Score"
per tre strategie:
    1. Random (stesso numero di POI del tour CSP)
    2. GreedyScore (sceglie i POI col punteggio più alto)
    3. CSP + A*  (il tuo tour finale)

Genera anche uno scatter PNG `fig_quality_vs_time.png`.
"""
from __future__ import annotations

import sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CITY = sys.argv[1] if len(sys.argv) > 1 else "Rome"
DATA = Path(__file__).resolve().parents[2] / "data"

POI_FILE   = DATA / f"poi_{CITY.lower()}_scored.csv"
MATRIX_FILE= DATA / f"distance_matrix_{CITY.lower()}.npy"
ROUTE_FILE = DATA / f"route_{CITY.lower()}.csv"
FIG_FILE   = DATA / "fig_quality_vs_time.png"

# --- load datasets ----------------------------------------------------------
poi_df = pd.read_csv(POI_FILE)
D      = np.load(MATRIX_FILE)
route  = pd.read_csv(ROUTE_FILE)
uri2idx = dict(zip(poi_df.uri, poi_df.index))

k = len(route)
sel_route_idx = [uri2idx[u] for u in route.uri]

# --- helper to compute walk time -------------------------------------------
def path_time(indices:list[int]) -> float:
    s=0
    for a,b in zip(indices[:-1],indices[1:]):
        d=D[a,b]
        s+= d if np.isfinite(d) else 0
    return s/60  # minutes

# --- 1) CSP+A* --------------------------------------------------------------
score_cspa = route.score.sum()
time_cspa = path_time(sel_route_idx)

# --- 2) GreedyScore ---------------------------------------------------------
best_idx = poi_df.sort_values("score", ascending=False).head(k).index.tolist()
# simple NN ordering
ordered=[best_idx[0]]
rem=set(best_idx[1:])
while rem:
    last=ordered[-1]
    nxt=min(rem, key=lambda j: D[last,j] if np.isfinite(D[last,j]) else 1e9)
    ordered.append(nxt); rem.remove(nxt)

time_greedy = path_time(ordered)
score_greedy= poi_df.loc[ordered,'score'].sum()

# --- 3) Random --------------------------------------------------------------
random_idx = random.sample(list(poi_df.index), k)
time_rand  = path_time(random_idx)
score_rand = poi_df.loc[random_idx,'score'].sum()

# --- print summary ----------------------------------------------------------
print("\nTempo (min)  |  Score")
print("Random      {:>6.0f}   {:>6.1f}".format(time_rand, score_rand))
print("GreedyScore {:>6.0f}   {:>6.1f}".format(time_greedy, score_greedy))
print("CSP + A*    {:>6.0f}   {:>6.1f}".format(time_cspa, score_cspa))

# --- scatter plot -----------------------------------------------------------
plt.figure(figsize=(6,4))
plt.scatter(time_rand, score_rand, label="Random", marker='x', s=80)
plt.scatter(time_greedy, score_greedy, label="Greedy", marker='s', s=80)
plt.scatter(time_cspa,  score_cspa,  label="CSP+A*", marker='o', s=80)
plt.xlabel("Tempo di cammino (min)")
plt.ylabel("Score totale")
plt.title(f"Qualità vs Tempo – {CITY.capitalize()}")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_FILE, dpi=120)
print("\n✅  Figura salvata →", FIG_FILE.relative_to(DATA.parent))
