#!/usr/bin/env python3
"""RF6 – apprendiamo le preferenze utente (1‑5) e calcoliamo lo score.
Fix finale: gestiamo sia matrice sparse che ndarray.
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.ensemble import GradientBoostingRegressor

try:
    from pyproj import Transformer
except ImportError:
    sys.exit("💥  Installare pyproj:  pip install pyproj")

parser = argparse.ArgumentParser(description="Apprende le preferenze utente sui POI (1-5)")
parser.add_argument("city")
parser.add_argument("--samples", type=int, default=10)
args = parser.parse_args()
city = args.city.lower(); K = args.samples

RAWFILE   = Path(f"data/poi_{city}.csv")
PIPEFILE  = Path(f"data/pipeline_{city}.pkl")
OUTFILE   = Path(f"data/poi_{city}_scored.csv")

if not RAWFILE.exists() or not PIPEFILE.exists():
    sys.exit("💥  File mancanti: assicurati di aver eseguito harvest & preprocess.")

df = pd.read_csv(RAWFILE)

# --- ricrea feature x,y,open_sin/cos ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
x_m, y_m = transformer.transform(df['lon'].to_numpy(), df['lat'].to_numpy())
df['x'] = x_m; df['y'] = y_m
if 'open_mean' in df.columns:
    theta = 2 * np.pi * df['open_mean'].fillna(0) / 1440
else:
    theta = np.zeros(len(df))          # niente orari → vettore di zeri
df['open_sin'] = np.sin(theta); df['open_cos'] = np.cos(theta)

pipe = joblib.load(PIPEFILE)
X = pipe.transform(df)  # sparse or ndarray

# helper to get dense rows
get_row = (lambda m, i: m[i].toarray()[0]) if hasattr(X, 'toarray') else (lambda m, i: m[i])
X_dense = X.toarray() if hasattr(X, 'toarray') else X

# --- k-medoids sampling ---
random.seed(0)
centroids_idx = [random.randrange(X.shape[0])]
remaining = set(range(X.shape[0])) - set(centroids_idx)
while len(centroids_idx) < K:
    _, dist = pairwise_distances_argmin_min(X[list(remaining)], X[centroids_idx])
    next_idx = list(remaining)[int(np.argmax(dist))]
    centroids_idx.append(next_idx); remaining.remove(next_idx)

print("\nDai un voto 1–5 ai seguenti luoghi:\n")
X_train, y_train = [], []
for idx in centroids_idx:
    label = df.loc[idx, 'label']
    voto = None
    while voto not in {'1','2','3','4','5'}:
        voto = input(f"{label}: [1-5] ").strip()
    y_train.append(int(voto)); X_train.append(get_row(X, idx))

X_train = np.vstack(X_train)

model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)

scores = model.predict(X_dense)
scores_norm = ((np.clip(scores, 1, 5) - 1) / 4).round(3)

df_out = df.copy(); df_out['score'] = scores_norm
df_out.to_csv(OUTFILE, index=False)
print(f"✅  Punteggi salvati in {OUTFILE}")
