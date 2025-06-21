#!/usr/bin/env python3
"""Pre-processing dei POI: coordinate in metri, feature cicliche per l'orario,
   pipeline ri-usabile salvata con joblib.

   Uso:
       python preprocess.py Rome

   Crea:
       data/poi_<city>_prep.csv   (dati trasformati, solo per ispezione rapida)
       data/pipeline_<city>.pkl   (pipeline sklearn serializzata)
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
from pyproj import Transformer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------- CLI ------------------------------------------
parser = argparse.ArgumentParser(description="Pre-processing del CSV dei POI per una data città")
parser.add_argument("city", help="Nome città (es. Rome, Florence, Bari)")
args = parser.parse_args()
city = args.city.lower()

# --------------------------- Percorsi file ----------------------------------
RAW  = Path(f"data/poi_{city}.csv")
PREP = Path(f"data/poi_{city}_prep.csv")
PIPE = Path(f"data/pipeline_{city}.pkl")
PREP.parent.mkdir(parents=True, exist_ok=True)

# --------------------------- Lettura dati -----------------------------------
if not RAW.exists():
    raise FileNotFoundError(f"Non trovo il file {RAW}; hai eseguito harvest_poi.py?")

df = pd.read_csv(RAW)

# ----------------- Coordinate: lat/lon → metri, scaler ----------------------
if {'lat', 'lon'}.issubset(df.columns):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x_m, y_m = transformer.transform(df['lon'].to_numpy(), df['lat'].to_numpy())
    df['x'] = x_m
    df['y'] = y_m
    df.drop(columns=['lat', 'lon'], inplace=True)
else:
    raise ValueError("Il CSV deve contenere colonne 'lat' e 'lon'.")

# ----------------- Orari: open_mean → feature cicliche ----------------------
if 'open_mean' in df.columns:
    if df['open_mean'].dtype == object:
        # Interpreta stringhe HH:MM
        times = pd.to_datetime(df['open_mean'], format='%H:%M', errors='coerce')
        df['open_mean'] = times.dt.hour * 60 + times.dt.minute
    theta = 2 * np.pi * df['open_mean'].fillna(0) / 1440
    df['open_sin'] = np.sin(theta)
    df['open_cos'] = np.cos(theta)
    df.drop(columns=['open_mean'], inplace=True)
else:
    df['open_sin'] = 0.0
    df['open_cos'] = 0.0

# ----------------------- Pipeline sklearn -----------------------------------
num_cols = ['x', 'y', 'open_sin', 'open_cos']
cat_cols = [c for c in df.columns if c not in num_cols and c not in {'uri', 'label'}]

numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])

categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

prep = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# ----------------------- Fit & transform ------------------------------------
X = prep.fit_transform(df)

# ----------------------- Persistenza ----------------------------------------
joblib.dump(prep, PIPE)

# Il CSV trasformato è solo a scopo di debug / ispezione: se X è sparse lo densifichiamo.
if hasattr(X, 'todense'):
    X_dense = X.todense()
else:
    X_dense = X

pd.DataFrame(np.asarray(X_dense)).to_csv(PREP, index=False)

print("✅  Pre-processing completato. File salvati:\n    • Dati:   {}\n    • Pipeline: {}".format(PREP, PIPE))