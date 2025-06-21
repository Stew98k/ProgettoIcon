#!/usr/bin/env python3
"""
Seleziona l’insieme ottimo di POI (RF10) con OR-Tools CP-SAT
– legge i punteggi personalizzati (poi_<city>_scored.csv)
– accetta POI senza orari: li considera “sempre aperti”
– opzionale: porta con sé il cluster (se presente) – utile nei post-check
"""

import sys
from pathlib import Path
import pandas as pd
from ortools.sat.python import cp_model

# ─────────────────────────── parametri base
CITY       = sys.argv[1] if len(sys.argv) > 1 else "Rome"
START_H, END_H = 9, 18             # slot orari (9-10, 10-11, … 17-18)
SOLVER_TL  = 10                    # secondi di time-limit

DATA = Path(__file__).resolve().parents[2] / "data"
POI  = pd.read_csv(DATA / f"poi_{CITY.lower()}_scored.csv")   # ← nuovo file

# 1) aggiungi cluster se serve (facoltativo – commenta se non ti occorre)
cluster_file = DATA / f"poi_{CITY.lower()}_cluster.csv"
if cluster_file.exists():
    clusters = pd.read_csv(cluster_file)[["uri", "cluster"]]
    POI = POI.merge(clusters, on="uri", how="left")

# 2) gestisci eventuali colonne open / close mancanti
if "open" not in POI.columns:
    POI["open"]  = "00:00"
    POI["close"] = "24:00"
POI["open"]  = POI["open"].fillna("00:00")
POI["close"] = POI["close"].fillna("24:00")

# ─────────────────────────── modello CP-SAT
SLOTS = list(range(START_H, END_H))            # 9…17 inclusi (9 slot)
model = cp_model.CpModel()
sel   = {(s, p): model.NewBoolVar(f"x_{s}_{p}")
         for s in SLOTS for p in POI.index}

# 3) vincoli: un POI per slot, un solo slot per POI
for s in SLOTS:
    model.Add(sum(sel[s, p] for p in POI.index) <= 1)
for p in POI.index:
    model.Add(sum(sel[s, p] for s in SLOTS) <= 1)

# 3-bis) vieta tre POI consecutivi dello stesso 'type'
for t in POI['type'].unique():
    idx_same = [p for p in POI.index if POI.loc[p, 'type'] == t]
    # scorre finestre di 3 slot (0-1-2, 1-2-3, …)
    for s0, s1, s2 in zip(SLOTS, SLOTS[1:], SLOTS[2:]):
        model.Add(
            sum(sel[s, p] for s in (s0, s1, s2) for p in idx_same) <= 2
        )

# 4) vincolo di apertura/chiusura
for s in SLOTS:
    for p, row in POI.iterrows():
        o = int(str(row["open"]).split(":")[0])     # ora di apertura
        c = int(str(row["close"]).split(":")[0])    # ora di chiusura
        if not (o <= s < c and s + 1 <= c):
            model.Add(sel[s, p] == 0)

# 5) obiettivo: massimizzare la somma dei punteggi
model.Maximize(sum(int(row["score"] * 100) * sel[s, p]
                   for s in SLOTS for p, row in POI.iterrows()))

# ─────────────────────────── solve
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = SOLVER_TL
status = solver.Solve(model)
if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    sys.exit("⚠️  Nessuna soluzione trovata")

# ─────────────────────────── export tour
rows = []
for s in SLOTS:
    chosen = [p for p in POI.index if solver.BooleanValue(sel[s, p])]
    if not chosen:
        continue
    p = chosen[0]
    rows.append({
        "slot":   f"{s:02d}:00–{s+1:02d}:00",
        "label":  POI.loc[p, "label"],
        "uri":    POI.loc[p, "uri"],
        "idx":    int(p),
        "type":   POI.loc[p, "type"],
        "score":  round(POI.loc[p, "score"], 3),
        "cluster": POI.loc[p, "cluster"] if "cluster" in POI.columns else None
    })

out = DATA / f"tour_{CITY.lower()}.csv"
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
print("✅  tour salvato →", out.relative_to(Path.cwd()))
