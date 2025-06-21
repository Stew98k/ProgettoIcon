#!/usr/bin/env python3
"""A* ordering (RF10 – parte B)

• Usa la matrice NumPy distance_matrix_<city>.npy (float32, np.inf)
• Ri-ordina il tour_<city>.csv minimizzando il cammino a piedi.
• URI mappati alle righe/colonne corrette tramite poi_<city>_cluster.csv.
• Salta POI isolati (tutti archi infiniti) per evitare percorsi impossibili.
"""
from __future__ import annotations

import sys, heapq
from pathlib import Path
import numpy as np
import pandas as pd

CITY = sys.argv[1] if len(sys.argv) > 1 else "Rome"
DATA = Path(__file__).resolve().parents[2] / "data"
TOUR_IN   = DATA / f"tour_{CITY.lower()}.csv"
MAT_FILE  = DATA / f"distance_matrix_{CITY.lower()}.npy"
CLUST_IN  = DATA / f"poi_{CITY.lower()}_cluster.csv"
ROUTE_OUT = DATA / f"route_{CITY.lower()}.csv"

# ---------- checks ----------
for f,p in [("tour",TOUR_IN),("matrix",MAT_FILE),("cluster",CLUST_IN)]:
    if not p.exists():
        sys.exit(f"💥  File {f} mancante: {p}")

# ---------- load ------------
D = np.load(MAT_FILE)
cluster_df = pd.read_csv(CLUST_IN)
uri2idx = dict(zip(cluster_df['uri'], cluster_df.index))

orig_tour = pd.read_csv(TOUR_IN)
# filtra URI non presenti nella matrice (es. rari mismatch)
orig_tour = orig_tour[orig_tour['uri'].isin(uri2idx)].reset_index(drop=True)

idx_list = [uri2idx[u] for u in orig_tour['uri']]
N = len(idx_list)
if N == 0:
    sys.exit("💥  Nessun POI del tour presente nella matrice.")

# ---------- costo -----------
def cost(a:int,b:int)->float:
    i,j = idx_list[a], idx_list[b]
    val = D[i,j]
    return float(val) if np.isfinite(val) else None  # None = impassable

# ---------- pre-check isolate nodes ----------
reachable = [any(np.isfinite(D[idx_list[i], idx_list[j]]) for j in range(N) if j!=i)
             for i in range(N)]
if not all(reachable):
    drop = [i for i,r in enumerate(reachable) if not r]
    print("⚠️  POI isolati rimossi:", len(drop))
    keep = [i for i in range(N) if i not in drop]
    orig_tour = orig_tour.iloc[keep].reset_index(drop=True)
    idx_list  = [idx_list[i] for i in keep]
    N = len(idx_list)
    if N<2:
        sys.exit("💥  Troppi isolati, impossibile ordinare.")

# ---------- heuristic (min finite edge) -------
finite_edges = [[cost(i,j) for j in range(N) if i!=j and cost(i,j) is not None]
                for i in range(N)]
min_out = [min(ed) for ed in finite_edges]

# ---------- A* -------------------------------
goal_mask = (1<<N)-1
pq=[(0,0,1,0,[0])]  # (f,g,mask,last,path)
best={}
while pq:
    f,g,mask,last,path = heapq.heappop(pq)
    if mask==goal_mask:
        best_path,tot=g,path; break
    if best.get((mask,last),1e18)<=g:continue
    best[(mask,last)]=g
    for nxt in range(N):
        if mask&(1<<nxt):continue
        c = cost(last,nxt)
        if c is None: continue
        g2=g+c; h=min_out[nxt]
        heapq.heappush(pq,(g2+h,g2,mask|(1<<nxt),nxt,path+[nxt]))
else:
    sys.exit("💥  Nessun percorso percorribile trovato (grafo disconnesso).")

# ---------- export ---------------------------
ordered = orig_tour.iloc[path].reset_index(drop=True)
cum=[0]
for a,b in zip(path[:-1],path[1:]):
    cum.append(cum[-1]+cost(a,b))
ordered['cum_walk_s']=cum
ordered.to_csv(ROUTE_OUT,index=False,encoding='utf-8')
print(f"✅  Route ottimizzata → {ROUTE_OUT.relative_to(Path.cwd())}")
print(f"   Tempo totale di cammino: {cum[-1]/60:.0f} min")