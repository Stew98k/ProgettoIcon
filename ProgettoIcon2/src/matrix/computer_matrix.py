#!/usr/bin/env python3
"""Calcola (o aggiorna) la matrice dei tempi di percorrenza fra i POI.

• legge poi_<city>_cluster.csv
• chiama OSRM profilo foot (async se aiohttp presente)
• sostituisce gli archi mancanti (np.inf) con una stima Haversine a 5 km/h
• salva distance_matrix_<city>.npy   (float32, senza inf)
"""
from __future__ import annotations

import argparse, asyncio, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests, math

try:
    import aiohttp, async_timeout
    ASYNC=True
except ImportError:
    ASYNC=False

OSRM_URL="https://router.project-osrm.org/table/v1/{profile}/"
MAX_BATCH=100
HEADERS={"User-Agent":"SmartTour-Matrix/2.0"}

# ─── CLI ───────────────────────────────────────────────────────────
par=argparse.ArgumentParser(description="Genera matrice tempi fra POI")
par.add_argument("city")
par.add_argument("--profile",choices=["foot","bike","car"],default="foot")
par.add_argument("--rebuild",action="store_true")
args=par.parse_args(); CITY=args.city.lower(); PROF=args.profile

CSV=Path(f"data/poi_{CITY}_cluster.csv"); NPY=Path(f"data/distance_matrix_{CITY}.npy")
if not CSV.exists(): sys.exit("💥  Prima esegui clustering")
if NPY.exists() and not args.rebuild:
    print("✅  Matrice già presente – usa --rebuild per rigenerare"); sys.exit(0)

# ─── util ──────────────────────────────────────────────────────────
chunk=lambda it,s: (it[i:i+s] for i in range(0,len(it),s))
coords2str=lambda c: ";".join(f"{lon},{lat}" for lat,lon in c)

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0
    p1,p2=np.radians(lat1),np.radians(lat2)
    dphi=p2-p1; dlamb=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlamb/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ─── fetch helpers ────────────────────────────────────────────────
async def fetch(session,url):
    from async_timeout import timeout
    async with timeout(60):
        async with session.get(url,headers=HEADERS) as r:
            if r.status!=200: raise RuntimeError(r.status)
            return await r.json()

aSync=lambda crd: asyncio.run(build_async(crd)) if ASYNC else build_sync(crd)

def build_sync(crd):
    N=len(crd); M=np.full((N,N),np.inf,float)
    for batch in chunk(list(range(N)),MAX_BATCH):
        subset=[crd[i] for i in batch]
        url=OSRM_URL.format(profile=PROF)+coords2str(subset)
        r=requests.get(url,headers=HEADERS,timeout=60); r.raise_for_status()
        dur=r.json()["durations"]
        for i, row in enumerate(dur):
            for j,val in enumerate(row):
                if val is not None: M[batch[i],batch[j]]=val
    return M

async def build_async(crd):
    N=len(crd); M=np.full((N,N),np.inf,float); tasks=[]
    async with aiohttp.ClientSession() as sess:
        for batch in chunk(list(range(N)),MAX_BATCH):
            subset=[crd[i] for i in batch]
            url=OSRM_URL.format(profile=PROF)+coords2str(subset)
            tasks.append(fetch(sess,url))
        res=await asyncio.gather(*tasks)
    base=0
    for part in res:
        for i,row in enumerate(part["durations"]):
            for j,val in enumerate(row):
                if val is not None: M[base+i,base+j]=val
        base+=len(part["durations"])
    return M

# ─── main ─────────────────────────────────────────────────────────
print(f"🔄  Carico {CSV} …")
df=pd.read_csv(CSV); coords=list(zip(df.lat,df.lon)); N=len(coords)
print(f"→ {N} POI – costruzione matrice {N}×{N} con profilo {PROF} …")
mat=aSync(coords)

# ─── fallback per archi inf ───────────────────────────────────────
mask=np.isinf(mat)
if mask.any():
    print(f"ℹ️  {mask.sum()} archi mancanti – uso fallback Haversine 5 km/h")
    lat=df.lat.to_numpy(); lon=df.lon.to_numpy()
    i_idx,j_idx=np.where(mask)
    for i,j in zip(i_idx,j_idx):
        dist_km=haversine_km(lat[i],lon[i],lat[j],lon[j])
        mat[i,j]=(dist_km*1000)/1.388  # sec @5 km/h

np.save(NPY,mat.astype("float32"))
print(f"✅  Salvato {NPY}  (shape {mat.shape}, inf rimasti {np.isinf(mat).sum()})")