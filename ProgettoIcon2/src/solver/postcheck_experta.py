#!/usr/bin/env python3
"""RF11 – Experta post‑check.

Regole implementate:
1. ⚠️  Più di due POI consecutivi con lo stesso *type*.
2. ⚠️  Un tratto di cammino > 30 min fra due POI consecutivi.
3. ⚠️  POI di tipo "ArchaeologicalSite" visitato dopo le 16.

Input  : route_<city>.csv   (prodotto da astar_order.py)
Output : solo stdout (warning) – il file non viene modificato.
"""
from __future__ import annotations
import collections, collections.abc, sys, pathlib, pandas as pd
for _n in ("Mapping","MutableMapping","MutableSequence"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
from experta import *

CITY = sys.argv[1] if len(sys.argv) > 1 else "Rome"
ROUTE = pathlib.Path("data", f"route_{CITY.lower()}.csv")
if not ROUTE.exists():
    sys.exit("💥  Esegui prima astar_order.py per ottenere route_<city>.csv")

df = pd.read_csv(ROUTE)  # slot,label,uri,type,score,cum_walk_s

# ---------- build facts -----------------
class POIFact(Fact):
    """type, slot_idx, start_h, walk_to_next"""
    pass

fe = []
for i, row in df.iterrows():
    start_h = int(row['slot'][:2])
    walk = (df.loc[i+1,'cum_walk_s']-row['cum_walk_s']) if i < len(df)-1 else 0
    fe.append(POIFact(idx=i, type=row['type'], start_h=start_h, walk=walk))

# ---------- rules engine ---------------
class TourRules(KnowledgeEngine):
    @Rule(POIFact(type=MATCH.t, idx=MATCH.i1),
          POIFact(type=MATCH.t, idx=MATCH.i2),
          POIFact(type=MATCH.t, idx=MATCH.i3),
          TEST(lambda i1,i2,i3: i2==i1+1 and i3==i2+1))
    def triple_same_type(self, t, i1):
        print(f"⚠️  Tre POI consecutivi di tipo {t} a partire dallo slot {i1}.")

    @Rule(POIFact(walk=MATCH.w, idx=MATCH.i), TEST(lambda w: w>1800))
    def long_walk(self, w, i):
        print(f"⚠️  Tratto di cammino >30 min fra slot {i} e {i+1} ({w/60:.1f} min).")

    @Rule(POIFact(type='ArchaeologicalSite', start_h=MATCH.h))
    def archeo_late(self, h):
        if h>=16:
            print(f"⚠️  Visita a sito archeologico dopo le 16 (slot {h}:00).")

# ---------- run ------------------------
eng = TourRules()
eng.reset()
for f in fe:
    eng.declare(f)
eng.run()
print("✅  Post‑check Experta completato")
