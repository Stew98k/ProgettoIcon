#!/usr/bin/env python3
"""Harvest POI da DBpedia con debug opzionale.

Esempio d'uso:
    python harvest_poi.py Rome --lang it --bbox --delta 0.08 --debug
"""
from __future__ import annotations

from pathlib import Path
import sys, argparse, textwrap, logging
from typing import Tuple

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import requests

try:
    from geopy.geocoders import Nominatim
except ImportError:  # fallback neutro
    Nominatim = None

# ------------------------- CLI ---------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Scarica musei, chiese, parchi, monumenti ecc. da DBpedia."
)
parser.add_argument("city", help="Nome risorsa DBpedia (es. Rome, Florence, Bari)")
parser.add_argument("--lang",  default="en", help="Lingua delle label (en/it/…)")
parser.add_argument("--limit", type=int, default=500, help="Limite righe SPARQL")
parser.add_argument("--bbox",  action="store_true", help="Applica filtro bounding-box")
parser.add_argument("--delta", type=float, default=0.10, help="Ampiezza bbox in gradi (default 0.10)")
parser.add_argument("--debug", action="store_true", help="Stampa query e diagnostica")
args = parser.parse_args()

CITY, LANG, LIMIT, DEBUG = args.city, args.lang, args.limit, args.debug

# ------------------------- Logging -----------------------------------------
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(levelname)s│%(message)s")
log = logging.getLogger("harvest")

# ------------------------- Tipologie POI -----------------------------------
POI_TYPES = [
    "dbo:Museum", "dbo:Church", "dbo:Park", "dbo:HistoricBuilding",
    "dbo:Monument", "dbo:Gallery", "dbo:Theatre", "dbo:ArchaeologicalSite",
    "dbo:Bridge"
]

LOCAL_PROPS = ["dbo:location", "dbo:city", "dbo:municipality", "dbo:isPartOf"]

# ------------------------- Utility -----------------------------------------
HEADERS = {"User-Agent": "SmartTour-CSP/1.0 (harvester)"}


def get_bbox(city: str, delta_deg: float) -> Tuple[float, float, float, float]:
    """Restituisce (lat_min, lat_max, lon_min, lon_max) oppure uscita con sys.exit"""
    if Nominatim is None:
        sys.exit("💥  geopy non installato: pip install geopy  oppure ometti --bbox")
    geo = Nominatim(user_agent="smarttour")
    loc = geo.geocode(f"{city}, Italy")
    if loc is None:
        sys.exit(f"💥  Nominatim non trova la città {city}")
    lat, lon = loc.latitude, loc.longitude
    log.debug("Centro geocodificato: %.5f, %.5f", lat, lon)
    return lat - delta_deg, lat + delta_deg, lon - delta_deg, lon + delta_deg


def build_query() -> str:
    type_list = ", ".join(POI_TYPES)
    union_props = " UNION\n        ".join(f"{{ ?poi {p} dbr:{CITY} }}" for p in LOCAL_PROPS)

    bbox_filter = ""
    if args.bbox:
        lat_min, lat_max, lon_min, lon_max = get_bbox(CITY, args.delta)
        bbox_filter = textwrap.dedent(f"""
            FILTER (?lat > {lat_min:.5f} && ?lat < {lat_max:.5f} &&
                    ?lon > {lon_min:.5f} && ?lon < {lon_max:.5f})
        """)
    query = textwrap.dedent(f"""
        PREFIX dbo:  <http://dbpedia.org/ontology/>
        PREFIX geo:  <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?poi ?label ?lat ?lon ?type
        WHERE {{
            {union_props}
            ?poi rdf:type ?type ;
                 geo:lat  ?lat ;
                 geo:long ?lon ;
                 rdfs:label ?label .
            FILTER (?type IN ({type_list}))
            FILTER (lang(?label) = '{LANG}')
            {bbox_filter}
        }}
        LIMIT {LIMIT}
    """)
    if DEBUG:
        log.debug("\n======= SPARQL QUERY =======\n%s\n============================", query)
    return query


# ------------------------- Main --------------------------------------------

def run_sparql(query: str):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert()
    except Exception as exc:
        log.error("Errore SPARQL: %s", exc)
        sys.exit(1)


def main():
    log.info("Scarico POI per %s (lang=%s)…", CITY, LANG)
    query = build_query()
    raw = run_sparql(query)

    bindings = raw.get("results", {}).get("bindings", [])
    log.info("Risultati ricevuti: %d", len(bindings))

    rows = [
        dict(
            uri=b["poi"]["value"],
            label=b["label"]["value"],
            lat=float(b["lat"]["value"]),
            lon=float(b["lon"]["value"]),
            type=b["type"]["value"].split("/")[-1],
        )
        for b in bindings
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset="uri").reset_index(drop=True)

    outdir = Path.cwd() / "data"
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"poi_{CITY.lower()}.csv"
    df.to_csv(outfile, index=False, encoding="utf-8")
    log.info("Salvato %d POI in %s", len(df), outfile.relative_to(Path.cwd()))


if __name__ == "__main__":
    main()
