import sys
import re
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

# ------------------------------- CONFIG ------------------------------------
# Definiamo la cartella data/ alla radice del progetto
base = Path.cwd() / "data"

# Leggiamo la città da CLI (default "Rome")
CITY = sys.argv[1] if len(sys.argv) > 1 else "Rome"
city_lower = CITY.lower()

# Percorsi di input e output
INFILE  = base / f"poi_{city_lower}.csv"
OUTFILE = base / f"poi_{city_lower}_hours.csv"

# Orari di fallback
FALLBACK_OPEN, FALLBACK_CLOSE = "09:00", "18:00"

# Header personalizzato per le richieste HTTP   
HEADERS = {"User-Agent": "SmartTour-CSP/1.0 (stefano@studenti.uniba.it)"}
# ---------------------------------------------------------------------------

def guess_hours_from_infobox(html: str):
    """
    Cerca due orari hh:mm (o h.mm) entro 30 caratteri di distanza nel blocco HTML.
    Restituisce (open_time, close_time) o fallback se non trova nulla.
    """
    pattern = r"([0-2]?\d[:.][0-5]\d).{0,30}?([0-2]?\d[:.][0-5]\d)"
    m = re.search(pattern, html)
    if m:
        # Uniformiamo il formato sostituendo eventuali punti con due punti
        open_time  = m.group(1).replace(".", ":")
        close_time = m.group(2).replace(".", ":")
        return open_time, close_time
    return FALLBACK_OPEN, FALLBACK_CLOSE

def fetch_hours(title: str):
    """
    Richiama l'endpoint REST di Wikipedia per la pagina 'title',
    prende il codice HTML e ne estrae gli orari tramite regex.
    """
    url = f"https://it.wikipedia.org/api/rest_v1/page/html/{title}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 404:
            # Pagina non trovata: fallback
            return FALLBACK_OPEN, FALLBACK_CLOSE
        return guess_hours_from_infobox(r.text)
    except requests.exceptions.RequestException:
        # Errori di rete o timeout: fallback
        return FALLBACK_OPEN, FALLBACK_CLOSE

def main():
    # Controllo che il file di input esista
    if not INFILE.exists():
        print(f"❌ File non trovato: {INFILE}")
        sys.exit(1)

    # Carichiamo il CSV base dei POI
    df = pd.read_csv(INFILE)

    opens, closes = [], []
    total = len(df)
    print(f"⏳ Recupero orari per {total} POI…")

    # Iteriamo sulle label, trasformando in titolo URL-friendly
    for label in tqdm(df["label"], desc="Fetching hours", ncols=80):
        title = label.replace(" ", "_")
        open_h, close_h = fetch_hours(title)
        opens.append(open_h)
        closes.append(close_h)
        # Pausa per non sovraccaricare l'API
        time.sleep(0.3)

    # Aggiungiamo le colonne al DataFrame
    df["open"]  = opens
    df["close"] = closes

    # Salviamo il CSV arricchito
    df.to_csv(OUTFILE, index=False, encoding="utf-8")
    print(f"✅ Salvato file con orari → {OUTFILE.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()
