"""
download_data.py
----------------
Moduł odpowiedzialny za pobieranie surowych szeregów czasowych GNSS.

Skrypt łączy się z serwerem Nevada Geodetic Laboratory (NGL) i pobiera pliki
w formacie `.tenv3` dla zdefiniowanej listy stacji. Dane pochodzą z rozwiązań
w układzie odniesienia 'EU Plate Fixed' (płyta europejska), co eliminuje
główny trend tektoniczny.

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import requests
from pathlib import Path
import time

# Adres bazowy do produktów NGL (Nevada Geodetic Laboratory).
# Używamy katalogu 'plates/EU', aby pobrać współrzędne zredukowane o ruch płyty eurazjatyckiej.
BASE_URL = "https://geodesy.unr.edu/gps_timeseries/tenv3/plates/EU/"

# Lista kodów stacji GNSS (4-znakowe kody IGS/EPN) do analizy.
STATIONS = [
    "BOR1", "KRAW", "BOGI", "JOZE", "WROC",
    "LAMA", "KATO", "BYDG", "REDZ", "ZYWI"
]

# Konfiguracja ścieżek systemowych
# PROJECT_ROOT wskazuje na główny folder projektu (jeden poziom wyżej niż folder src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Upewniamy się, że katalog docelowy istnieje przed próbą zapisu
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_station(station_code: str) -> None:
    """
    Pobiera plik .tenv3 dla pojedynczej stacji i zapisuje go na dysku.

    Parametry:
    ----------
    station_code : str
        4-znakowy kod stacji (np. 'KRA1').
    """
    station = station_code.upper()

    # Konstrukcja URL zgodnie ze schematem nazewnictwa NGL
    # Przykład: KRA1.EU.tenv3
    filename = f"{station}.EU.tenv3"
    url = f"{BASE_URL}{filename}"

    # Ścieżka docelowa na dysku lokalnym
    dest = RAW_DIR / f"{station}.tenv3"

    # Nagłówki HTTP, aby skrypt przedstawiał się jako przeglądarka/klient (unikamy blokad anty-botowych)
    headers = {
        "User-Agent": "GNSS-Student-Project/1.0 (Educational Purpose)"
    }

    print(f"Pobieranie: {station} ...", end=" ")

    try:
        # Wysłanie żądania GET z timeoutem 30s (na wypadek wolnego łącza)
        resp = requests.get(url, headers=headers, timeout=30)

        # Sprawdzenie statusu HTTP (rzuca wyjątek, jeśli kod != 200)
        resp.raise_for_status()

        # --- WALIDACJA ZAWARTOŚCI ---
        # Serwer NGL czasem zwraca status 200 OK, ale w treści przesyła stronę HTML z błędem
        # "File not found". Sprawdzamy, czy plik wygląda na poprawny plik tekstowy z danymi.
        if "Nr of data points" not in resp.text and len(resp.content) < 1000:
            print(f"BŁĄD: Pobrany plik wygląda na uszkodzony lub pusty (serwer zwrócił błąd).")
            return

        # Zapis w trybie binarnym (wb), aby zachować oryginalne kodowanie znaków
        with open(dest, "wb") as f:
            f.write(resp.content)

        print(f"Zapisano: {dest.name}")

    except requests.exceptions.RequestException as e:
        # Obsługa błędów sieciowych (brak internetu, timeout, błąd 404)
        print(f"\nNie udało się pobrać {station}. Szczegóły: {e}")


def download_all(stations: list[str]) -> None:
    """
    Iteruje przez listę stacji i uruchamia pobieranie dla każdej z nich.
    """
    print(f"=== Rozpoczynam pobieranie danych dla {len(stations)} stacji ===")

    for st in stations:
        download_station(st)
        # Krótka pauza, żeby nie bombardować serwera naukowego zbyt dużą liczbą zapytań
        time.sleep(0.5)

    print("=== Zakończono pobieranie ===")


if __name__ == "__main__":
    # Ten blok wykonuje się tylko przy bezpośrednim uruchomieniu skryptu
    download_all(STATIONS)