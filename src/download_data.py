"""
download_data.py
----------------
Pobiera pliki GNSS (.tenv3) z serwera Nevada Geodetic Laboratory (UNR)
i zapisuje je do folderu `data/raw/`.

Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import requests
from pathlib import Path

BASE_URL = "https://geodesy.unr.edu/gps_timeseries/tenv3/plates/EU/"

# Lista kodów stacji GNSS, dla których chcemy pobrać pliki .tenv3
STATIONS = ["BOR1", "KRAW", "BOGI", "JOZE", "WROC",
            "LAMA", "KATO", "BYDG", "REDZ", "ZYWI"]  # <- Można dodać więcej stacji w tym miejscu

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_station(station_code: str):
    station = station_code.upper()

    # Tworzymy pełny adres URL do pliku, np.:
    # https://geodesy.unr.edu/gps_timeseries/tenv3/plates/EU/KRA1.EU.tenv3
    url = f"{BASE_URL}{station}.EU.tenv3"
    dest = RAW_DIR / f"{station}.tenv3"

    print(f"Pobieranie {url}")
    try:

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        if "Nr of data points" not in resp.text and len(resp.content) < 1000:
            print(f"BŁĄD: Pobrany plik wygląda na uszkodzony lub pusty.")
            return

        with open(dest, "wb") as f:
            f.write(resp.content)

        print(f"Zapisano: {dest}")

    except requests.exceptions.RequestException as e:
        print(f"Nie udało się pobrać {station}: {e}")


def download_all(stations: list[str]):
    for st in stations:
        download_station(st)


if __name__ == "__main__":
    print("Pobieranie danych GNSS z UNR (IGS20 / tenv3)")
    download_all(STATIONS)
    print("Zakończono pobieranie")