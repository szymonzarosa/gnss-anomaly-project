"""
data_loader.py
----------------
Odpowiada za wczytywanie i parsowanie danych GNSS.
Funkcje pomocnicze do wczytywania i przetwarzania danych GNSS (.tenv3)
z serwera Nevada Geodetic Laboratory (UNR).
    
Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import pandas as pd
from pathlib import Path

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_tenv3_file(file_path: Path) -> pd.DataFrame:
    """
    Wczytuje plik .tenv3 używając Pandas.
    Naprawia nazwy kolumn i przelicza MJD na Datetime.
    """
    # Definiujemy, które kolumny chcemy wczytać (bazując na pliku):

    try:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            skiprows=1,
            usecols=[3, 8, 10, 12, 14, 15, 16],
            names=['mjd', 'east', 'north', 'up', 'sigmaE', 'sigmaN', 'sigmaU'],
            on_bad_lines='skip'
        )

        # Konwersja MJD na Datetime (MJD 0 = 1858-11-17)
        # Standard astronomiczny
        df['date'] = pd.to_datetime(df['mjd'], unit='D', origin=pd.Timestamp('1858-11-17'))

        # Ustawiamy datę jako indeks
        df = df.set_index('date').sort_index()
        df = df.drop(columns=['mjd'])

        return df

    except Exception as e:
        print(f"Błąd przy wczytywaniu {file_path.name}: {e}")
        return pd.DataFrame()


def process_raw_to_csv():
    """
    Przetwarza wszystkie pliki .tenv3 z data/raw i zapisuje czyste CSV w data/processed.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = list(RAW_DIR.glob("*.tenv3"))

    if not files:
        print("Brak plików .tenv3 w data/raw/. Uruchom najpierw download_data.py")
        return

    print(f"Rozpoczynam przetwarzanie {len(files)} plików...")

    for file_path in files:
        df = load_tenv3_file(file_path)

        if not df.empty:
            # Zapis jako CSV
            output_path = PROCESSED_DIR / f"{file_path.stem}.csv"
            df.to_csv(output_path)
            print(f"Przetworzono: {file_path.name} -> {output_path.name} (Dni: {len(df)})")
        else:
            print(f"Pusty lub błędny plik: {file_path.name}")


if __name__ == "__main__":
    process_raw_to_csv()
    print("Zakończono")
