"""
data_loader.py
----------------
Moduł odpowiedzialny za transformację surowych danych GNSS do formatu analitycznego.

Główne zadania:
1. Wczytanie specyficznego formatu .tenv3 (rozdzielany spacjami).
2. Ekstrakcja kluczowych kolumn (MJD, East, North, Up, Sigmas).
3. Konwersja czasu astronomicznego (MJD) na datę kalendarzową (datetime).
4. Zapis ustandaryzowanych plików CSV gotowych do dalszej obróbki.

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import pandas as pd
from pathlib import Path

# KONFIGURACJA STRUKTURY FOLDERÓW
# PROJECT_ROOT wskazuje na główny folder projektu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Upewniamy się, że folder wyjściowy istnieje
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_tenv3_file(file_path: Path) -> pd.DataFrame:
    """
    Wczytuje pojedynczy plik .tenv3, selekcjonuje kolumny i konwertuje czas.

    Parametry:
    ----------
    file_path : Path
        Ścieżka do surowego pliku .tenv3.

    Zwraca:
    -------
    pd.DataFrame
        DataFrame z indeksem czasowym (Datetime) i kolumnami [east, north, up, sigmaE...].
        Zwraca pusty DataFrame w przypadku błędu.
    """
    try:
        # Wczytywanie pliku o nieregularnych odstępach (sep='\s+')
        # Format NGL tenv3 nie ma standardowego nagłówka CSV, więc pomijamy 1. linię
        # i nadajemy nazwy ręcznie.

        # Indeksy kolumn w formacie NGL (liczone od 0):
        # 3: MJD (Modified Julian Date)
        # 8: East (m), 10: North (m), 12: Up (m) -> Wersja "clean" (bez _e0)
        # 14: Sigma East, 15: Sigma North, 16: Sigma Up
        df = pd.read_csv(
            file_path,
            sep=r'\s+',       # Separator: dowolna liczba spacji/tabulacji
            header=None,      # Brak nagłówka w pliku
            skiprows=1,       # Pomijamy pierwszą linię tekstową
            usecols=[3, 8, 10, 12, 14, 15, 16], # Wybieramy tylko to, co nas interesuje
            names=['mjd', 'east', 'north', 'up', 'sigmaE', 'sigmaN', 'sigmaU'],
            on_bad_lines='skip' # Ignorujemy uszkodzone linie
        )

        # Konwersja MJD (Modified Julian Date) na Datetime.
        # MJD to liczba dni, które upłynęły od północy 17 listopada 1858 roku.
        # Jest to standardowy format czasu w astronomii i geodezji satelitarnej.
        df['date'] = pd.to_datetime(df['mjd'], unit='D', origin=pd.Timestamp('1858-11-17'))

        # Ustawiamy datę jako indeks (kluczowe dla szeregów czasowych)
        df = df.set_index('date').sort_index()

        # Kolumna MJD nie jest już potrzebna po konwersji
        df = df.drop(columns=['mjd'])

        return df

    except Exception as e:
        print(f"Błąd przy wczytywaniu {file_path.name}: {e}")
        return pd.DataFrame()


def process_raw_to_csv() -> None:
    """
    Funkcja główna: przetwarza wszystkie pliki .tenv3 z folderu raw/
    i zapisuje czyste pliki .csv w folderze processed/.
    """
    # Znajdujemy wszystkie pliki .tenv3
    files = list(RAW_DIR.glob("*.tenv3"))

    if not files:
        print("Brak plików .tenv3 w data/raw/. Uruchom najpierw download_data.py")
        return

    print(f"=== Rozpoczynam parsowanie {len(files)} plików .tenv3 ===")

    count_success = 0
    for file_path in files:
        df = load_tenv3_file(file_path)

        if not df.empty:
            # Zapisujemy jako standardowy CSV (łatwiejszy do odczytu dla innych narzędzi)
            output_path = PROCESSED_DIR / f"{file_path.stem}.csv"
            df.to_csv(output_path)

            print(f"Przetworzono: {file_path.name} -> {output_path.name} (Dni: {len(df)})")
            count_success += 1
        else:
            print(f"Pusty lub błędny plik: {file_path.name}")

    print(f"=== Zakończono. Poprawnie przetworzono: {count_success}/{len(files)} plików ===")


if __name__ == "__main__":
    # Ten blok wykonuje się tylko przy bezpośrednim uruchomieniu skryptu
    process_raw_to_csv()