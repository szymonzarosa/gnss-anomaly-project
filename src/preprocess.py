"""
preprocess.py
-------------
Moduł przetwarzania wstępnego danych GNSS (ETL).

Realizuje kluczowe zadania przygotowujące surowe szeregi czasowe do analizy ML:
1. Synchronizacja czasowa wielu stacji (wspólny indeks).
2. Uzupełnianie małych luk w danych (interpolacja liniowa).
3. Dekompozycja szeregu: usunięcie trendu tektonicznego i sezonowości rocznej.
   Cel: uzyskanie czystych residuów (błędów/szumu), które są przedmiotem detekcji.
4. Normalizacja danych (Z-score) przy zachowaniu oryginalnych wartości fizycznych [mm].

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Konfiguracja ścieżek
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "model_input"

# Upewniamy się, że folder wyjściowy istnieje
MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_stations() -> dict[str, pd.DataFrame]:
    """
    Wczytuje wszystkie pliki CSV z folderu processed/ do słownika.

    Zwraca:
    -------
    dict: { 'NAZWA_STACJI': DataFrame z indeksem czasowym }
    """
    files = list(PROCESSED_DIR.glob("*.csv"))
    if not files:
        print("Brak plików w folderze data/processed. Uruchom najpierw data_loader.py.")
        return {}

    station_data = {}
    for f in files:
        # Parsowanie daty jest kluczowe dla analizy szeregów czasowych
        df = pd.read_csv(f, parse_dates=["date"])
        station_data[f.stem] = df.set_index("date").sort_index()
        print(f"Wczytano {f.stem}: {len(df)} pomiarów dziennych")

    return station_data


def remove_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa składowe geofizyczne (Trend + Sezonowość), pozostawiając Residua.

    Wykorzystuje model addytywny:
    Sygnał(t) = Trend(t) + Sezon(t) + Residuum(t)

    Zwraca:
    -------
    DataFrame zawierający tylko Residuum (szum + potencjalne anomalie).
    """
    df_clean = df.copy()
    components = ['east', 'north', 'up']

    # seasonal_decompose wymaga ciągłości danych (brak NaN).
    # Uzupełniamy małe luki (do 5 dni) interpolacją liniową.
    # Większe luki zostaną, co może spowodować błąd dekompozycji (obsłużony w try-except).
    df_filled = df_clean.interpolate(method='linear', limit=5)

    for col in components:
        if col not in df_filled.columns:
            continue

        try:
            # 1. Dekompozycja sezonowa (okres roczny = 365 dni)
            # extrapolate_trend='freq' pozwala wyznaczyć trend także na krańcach przedziału
            res = seasonal_decompose(
                df_filled[col].dropna(),
                model='additive',
                period=365,
                extrapolate_trend='freq'
            )

            # Zapisujemy same reszty (Residua).
            # Używamy reindex, aby dopasować wyniki do oryginalnego indeksu (z NaNami).
            df_clean[col] = res.resid.reindex(df_clean.index)

        except Exception:
            # Fallback: Jeśli szereg jest za krótki (< 2 lata) lub ma za dużo dziur,
            # seasonal_decompose rzuci błąd. Wtedy robimy tylko prosty detrend liniowy.
            print(f"Nie udało się wykonać pełnej dekompozycji dla {col}. Wykonuję prosty detrend.")

            vals = df_filled[col].dropna()
            if len(vals) > 10:
                # Dopasowanie wielomianu 1. stopnia (prosta y = ax + b)
                z = np.polyfit(range(len(vals)), vals, 1)
                p = np.poly1d(z)
                trend = p(range(len(vals)))

                # Odejmujemy trend od sygnału
                df_clean.loc[vals.index, col] = vals - trend

    return df_clean


def synchronize_and_clean(station_data: dict) -> pd.DataFrame:
    """
    Przetwarza każdą stację z osobna (usuwanie fizyki) i łączy je w jedną macierz.
    """
    processed_dfs = []

    print("\nRozpoczynam usuwanie trendu i sezonowości (Dekompozycja)...")

    for name, df in station_data.items():
        # Krok 1: Usunięcie fizyki
        df_resid = remove_physics(df)

        # Krok 2: Resampling do siatki dziennej (1D)
        # To zapewnia, że wszystkie stacje będą miały ten sam indeks czasowy
        df_resampled = df_resid.resample("1D").mean()

        # Krok 3: Dodanie prefiksu stacji (np. 'up' -> 'KRA1_up')
        df_resampled = df_resampled.add_prefix(f"{name}_")

        processed_dfs.append(df_resampled)

    # Krok 4: Złączenie (outer join) według daty
    combined = pd.concat(processed_dfs, axis=1)
    return combined


def preprocess_all() -> None:
    """
    Główna funkcja sterująca procesem ETL.
    """
    # 1. Wczytanie
    station_data = load_all_stations()
    if not station_data: return

    # 2. Przetwarzanie fizyczne
    combined = synchronize_and_clean(station_data)

    # 3. Konwersja jednostek (metry -> milimetry)
    # Isolation Forest działa lepiej na liczbach rzędu 1-10 niż 0.001
    cols_to_scale = [c for c in combined.columns if "sigma" not in c]
    combined[cols_to_scale] = combined[cols_to_scale] * 1000.0
    print("Przeliczono jednostki na milimetry [mm].")

    # 4. Normalizacja (Feature Scaling)
    # Tworzymy kopię danych znormalizowanych (Z-score) dla algorytmu ML.
    # Oryginalne kolumny (w mm) zostawiamy do wizualizacji.
    scaler = StandardScaler()

    # Wybieramy tylko dane (bez sigm) do normalizacji
    data_cols = [c for c in combined.columns if "sigma" not in c]

    # Wypełniamy NaN zerami (średnia) przed normalizacją
    combined_filled = combined.fillna(0)

    # Obliczamy Z-score: (x - mean) / std
    norm_values = scaler.fit_transform(combined_filled[data_cols])

    # Tworzymy nowe kolumny z sufiksem '_norm'
    norm_df = pd.DataFrame(
        norm_values,
        index=combined.index,
        columns=[f"{c}_norm" for c in data_cols]
    )

    # Łączymy wszystko w jeden duży plik wejściowy
    final_df = pd.concat([combined, norm_df], axis=1)

    # 5. Zapis wyniku
    out_path = MODEL_INPUT_DIR / "gnss_model_input.csv"
    final_df.to_csv(out_path)

    print(f"\nDane przygotowane: {out_path}")
    print("Plik zawiera kolumny fizyczne [mm] (do wykresów) i znormalizowane [_norm] (do AI).")


if __name__ == "__main__":
    preprocess_all()