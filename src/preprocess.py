"""
preprocess.py
----------------
Przygotowanie danych GNSS do detekcji anomalii przy użyciu Isolation Forest.

- Synchronizacja stacji
- Filtracja szumów
- Interpolacja braków
- Normalizacja
- Eksport do data/model_input/

Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "model_input"

MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_stations():

    files = list(PROCESSED_DIR.glob("*.csv"))
    if not files:
        print("Brak plików w folderze data/processed.")
        return {}

    station_data = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        station_data[f.stem] = df.set_index("date").sort_index()
        print(f"Wczytano {f.stem}: {len(df)} wierszy")
    return station_data


def remove_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kluczowa funkcja.
    Usuwa naturalny trend liniowy i sezonowość roczną, zostawiając tylko reszty.
    """
    df_clean = df.copy()
    components = ['east', 'north', 'up']

    # Najpierw uzupełniamy małe braki danych
    # Limit 5 dni - większe braki zostawiamy (będą obsłużone później)
    df_filled = df_clean.interpolate(method='linear', limit=5)

    for col in components:
        if col not in df_filled.columns:
            continue

        try:
            # 1. Sezonowość (okres 365 dni)
            # Model addytywny: Wynik = Trend + Sezon + Reszta
            res = seasonal_decompose(df_filled[col].dropna(), model='additive', period=365, extrapolate_trend='freq')

            # Nadpisujemy kolumnę samymi resztami
            # Używamy reindex, żeby dopasować do oryginalnego indeksu
            df_clean[col] = res.resid.reindex(df_clean.index)

        except Exception as e:
            # Jeśli szereg jest za krótki (< 2 lata), seasonal_decompose wypisze błąd.
            # Wtedy robimy tylko prosty detrend liniowy (wielomian 1. stopnia)
            print(f"Nie udało się usunąć sezonowości dla {col}. Robię prosty detrend.")
            vals = df_filled[col].dropna()
            if len(vals) > 10:
                z = np.polyfit(range(len(vals)), vals, 1)
                p = np.poly1d(z)
                trend = p(range(len(vals)))
                df_clean.loc[vals.index, col] = vals - trend

    return df_clean


def synchronize_and_clean(station_data: dict) -> pd.DataFrame:
    """
    Synchronizuje stacje, usuwa fizykę (trend/sezon) i łączy w jeden DataFrame.
    """
    processed_dfs = []

    print("Rozpoczynam usuwanie trendu i sezonowości...")

    for name, df in station_data.items():
        # 1. Najpierw usuwamy fizykę (dla każdej stacji osobno!)
        # Robimy to PRZED łączeniem, żeby algorytm widział ciągłość konkretnej stacji
        df_resid = remove_physics(df)

        # 2. Resampling do 1 dnia (na wypadek braków)
        df_resampled = df_resid.resample("1D").mean()

        # 3. Zmieniamy nazwy kolumn (np. up -> KRA1_up)
        df_resampled = df_resampled.add_prefix(f"{name}_")

        processed_dfs.append(df_resampled)

    # 4. Łączymy wszystko w jedną wielką macierz
    combined = pd.concat(processed_dfs, axis=1)
    return combined


def filter_noise(df: pd.DataFrame, window=3) -> pd.DataFrame:
    """Filtracja szumów"""
    return df.rolling(window, min_periods=1, center=True).median()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizacja"""
    scaler = StandardScaler()
    df_filled = df.fillna(0)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df.columns, index=df.index)
    return df_scaled


def preprocess_all():
    station_data = load_all_stations()
    if not station_data: return

    # 2. Fizyka + Synchronizacja (Wartości w METRACH)
    combined = synchronize_and_clean(station_data)

    # 3. Konwersja na MILIMETRY
    cols_to_scale = [c for c in combined.columns if not "sigma" in c]
    combined[cols_to_scale] = combined[cols_to_scale] * 1000.0  # m -> mm
    print("Przeliczono jednostki na milimetry [mm].")

    # 4. Tworzenie wersji znormalizowanej
    scaler = StandardScaler()
    data_cols = [c for c in combined.columns if "sigma" not in c]
    combined_filled = combined.fillna(0)
    norm_values = scaler.fit_transform(combined_filled[data_cols])
    norm_df = pd.DataFrame(norm_values, index=combined.index, columns=[f"{c}_norm" for c in data_cols])

    # Łączymy: [Dane w mm] + [Sigm-y] + [Dane znormalizowane]
    final_df = pd.concat([combined, norm_df], axis=1)

    out_path = MODEL_INPUT_DIR / "gnss_model_input.csv"
    final_df.to_csv(out_path)
    print(f"Gotowe: {out_path}")
    print("Zawiera kolumny w [mm] i kolumny _norm.")


if __name__ == "__main__":
    preprocess_all()
