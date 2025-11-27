"""
detect_anomalies.py
-------------------
Moduł uczenia maszynowego (ML) odpowiedzialny za detekcję anomalii.
Trenuje model TYLKO na aktywnych dniach stacji.

Sercem tego modułu jest algorytm Isolation Forest (Las Izolacji).
Ten skrypt analizuje wektor 3D (East, North, Up) jednocześnie.

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib

# KONFIGURACJA ŚCIEŻEK
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_INPUT_FILE = PROJECT_ROOT / "data" / "model_input" / "gnss_model_input.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Upewniamy się, że foldery wynikowe istnieją
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def detect_anomalies_per_station(contamination: float = 0.004) -> None:
    """
    Główna pętla detekcji. Trenuje osobny model dla każdej stacji GNSS,
    biorąc pod uwagę tylko okresy aktywności stacji (ignoruje wypełnione zera).
    """
    if not MODEL_INPUT_FILE.exists():
        print(f"Brak pliku wejściowego: {MODEL_INPUT_FILE}")
        print("Uruchom najpierw skrypt 'preprocess.py'.")
        return

    # Wczytanie danych
    df = pd.read_csv(MODEL_INPUT_FILE, parse_dates=["date"], index_col="date")
    print(f"Wczytano pełną macierz danych: {df.shape} (wszystkie daty)")

    # Wykrywanie stacji
    station_names = sorted({c.replace("_east_norm", "") for c in df.columns if c.endswith("_east_norm")})
    print(f"Znaleziono {len(station_names)} stacji do analizy.")

    # Kopia DataFrame na wyniki. Domyślnie wypełniamy 1 (Norma).
    # -1 = Anomalia, 1 = Norma.
    final_results = df.copy()

    # Inicjalizujemy kolumny anomalii wartością 1 (Norma) dla wszystkich
    for station in station_names:
        final_results[f"{station}_anomaly"] = 1

    total_anomalies = 0

    print(f"Rozpoczynam trening Isolation Forest (Contamination={contamination})...")

    for station in station_names:
        train_cols = [f"{station}_east_norm", f"{station}_north_norm", f"{station}_up_norm"]

        if not all(c in df.columns for c in train_cols):
            print(f"Pominięto {station} - brak danych.")
            continue

        # --- KLUCZOWA POPRAWKA ---
        # 1. Pobieramy dane dla stacji
        station_data = df[train_cols]

        # 2. Wykrywamy dni AKTYWNE (niebędące sztucznym zerem)
        # W preprocessingu puste miejsca zostały zamienione na 0.
        # Sprawdzamy sumę modułów. Jeśli > 0.000001 to znaczy, że są tam dane.
        # (Używamy surowych danych w mm do sprawdzenia aktywności, bo znormalizowane też mogą być bliskie 0)
        # Ale tutaj mamy dostęp tylko do _norm i _mm w df.
        # Bezpieczniej: Sprawdzamy czy kolumny _norm nie są idealnym zerem wszystkie naraz.

        # Pobieramy odpowiadające kolumny w mm (bez _norm), żeby sprawdzić aktywność
        cols_mm = [c.replace("_norm", "") for c in train_cols]
        magnitude = df[cols_mm].abs().sum(axis=1)

        # Maska aktywności: Dni, kiedy stacja faktycznie mierzyła
        active_mask = magnitude > 0.0001

        # 3. Tworzymy zbiór treningowy TYLKO z aktywnych dni
        X_train = station_data[active_mask]

        if X_train.empty:
            print(f"{station}: Brak aktywnych danych (same zera). Pomijam.")
            continue

        # --- KONFIGURACJA MODELU ---
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        # 4. Trenujemy i przewidujemy TYLKO na aktywnych danych
        preds = model.fit_predict(X_train)

        # 5. Zapisujemy wyniki w odpowiednich miejscach (według indeksu active_mask)
        # Domyślnie w final_results jest 1 (Norma). Nadpisujemy tylko tam, gdzie stacja działała.
        final_results.loc[active_mask, f"{station}_anomaly"] = preds

        # Zapis modelu
        joblib.dump(model, MODELS_DIR / f"{station}_iforest.pkl")

        # Raportowanie
        n_anom = (preds == -1).sum()
        total_anomalies += n_anom
        n_days = len(X_train)

        # Teraz liczba anomalii zależy od długości serii!
        print(f"{station}: {n_anom} anomalii (na {n_days} dni obs).")

    # Eksport wyników
    final_results.to_csv(RESULTS_FILE)

    print(f"\nSukces! Zapisano wyniki do: {RESULTS_FILE}")
    print(f"Łącznie wykryto {total_anomalies} anomalii.")


if __name__ == "__main__":
    detect_anomalies_per_station(contamination=0.004)