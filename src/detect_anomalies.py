"""
detect_anomalies.py
----------------
Detekcja anomalii GNSS przy użyciu IsolationForest.
Analizuje wektor 3D (East, North, Up) jednocześnie.

- Wczytuje znormalizowane reszty z preprocessingu
- Trenuje Isolation Forest dla każdej stacji na danych wielowymiarowych
- Zapisuje wyniki (-1 anomalia, 1 norma)

Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_INPUT_FILE = PROJECT_ROOT / "data" / "model_input" / "gnss_model_input.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def detect_anomalies_per_station(contamination=0.004):
    if not MODEL_INPUT_FILE.exists():
        print(f"Brak pliku: {MODEL_INPUT_FILE}")
        return

    df = pd.read_csv(MODEL_INPUT_FILE, parse_dates=["date"], index_col="date")
    print(f"Wczytano dane: {df.shape}")

    # Szukamy stacji
    station_names = sorted({c.split("_")[0] for c in df.columns if "_" in c})

    final_results = df.copy()
    total_anomalies = 0

    print(f"Szukam anomalii w danych znormalizowanych (contamination={contamination})...")

    for station in station_names:
        # Do treningu bierzemy TYLKO kolumny znormalizowane (_norm)
        train_cols = [f"{station}_east_norm", f"{station}_north_norm", f"{station}_up_norm"]

        if not all(c in df.columns for c in train_cols):
            continue

        X = df[train_cols].fillna(0)

        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
        preds = model.fit_predict(X)

        # Zapis wyniku
        final_results[f"{station}_anomaly"] = preds

        # Opcjonalnie zapis modelu
        joblib.dump(model, MODELS_DIR / f"{station}_iforest.pkl")

        n_anom = (preds == -1).sum()
        total_anomalies += n_anom
        print(f"{station}: {n_anom} anomalii.")

    final_results.to_csv(RESULTS_FILE)
    print(f"Zapisano wyniki (jednostki [mm] zachowane) do: {RESULTS_FILE}")


if __name__ == "__main__":
    detect_anomalies_per_station(contamination=0.004)