"""
benchmark_synthetic.py
----------------------
Moduł walidacji i benchmarkingu algorytmu.

Zadaniem tego modułu jest sprawdzenie "czułości" algorytmu na sztucznych danych,
gdzie znamy dokładne położenie anomalii (Ground Truth).

Generuje:
1. Syntetyczny sygnał GNSS (sinusoida roczna + szum).
2. Wstrzykuje 3 rodzaje awarii:
   - STEP (nagły skok)
   - RAMP (powolny dryf/osuwisko)
   - NOISE (zwiększony szum)
3. Rysuje krzywą skuteczności (Recall) w funkcji prędkości narastania awarii.

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# KONFIGURACJA ŚCIEŻEK
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_PLOTS_DIR = PROJECT_ROOT / "data" / "simulated"
SIM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# KONFIGURACJA SYMULACJI
DAYS = 365 * 4       # 4 lata danych
NOISE_LEVEL = 2.0    # Poziom szumu [mm]
SEASONAL_AMP = 6.0   # Amplituda roczna [mm]


def generate_clean_signal() -> pd.DataFrame:
    """Generuje czysty, zaszumiony sygnał z sezonowością."""
    t = np.arange(DAYS)
    dates = pd.date_range(start="2020-01-01", periods=DAYS, freq="D")

    # Model: Sinusoida + Szum Gaussa
    season = SEASONAL_AMP * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, NOISE_LEVEL, DAYS)

    return pd.DataFrame({
        "date": dates,
        "signal": season + noise,
        "is_anomaly": 0
    }).set_index("date")


def inject_anomalies(df: pd.DataFrame, ramp_slope: float = 0.5) -> pd.DataFrame:
    """
    Wstrzykuje trzy typy anomalii do sygnału.
    Używa bezpiecznego indeksowania .iloc, aby uniknąć ostrzeżeń Pandas.
    """
    data = df.copy()
    L = len(data)

    # Pobieramy indeksy numeryczne kolumn dla .iloc
    sig_col = data.columns.get_loc('signal')
    anom_col = data.columns.get_loc('is_anomaly')

    # 1. STEP (Nagły skok +15mm) - w 25% czasu
    t1 = int(L * 0.25)
    data.iloc[t1:, sig_col] += 15.0
    data.iloc[t1:t1+3, anom_col] = 1

    # 2. RAMP (Dryf liniowy) - w 50% czasu
    t2 = int(L * 0.50)
    duration = 60
    ramp_vals = np.arange(duration) * ramp_slope

    data.iloc[t2:t2+duration, sig_col] += ramp_vals
    data.iloc[t2:t2+duration, anom_col] = 1

    # 3. NOISE (Szum x4) - w 75% czasu
    t3 = int(L * 0.75)
    duration_noise = 20
    noise_burst = np.random.normal(0, NOISE_LEVEL * 4, duration_noise)

    data.iloc[t3:t3+duration_noise, sig_col] += noise_burst
    data.iloc[t3:t3+duration_noise, anom_col] = 1

    return data


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy dodatkowe cechy (prędkość, przyspieszenie, lokalny szum)."""
    X = pd.DataFrame(index=df.index)
    X['val'] = df['signal']
    X['velocity'] = df['signal'].diff().fillna(0)       # Wykrywa skoki i rampy
    X['accel'] = X['velocity'].diff().fillna(0)         # Wykrywa zmiany trendu
    X['rolling_std'] = df['signal'].rolling(7, center=True).std().fillna(0) # Wykrywa szum
    return X.fillna(0)


def plot_simulation_result(df, title, filename):
    """Rysuje wykres pojedynczego eksperymentu (Sygnał vs Wykrycie)."""
    plt.figure(figsize=(14, 7))

    # Sygnał
    plt.plot(df.index, df['signal'], label='Sygnał GNSS (Sztuczny)', color='royalblue', alpha=0.6, linewidth=1)

    # Ground Truth (Zielone tło)
    y_min, y_max = df['signal'].min(), df['signal'].max()
    margin = (y_max - y_min) * 0.05
    plt.fill_between(df.index, y_min - margin, y_max + margin,
                     where=df['is_anomaly'] == 1,
                     color='green', alpha=0.15, label='Prawdziwa Anomalia')

    # Detekcja AI (Czerwone punkty)
    det = df[df['detected'] == 1]
    plt.scatter(det.index, det['signal'], color='red', s=25,
                label='Wykryta Anomalia', zorder=5, edgecolors='black', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Up [mm]", fontweight='bold')
    plt.xlabel("Data", fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = SIM_PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300)
    print(f"Zapisano przykład: {out_path.name}")
    plt.close()


def run_single_simulation_and_plot(slope_val):
    """Uruchamia jedną symulację i generuje wykres."""
    df_clean = generate_clean_signal()
    df_dirty = inject_anomalies(df_clean, ramp_slope=slope_val)
    features = feature_engineering(df_dirty)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Trenujemy model na tych konkretnych sztucznych danych
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    df_dirty['pred'] = model.fit_predict(X_scaled)
    df_dirty['detected'] = (df_dirty['pred'] == -1).astype(int)

    # Liczymy skuteczność
    overlap = (df_dirty['is_anomaly'] & df_dirty['detected']).sum()
    total = df_dirty['is_anomaly'].sum()
    recall = (overlap / total) * 100 if total > 0 else 0

    title = f"Przykład Detekcji: Dynamika = {slope_val:.1f} mm/dzień (Recall: {recall:.1f}%)"
    filename = f"sim_example_slope_{slope_val:.1f}.png"
    plot_simulation_result(df_dirty, title, filename)


def generate_demonstration_examples():
    """Generuje 3 przykłady dla różnych prędkości rampy."""
    print("\n=== GENEROWANIE PRZYKŁADOWYCH WYKRESÓW ===")
    # 0.3 = Niewykrywalne (szum)
    # 1.0 = Wykrywalne z opóźnieniem
    # 2.5 = Wykrywalne natychmiast
    example_slopes = [0.3, 1.0, 2.5]
    for s in example_slopes:
        run_single_simulation_and_plot(s)


def calculate_recall_only(slope_val):
    """Pomocnicza funkcja do szybkiego liczenia Recall (bez rysowania)."""
    df_clean = generate_clean_signal()
    df_dirty = inject_anomalies(df_clean, ramp_slope=slope_val)
    features = feature_engineering(df_dirty)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    preds = model.fit_predict(X_scaled)

    detected = (preds == -1).astype(int)
    overlap = (df_dirty['is_anomaly'] & detected).sum()
    return (overlap / df_dirty['is_anomaly'].sum()) * 100


def run_sensitivity_analysis():
    """Generuje główny wykres krzywej czułości."""
    print("\n=== ROZPOCZYNAM GŁÓWNĄ ANALIZĘ CZUŁOŚCI ===")
    slopes = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    recalls = []

    for s in slopes:
        rec = calculate_recall_only(s)
        recalls.append(rec)
        print(f" -> Slope: {s:.1f} mm/d  => Recall: {rec:.1f}%")

    # Rysowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(slopes, recalls, marker='o', linestyle='-', color='darkblue', linewidth=2, label='Krzywa Czułości Detekcji')

    plt.title("Charakterystyka Czułości Detekcji", fontsize=14, fontweight='bold')
    plt.xlabel("Dynamika zjawiska [mm/dzień]", fontsize=12)
    plt.ylabel("Skuteczność detekcji [%]", fontsize=12)

    # Elementy pomocnicze
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axhline(y=90, color='green', linestyle=':', label='Wysoka skuteczność')
    plt.legend(loc='lower right')

    # Adnotacja o strefie ostrzegawczej
    plt.text(1.5, 40, "Strefa 'Early Warning'\n(Wykrycie z opóźnieniem)",
             bbox=dict(facecolor='yellow', alpha=0.2))

    plt.tight_layout()

    out_path = SIM_PLOTS_DIR / "sensitivity_curve.png"
    plt.savefig(out_path, dpi=300)
    print(f"Zapisano główny wykres: {out_path.name}")

    plt.show()


if __name__ == "__main__":
    generate_demonstration_examples()
    run_sensitivity_analysis()