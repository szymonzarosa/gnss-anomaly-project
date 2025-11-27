"""
visualise.py
------------
Moduł wizualizacji danych GNSS.

Generuje wykresy szeregów czasowych (scatter plots) dla każdej stacji,
przedstawiając:
1. Pomiary (niebieskie punkty) - residua w [mm] po usunięciu trendu/sezonowości.
2. Linię trendu (czarna linia) - 30-dniowa średnia krocząca.
3. Anomalie (czerwone punkty) - dni oflagowane przez algorytm jako nietypowe.

Wykresy są automatycznie skalowane i przycinane do okresu dostępności danych
(usuwanie "martwych stref").

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np

# KONFIGURACJA ŚCIEŻEK
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"

# Tworzymy folder na wykresy, jeśli nie istnieje
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_station_components(df: pd.DataFrame, station: str) -> tuple[list[str], list[str]]:
    """
    Pomocnicza funkcja filtrująca kolumny dla danej stacji.
    Zwraca osobno nazwy kolumn z danymi (E,N,U) i kolumn z błędami (Sigma).
    """
    # Pobieramy wszystkie kolumny zaczynające się od nazwy stacji
    # Dodajemy podkreślnik, żeby uniknąć pomyłek (np. żeby 'SIM' nie łapało 'SIM_STEP')
    prefix = f"{station}_"
    station_cols = [c for c in df.columns if c.startswith(prefix)]

    # Filtrujemy kolumny z danymi (E, N, U)
    # Odrzucamy kolumny techniczne (_anomaly, _norm, _sigma)
    data_cols = [c for c in station_cols
                 if "anomaly" not in c
                 and "norm" not in c
                 and "sigma" not in c]

    # Filtrujemy kolumny z niepewnościami (Sigma)
    sigma_cols = [c for c in station_cols if "sigma" in c]

    return sorted(data_cols), sorted(sigma_cols)


def visualise_anomalies(show_plots: bool = True,
                        save_plots: bool = True,
                        filter_sigma_threshold: bool = True,
                        n_sigma: int = 3) -> None:
    """
    Główna funkcja generująca wykresy dla wszystkich stacji.

    Parametry:
    ----------
    show_plots : bool
        Czy wyświetlać okna z wykresami (przydatne przy debugowaniu).
    save_plots : bool
        Czy zapisywać wykresy do plików PNG.
    filter_sigma_threshold : bool
        Czy stosować dodatkowy filtr 3-sigma (ukrywa anomalie o małej amplitudzie).
    n_sigma : int
        Mnożnik odchylenia standardowego dla filtru (domyślnie 3).
    """
    if not RESULTS_FILE.exists():
        print("Brak pliku z wynikami. Uruchom najpierw 'detect_anomalies.py'.")
        return

    print(f"Wczytywanie wyników z: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"], index_col="date")

    # --- WYKRYWANIE NAZW STACJI ---
    # Szukamy kolumn fizycznych (kończących się na _east), ignorując kolumny _norm
    # To pozwala poprawnie zidentyfikować stacje, nawet te z symulacji (SIM_STEP)
    station_names = sorted({c.replace("_east", "") for c in df.columns if c.endswith("_east")})

    print(f"Rysowanie wykresów dla {len(station_names)} stacji...")

    for station in station_names:
        data_cols, sigma_cols = _get_station_components(df, station)
        if not data_cols:
            continue

        # --- USUWANIE MARTWYCH STREF ---
        # Jeśli w danym dniu suma modułów (E+N+U) jest bliska 0, to znaczy,
        # że dane zostały sztucznie uzupełnione zerami w preprocessingu.
        # Usuwamy te wiersze, żeby na wykresie były puste przerwy.
        magnitude = df[data_cols].abs().sum(axis=1)
        station_df = df[magnitude > 0.0001].copy()

        if station_df.empty:
            print(f"Stacja {station} nie posiada aktywnych danych. Pomijam.")
            continue

        # Przygotowanie płótna (Subplots)
        comp_map = {c: c.split("_")[-1] for c in data_cols} # np. 'east', 'north'
        n_comp = len(data_cols)
        fig, axes = plt.subplots(n_comp, 1, figsize=(14, 4 * n_comp), sharex=True)
        if n_comp == 1: axes = [axes]

        anomaly_legend_plotted = False

        # Sprawdzamy czy kolumna anomalii istnieje
        anom_col = f"{station}_anomaly"
        if anom_col not in station_df.columns:
            continue

        anom_mask = station_df[anom_col] == -1

        for ax, col in zip(axes, data_cols):
            # 1. Rysowanie pomiarów (Scatter plot)
            ax.plot(station_df.index, station_df[col],
                    label="Pomiary [mm]", color='royalblue',
                    marker='.', linestyle='None', markersize=2, alpha=0.4)

            # 2. Rysowanie trendu (Rolling Mean)
            # Pomaga wzrokowo ocenić, czy stacja "pływa" (dryfuje)
            rolling_mean = station_df[col].rolling(window=30, center=True).mean()
            ax.plot(rolling_mean.index, rolling_mean,
                    color='black', linewidth=0.8, alpha=0.7, label="Trend (30d)")

            # 3. Rysowanie anomalii
            if filter_sigma_threshold:
                # Hybrydowe podejście: Anomalia ML musi być też anomalią fizyczną (> 3 sigma)
                val = station_df[col]
                mean_val = val.mean()
                std_dev = val.std()

                is_outlier = (val > mean_val + n_sigma * std_dev) | \
                             (val < mean_val - n_sigma * std_dev)

                final_anom_mask = anom_mask & is_outlier
                anom_points = val[final_anom_mask]
            else:
                anom_points = station_df[col][anom_mask]

            if not anom_points.empty:
                lbl = "Wykryta Anomalia" if not anomaly_legend_plotted else ""
                # Rysujemy czerwone punkty z obwódką
                ax.scatter(anom_points.index, anom_points.values,
                           color='red', s=35, zorder=10, label=lbl,
                           edgecolors='black', linewidth=0.5)
                anomaly_legend_plotted = True

            # Kosmetyka osi
            comp_name = comp_map[col].capitalize() # np. 'East'
            ax.set_ylabel(f"{comp_name} residuum [mm]", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Dynamiczna skala Y (z lekkim marginesem)
            y_vals = station_df[col]
            if not y_vals.empty:
                y_min, y_max = y_vals.min(), y_vals.max()
                margin = max((y_max - y_min) * 0.1, 1.0) # Min 1mm marginesu
                ax.set_ylim(y_min - margin, y_max + margin)

        # Formatowanie osi X (Daty)
        ax_last = axes[-1]
        ax_last.set_xlabel("Data", fontweight='bold')

        # Inteligentne dobieranie etykiet dat (lata/miesiące)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax_last.xaxis.set_major_locator(locator)
        ax_last.xaxis.set_major_formatter(formatter)

        # Legenda
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        # Usuwanie duplikatów w legendzie
        by_label = dict(zip(labels, handles))

        # Umieszczenie legendy na zewnątrz wykresu
        fig.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.95))

        fig.suptitle(f"Analiza Anomalii GNSS: Stacja {station}", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Zapis do pliku
        out_file = PLOTS_DIR / f"{station}_anomalies_mm.png"
        if save_plots:
            fig.savefig(out_file, dpi=300)
            print(f"Zapisano: {out_file.name}")


        plt.close(fig)


if __name__ == "__main__":
    visualise_anomalies()