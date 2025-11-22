"""
visualise.py
----------------
Wizualizacja wyników detekcji anomalii GNSS (per-station).
Wyniki w [mm]

Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_station_components(df: pd.DataFrame, station: str):

    station_cols = [c for c in df.columns if c.startswith(station)]

    # Filtrujemy:
    # 1. Bez 'anomaly'
    # 2. Bez 'norm'
    # 3. Bez 'sigma'
    data_cols = [c for c in station_cols
                 if "anomaly" not in c
                 and "norm" not in c
                 and "sigma" not in c]

    sigma_cols = [c for c in station_cols if "sigma" in c]
    return sorted(data_cols), sorted(sigma_cols)


def visualise_anomalies(show_plots=True, save_plots=True, filter_sigma_threshold=True, n_sigma=3):
    if not RESULTS_FILE.exists():
        print("Brak pliku. Uruchom detect_anomalies.py.")
        return

    print(f"Wczytywanie: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"], index_col="date")

    station_names = sorted({c.split("_")[0] for c in df.columns if "_" in c})
    print(f"Rysowanie wykresów w [mm] dla {len(station_names)} stacji...")

    for station in station_names:
        data_cols, sigma_cols = _get_station_components(df, station)
        if not data_cols: continue

        magnitude = df[data_cols].abs().sum(axis=1)
        station_df = df[magnitude > 0.0001].copy()

        if station_df.empty: continue

        comp_map = {c: c.split("_")[1] for c in data_cols}
        n_comp = len(data_cols)
        fig, axes = plt.subplots(n_comp, 1, figsize=(14, 4 * n_comp), sharex=True)
        if n_comp == 1: axes = [axes]

        anomaly_legend_plotted = False
        anom_mask = station_df[station + "_anomaly"] == -1

        for ax, col in zip(axes, data_cols):
            # Wykres w mm
            ax.plot(station_df.index, station_df[col],
                    label="Dane [mm]", color='royalblue',
                    marker='.', linestyle='None', markersize=2, alpha=0.4)

            # Trend
            rolling_mean = station_df[col].rolling(window=30, center=True).mean()
            ax.plot(rolling_mean.index, rolling_mean,
                    color='black', linewidth=0.8, alpha=0.7, label="Linia trendu (30-dniowy)")

            # Anomalie
            if filter_sigma_threshold:
                val = station_df[col]
                threshold = n_sigma * val.std()
                mean_val = val.mean()
                final_anom_mask = anom_mask & ((val > mean_val + threshold) | (val < mean_val - threshold))
                anom_points = val[final_anom_mask]
            else:
                anom_points = station_df[col][anom_mask]

            if not anom_points.empty:
                lbl = "Wykryte anomalie" if not anomaly_legend_plotted else ""
                ax.scatter(anom_points.index, anom_points.values,
                           color='red', s=35, zorder=10, label=lbl, edgecolors='black', linewidth=0.5)
                anomaly_legend_plotted = True

            # Opisy osi
            comp_clean = comp_map[col].capitalize()
            ax.set_ylabel(f"{comp_clean} [mm]", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Skala Y
            y_vals = station_df[col]
            if not y_vals.empty:
                y_min, y_max = y_vals.min(), y_vals.max()
                margin = max((y_max - y_min) * 0.1, 1.0)  # Min margin 1mm
                ax.set_ylim(y_min - margin, y_max + margin)

        # Oś X
        ax_last = axes[-1]
        ax_last.set_xlabel("Data", fontweight='bold')
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax_last.xaxis.set_major_locator(locator)
        ax_last.xaxis.set_major_formatter(formatter)

        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.95))

        fig.suptitle(f"Stacja {station} ", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_file = PLOTS_DIR / f"{station}.png"
        if save_plots:
            fig.savefig(out_file, dpi=300)
            print(f"Zapisano: {out_file.name}")
        if show_plots:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    visualise_anomalies()
