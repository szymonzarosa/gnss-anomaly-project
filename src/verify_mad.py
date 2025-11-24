"""
src/verify_mad.py
----------------
Kompleksowa weryfikacja wyników: Isolation Forest (AI) vs MAD (Statystyka).
Generuje raport Excel z porównaniem skuteczności obu metod.

Wzór: Modified Z-Score = 0.6745 * (X - Median) / MAD
Próg detekcji: > 3.5

Autor: Szymon Zarosa
Data: 2025-11-22
"""

# Import bibliotek
import pandas as pd
import numpy as np
from pathlib import Path

# KONFIGURACJA STRUKTURY FOLDERÓW
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
OUTPUT_EXCEL = PROJECT_ROOT / "data" / "results" / "raport_porownawczy.xlsx"


def calculate_mad_zscore(series):
    """Oblicza Modified Z-Score (statystyka odporna)."""
    median = series.median()
    deviations = np.abs(series - median)
    mad_value = deviations.median()

    if mad_value == 0:
        return np.zeros(len(series))

    modified_z_score = 0.6745 * (series - median) / mad_value
    return modified_z_score


def verify_results():
    if not RESULTS_FILE.exists():
        print("Brak pliku z wynikami (gnss_anomalies.csv). Uruchom najpierw detect_anomalies.py")
        return

    print(f"Wczytywanie wyników AI z: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"], index_col="date")

    # Znajdujemy stacje (szukamy kolumn z końcówką _anomaly)
    cols = [c for c in df.columns if "_anomaly" in c]
    station_names = sorted([c.replace("_anomaly", "") for c in cols])

    summary_list = []

    print("Generowanie raportu porównawczego Excel...")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

        for station in station_names:
            # 1. Pobieramy dane
            col_data = f"{station}_up"  # Dane w mm
            col_ai = f"{station}_anomaly"  # Wynik AI (-1/1)

            if col_data not in df.columns:
                continue

            # Przygotowanie serii
            series_mm = df[col_data]
            ai_preds = df[col_ai]  # -1 to anomalia

            # 2. Obliczamy MAD (Statystyka)
            mad_scores = calculate_mad_zscore(series_mm)
            # Definicja anomalii wg MAD: |Z-score| > 3.5
            mad_anomalies = np.abs(mad_scores) > 3.5

            # 3. Porównanie (Confusion Matrix logic)
            # AI Anomaly: (ai_preds == -1)
            # MAD Anomaly: (mad_anomalies == True)

            is_ai_anom = (ai_preds == -1)
            is_mad_anom = mad_anomalies

            # Tworzymy kolumnę statusu dla Excela
            status_list = []
            for ai, mad in zip(is_ai_anom, is_mad_anom):
                if ai and mad:
                    status_list.append("POTWIERDZONA (Obie)")
                elif ai and not mad:
                    status_list.append("Tylko AI")
                elif not ai and mad:
                    status_list.append("Tylko Statystyka")
                else:
                    status_list.append("OK")

            # 4. Zapis szczegółowy dla stacji
            detail_df = pd.DataFrame({
                "Wysokość [mm]": series_mm,
                "Wynik AI": ai_preds.apply(lambda x: "ANOMALIA" if x == -1 else "OK"),
                "Wynik MAD": np.where(mad_anomalies, "ANOMALIA", "OK"),
                "MAD Score": mad_scores,
                "STATUS PORÓWNANIA": status_list
            })

            # Zapisujemy tylko dni, gdzie COKOLWIEK się dzieje (żeby plik nie był pusty),
            # albo całość. Tu zapiszemy całość, ale możesz odfiltrować 'OK'.
            detail_df.to_excel(writer, sheet_name=station)

            # 5. Statystyki do podsumowania
            count_ai = is_ai_anom.sum()
            count_mad = is_mad_anom.sum()
            count_both = (is_ai_anom & is_mad_anom).sum()

            # Jaccard Index (podobieństwo zbiorów): Przecięcie / Suma
            union = (is_ai_anom | is_mad_anom).sum()
            similarity = (count_both / union * 100) if union > 0 else 0.0

            summary_list.append({
                "Stacja": station,
                "Liczba Dni": len(series_mm),
                "Wykryte przez AI": count_ai,
                "Wykryte przez MAD": count_mad,
                "Wspólne (Potwierdzone)": count_both,
                "Tylko AI (Nadczułość?)": count_ai - count_both,
                "Tylko MAD (Przeoczone?)": count_mad - count_both,
                "Zgodność Metod [%]": f"{similarity:.1f}%"
            })

            print(f" -> {station}: AI={count_ai}, MAD={count_mad}, Wspólne={count_both}")

        # 6. Zapis arkusza zbiorczego na początku
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_excel(writer, sheet_name="PORÓWNANIE_METOD", index=False)

    print(f"\nRaport gotowy: {OUTPUT_EXCEL}")
    print("Otwórz arkusz 'PORÓWNANIE_METOD' by zobaczyć zestawienie.")


if __name__ == "__main__":
    verify_results()