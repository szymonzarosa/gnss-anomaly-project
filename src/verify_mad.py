"""
verify_mad.py
-------------
Moduł weryfikacji statystycznej (Cross-Validation).

Porównuje wyniki uzyskane metodą Machine Learning (Isolation Forest)
z klasyczną, odporną metodą statystyczną (Median Absolute Deviation - MAD).

Cel: Sprawdzenie, czy AI wykrywa te same anomalie co fizyka,
oraz identyfikacja przypadków nadczułości (False Positives).

Metoda: Modified Z-Score (Iglewicz & Hoaglin).
Próg detekcji: > 3.5 (odpowiada ok. 3.5 sigma w rozkładzie normalnym).

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import pandas as pd
import numpy as np
from pathlib import Path

# KONFIGURACJA ŚCIEŻEK
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
OUTPUT_EXCEL = PROJECT_ROOT / "data" / "results" / "raport_porownawczy.xlsx"


def calculate_mad_zscore(series: pd.Series) -> pd.Series:
    """
    Oblicza Modified Z-Score dla szeregu danych.
    W przeciwieństwie do zwykłego Z-Score, ta miara jest odporna na outliery.

    Wzór: M = 0.6745 * (x - mediana) / MAD
    Stała 0.6745 sprawia, że dla rozkładu normalnego M ~= Z-Score.
    """
    median = series.median()
    # MAD = Mediana z wartości bezwzględnych odchyleń
    deviations = np.abs(series - median)
    mad_value = deviations.median()

    if mad_value == 0:
        # Jeśli zmienność jest zerowa, zwracamy zera (unikamy dzielenia przez 0)
        return pd.Series(0, index=series.index)

    modified_z_score = 0.6745 * (series - median) / mad_value
    return modified_z_score


def verify_results() -> None:
    """
    Główna funkcja generująca raport Excel porównujący AI i Statystykę.
    """
    if not RESULTS_FILE.exists():
        print("Brak pliku z wynikami (gnss_anomalies.csv). Uruchom najpierw 'detect_anomalies.py'.")
        return

    print(f"Wczytywanie wyników AI z: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"], index_col="date")

    # Wykrywanie stacji (szukamy kolumn z końcówką _anomaly)
    cols = [c for c in df.columns if "_anomaly" in c]
    # Filtrujemy symulacje (opcjonalnie, tu bierzemy wszystko co nie ma 'SIM' w nazwie,
    # lub wszystko jeśli chcesz też sprawdzać symulacje)
    station_names = sorted([c.replace("_anomaly", "") for c in cols if "SIM" not in c])

    summary_list = []

    print("Generowanie raportu porównawczego Excel...")

    # Używamy 'openpyxl' jako silnika do zapisu .xlsx
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

        for station in station_names:
            # 1. Pobieramy dane
            col_data = f"{station}_up"       # Dane fizyczne (wysokość w mm)
            col_ai = f"{station}_anomaly"    # Wynik AI (-1 = anomalia, 1 = norma)

            if col_data not in df.columns:
                continue

            series_mm = df[col_data]
            ai_preds = df[col_ai]

            # 2. Obliczamy MAD (Statystyka)
            mad_scores = calculate_mad_zscore(series_mm)
            # Anomalia statystyczna: |Z-score| > 3.5
            mad_anomalies = np.abs(mad_scores) > 3.5

            # 3. Logika Porównania (Macierz pomyłek / Confusion Matrix)
            is_ai_anom = (ai_preds == -1)
            is_mad_anom = mad_anomalies

            # Tworzymy opis słowny dla każdego dnia
            status_list = []
            for ai, mad in zip(is_ai_anom, is_mad_anom):
                if ai and mad:
                    status_list.append("POTWIERDZONA (Obie)")
                elif ai and not mad:
                    status_list.append("Tylko AI (Nadczułość?)")
                elif not ai and mad:
                    status_list.append("Tylko Statystyka (Przeoczenie?)")
                else:
                    status_list.append("OK")

            # 4. Zapis szczegółowy (karta stacji)
            detail_df = pd.DataFrame({
                "Wysokość [mm]": series_mm,
                "Wynik AI": ai_preds.apply(lambda x: "ANOMALIA" if x == -1 else "OK"),
                "Wynik MAD": np.where(mad_anomalies, "ANOMALIA", "OK"),
                "MAD Z-Score": mad_scores,
                "STATUS": status_list
            })

            # Zapisujemy do arkusza o nazwie stacji
            detail_df.to_excel(writer, sheet_name=station)

            # 5. Statystyki zbiorcze
            count_ai = is_ai_anom.sum()
            count_mad = is_mad_anom.sum()
            count_both = (is_ai_anom & is_mad_anom).sum()

            # Jaccard Index (IoU): Miara podobieństwa dwóch zbiorów (0% - 100%)
            union = (is_ai_anom | is_mad_anom).sum()
            similarity = (count_both / union * 100) if union > 0 else 0.0

            summary_list.append({
                "Stacja": station,
                "Liczba Dni": len(series_mm),
                "Wykryte przez AI": count_ai,
                "Wykryte przez MAD": count_mad,
                "Wspólne (Potwierdzone)": count_both,
                "Tylko AI": count_ai - count_both,
                "Tylko MAD": count_mad - count_both,
                "Zgodność Metod [%]": f"{similarity:.1f}%"
            })

            print(f"   -> {station}: AI={count_ai}, MAD={count_mad}, Wspólne={count_both}")

        # 6. Zapis podsumowania na pierwszym arkuszu
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_excel(writer, sheet_name="PORÓWNANIE_METOD", index=False)

    print(f"\nRaport gotowy: {OUTPUT_EXCEL}")
    print("   Otwórz arkusz 'PORÓWNANIE_METOD', aby zobaczyć tabelę zbiorczą.")


if __name__ == "__main__":
    verify_results()