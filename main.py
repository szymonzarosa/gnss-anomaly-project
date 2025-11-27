"""
main.py
-------
Główny punkt wejścia (Entry Point) dla projektu GNSS Anomaly Detection.

Ten skrypt integruje wszystkie moduły w jeden spójny potok przetwarzania (pipeline):
1. Pobieranie i parsowanie danych (ETL).
2. Przetwarzanie sygnału (usuwanie trendu/sezonowości).
3. Detekcja anomalii algorytmem Isolation Forest.
4. Wizualizacja wyników (wykresy i mapy).
5. Walidacja statystyczna i testy czułości.

Uruchomienie:
    python main.py

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import sys
from src.download_data import download_all, STATIONS
from src.data_loader import process_raw_to_csv
from src.preprocess import preprocess_all
from src.detect_anomalies import detect_anomalies_per_station
from src.visualize import visualise_anomalies
from src.create_map import create_gnss_map
from src.verify_mad import verify_results
from src.benchmark_synthetic import run_sensitivity_analysis, generate_demonstration_examples

def main():
    print("\n" + "="*60)
    print("GNSS ANOMALY DETECTION SYSTEM - PIPELINE START")
    print("="*60 + "\n")

    try:
        # --- ETAP 1: DANE (ETL) ---
        print("--- [KROK 1/7] POZYSKIWANIE DANYCH ---")
        # download_all(STATIONS) # Odkomentuj, aby pobrać najnowsze dane z NGL
        print(">>> Pominięto pobieranie (używam danych lokalnych).")

        print("\n--- [KROK 2/7] PARSOWANIE DANYCH ---")
        # Konwersja surowych plików .tenv3 na czytelne CSV
        process_raw_to_csv()

        print("\n--- [KROK 3/7] PREPROCESSING ---")
        # Usuwanie fizyki (trend/sezonowość) i normalizacja
        preprocess_all()

        # --- ETAP 2: ANALIZA ML ---
        print("\n--- [KROK 4/7] DETEKCJA ANOMALII (AI) ---")
        # Isolation Forest z założeniem 0.4% zanieczyszczenia danych
        detect_anomalies_per_station(contamination=0.004)

        # --- ETAP 3: WIZUALIZACJA ---
        print("\n--- [KROK 5/7] GENEROWANIE WIZUALIZACJI ---")
        # Wykresy PNG (nie blokujemy okienkami - show_plots=False)
        visualise_anomalies(show_plots=False, save_plots=True)
        # Mapa HTML
        create_gnss_map()

        # --- ETAP 4: WALIDACJA ---
        print("\n--- [KROK 6/7] WERYFIKACJA STATYSTYCZNA (MAD) ---")
        # Generowanie raportu Excel porównującego AI ze statystyką
        verify_results()

        print("\n--- [KROK 7/7] BENCHMARK SYNTETYCZNY ---")
        # Testy czułości na sztucznych danych (Step, Ramp, Noise)
        generate_demonstration_examples()
        run_sensitivity_analysis()

        # --- PODSUMOWANIE ---
        print("\n" + "="*60)
        print("SUKCES: PRZETWARZANIE ZAKOŃCZONE POMYŚLNIE")
        print("="*60)
        print("LOKALIZACJA WYNIKÓW:")
        print(f"Wykresy stacji:    data/plots/")
        print(f"Mapa sieci:        data/gnss_network_map.html")
        print(f"Raport Excel:      data/results/raport_porownawczy.xlsx")
        print(f"Walidacja (Sim):   data/simulated/")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\nPrzerwano przez użytkownika (Ctrl+C). Zamykanie...")
        sys.exit(0)
    except Exception as e:
        print(f"\nWYSTĄPIŁ BŁĄD KRYTYCZNY: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()