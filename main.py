"""
Główny plik uruchomieniowy projektu GNSS Anomaly Detection.
Steruje całym procesem: od pobrania danych, przez ML, aż po walidację i raporty.

Autor: Szymon Zarosa
Data: 2025-11-22
"""

from src.download_data import download_all, STATIONS
from src.data_loader import process_raw_to_csv
from src.preprocess import preprocess_all
from src.detect_anomalies import detect_anomalies_per_station
from src.visualize import visualise_anomalies
from src.create_map import create_gnss_map
from src.verify_mad import verify_results
from src.benchmark_synthetic import run_sensitivity_analysis, generate_demonstration_examples

def main():
    print("=== START: GNSS Anomaly Detection Pipeline ===\n")

    # --- ETAP 1: DANE ---
    # Krok 1: Pobieranie danych (zakomentowane, żeby nie pobierać za każdym razem)
    print(">>> 1. Pobieranie danych z NGL...")
    # download_all(STATIONS)

    # Krok 2: Parsowanie surowych plików
    print("\n>>> 2. Parsowanie plików .tenv3 do CSV...")
    process_raw_to_csv()

    # Krok 3: Preprocessing (Usuwanie trendu/sezonowości, Normalizacja)
    print("\n>>> 3. Preprocessing i czyszczenie danych...")
    preprocess_all()

    # --- ETAP 2: ANALIZA ML ---
    # Krok 4: Detekcja (Isolation Forest)
    print("\n>>> 4. Uruchamianie Isolation Forest (AI)...")
    detect_anomalies_per_station(contamination=0.004)

    # --- ETAP 3: WIZUALIZACJA ---
    # Krok 5: Generowanie wykresów czasowych i mapy
    print("\n>>> 5. Generowanie wizualizacji...")
    visualise_anomalies(show_plots=False, save_plots=True)
    create_gnss_map()

    # --- ETAP 4: WALIDACJA I WERYFIKACJA ---
    # Krok 6: Weryfikacja statystyczna (MAD vs IsolationForest)
    print("\n>>> 6. Generowanie raportu porównawczego (AI vs MAD)...")
    verify_results()

    # Krok 7: Benchmark syntetyczny (Analiza czułości)
    print("\n>>> 7. Uruchamianie benchmarku syntetycznego (Testy Czułości Detekcji)...")
    generate_demonstration_examples() # Generuje przykłady wykresów
    run_sensitivity_analysis()        # Generuje krzywą

    print("\n===================================================")
    print("KONIEC: Wszystkie zadania wykonane pomyślnie.")
    print("===================================================")
    print("📂 Wyniki znajdziesz w:")
    print("   - Wykresy stacji:  data/plots/")
    print("   - Mapa sieci:      data/gnss_network_map.html")
    print("   - Raport Excel:    data/results/raport_porownawczy.xlsx")
    print("   - Walidacja:       data/simulated/")

if __name__ == "__main__":
    main()