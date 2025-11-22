# Zastosowanie metod detekcji anomalii w szeregach czasowych GNSS do analizy stabilności obiektów i sieci geodezyjnych

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Science](https://img.shields.io/badge/Geodesy-GNSS-green)
![ML](https://img.shields.io/badge/ML-Isolation%20Forest-orange)

Projekt badawczy mający na celu automatyczną detekcję nietypowych zachowań stacji GNSS (awarie sprzętowe, ruchy geodynamiczne, uszkodzenia konstrukcji) przy użyciu algorytmu **Isolation Forest**.

System analizuje dane w układzie współrzędnych topocentrycznych (ENU), uwzględnia usuwanie trendu i sezonowości, a także oferuje moduł walidacji poprzez symulację uszkodzeń inżynierskich.

## Funkcjonalności

* **Pobieranie danych:** Automatyczny import szeregów czasowych (format `.tenv3`) z *Nevada Geodetic Laboratory* (IGS20/EPN).
* **Parsowanie danych:** Konwersja surowych plików do formatu `pandas.DataFrame`.
* **Preprocessing:**
    * Interpolacja braków danych.
    * Usuwanie trendu liniowego i sezonowości (dekompozycja addytywna).
    * Normalizacja danych (Z-score) przy zachowaniu fizycznych jednostek [mm].
* **Detekcja anomalii:** Zastosowanie uczenia nienadzorowanego (Isolation Forest) na wektorze 3D (East, North, Up) oraz cechach pochodnych (prędkość przemieszczeń).
* **Analiza czułości:** Wyznaczanie krzywej skuteczności (Recall) w zależności od dynamiki zjawiska (symulacja Step, Ramp, Noise).
* **Wizualizacja:** Generowanie wykresów punktowych z zaznaczonymi anomaliami i linią trendu.

## Wymagania i Instalacja

Projekt został napisany w języku Python. Do działania wymaga bibliotek wymienionych w `requirements.txt`.

1.  **Sklonuj repozytorium:**
    ```bash
    git clone <adres-repozytorium-muszę-dodać>
    cd gnss_anomaly_project
    ```

2.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```

## Instrukcja Użytkowania

Cały proces analizy podzielony jest na niezależne moduły w folderze `src/`. Należy uruchamiać je w następującej kolejności:

### 1. Pobieranie danych
Pobiera surowe pliki `.tenv3` z serwerów NGL dla zdefiniowanej listy stacji.
```bash
python src/download_data.py
```

### 2. Parsowanie (Data Loader)
Konwertuje specyficzny format .tenv3 na ustandaryzowane pliki .csv, przeliczając daty MJD na format kalendarzowy.
```bash
python src/data_loader.py
```

### 3. Preprocessing
Przygotowuje dane do analizy: usuwa składowe fizyczne (trend tektoniczny, sezonowość roczną), pozostawiając residua (reszty - szum oraz anomalie). Dokonuje normalizacji dla modelu ML.
```bash
python src/preprocess.py
```

Wynik: Plik data/model_input/gnss_model_input.csv zawierający residua w [mm] oraz znormalizowane cechy.

### 4. Detekcja Anomalii
Trenuje model Isolation Forest dla każdej stacji osobno i klasyfikuje dni jako "Norma" (1) lub "Anomalia" (-1). Algorytm uczy się na danych znormalizowanych (cechy _norm), ale wyniki zapisuje w kontekście wartości fizycznych [mm].
```bash
python src/detect_anomalies.py
```
### 5. Wizualizacja i Mapa
Generuje raporty graficzne w dwóch formach:

* Wykresy czasowe: Szczegółowa analiza każdej stacji.

* Mapa sieci: Plik HTML z mapą, na której kolor stacji (gradient od zielonego do czerwonego) odzwierciedla procent wykrytych awarii (filtr: ML + 3σ).
```bash
python src/visualize.py
python src/create_map.py
```
Wynik: Wykresy PNG w folderze data/plots/ oraz interaktywna mapa data/gnss_network_map.html.

### 6. Walidacja i Analiza Czułości
Projekt zawiera moduł (benchmark_synthetic.py), który bada skuteczność algorytmu w funkcji prędkości narastania awarii (np. osuwiska).

Aby wygenerować Krzywą Czułości Detekcji oraz przykładowe wykresy detekcji.
```bash
python src/benchmark_synthetic.py
```
Skrypt wygeneruje wykresy w folderze data/simulated/, pokazując przy jakim tempie deformacji (mm/dzień) algorytm osiąga zadowalającą skuteczność.

## Wnioski z analizy czułości

**Step/Noise**: Wykrywalność bliska 100% dla nagłych zdarzeń.

**Ramp (Dryf)**: Skuteczna detekcja (>65%) dla prędkości przemieszczeń powyżej 1.5 mm/dobę.

## Struktura projektu

```text
gnss_anomaly_project/
├── data/                  # Dane
│   ├── raw/               # Pobrane pliki .tenv3
│   ├── processed/         # Wstępnie sparsowane pliki CSV
│   ├── model_input/       # Dane gotowe dla modelu (residua + norm)
│   ├── results/           # Wyniki detekcji (flagi anomalii)
│   ├── plots/             # Wykresy rzeczywiste
│   ├── simulated/         # Wykresy z benchmarku syntetycznego
│   └── gnss_network_map.html # Interaktywna mapa awaryjności
├── models/                # Zapisane modele .pkl (joblib)
├── src/                   # Kody źródłowe
│   ├── download_data.py   # Pobieranie z NGL
│   ├── data_loader.py     # Parsowanie formatu .tenv3 do CSV
│   ├── preprocess.py      # ETL, usuwanie sezonowości, konwersja do mm
│   ├── detect_anomalies.py# Logika ML (Isolation Forest 3D)
│   ├── benchmark_synthetic.py # Testy syntetyczne i krzywa Recall
│   ├── visualize.py       # Wizualizacja (Scatter plot + Trend)
│   └── create_map.py      # Generowanie mapy Folium z gradientem
├── requirements.txt       # Lista bibliotek
└── README.md              # Dokumentacja
```

### Autor
Szymon Zarosa - student kierunku Geodezja i Kartografia na Wydziale Geodezji Górniczej i Inżynierii Środowiska, AGh.