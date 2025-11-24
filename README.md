# Zastosowanie metod detekcji anomalii w szeregach czasowych GNSS do analizy stabilności obiektów i sieci geodezyjnych

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Science](https://img.shields.io/badge/Geodesy-GNSS-green)
![ML](https://img.shields.io/badge/ML-Isolation%20Forest-orange)

Projekt badawczy mający na celu automatyczną detekcję nietypowych zachowań (anomalii) stacji GNSS (np. awarie sprzętowe, ruchy geodynamiczne, uszkodzenia konstrukcji) przy użyciu algorytmu [**Isolation Forest**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

System analizuje dane w układzie współrzędnych topocentrycznych (ENU), uwzględnia usuwanie trendu i sezonowości, a także oferuje moduł walidacji poprzez symulację uszkodzeń inżynierskich.

## Funkcjonalności

* **Pobieranie danych:** Automatyczny import szeregów czasowych (format `.tenv3`) z *Nevada Geodetic Laboratory* (IGS20/EPN).
* **Parsowanie danych:** Konwersja surowych plików do formatu `pandas.DataFrame`.
* **Preprocessing:**
    * Interpolacja braków danych.
    * Usuwanie trendu liniowego i sezonowości (dekompozycja addytywna).
    * Normalizacja danych oraz konwersja jednostek [z metrów na milimetry].
* **Detekcja anomalii:** Zastosowanie uczenia nienadzorowanego (algorytm Isolation Forest) na wektorze 3D (East, North, Up) oraz cechach pochodnych (prędkość przemieszczeń).
* **Analiza czułości:** Wyznaczanie krzywej skuteczności (Recall) w zależności od dynamiki zjawiska (symulacja Step, Ramp, Noise).
* **Wizualizacja:** Generowanie wykresów punktowych z zaznaczonymi anomaliami i linią trendu.
* **Mapa:** Stworzenie interaktywnej mapy (HTML), pokazującej procent wykrytych anomalii.
* **Weryfikacja wyników:** Porównanie **Isolation Forest** z metodą **MAD** (Median Absolute Deviation)

## Wymagania i Instalacja

Projekt został napisany w języku Python. Do działania wymaga bibliotek wymienionych w [`requirements.txt`](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/requirements.txt).

1.  **Sklonuj repozytorium:**
    ```bash
    git clone <https://github.com/szymonzarosa/gnss-anomaly-project>
    cd gnss_anomaly_project
    ```

2.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```

## Instrukcja Użytkowania

Cały proces analizy podzielony jest na niezależne moduły w folderze `src/`. Należy uruchamiać je w następującej kolejności:

### 1. [Pobieranie danych](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/download_data.py)
Pobiera surowe pliki `.tenv3` z serwerów NGL dla zdefiniowanej listy stacji.
```bash
python src/download_data.py
```

### 2. [Parsowanie (Data Loader)](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/data_loader.py)
Konwertuje specyficzny format .tenv3 na ustandaryzowane pliki *.csv*, przeliczając daty MJD na format kalendarzowy.
```bash
python src/data_loader.py
```

### 3. [Preprocessing](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/preprocess.py)
Przygotowuje dane do analizy: usuwa składowe fizyczne (trend tektoniczny, sezonowość roczną), pozostawiając residua (reszty - szum oraz anomalie). Dokonuje normalizacji dla modelu ML.
```bash
python src/preprocess.py
```

Wynik: Plik *data/model_input/gnss_model_input.csv* zawierający residua (reszty) w [mm] oraz znormalizowane cechy.

### 4. [Detekcja Anomalii](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/detect_anomalies.py)
Trenuje model Isolation Forest dla każdej stacji osobno i klasyfikuje dni jako "Norma" (1) lub "Anomalia" (-1). Algorytm uczy się na danych znormalizowanych (cechy _norm), ale wyniki zapisuje w kontekście wartości fizycznych [mm].
```bash
python src/detect_anomalies.py
```
### 5. [Wizualizacja](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/visualize.py) i [Mapa](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/create_map.py)
Generuje raporty graficzne w dwóch formach:

* **Wykresy czasowe:** Szczegółowa analiza każdej stacji.

* **Mapa sieci:** Plik HTML z mapą, na której kolor stacji (gradient od zielonego do czerwonego) odzwierciedla procent wykrytych awarii (filtr: ML + 3σ).
```bash
python src/visualize.py
python src/create_map.py
```
Wynik: Wykresy PNG w folderze *data/plots/* oraz interaktywna mapa *data/gnss_network_map.html.*

### 6. [Walidacja i Analiza Czułości](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/benchmark_synthetic.py)
Projekt zawiera moduł (*benchmark_synthetic.py*), który bada skuteczność algorytmu w funkcji prędkości narastania awarii (np. osuwiska).

Aby wygenerować **Krzywą Czułości Detekcji** oraz przykładowe wykresy detekcji.
```bash
python src/benchmark_synthetic.py
```
Skrypt wygeneruje wykresy w folderze *data/simulated/*, pokazując przy jakim tempie deformacji (mm/dzień) algorytm osiąga zadowalającą skuteczność.

**Wnioski z analizy czułości:**

* **Step i Noise**: Wykrywalność bliska 100% dla nagłych zdarzeń.

* **Ramp (Dryf)**: Skuteczna detekcja (>65%) dla prędkości przemieszczeń powyżej 1.5 mm/dobę.

### 7. [Weryfikacja Wyników: Isolation Forest vs MAD (metoda statystyczna)](https://github.com/szymonzarosa/gnss-anomaly-project/blob/main/src/verify_mad.py)
W celu oceny wiarygodności detekcji przeprowadzono analizę porównawczą z wykorzystaniem statystyki odpornej – **MAD** (*Median Absolute Deviation*).

Dla każdej stacji obliczono wskaźnik *Modified Z-Score*. Dni, w których wartość wskaźnika przekroczyła próg **3.5**, uznano za anomalie statystyczne i zestawiono z wynikami algorytmu Isolation Forest.

**Wnioski z porównania:**
1.  **Różnica podejść:** Metoda statystyczna (MAD) działa w sposób **bezwzględny** (wykrywa tylko przekroczenia amplitudy), podczas gdy Isolation Forest działa w sposób **względny** (szuka najmniej prawdopodobnych wzorców w wielowymiarowej przestrzeni danych).
2.  **Stacje stabilne:** Dla stacji o wysokiej jakości danych (np. BYDG), algorytm wykazuje tendencję do nadczułości (wymusza znalezienie określonego procenta anomalii w szumie), podczas gdy MAD nie wskazuje błędów.
3.  **Stacje awaryjne:** W przypadku wystąpienia silnych zakłóceń (np. KRA1, REDZ), obie metody wykazują wysoką zgodność detekcji.
4.  **Rekomendacja:** Wyniki sugerują, że optymalnym rozwiązaniem inżynierskim jest podejście hybrydowe, tj. wykorzystanie Isolation Forest do wykrywania subtelnych anomalii przestrzennych oraz MAD do filtrowania fałszywych alarmów o niskiej amplitudzie.

Aby wygenerować szczegółowy raport porównawczy w formacie Excel:
```bash
python src/verify_mad.py
```
Wynik: Plik *data/results/raport_porownawczy.xlsx* zawierający macierz zgodności dla każdej stacji.

## Struktura projektu

```text
gnss_anomaly_project/
├── data/                       # Dane
│   ├── raw/                    # Pobrane pliki .tenv3
│   ├── processed/              # Wstępnie sparsowane pliki CSV
│   ├── model_input/            # Dane gotowe dla modelu (residua + norm)
│   ├── results/                # Wyniki detekcji (flagi anomalii)
│   ├── plots/                  # Wykresy rzeczywiste
│   ├── simulated/              # Wykresy symulowane
│   └── gnss_network_map.html   # Interaktywna mapa awaryjności
├── models/                     # Zapisane modele .pkl (joblib)
├── src/                        # Kody źródłowe
│   ├── download_data.py        # Pobieranie z NGL
│   ├── data_loader.py          # Parsowanie formatu .tenv3 do CSV
│   ├── preprocess.py           # ETL, usuwanie sezonowości, konwersja do mm
│   ├── detect_anomalies.py     # Logika ML (Isolation Forest 3D)
│   ├── benchmark_synthetic.py  # Testy syntetyczne i krzywa czułości detekcji
│   ├── visualize.py            # Wizualizacja (wykresy + Trend)
│   └── create_map.py           # Generowanie mapy Folium z gradientem
├── requirements.txt            # Lista bibliotek
└── README.md                   # Dokumentacja
```

### O autorze
Szymon Zarosa - student kierunku Geodezja i Kartografia na Wydziale Geodezji Górniczej i Inżynierii Środowiska, AGH.

### Opiekun projektu
[dr hab. inż. Kamil Maciuk](https://home.agh.edu.pl/~maciuk/)

### Bibliografia i źródła wykorzystane w projekcie
**Literatura przedmiotu:**
* Ganczarek-Gamrot, A. (2014). Analiza szeregów czasowych. Wydawnictwo Uniwersytetu Ekonomicznego w Katowicach.
* [Brusak, I., Maciuk, K., Haidus, O. (2025). Detection of Geodynamic Anomalies in GNSS Time Series using Machine Learning Methods.](https://www.researchgate.net/publication/393506656_Detection_of_geodynamic_anomalies_in_GNSS_time_series_using_machine_learning_methods)
* [Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. Eighth IEEE International Conference on Data Mining.](https://www.researchgate.net/publication/224384174_Isolation_Forest) 
* Iglewicz, B., & Hoaglin, D. C. (1993). How to Detect and Handle Outliers. ASQC Quality Press.

**Żródła danych i standardy:**
* [Nevada Geodetic Laboratory (NGL): Plug and Play GPS Data Products.](https://geodesy.unr.edu)
* [NIST/SEMATECH: e-Handbook of Statistical Methods. (Sekcja 1.3.5.17: Detection of Outliers).](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)
