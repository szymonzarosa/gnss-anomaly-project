"""
src/create_map.py
-----------------
Moduł wizualizacji przestrzennej (GIS).

Generuje interaktywny plik HTML z mapą stacji GNSS.
Kolor każdej stacji jest wyznaczany dynamicznie na podstawie wskaźnika awaryjności
(procent dni anomalnych), wykorzystując płynny gradient (Heatmap approach).

Metoda oceny: Hybrydowa (Isolation Forest + 3-Sigma).
Stacja jest "czerwona" tylko wtedy, gdy AI wykryje anomalię ORAZ ma ona dużą amplitudę fizyczną.

Autor: Szymon Zarosa
Data: 2025-11-27
"""

import folium
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# KONFIGURACJA ŚCIEŻEK
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
OUTPUT_MAP = PROJECT_ROOT / "data" / "gnss_network_map.html"

# KONFIGURACJA SKALI KOLORYSTYCZNEJ
# Wartość wskaźnika (%), powyżej której stacja uznawana jest za krytyczną (100% czerwieni).
MAX_RATIO = 1.2


def get_color_from_value(value: float, max_val: float) -> str:
    """
    Zamienia wartość liczbową (0 - max_val) na kolor HEX z gradientu Zielony->Żółty->Czerwony.
    Wykorzystuje paletę 'RdYlGn_r' z biblioteki matplotlib.
    """
    # Normalizacja wartości do zakresu 0.0 - 1.0
    norm = value / max_val
    if norm > 1.0: norm = 1.0
    if norm < 0.0: norm = 0.0

    # Pobieramy mapę kolorów (Red-Yellow-Green reversed)
    # 0.0 = Zielony, 0.5 = Żółty, 1.0 = Czerwony
    cmap = plt.get_cmap('RdYlGn_r')

    # Pobieramy kolor RGBA
    rgba = cmap(norm)

    # Konwersja na format HEX (np. #FF0505) dla HTML/CSS
    return mcolors.to_hex(rgba)


def normalize_longitude(lon: float) -> float:
    """
    Normalizuje długość geograficzną do przedziału [-180, 180].
    NGL często podaje np. 338 stopni zamiast -22 stopni.
    """
    while lon < -180: lon += 360
    while lon > 180: lon -= 360
    return lon


def get_station_coordinates(station_name: str) -> tuple[float, float] | tuple[None, None]:
    """
    Pobiera współrzędne stacji z surowego pliku .tenv3.
    Jest odporna na różne formatowanie (szuka kolumn od końca wiersza).
    """
    pattern = str(RAW_DIR / f"*{station_name}*.tenv3")
    files = glob.glob(pattern)

    if not files:
        return None, None

    try:
        with open(files[0], "r") as f:
            for line in f:
                # Pomijamy nagłówki
                if "site" in line.lower() or "yy" in line.lower(): continue

                parts = line.split()
                # Sprawdzamy, czy to linia z danymi dla naszej stacji
                if len(parts) > 10 and station_name in parts[0]:
                    # Format NGL: ... Lat Lon Height (na końcu linii)
                    lat = float(parts[-3])
                    lon = float(parts[-2])
                    return lat, normalize_longitude(lon)
    except Exception:
        return None, None

    return None, None


def calculate_real_anomalies(df: pd.DataFrame, station_name: str, sigma_thresh: int = 3):
    """
    Oblicza liczbę dni anomalnych metodą hybrydową.

    Warunek awarii:
    1. Isolation Forest zwrócił -1 (Anomalia).
    2. ORAZ wartość (E/N/U) przekroczyła 3 odchylenia standardowe (sigma_thresh).

    Eliminuje to "szum" wykrywany przez AI na bardzo stabilnych stacjach.
    """
    cols_data = [f"{station_name}_east", f"{station_name}_north", f"{station_name}_up"]
    col_anom = f"{station_name}_anomaly"

    if col_anom not in df.columns or not all(c in df.columns for c in cols_data):
        return 0, 0

    # Ignorujemy dni, gdzie stacja nie działała (sztuczne zera)
    magnitude = df[cols_data].abs().sum(axis=1)
    df_active = df[magnitude > 0.0001].copy()

    if df_active.empty:
        return 0, 0

    # Maska AI
    mask_model = (df_active[col_anom] == -1)

    # Maska Fizyczna (3 Sigma)
    mask_physics = pd.Series(False, index=df_active.index)
    for col in cols_data:
        val = df_active[col]
        mean = val.mean()
        std = val.std()
        is_outlier = (val > mean + sigma_thresh * std) | \
                     (val < mean - sigma_thresh * std)
        mask_physics = mask_physics | is_outlier

    # Koniunkcja warunków (AND)
    count = (mask_model & mask_physics).sum()
    return count, len(df_active)


def create_gnss_map():
    if not RESULTS_FILE.exists():
        print("Brak pliku wyników. Uruchom 'detect_anomalies.py'.")
        return

    print(f"Generowanie mapy sieci z: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=['date'], index_col='date')

    # Filtrowanie stacji (bez symulacji SIM_)
    cols = [c for c in df.columns if "_anomaly" in c]
    real_stations = sorted([c.replace("_anomaly", "") for c in cols if "SIM" not in c])

    # Inicjalizacja mapy (Centrum: Polska)
    m = folium.Map(location=[52.0, 19.5], zoom_start=6, tiles="CartoDB positron")

    # --- LEGENDA CSS (Pasek gradientowy) ---
    legend_html = f'''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 250px; height: 130px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity:0.95; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
     <b>Wskaźnik Awaryjności Stacji:</b><br>
     <small style="color:gray;">(Procent dni, w których występują anomalie)</small>
     <div style="margin-top: 10px; margin-bottom: 5px; 
          background: linear-gradient(to right, green, yellow, red); 
          width: 100%; height: 15px; border-radius: 3px; border: 1px solid #ccc;"></div>
     <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span>0.0% (Idealna)</span>
        <span>&ge; {MAX_RATIO}% (Awaria)</span>
     </div>
     <hr style="margin: 10px 0;">
     <small>Metoda: Hybrydowa (IF + 3&sigma;)</small>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    count = 0
    for station in real_stations:
        lat, lon = get_station_coordinates(station)
        if lat is None: continue

        n_anom, n_total = calculate_real_anomalies(df, station, sigma_thresh=3)
        if n_total == 0: ratio = 0
        else: ratio = (n_anom / n_total) * 100

        # Pobranie koloru z gradientu
        color_hex = get_color_from_value(ratio, max_val=MAX_RATIO)

        # Treść dymka (Popup)
        popup_text = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h3 style="margin:0;">📡 {station}</h3>
            <p style="margin: 5px 0; color: gray; font-size: 11px;">Lat: {lat:.3f}, Lon: {lon:.3f}</p>
            <hr style="border: 0; border-top: 1px solid #eee;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Anomalie:</span>
                <span style="font-weight: bold; font-size: 14px;">{n_anom} dni</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                <span>Wskaźnik:</span>
                <span style="font-weight: bold; font-size: 16px; color: {color_hex};">{ratio:.2f}%</span>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + (ratio * 3), # Skalowanie wielkości kropki
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{station}: {ratio:.2f}%",
            color="#333333",
            weight=1.5,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.9
        ).add_to(m)

        count += 1
        print(f" -> {station}: {ratio:.2f}% (Kolor: {color_hex})")

    m.save(OUTPUT_MAP)
    print(f"\nMapa Gradientowa gotowa: {OUTPUT_MAP} ({count} stacji)")

if __name__ == "__main__":
    create_gnss_map()