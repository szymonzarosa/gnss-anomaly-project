"""
src/create_map.py
-----------------
Generuje interaktywną mapę stacji GNSS.

Autor: Szymon Zarosa
Data: 2025-11-22
"""

import folium
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ścieżki
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "gnss_anomalies.csv"
OUTPUT_MAP = PROJECT_ROOT / "data" / "gnss_network_map.html"

# KONFIGURACJA SKALI KOLORYSTYCZNEJ
MAX_RATIO = 1.2

def get_color_from_value(value, max_val):
    """
    Zamienia wartość liczbową (0 - max_val) na kolor HEX z gradientu Zielony->Żółty->Czerwony.
    """
    # Normalizacja wartości do zakresu 0.0 - 1.0
    norm = value / max_val
    if norm > 1.0: norm = 1.0
    if norm < 0.0: norm = 0.0

    # Pobieramy mapę kolorów z matplotlib
    # RdYlGn_r = Red-Yellow-Green (Reversed) -> czyli 0=Zielony, 1=Czerwony
    cmap = plt.get_cmap('RdYlGn_r')

    # Pobieramy kolor RGBA
    rgba = cmap(norm)

    # Konwersja na HEX (np. #FF0505)
    return mcolors.to_hex(rgba)

def normalize_longitude(lon):
    while lon < -180: lon += 360
    while lon > 180: lon -= 360
    return lon

def get_station_coordinates(station_name):
    pattern = str(RAW_DIR / f"*{station_name}*.tenv3")
    files = glob.glob(pattern)
    if not files: return None, None
    try:
        with open(files[0], "r") as f:
            for line in f:
                if "site" in line.lower() or "yy" in line.lower(): continue
                parts = line.split()
                if len(parts) > 10 and station_name in parts[0]:
                    return float(parts[-3]), normalize_longitude(float(parts[-2]))
    except Exception: return None, None
    return None, None

def calculate_real_anomalies(df, station_name, sigma_thresh=3):
    cols_data = [f"{station_name}_east", f"{station_name}_north", f"{station_name}_up"]
    col_anom = f"{station_name}_anomaly"
    if col_anom not in df.columns or not all(c in df.columns for c in cols_data):
        return 0, 0

    magnitude = df[cols_data].abs().sum(axis=1)
    df_active = df[magnitude > 0.0001].copy()
    if df_active.empty: return 0, 0

    mask_model = (df_active[col_anom] == -1)
    mask_physics = pd.Series(False, index=df_active.index)
    for col in cols_data:
        val = df_active[col]
        is_outlier = (val > val.mean() + sigma_thresh * val.std()) | \
                     (val < val.mean() - sigma_thresh * val.std())
        mask_physics = mask_physics | is_outlier

    return (mask_model & mask_physics).sum(), len(df_active)

def create_gnss_map():
    if not RESULTS_FILE.exists():
        print("Brak wyników.")
        return

    print(f"Wczytywanie: {RESULTS_FILE.name}...")
    df = pd.read_csv(RESULTS_FILE, parse_dates=['date'], index_col='date')

    cols = [c for c in df.columns if "_anomaly" in c]
    real_stations = sorted([c.replace("_anomaly", "") for c in cols if "SIM" not in c])

    m = folium.Map(location=[52.0, 19.5], zoom_start=6, tiles="CartoDB positron")

    # --- LEGENDA Z GRADIENTEM (CSS) ---
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
        <span>0.0%</span>
        <span>&ge; {MAX_RATIO}%</span>
     </div>
     <hr style="margin: 10px 0;">
     <small>Analiza: IF + 3&sigma;</small>
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

        # --- DYNAMICZNY KOLOR ---
        color_hex = get_color_from_value(ratio, max_val=MAX_RATIO)

        popup_text = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h3 style="margin:0;">{station}</h3>
            <p style="margin: 5px 0; color: gray; font-size: 11px;">Latitude: {lat:.3f}, Longitude: {lon:.3f}</p>
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
            radius=8 + (ratio * 3), # Im gorsza stacja, tym większa kropka
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{station}: {ratio:.2f}%",
            color="#333333", # Ciemnoszara obwódka
            weight=1.5,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.9
        ).add_to(m)

        count += 1
        print(f" -> {station}: {ratio:.2f}% (Kolor: {color_hex})")

    m.save(OUTPUT_MAP)
    print(f"\nMapa Gradientowa gotowa: {OUTPUT_MAP}")

if __name__ == "__main__":
    create_gnss_map()