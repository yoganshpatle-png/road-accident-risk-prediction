"""Section 8 - Interactive map (folium): heatmap + ranked hotspot markers."""
import folium
from folium.plugins import HeatMap

from src.config import OUTPUT_DIR


def build_map(df, hotspots, top_n=10):
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    HeatMap(df[["latitude", "longitude"]].values.tolist(), radius=12, blur=15,
            name="Accident density").add_to(m)

    layer = folium.FeatureGroup(name="Top hotspots").add_to(m)
    for _, r in hotspots.head(top_n).iterrows():
        color = "darkred" if r["severe_rate"] > 0.5 else "orange"
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8 + min(r["accidents"] / 40, 14),
            color=color, fill=True, fill_opacity=0.6,
            tooltip=f"Hotspot #{int(r['rank'])}",
            popup=(f"<b>Hotspot #{int(r['rank'])}</b><br>Accidents: {int(r['accidents'])}"
                   f"<br>Severe rate: {r['severe_rate']:.0%}<br>Score: {r['score']}"),
        ).add_to(layer)
    folium.LayerControl().add_to(m)
    return m


def save_map(m, name="hotspot_map.html"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    m.save(str(path))
    return path
