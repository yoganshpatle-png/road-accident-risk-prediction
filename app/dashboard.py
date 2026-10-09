"""Section 9 - Streamlit dashboard.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.config import CITY_CENTER, PROCESSED_DATA
from src.hotspot_analysis import find_hotspots
from src.map_visualization import build_map
from src.predict import load_model, predict_risk

st.set_page_config(page_title="Road Accident Risk & Hotspots", layout="wide")
st.title("AI-Based Road Accident Risk Prediction & Hotspot Analysis")

if not PROCESSED_DATA.exists():
    st.error("Run `python main.py` first to create data and train the model.")
    st.stop()


@st.cache_data
def get_data():
    return pd.read_csv(PROCESSED_DATA)


@st.cache_resource
def get_model():
    return load_model()


df, model = get_data(), get_model()
tab1, tab2 = st.tabs(["Risk prediction", "Hotspot map"])

with tab1:
    c1, c2, c3 = st.columns(3)
    hour = c1.slider("Hour of day", 0, 23, 22)
    day = c1.selectbox("Day of week (0=Mon)", list(range(7)), index=5)
    month = c1.slider("Month", 1, 12, 11)
    weather = c2.selectbox("Weather", ["clear", "rain", "fog", "cloudy"], index=2)
    road = c2.selectbox("Road type", ["highway", "urban", "rural", "junction"])
    light = c2.selectbox("Lighting", ["daylight", "dark_lit", "dark_unlit"], index=2)
    vehicle = c3.selectbox("Vehicle", ["two_wheeler", "car", "truck", "bus"])
    traffic = c3.selectbox("Traffic", ["low", "medium", "high"], index=1)
    speed = c3.selectbox("Speed limit", [30, 40, 50, 60, 80, 100], index=4)
    holiday = st.checkbox("Holiday")
    lat = st.number_input("Latitude", value=float(CITY_CENTER[0]), format="%.4f")
    lon = st.number_input("Longitude", value=float(CITY_CENTER[1]), format="%.4f")

    if st.button("Predict risk"):
        row = pd.DataFrame([{"latitude": lat, "longitude": lon, "hour": hour, "day_of_week": day,
                             "month": month, "weather": weather, "road_type": road,
                             "lighting": light, "vehicle_type": vehicle,
                             "traffic_density": traffic, "speed_limit": speed,
                             "is_holiday": int(holiday)}])
        res = predict_risk(row, model).iloc[0]
        st.metric("Risk level", res["risk_level"],
                  f"{res['risk_probability']:.0%} probability of severe accident")

with tab2:
    eps = st.slider("Cluster radius (km)", 0.1, 1.5, 0.5, 0.1)
    min_pts = st.slider("Minimum accidents per hotspot", 5, 50, 15)
    clustered, hotspots = find_hotspots(df, eps_km=eps, min_samples=min_pts)
    st.write(f"**{len(hotspots)} hotspots found**")
    st_folium(build_map(clustered, hotspots), width=1000, height=520)
    st.dataframe(hotspots.head(10), use_container_width=True)
