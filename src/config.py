"""Central configuration: paths, constants and risk thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "accidents.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "accidents_clean.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TARGET = "severe"            # 1 = serious/fatal accident, 0 = minor accident
N_SYNTHETIC_ROWS = 6000
CITY_CENTER = (28.6139, 77.2090)  # change to your city (lat, lon)

CATEGORICAL = ["weather", "road_type", "lighting", "vehicle_type", "traffic_density"]
NUMERIC = ["hour", "day_of_week", "month", "speed_limit", "is_holiday",
           "is_night", "is_rush_hour", "is_weekend", "hour_sin", "hour_cos",
           "latitude", "longitude"]

# Probability thresholds -> risk level
RISK_LEVELS = [(0.25, "Low"), (0.50, "Moderate"), (0.75, "High"), (1.01, "Critical")]

# Hotspot detection (DBSCAN)
HOTSPOT_EPS_KM = 0.5
HOTSPOT_MIN_SAMPLES = 15
