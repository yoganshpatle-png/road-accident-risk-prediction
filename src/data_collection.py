"""Section 1 - Data collection.

Loads a real accident CSV if one exists in data/raw/, otherwise generates a
realistic synthetic dataset so the whole project runs end to end.

Real data sources: MoRTH / data.gov.in (India), US Accidents (Kaggle),
UK STATS19. Map your columns to the names used below.
"""
import numpy as np
import pandas as pd

from src.config import CITY_CENTER, N_SYNTHETIC_ROWS, RANDOM_STATE, RAW_DATA

REQUIRED_COLUMNS = ["latitude", "longitude", "hour", "day_of_week", "month",
                    "weather", "road_type", "lighting", "vehicle_type",
                    "traffic_density", "speed_limit", "is_holiday", "severe"]


def generate_synthetic_data(n=N_SYNTHETIC_ROWS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    lat0, lon0 = CITY_CENTER

    # Three accident hotspots plus background scatter
    centers = [(lat0 + 0.03, lon0 + 0.02), (lat0 - 0.04, lon0 - 0.03), (lat0 + 0.01, lon0 - 0.06)]
    per_hot = int(n * 0.6) // len(centers)
    pts = [np.c_[rng.normal(la, 0.004, per_hot), rng.normal(lo, 0.004, per_hot)] for la, lo in centers]
    k_bg = n - per_hot * len(centers)
    pts.append(np.c_[rng.normal(lat0, 0.07, k_bg), rng.normal(lon0, 0.07, k_bg)])
    coords = np.vstack(pts)
    n = len(coords)

    df = pd.DataFrame({
        "latitude": coords[:, 0], "longitude": coords[:, 1],
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "weather": rng.choice(["clear", "rain", "fog", "cloudy"], n, p=[0.55, 0.2, 0.1, 0.15]),
        "road_type": rng.choice(["highway", "urban", "rural", "junction"], n, p=[0.25, 0.4, 0.15, 0.2]),
        "lighting": rng.choice(["daylight", "dark_lit", "dark_unlit"], n, p=[0.6, 0.25, 0.15]),
        "vehicle_type": rng.choice(["two_wheeler", "car", "truck", "bus"], n, p=[0.4, 0.35, 0.15, 0.1]),
        "traffic_density": rng.choice(["low", "medium", "high"], n, p=[0.3, 0.45, 0.25]),
        "speed_limit": rng.choice([30, 40, 50, 60, 80, 100], n),
        "is_holiday": rng.choice([0, 1], n, p=[0.92, 0.08]),
    })

    # Severity depends on conditions (so models have something to learn)
    z = (-1.6
         + 0.9 * df["weather"].isin(["rain", "fog"])
         + 0.8 * (df["lighting"] == "dark_unlit")
         + 0.5 * (df["road_type"] == "highway")
         + 0.4 * (df["vehicle_type"] == "two_wheeler")
         + 0.5 * (df["vehicle_type"] == "truck")
         + 0.012 * (df["speed_limit"] - 50)
         + 0.5 * df["hour"].isin([22, 23, 0, 1, 2, 3, 4])
         + 0.3 * df["is_holiday"])
    p = 1 / (1 + np.exp(-z.astype(float)))
    df["severe"] = (rng.random(n) < p).astype(int)

    # Inject a few missing values to exercise the cleaning step
    for col in ["weather", "speed_limit"]:
        df.loc[rng.choice(n, 40, replace=False), col] = np.nan
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data(path=RAW_DATA):
    """Load raw accident data; create synthetic data if the file is missing."""
    if path.exists():
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        print(f"[data] loaded {len(df)} rows from {path}")
        return df
    print("[data] no raw CSV found, generating synthetic data")
    df = generate_synthetic_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] saved synthetic data to {path}")
    return df


if __name__ == "__main__":
    print(load_data().head())
