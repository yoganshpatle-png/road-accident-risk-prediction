"""Section 7 - Accident hotspot analysis (DBSCAN clustering + KDE density)."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

from src.config import HOTSPOT_EPS_KM, HOTSPOT_MIN_SAMPLES, TARGET

EARTH_RADIUS_KM = 6371.0088
SUMMARY_COLUMNS = ["cluster", "accidents", "severe_rate", "latitude", "longitude", "score", "rank"]


def find_hotspots(df: pd.DataFrame, eps_km=HOTSPOT_EPS_KM, min_samples=HOTSPOT_MIN_SAMPLES):
    """Cluster accident locations with DBSCAN (haversine distance).

    Returns (df_with_cluster_column, hotspot_summary_table).
    Cluster -1 means the point is not part of any hotspot.
    """
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit_predict(coords)
    out = df.copy()
    out["cluster"] = labels

    clustered = out[out["cluster"] >= 0]
    if clustered.empty:
        return out, pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (clustered.groupby("cluster")
               .agg(accidents=("cluster", "size"),
                    severe_rate=(TARGET, "mean"),
                    latitude=("latitude", "mean"),
                    longitude=("longitude", "mean"))
               .reset_index())
    # Priority score: many accidents AND high severity
    summary["score"] = (summary["accidents"] * (0.5 + summary["severe_rate"])).round(1)
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary["severe_rate"] = summary["severe_rate"].round(3)
    return out, summary


def kde_density(df: pd.DataFrame, bandwidth_km=0.4, grid_size=80):
    """Kernel density estimate over a lat/lon grid. Returns (lats, lons, density)."""
    pts = np.radians(df[["latitude", "longitude"]].to_numpy())
    kde = KernelDensity(bandwidth=bandwidth_km / EARTH_RADIUS_KM, metric="haversine").fit(pts)
    lats = np.linspace(df["latitude"].min(), df["latitude"].max(), grid_size)
    lons = np.linspace(df["longitude"].min(), df["longitude"].max(), grid_size)
    grid = np.array([[la, lo] for la in lats for lo in lons])
    dens = np.exp(kde.score_samples(np.radians(grid))).reshape(grid_size, grid_size)
    return lats, lons, dens
