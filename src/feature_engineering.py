"""Section 3 - Feature engineering."""
import numpy as np
import pandas as pd

from src.config import CATEGORICAL, NUMERIC, TARGET


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features. Works for training and for single predictions."""
    df = df.copy()
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_rush_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclic encoding so 23:00 and 00:00 are close together
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def split_xy(df: pd.DataFrame):
    df = add_features(df)
    return df[NUMERIC + CATEGORICAL], df[TARGET]
