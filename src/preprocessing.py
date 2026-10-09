"""Section 2 - Data cleaning and preprocessing."""
import pandas as pd

from src.config import CATEGORICAL, PROCESSED_DATA


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    df = df[df["hour"].between(0, 23) & df["day_of_week"].between(0, 6) & df["month"].between(1, 12)]

    # Fill missing values: categorical -> mode, numeric -> median
    for col in df.columns:
        if df[col].isna().any():
            if col in CATEGORICAL or df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    df["speed_limit"] = df["speed_limit"].clip(10, 130)
    df = df.reset_index(drop=True)
    print(f"[clean] {before} -> {len(df)} rows")
    return df


def save_processed(df: pd.DataFrame, path=PROCESSED_DATA):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
