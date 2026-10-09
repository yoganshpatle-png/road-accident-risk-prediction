import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection import generate_synthetic_data
from src.feature_engineering import add_features, split_xy
from src.hotspot_analysis import find_hotspots
from src.predict import probability_to_level
from src.preprocessing import clean_data


def test_cleaning_removes_missing():
    df = clean_data(generate_synthetic_data(1500))
    assert df.isna().sum().sum() == 0


def test_features_added():
    df = clean_data(generate_synthetic_data(500))
    assert {"is_night", "hour_sin", "is_weekend"} <= set(add_features(df).columns)
    X, y = split_xy(df)
    assert len(X) == len(y)


def test_risk_levels():
    assert probability_to_level(0.1) == "Low"
    assert probability_to_level(0.4) == "Moderate"
    assert probability_to_level(0.6) == "High"
    assert probability_to_level(0.9) == "Critical"


def test_hotspots_found():
    df = clean_data(generate_synthetic_data(3000))
    _, summary = find_hotspots(df)
    assert len(summary) >= 1
