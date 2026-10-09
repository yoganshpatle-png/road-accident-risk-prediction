"""Section 6 - Risk prediction and risk levels."""
import joblib
import pandas as pd

from src.config import CATEGORICAL, MODEL_PATH, NUMERIC, RISK_LEVELS
from src.feature_engineering import add_features


def probability_to_level(p: float) -> str:
    for limit, label in RISK_LEVELS:
        if p < limit:
            return label
    return RISK_LEVELS[-1][1]


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python main.py` first.")
    return joblib.load(path)


def predict_risk(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return the input with risk_probability and risk_level columns."""
    model = model or load_model()
    feats = add_features(df)
    proba = model.predict_proba(feats[NUMERIC + CATEGORICAL])[:, 1]
    out = df.copy()
    out["risk_probability"] = proba.round(3)
    out["risk_level"] = [probability_to_level(p) for p in proba]
    return out


if __name__ == "__main__":
    example = pd.DataFrame([{
        "latitude": 28.64, "longitude": 77.23, "hour": 23, "day_of_week": 5, "month": 11,
        "weather": "fog", "road_type": "highway", "lighting": "dark_unlit",
        "vehicle_type": "two_wheeler", "traffic_density": "medium",
        "speed_limit": 80, "is_holiday": 0}])
    print(predict_risk(example)[["risk_probability", "risk_level"]])
