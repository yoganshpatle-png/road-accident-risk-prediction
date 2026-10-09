"""Section 4 - Model training and selection."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL, MODEL_PATH, NUMERIC, OUTPUT_DIR, RANDOM_STATE
from src.feature_engineering import split_xy


def build_candidates():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    try:  # optional dependency
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08,
                                          eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        pass
    return models


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def train_and_select(df: pd.DataFrame):
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, fitted = [], {}
    for name, model in build_candidates().items():
        pipe = make_pipeline(model).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results.append({"model": name,
                        "f1": f1_score(y_test, pipe.predict(X_test)),
                        "roc_auc": roc_auc_score(y_test, proba)})
        fitted[name] = pipe
        print(f"[train] {name:20s} F1={results[-1]['f1']:.3f}  AUC={results[-1]['roc_auc']:.3f}")

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = table.loc[0, "model"]
    best = fitted[best_name]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    joblib.dump(best, MODEL_PATH)
    print(f"[train] best model: {best_name} -> saved to {MODEL_PATH}")
    return best, table, (X_test, y_test)
