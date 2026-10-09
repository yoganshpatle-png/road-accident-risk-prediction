"""Section 5 - Model evaluation and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.config import OUTPUT_DIR, RANDOM_STATE


def evaluate_model(pipe, X_test, y_test):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    pd.Series(metrics).to_csv(OUTPUT_DIR / "metrics.csv", header=["value"])
    print(classification_report(y_test, pred))

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
    imp = permutation_importance(pipe, sample, y_test.loc[sample.index],
                                 n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc")
    s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    s.plot.barh(ax=ax, color="#2A9D8F"); ax.set_title("Top feature importances")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150); plt.close(fig)
    return metrics
