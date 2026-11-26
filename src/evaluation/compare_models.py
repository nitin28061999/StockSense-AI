
"""Compare baseline and tree models using validation data only."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = ROOT / "data/processed/splits/validation.csv"
BASELINE_DIR = ROOT / "artifacts/baselines"
TREE_DIR = ROOT / "artifacts/tree_models"
REPORT_DIR = ROOT / "reports/models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def classification_metrics(name, y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def regression_metrics(name, y_true, predictions):
    return {
        "model": name,
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
        "directional_accuracy": accuracy_score(
            (y_true > 0).astype(int),
            (predictions > 0).astype(int),
        ),
    }


def main():  # sourcery skip: extract-duplicate-method
    print("Loading validation data...")

    validation = pd.read_csv(VALIDATION_PATH)

    excluded = {"date", "Date", "target_return", "target_direction"}
    feature_columns = [
        column for column in validation.columns
        if column not in excluded
    ]

    X_val = validation[feature_columns]
    y_direction = validation["target_direction"].astype(int)
    y_return = validation["target_return"].astype(float)

    print(f"Validation records: {len(validation)}")
    print(f"Features: {len(feature_columns)}")

    # Classification models
    classifiers = {
        "Dummy Classifier": BASELINE_DIR / "dummy_classifier.joblib",
        "Logistic Regression": BASELINE_DIR / "logistic_regression.joblib",
        "Random Forest Classifier": TREE_DIR
        / "random_forest_classifier.joblib",
        "XGBoost Classifier": TREE_DIR / "xgboost_classifier.joblib",
    }

    classification_results = []
    threshold_results = []

    print("\nEvaluating classification models...")

    for name, path in classifiers.items():
        model = load_model(path)

        probabilities_all = model.predict_proba(X_val)
        positive_index = list(model.classes_).index(1)
        probabilities = probabilities_all[:, positive_index]

        classification_results.append(
            classification_metrics(
                name, y_direction, probabilities, threshold=0.5
            )
        )

        if name == "XGBoost Classifier":
            for threshold in np.round(
                np.arange(0.30, 0.701, 0.025), 3
            ):
                threshold_results.append(
                    classification_metrics(
                        name, y_direction, probabilities, threshold
                    )
                )

    classification_df = pd.DataFrame(classification_results)
    classification_df.to_csv(
        REPORT_DIR / "validation_classification_comparison.csv",
        index=False,
    )

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(
        REPORT_DIR / "xgboost_threshold_analysis.csv",
        index=False,
    )

    # Regression models
    regressors = {
        "Ridge Regression": BASELINE_DIR / "ridge_regression.joblib",
        "Random Forest Regressor": TREE_DIR
        / "random_forest_regressor.joblib",
        "XGBoost Regressor": TREE_DIR / "xgboost_regressor.joblib",
    }



    print("\nEvaluating regression models...")

    zero_baseline_metrics = regression_metrics(
        "Zero Return Baseline",
        y_return,
        np.zeros(len(y_return)),
    )

    zero_baseline_metrics["directional_accuracy"] = np.nan

    regression_results = [zero_baseline_metrics]

    for name, path in regressors.items():
        model = load_model(path)
        predictions = model.predict(X_val)

        regression_results.append(
            regression_metrics(name, y_return, predictions)
        )

    regression_df = pd.DataFrame(regression_results)
    regression_df.to_csv(
        REPORT_DIR / "validation_regression_comparison.csv",
        index=False,
    )

    # Display results
    print("\n========== CLASSIFICATION COMPARISON ==========")
    print(classification_df.round(4).to_string(index=False))

    print("\n========== XGBOOST THRESHOLD ANALYSIS ==========")
    print(
        threshold_df[
            ["threshold", "accuracy", "balanced_accuracy",
             "precision", "recall", "f1"]
        ].round(4).to_string(index=False)
    )

    best_row = threshold_df.loc[
        threshold_df["balanced_accuracy"].idxmax()
    ] # pyright: ignore[reportArgumentType, reportCallIssue]

    print("\nBest validation threshold by balanced accuracy:")
    print(f"Threshold: {best_row['threshold']:.3f}")
    print(f"Balanced accuracy: {best_row['balanced_accuracy']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"F1: {best_row['f1']:.4f}")

    print("\n========== REGRESSION COMPARISON ==========")
    print(regression_df.round(6).to_string(index=False))

    print("\nReports saved:")
    print(REPORT_DIR / "validation_classification_comparison.csv")
    print(REPORT_DIR / "xgboost_threshold_analysis.csv")
    print(REPORT_DIR / "validation_regression_comparison.csv")

    print("\nValidation comparison completed. Test data was not used.")


if __name__ == "__main__":
    main()
