
"""Final evaluation on the untouched chronological test set."""

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

TEST_PATH = ROOT / "data/processed/splits/test.csv"
BASELINE_DIR = ROOT / "artifacts/baselines"
TREE_DIR = ROOT / "artifacts/tree_models"
REPORT_DIR = ROOT / "reports/models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def evaluate_classifier(name, model, X, y):
    probabilities_all = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    probabilities = probabilities_all[:, positive_index]

    predictions = (probabilities >= 0.50).astype(int)

    return {
        "model": name,
        "threshold": 0.50,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def evaluate_regressor(name, y, predictions):
    return {
        "model": name,
        "mae": mean_absolute_error(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions)),
        "r2": r2_score(y, predictions),
        "directional_accuracy": accuracy_score(
            (y > 0).astype(int),
            (predictions > 0).astype(int),
        ),
    }


def main():
    print("Loading untouched test data...")

    test = pd.read_csv(TEST_PATH)

    excluded = {"date", "Date", "target_return", "target_direction"}
    feature_columns = [
        column for column in test.columns
        if column not in excluded
    ]

    X_test = test[feature_columns]
    y_direction = test["target_direction"].astype(int)
    y_return = test["target_return"].astype(float)

    print(f"Test records: {len(test)}")
    print(f"Features: {len(feature_columns)}")
    date_column = "date" if "date" in test.columns else "Date"

    print(
        f"Test period: {test[date_column].iloc[0]} "
        f"to {test[date_column].iloc[-1]}"
    )

    # Classification
    classifiers = {
        "Dummy Classifier": BASELINE_DIR / "dummy_classifier.joblib",
        "Logistic Regression": BASELINE_DIR / "logistic_regression.joblib",
        "Random Forest Classifier": TREE_DIR
        / "random_forest_classifier.joblib",
        "XGBoost Classifier": TREE_DIR / "xgboost_classifier.joblib",
    }

    classification_results = []

    print("\nEvaluating classification models...")

    for name, path in classifiers.items():
        model = load_model(path)
        classification_results.append(
            evaluate_classifier(name, model, X_test, y_direction)
        )

    classification_df = pd.DataFrame(classification_results)

    classification_df.to_csv(
        REPORT_DIR / "final_test_classification.csv",
        index=False,
    )

    # Regression
    regressors = {
        "Ridge Regression": BASELINE_DIR / "ridge_regression.joblib",
        "Random Forest Regressor": TREE_DIR
        / "random_forest_regressor.joblib",
        "XGBoost Regressor": TREE_DIR / "xgboost_regressor.joblib",
    }

    zero_predictions = np.zeros(len(y_return))
    zero_result = evaluate_regressor(
        "Zero Return Baseline", y_return, zero_predictions
    )
    zero_result["directional_accuracy"] = np.nan
    regression_results = [zero_result]
    print("\nEvaluating regression models...")

    for name, path in regressors.items():
        model = load_model(path)
        predictions = model.predict(X_test)

        regression_results.append(
            evaluate_regressor(name, y_return, predictions)
        )

    regression_df = pd.DataFrame(regression_results)

    regression_df.to_csv(
        REPORT_DIR / "final_test_regression.csv",
        index=False,
    )

    print("\n========== FINAL TEST CLASSIFICATION ==========")
    print(classification_df.round(4).to_string(index=False))

    print("\n========== FINAL TEST REGRESSION ==========")
    print(regression_df.round(6).to_string(index=False))

    print("\nReports saved:")
    print(REPORT_DIR / "final_test_classification.csv")
    print(REPORT_DIR / "final_test_regression.csv")

    print("\nFinal test evaluation completed.")
    print("Do not tune models using these test results.")


if __name__ == "__main__":
    main()
