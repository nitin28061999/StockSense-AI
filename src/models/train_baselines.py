
"""Train baseline classification and regression models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed" / "splits"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "baselines"
REPORT_DIR = ROOT_DIR / "reports" / "models"

FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "sma_10_distance",
    "ema_10_distance",
    "sma_20_distance",
    "ema_20_distance",
    "sma_50_distance",
    "ema_50_distance",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "atr_14_pct",
    "bb_upper_distance",
    "bb_lower_distance",
    "bb_width",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "day_of_week",
    "month",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train[FEATURE_COLUMNS]
    X_val = validation[FEATURE_COLUMNS]

    y_train_direction = train["target_direction"]
    y_val_direction = validation["target_direction"]

    y_train_return = train["target_return"]
    y_val_return = validation["target_return"]

    return (
        X_train,
        X_val,
        y_train_direction,
        y_val_direction,
        y_train_return,
        y_val_return,
    )


def evaluate_classifier(name, model, X_val, y_true):
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    return {
        "task": "classification",
        "model": name,
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
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def evaluate_regressor(name, model, X_val, y_true):
    predictions = model.predict(X_val)

    actual_direction = np.sign(y_true)
    predicted_direction = np.sign(predictions)

    return {
        "task": "regression",
        "model": name,
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
        "directional_accuracy": np.mean(
            actual_direction == predicted_direction
        ),
    }


def main():
    print("Loading chronological datasets...")

    (
        X_train,
        X_val,
        y_train_direction,
        y_val_direction,
        y_train_return,
        y_val_return,
    ) = load_data()

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    print("\nTraining classification baselines...")

    classifiers = {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    for name, model in classifiers.items():
        print(f"Fitting {name}...")

        model.fit(X_train, y_train_direction)

        metrics = evaluate_classifier(
            name,
            model,
            X_val,
            y_val_direction,
        )

        results.append(metrics)

        joblib.dump(
            model,
            ARTIFACT_DIR / f"{name.lower().replace(' ', '_')}.joblib",
        )

    # --------------------------------------------------
    # REGRESSION
    # --------------------------------------------------

    print("\nTraining regression baselines...")

    regressors = {
        "Zero Return Baseline": DummyRegressor(
            strategy="constant",
            constant=0.0,
        ),
        "Ridge Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    for name, model in regressors.items():
        print(f"Fitting {name}...")

        model.fit(X_train, y_train_return)

        metrics = evaluate_regressor(
            name,
            model,
            X_val,
            y_val_return,
        )

        results.append(metrics)

        joblib.dump(
            model,
            ARTIFACT_DIR / f"{name.lower().replace(' ', '_')}.joblib",
        )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n========== BASELINE MODEL RESULTS ==========")

    for task in ["classification", "regression"]:
        task_results = results_df[
            results_df["task"] == task
        ]

        print(f"\n{task.upper()}")
        print(
            task_results.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\nModels saved to:", ARTIFACT_DIR)
    print("Metrics saved to:", output_path)
    print("\nValidation completed. Test data was not used.")
    print("============================================")


if __name__ == "__main__":
    main()
