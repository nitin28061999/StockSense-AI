
"""Descriptive model performance across market regimes.

Uses the previously evaluated test set and saved models.
Does not train, tune, or select models.
"""

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

OUTPUT_DIR = ROOT / "reports/market_regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

CLASSIFIERS = {
    "Dummy Classifier": BASELINE_DIR / "dummy_classifier.joblib",
    "Logistic Regression": BASELINE_DIR / "logistic_regression.joblib",
    "Random Forest Classifier": (
        TREE_DIR / "random_forest_classifier.joblib"
    ),
    "XGBoost Classifier": TREE_DIR / "xgboost_classifier.joblib",
}

REGRESSORS = {
    "Ridge Regression": BASELINE_DIR / "ridge_regression.joblib",
    "Random Forest Regressor": (
        TREE_DIR / "random_forest_regressor.joblib"
    ),
    "XGBoost Regressor": TREE_DIR / "xgboost_regressor.joblib",
}


def load_model(path):
    """Load a saved model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def get_positive_probabilities(model, X):
    """Get predicted probabilities for the Up class."""
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Positive class 1 is missing.")

    return probabilities[:, classes.index(1)]


def assign_market_regimes(df):
    """Assign observations to four descriptive market regimes."""
    median_volatility = df["volatility_20d"].median()

    df = df.copy()

    df["trend_regime"] = np.where(
        df["sma_50_distance"] > 0,
        "Uptrend",
        "Downtrend",
    )

    df["volatility_regime"] = np.where(
        df["volatility_20d"] >= median_volatility,
        "High Volatility",
        "Low Volatility",
    )

    df["market_regime"] = (
        df["volatility_regime"]
        + " / "
        + df["trend_regime"]
    )

    return df, median_volatility


def classification_metrics(
    y_true,
    probabilities,
    predictions,
):
    """Calculate classification metrics for one regime."""
    result = {
        "observations": len(y_true),
        "actual_up_ratio": np.mean(y_true),
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
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true, probabilities
        )
    else:
        result["roc_auc"] = np.nan

    return result


def regression_metrics(y_true, predictions):
    """Calculate regression metrics for one regime."""
    residuals = y_true - predictions

    return {
        "observations": len(y_true),
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "r2": (
            r2_score(y_true, predictions)
            if len(y_true) > 1
            else np.nan
        ),
        "directional_accuracy": accuracy_score(
            (y_true > 0).astype(int),
            (predictions > 0).astype(int),
        ),
        "mean_residual": np.mean(residuals),
    }


def main():
    print("Loading previously evaluated test data...")

    df = pd.read_csv(TEST_PATH)

    required_columns = {
        "target_direction",
        "target_return",
        "sma_50_distance",
        "volatility_20d",
    }

    if missing := required_columns - set(df.columns):
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    date_column = "date" if "date" in df.columns else "Date"

    if date_column not in df.columns:
        raise ValueError("Date column not found.")

    excluded = {
        "date",
        "Date",
        "target_return",
        "target_direction",
    }

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    X = df[feature_columns]

    if X.isnull().any().any():
        raise ValueError("Missing values in model features.")

    y_direction = df["target_direction"].astype(int).to_numpy()
    y_return = df["target_return"].astype(float).to_numpy()

    df, volatility_threshold = assign_market_regimes(df)

    print(f"Test observations: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Test period: {df[date_column].iloc[0]} "
        f"to {df[date_column].iloc[-1]}"
    )
    print(
        f"Median 20-day volatility threshold: "
        f"{volatility_threshold:.6f}"
    )

    # Save regime distribution.
    regime_counts = (
        df.groupby("market_regime", observed=True)
        .agg(
            observations=("target_return", "size"),
            mean_volatility=("volatility_20d", "mean"),
            mean_sma_50_distance=("sma_50_distance", "mean"),
            actual_up_ratio=("target_direction", "mean"),
            mean_actual_return=("target_return", "mean"),
        )
        .reset_index()
    )

    regime_counts.to_csv(
        OUTPUT_DIR / "market_regime_distribution.csv",
        index=False,
    )

    print("\n========== MARKET REGIME DISTRIBUTION ==========")
    print(regime_counts.round(6).to_string(index=False))

    classification_records = []
    regression_records = []
    # Zero-return baseline for each market regime.
    for regime in regime_counts["market_regime"]:
        mask = (
            df["market_regime"].to_numpy() == regime
        )

        zero_predictions = np.zeros(mask.sum())

        metrics = regression_metrics(
            y_return[mask],
            zero_predictions,
        )

        metrics["model"] = "Zero Return Baseline"
        metrics["market_regime"] = regime
        metrics["directional_accuracy"] = np.nan

        regression_records.append(metrics)
    print("\nRunning classification regime analysis...")

    for model_name, model_path in CLASSIFIERS.items():
        print(f"Analyzing: {model_name}")

        model = load_model(model_path)
        probabilities = get_positive_probabilities(model, X)
        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        for regime in regime_counts["market_regime"]:
            mask = (
                df["market_regime"].to_numpy() == regime
            )

            metrics = classification_metrics(
                y_direction[mask],
                probabilities[mask],
                predictions[mask],
            )

            metrics["model"] = model_name
            metrics["market_regime"] = regime
            metrics["threshold"] = THRESHOLD

            classification_records.append(metrics)

    classification_df = pd.DataFrame(
        classification_records
    )

    classification_df = classification_df[
        [
            "model",
            "market_regime",
            "observations",
            "threshold",
            "actual_up_ratio",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    ]

    classification_df.to_csv(
        OUTPUT_DIR / "classification_by_regime.csv",
        index=False,
    )

    print("\nRunning regression regime analysis...")

    for model_name, model_path in REGRESSORS.items():
        print(f"Analyzing: {model_name}")

        model = load_model(model_path)
        predictions = model.predict(X)

        for regime in regime_counts["market_regime"]:
            mask = (
                df["market_regime"].to_numpy() == regime
            )

            metrics = regression_metrics(
                y_return[mask],
                predictions[mask],
            )

            metrics["model"] = model_name
            metrics["market_regime"] = regime

            regression_records.append(metrics)

    regression_df = pd.DataFrame(regression_records)

    regression_df = regression_df[
        [
            "model",
            "market_regime",
            "observations",
            "mae",
            "rmse",
            "r2",
            "directional_accuracy",
            "mean_residual",
        ]
    ]

    regression_df.to_csv(
        OUTPUT_DIR / "regression_by_regime.csv",
        index=False,
    )

    print("\n========== CLASSIFICATION BY REGIME ==========")
    print(classification_df.round(4).to_string(index=False))

    print("\n========== REGRESSION BY REGIME ==========")
    print(regression_df.round(6).to_string(index=False))

    print("\nReports saved to:")
    print(OUTPUT_DIR)

    print("\nMarket regime analysis completed.")
    print("No model was trained or tuned.")
    print("Test results are descriptive only.")


if __name__ == "__main__":
    main()
