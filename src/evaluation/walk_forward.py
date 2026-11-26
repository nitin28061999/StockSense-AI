
"""Expanding-window walk-forward validation for StockSense AI.

Uses only training and validation data.
The final chronological test set is never loaded.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_DIR = ROOT / "reports" / "models"

N_SPLITS = 5
TEST_SIZE = 395
CLASSIFICATION_THRESHOLD = 0.50
RANDOM_STATE = 42


def load_pretest_data():
    """Load only training and validation datasets."""

    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    data = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    date_column = (
        "date" if "date" in data.columns else "Date"
    )

    if date_column not in data.columns:
        raise ValueError("No date column found.")

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column).reset_index(drop=True)

    if data[date_column].duplicated().any():
        raise ValueError("Duplicate dates found.")

    excluded_columns = {
        "date",
        "Date",
        "target_return",
        "target_direction",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    required_columns = {
        "target_return",
        "target_direction",
    }

    if not required_columns.issubset(data.columns):
        raise ValueError("Required target columns are missing.")

    if len(feature_columns) != 24:
        raise ValueError(
            f"Expected 24 features, found {len(feature_columns)}."
        )

    if data[feature_columns].isnull().any().any():
        raise ValueError("Missing feature values detected.")

    if data[list(required_columns)].isnull().any().any():
        raise ValueError("Missing target values detected.")

    if not all(
        pd.api.types.is_numeric_dtype(data[column])
        for column in feature_columns
    ):
        raise ValueError("Non-numeric feature detected.")

    return data, date_column, feature_columns


def create_classification_models(y_train):
    """Create fresh classification models for each fold."""

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training fold contains only one class.")

    return {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent",
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2,
            eval_metric="logloss",
            scale_pos_weight=negative_count / positive_count,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def create_regression_models():
    """Create fresh regression models for each fold."""

    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def classification_metrics(
    model_name,
    fold_number,
    y_true,
    probabilities,
    dates,
):
    """Calculate classification metrics at the fixed threshold."""

    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    return {
        "fold": fold_number,
        "model": model_name,
        "start_date": dates.iloc[0].date(),
        "end_date": dates.iloc[-1].date(),
        "observations": len(y_true),
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


def regression_metrics(
    model_name,
    fold_number,
    y_true,
    predictions,
    dates,
):
    """Calculate regression and directional metrics."""

    return {
        "fold": fold_number,
        "model": model_name,
        "start_date": dates.iloc[0].date(),
        "end_date": dates.iloc[-1].date(),
        "observations": len(y_true),
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "r2": r2_score(y_true, predictions),
        "directional_accuracy": accuracy_score(
            (y_true > 0).astype(int),
            (predictions > 0).astype(int),
        ),
    }


def summarize_results(results, group_columns):
    """Calculate mean and standard deviation across folds."""

    metrics = [
        column
        for column in results.select_dtypes(include="number").columns
        if column not in {"fold", "observations"}
    ]

    summary = (
        results.groupby(group_columns)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary


def main():  # sourcery skip: extract-duplicate-method
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pre-test historical data...")

    data, date_column, feature_columns = load_pretest_data()

    X = data[feature_columns]
    y_direction = data["target_direction"].astype(int)
    y_return = data["target_return"].astype(float)
    dates = data[date_column]

    print(f"Total pre-test observations: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Historical period: {dates.iloc[0].date()} "
        f"to {dates.iloc[-1].date()}"
    )

    if len(data) <= N_SPLITS * TEST_SIZE:
        raise ValueError(
            "Not enough data for the requested walk-forward folds."
        )

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
    )

    classification_records = []
    regression_records = []

    print("\nStarting expanding-window validation...")

    for fold_number, (train_idx, eval_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_eval = X.iloc[eval_idx]

        y_direction_train = y_direction.iloc[train_idx]
        y_direction_eval = y_direction.iloc[eval_idx]

        y_return_train = y_return.iloc[train_idx]
        y_return_eval = y_return.iloc[eval_idx]

        eval_dates = dates.iloc[eval_idx]

        print(f"\n{'=' * 55}")
        print(f"FOLD {fold_number}")
        print(f"Training records: {len(train_idx)}")
        print(f"Evaluation records: {len(eval_idx)}")
        print(
            f"Training period: {dates.iloc[train_idx[0]].date()} "
            f"to {dates.iloc[train_idx[-1]].date()}"
        )
        print(
            f"Evaluation period: {eval_dates.iloc[0].date()} "
            f"to {eval_dates.iloc[-1].date()}"
        )

        # Classification
        print("\nTraining classification models...")

        classifiers = create_classification_models(
            y_direction_train
        )

        for name, model in classifiers.items():
            model.fit(X_train, y_direction_train)

            probabilities_all = model.predict_proba(X_eval)
            positive_index = list(model.classes_).index(1)
            probabilities = probabilities_all[:, positive_index]

            record = classification_metrics(
                name,
                fold_number,
                y_direction_eval,
                probabilities,
                eval_dates,
            )

            classification_records.append(record)

            print(
                f"{name}: "
                f"Balanced Accuracy="
                f"{record['balanced_accuracy']:.4f}, "
                f"ROC-AUC={record['roc_auc']:.4f}"
            )

        # Regression
        print("\nTraining regression models...")

        zero_predictions = np.zeros(len(y_return_eval))

        zero_result = regression_metrics(
            "Zero Return Baseline",
            fold_number,
            y_return_eval,
            zero_predictions,
            eval_dates,
        )

        zero_result["directional_accuracy"] = np.nan
        regression_records.append(zero_result)

        regressors = create_regression_models()

        for name, model in regressors.items():
            model.fit(X_train, y_return_train)

            predictions = model.predict(X_eval)

            record = regression_metrics(
                name,
                fold_number,
                y_return_eval,
                predictions,
                eval_dates,
            )

            regression_records.append(record)

            print(
                f"{name}: "
                f"MAE={record['mae']:.6f}, "
                f"Directional Accuracy="
                f"{record['directional_accuracy']:.4f}"
            )

    # Save fold-level results
    classification_df = pd.DataFrame(
        classification_records
    )

    regression_df = pd.DataFrame(
        regression_records
    )

    classification_df.to_csv(
        REPORT_DIR / "walk_forward_classification.csv",
        index=False,
    )

    regression_df.to_csv(
        REPORT_DIR / "walk_forward_regression.csv",
        index=False,
    )

    # Save aggregate summaries
    classification_summary = summarize_results(
        classification_df,
        ["model"],
    )

    regression_summary = summarize_results(
        regression_df,
        ["model"],
    )

    classification_summary.to_csv(
        REPORT_DIR / "walk_forward_classification_summary.csv",
        index=False,
    )

    regression_summary.to_csv(
        REPORT_DIR / "walk_forward_regression_summary.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("WALK-FORWARD CLASSIFICATION SUMMARY")
    print("=" * 65)
    print(
        classification_summary.round(4).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("WALK-FORWARD REGRESSION SUMMARY")
    print("=" * 65)
    print(
        regression_summary.round(6).to_string(index=False)
    )

    print("\nReports saved in:")
    print(REPORT_DIR)

    print("\nWalk-forward validation completed.")
    print("The untouched final test dataset was not loaded.")


if __name__ == "__main__":
    main()
