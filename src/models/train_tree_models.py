
"""Train Random Forest and XGBoost models for NIFTY 50 forecasting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
from xgboost import XGBClassifier, XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed" / "splits"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "tree_models"
REPORT_DIR = ROOT_DIR / "reports" / "models"

TARGET_CLASSIFICATION = "target_direction"
TARGET_REGRESSION = "target_return"

RANDOM_STATE = 42


def load_data():
    """Load chronological training and validation datasets."""

    print("Loading chronological datasets...")

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    excluded_columns = [
        "Date",
        TARGET_CLASSIFICATION,
        TARGET_REGRESSION,
    ]

    feature_columns = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    X_train = train[feature_columns]
    X_val = validation[feature_columns]

    y_train_class = train[TARGET_CLASSIFICATION].astype(int)
    y_val_class = validation[TARGET_CLASSIFICATION].astype(int)

    y_train_reg = train[TARGET_REGRESSION].astype(float)
    y_val_reg = validation[TARGET_REGRESSION].astype(float)

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation features do not match.")

    if X_train.isnull().any().any() or X_val.isnull().any().any():
        raise ValueError("Missing feature values detected.")

    print(f"Training records: {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Features: {len(feature_columns)}")

    return (
        X_train,
        X_val,
        y_train_class,
        y_val_class,
        y_train_reg,
        y_val_reg,
    )


def classification_metrics(y_true, predictions, probabilities, model_name):
    """Calculate classification evaluation metrics."""

    return {
        "task": "classification",
        "model": model_name,
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
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "directional_accuracy": np.nan,
    }


def regression_metrics(y_true, predictions, model_name):
    """Calculate regression evaluation metrics."""

    directional_accuracy = (
        np.nan
        if np.allclose(predictions, 0)
        else accuracy_score(
            np.asarray(y_true) > 0,
            np.asarray(predictions) > 0,
        )
    )

    return {
        "task": "regression",
        "model": model_name,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
        "directional_accuracy": directional_accuracy,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        X_val,
        y_train_class,
        y_val_class,
        y_train_reg,
        y_val_reg,
    ) = load_data()

    results = []

    # --------------------------------------------------
    # Classification models
    # --------------------------------------------------

    print("\nTraining classification models...")

    classification_models = {
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
            reg_lambda=2.0,
            eval_metric="logloss",
            scale_pos_weight=(
                (y_train_class == 0).sum()
                / (y_train_class == 1).sum()
            ),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for name, model in classification_models.items():
        print(f"Fitting {name}...")

        model.fit(X_train, y_train_class)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        results.append(
            classification_metrics(
                y_val_class,
                predictions,
                probabilities,
                name,
            )
        )

        joblib.dump(model, ARTIFACT_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # --------------------------------------------------
    # Regression models
    # --------------------------------------------------

    print("\nTraining regression models...")

    regression_models = {
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
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for name, model in regression_models.items():
        print(f"Fitting {name}...")

        model.fit(X_train, y_train_reg)

        predictions = model.predict(X_val)

        results.append(
            regression_metrics(
                y_val_reg,
                predictions,
                name,
            )
        )

        joblib.dump(model, ARTIFACT_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "tree_model_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n========== TREE MODEL RESULTS ==========")

    print("\nCLASSIFICATION")
    print(
        results_df[results_df["task"] == "classification"][
            ["model", "accuracy", "balanced_accuracy",
             "precision", "recall", "f1", "roc_auc"]
        ].round(4).to_string(index=False)
    )

    print("\nREGRESSION")
    print(
        results_df[results_df["task"] == "regression"][
            ["model", "mae", "rmse", "r2", "directional_accuracy"]
        ].round(6).to_string(index=False)
    )

    print(f"\nModels saved to: {ARTIFACT_DIR}")
    print(f"Metrics saved to: {output_path}")
    print("\nValidation completed. Test data was not used.")
    print("========================================")


if __name__ == "__main__":
    main()
