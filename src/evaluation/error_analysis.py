
"""Descriptive error analysis on the previously evaluated test set.

This module does not train, tune, or select models.
It only analyzes predictions from saved model artifacts.
"""


import itertools
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
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

OUTPUT_DIR = ROOT / "reports/error_analysis"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50
RANDOM_SEED = 42


def load_model(path):
    """Load an existing model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def load_test_data():
    """Load the existing chronological test set."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    test = pd.read_csv(TEST_PATH)

    date_column = "date" if "date" in test.columns else "Date"

    if date_column not in test.columns:
        raise ValueError("Test dataset must contain date or Date.")

    excluded = {
        "date",
        "Date",
        "target_return",
        "target_direction",
    }

    feature_columns = [
        column for column in test.columns
        if column not in excluded
    ]

    X = test[feature_columns]

    if X.isnull().any().any():
        raise ValueError("Missing values detected in test features.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values detected in test features.")

    y_direction = test["target_direction"].astype(int)
    y_return = test["target_return"].astype(float)

    if not set(y_direction.unique()).issubset({0, 1}):
        raise ValueError("Unexpected classification target values.")

    return test, date_column, X, y_direction, y_return


def get_positive_probabilities(model, X):
    """Extract probabilities for class 1."""
    probabilities_all = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Classifier does not contain positive class 1.")

    positive_index = classes.index(1)

    return probabilities_all[:, positive_index]


def classification_diagnostics(
    name,
    model,
    X,
    y_true,
    date_values,
):
    # sourcery skip: use-itertools-product  """Calculate classification diagnostics."""
    probabilities = get_positive_probabilities(model, X)
    predictions = (probabilities >= THRESHOLD).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": name,
        "threshold": THRESHOLD,
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
        "average_precision": average_precision_score(
            y_true, probabilities
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Down", "Up"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.insert(0, "model", name)

    # Save confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted Direction")
    ax.set_ylabel("Actual Direction")

    ax.set_xticks([0, 1], labels=["Down", "Up"])
    ax.set_yticks([0, 1], labels=["Down", "Up"])

    for row, col in itertools.product(range(2), range(2)):
        ax.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center",
            fontsize=13,
        )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(
        CONFUSION_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    prediction_df = pd.DataFrame({
        "date": date_values,
        "model": name,
        "actual_direction": y_true.to_numpy(),
        "predicted_direction": predictions,
        "probability_up": probabilities,
        "correct_prediction": (
            y_true.to_numpy() == predictions
        ),
    })

    return metrics, report_df, prediction_df, probabilities


def regression_diagnostics(
    name,
    model,
    X,
    y_true,
    date_values,
):
    """Calculate regression errors and prediction diagnostics."""
    predictions = model.predict(X)

    actual = y_true.to_numpy()
    residuals = actual - predictions

    metrics = {
        "model": name,
        "mae": mean_absolute_error(actual, predictions),
        "rmse": np.sqrt(
            mean_squared_error(actual, predictions)
        ),
        "r2": r2_score(actual, predictions),
        "directional_accuracy": accuracy_score(
            (actual > 0).astype(int),
            (predictions > 0).astype(int),
        ),
        "mean_residual": np.mean(residuals),
        "residual_std": np.std(residuals),
        "max_absolute_error": np.max(np.abs(residuals)),
    }

    prediction_df = pd.DataFrame({
        "date": date_values,
        "model": name,
        "actual_return": actual,
        "predicted_return": predictions,
        "residual": residuals,
        "absolute_error": np.abs(residuals),
        "correct_direction": (
            (actual > 0) == (predictions > 0)
        ),
    })

    return metrics, prediction_df


def plot_roc_curves(y_true, probability_results):
    """Plot ROC curves for all classifiers."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, probabilities in probability_results.items():
        RocCurveDisplay.from_predictions(
            y_true,
            probabilities,
            name=name,
            ax=ax,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_title("ROC Curves — Test Set")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "roc_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_precision_recall_curves(
    y_true,
    probability_results,
):
    """Plot precision-recall curves for all classifiers."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, probabilities in probability_results.items():
        PrecisionRecallDisplay.from_predictions(
            y_true,
            probabilities,
            name=name,
            ax=ax,
        )

    ax.set_title("Precision-Recall Curves — Test Set")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "precision_recall_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_regression_diagnostics(regression_predictions):
    """Plot regression residual distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, predictions in regression_predictions.items():
        residuals = (
            predictions["actual_return"]
            - predictions["predicted_return"]
        )

        ax.hist(
            residuals,
            bins=45,
            alpha=0.40,
            label=name,
        )

    ax.axvline(0, linestyle="--", linewidth=1.5)

    ax.set_title("Regression Residual Distribution")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "regression_residuals.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_actual_vs_predicted(regression_predictions):
    """Plot actual versus predicted returns."""
    model_count = len(regression_predictions)

    fig, axes = plt.subplots(
        1,
        model_count,
        figsize=(7 * model_count, 6),
        squeeze=False,
    )

    for ax, (name, predictions) in zip(
        axes[0],
        regression_predictions.items(),
    ):
        actual = predictions["actual_return"]
        predicted = predictions["predicted_return"]

        ax.scatter(
            actual,
            predicted,
            alpha=0.35,
            s=15,
        )

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
        )

        ax.set_title(name)
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.grid(alpha=0.2)

    fig.suptitle("Actual vs Predicted Returns")
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    print("Loading previously evaluated test data...")

    (
        test,
        date_column,
        X_test,
        y_direction,
        y_return,
    ) = load_test_data()

    dates = test[date_column]

    print(f"Test observations: {len(test)}")
    print(f"Features: {X_test.shape[1]}")
    print(f"Test period: {dates.iloc[0]} to {dates.iloc[-1]}")

    print("\nRunning classification diagnostics...")

    classifiers = {
        "Dummy Classifier": BASELINE_DIR
        / "dummy_classifier.joblib",
        "Logistic Regression": BASELINE_DIR
        / "logistic_regression.joblib",
        "Random Forest Classifier": TREE_DIR
        / "random_forest_classifier.joblib",
        "XGBoost Classifier": TREE_DIR
        / "xgboost_classifier.joblib",
    }

    classification_metrics = []
    classification_reports = []
    classification_predictions = []
    probability_results = {}

    for name, path in classifiers.items():
        print(f"Analyzing: {name}")

        model = load_model(path)

        (
            metrics,
            report,
            predictions,
            probabilities,
        ) = classification_diagnostics(
            name,
            model,
            X_test,
            y_direction,
            dates,
        )

        classification_metrics.append(metrics)
        classification_reports.append(report)
        classification_predictions.append(predictions)

        probability_results[name] = probabilities

    classification_df = pd.DataFrame(classification_metrics)

    classification_report_df = pd.concat(
        classification_reports,
        ignore_index=True,
    )

    classification_prediction_df = pd.concat(
        classification_predictions,
        ignore_index=True,
    )

    classification_df.to_csv(
        OUTPUT_DIR / "classification_diagnostics.csv",
        index=False,
    )

    classification_report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=False,
    )

    classification_prediction_df.to_csv(
        OUTPUT_DIR / "classification_predictions.csv",
        index=False,
    )

    plot_roc_curves(y_direction, probability_results)

    plot_precision_recall_curves(
        y_direction,
        probability_results,
    )

    print("\nRunning regression diagnostics...")

    regressors = {
        "Ridge Regression": BASELINE_DIR
        / "ridge_regression.joblib",
        "Random Forest Regressor": TREE_DIR
        / "random_forest_regressor.joblib",
        "XGBoost Regressor": TREE_DIR
        / "xgboost_regressor.joblib",
    }

    regression_metrics = []
    regression_predictions = {}

    for name, path in regressors.items():
        print(f"Analyzing: {name}")

        model = load_model(path)

        metrics, predictions = regression_diagnostics(
            name,
            model,
            X_test,
            y_return,
            dates,
        )

        regression_metrics.append(metrics)
        regression_predictions[name] = predictions

    regression_df = pd.DataFrame(regression_metrics)

    regression_prediction_df = pd.concat(
        regression_predictions.values(),
        ignore_index=True,
    )

    regression_df.to_csv(
        OUTPUT_DIR / "regression_diagnostics.csv",
        index=False,
    )

    regression_prediction_df.to_csv(
        OUTPUT_DIR / "regression_predictions.csv",
        index=False,
    )

    plot_regression_diagnostics(regression_predictions)

    plot_actual_vs_predicted(regression_predictions)

    print("\n========== CLASSIFICATION DIAGNOSTICS ==========")
    print(classification_df.round(4).to_string(index=False))

    print("\n========== REGRESSION DIAGNOSTICS ==========")
    print(regression_df.round(6).to_string(index=False))

    print("\nAll diagnostic reports saved to:")
    print(OUTPUT_DIR)

    print("\nAnalysis completed.")
    print("No model was trained or tuned.")
    print("The final test set remains a descriptive holdout.")


if __name__ == "__main__":
    main()
