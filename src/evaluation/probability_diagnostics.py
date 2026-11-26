
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "reports" / "probability_diagnostics"

MODEL_PATHS = {
    "Dummy Classifier": ARTIFACT_DIR / "baselines" / "dummy_classifier.joblib",
    "Logistic Regression": ARTIFACT_DIR / "baselines" / "logistic_regression.joblib",
    "Random Forest Classifier": (
        ARTIFACT_DIR / "tree_models" / "random_forest_classifier.joblib"
    ),
    "XGBoost Classifier": (
        ARTIFACT_DIR / "tree_models" / "xgboost_classifier.joblib"
    ),
}

N_BINS = 10


def load_test_data():
    """Load the existing holdout test dataset."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH)

    target_column = "target_direction"

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    date_column = next(
        (column for column in ["Date", "date"] if column in df.columns),
        None,
    )

    excluded_columns = {
        "target_direction",
        "target_return",
        "Date",
        "date",
    }

    feature_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No model features found in the test dataset.")

    X = df[feature_columns].copy()
    y = df[target_column].astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("target_direction must contain only 0 and 1.")

    dates = (
        df[date_column].astype(str).to_numpy()
        if date_column
        else np.arange(len(df))
    )

    print(f"Test observations: {len(df)}")
    print(f"Features: {len(feature_columns)}")

    if date_column:
        print(f"Test period: {dates[0]} to {dates[-1]}")

    return X, y, dates


def get_positive_probabilities(model, X):
    """Return the probability of the positive class (Up = 1)."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{type(model).__name__} does not support predict_proba."
        )

    probabilities = model.predict_proba(X)

    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) != 1:
        raise ValueError(
            f"Could not identify positive class 1. Classes: {classes}"
        )

    positive_probabilities = probabilities[:, positive_indices[0]]

    return np.clip(positive_probabilities, 1e-15, 1 - 1e-15)


def calculate_expected_calibration_error(y_true, probabilities):
    """Calculate a weighted calibration error using quantile bins."""
    calibration_bins = pd.DataFrame(
        {
            "actual": y_true,
            "probability": probabilities,
        }
    )

    calibration_bins["bin"] = pd.qcut(
        calibration_bins["probability"],
        q=N_BINS,
        duplicates="drop",
    )

    ece = 0.0
    bin_records = []

    for bin_name, group in calibration_bins.groupby(
        "bin",
        observed=True,
    ):
        mean_probability = group["probability"].mean()
        observed_frequency = group["actual"].mean()
        count = len(group)

        ece += (
            count / len(calibration_bins)
        ) * abs(observed_frequency - mean_probability)

        bin_records.append(
            {
                "probability_bin": str(bin_name),
                "observations": count,
                "mean_predicted_probability": mean_probability,
                "observed_up_frequency": observed_frequency,
                "absolute_calibration_error": abs(
                    observed_frequency - mean_probability
                ),
            }
        )

    return ece, bin_records


def evaluate_model(model_name, model, X, y, dates):
    """Calculate probability and confidence diagnostics."""
    probabilities = get_positive_probabilities(model, X)

    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    correct = (predictions == y).astype(int)

    brier = brier_score_loss(y, probabilities)
    cross_entropy = log_loss(y, probabilities, labels=[0, 1])

    try:
        auc = roc_auc_score(y, probabilities)
    except ValueError:
        auc = np.nan

    ece, calibration_bins = calculate_expected_calibration_error(
        y,
        probabilities,
    )

    metrics = {
        "model": model_name,
        "observations": len(y),
        "brier_score": brier,
        "log_loss": cross_entropy,
        "roc_auc": auc,
        "expected_calibration_error": ece,
        "mean_confidence": confidence.mean(),
        "high_confidence_ratio": (confidence >= 0.70).mean(),
        "prediction_accuracy": correct.mean(),
    }

    predictions_df = pd.DataFrame(
        {
            "date": dates,
            "model": model_name,
            "actual_direction": y,
            "probability_up": probabilities,
            "predicted_direction": predictions,
            "confidence": confidence,
            "correct": correct,
        }
    )

    bins_df = pd.DataFrame(calibration_bins)
    bins_df.insert(0, "model", model_name)

    return metrics, predictions_df, bins_df


def plot_reliability_curves(results, y):
    """Plot predicted probability against observed Up frequency."""
    plt.figure(figsize=(9, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for model_name, probabilities in results.items():
        observed, predicted = calibration_curve(
            y,
            probabilities,
            n_bins=N_BINS,
            strategy="quantile",
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            label=model_name,
        )

    plt.xlabel("Mean predicted probability of Up")
    plt.ylabel("Observed frequency of Up")
    plt.title("Classifier Probability Reliability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "reliability_curves.png",
        dpi=150,
    )
    plt.close()


def plot_confidence_distributions(results):
    """Compare the confidence distributions of classifiers."""
    plt.figure(figsize=(10, 6))

    for model_name, probabilities in results.items():
        confidence = np.maximum(
            probabilities,
            1 - probabilities,
        )

        plt.hist(
            confidence,
            bins=np.linspace(0.5, 1.0, 21), # pyright: ignore[reportArgumentType]
            alpha=0.35,
            label=model_name,
        )

    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of observations")
    plt.title("Prediction Confidence Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confidence_distributions.png",
        dpi=150,
    )
    plt.close()


def plot_confidence_accuracy(predictions_df):
    """Plot empirical accuracy across confidence intervals."""
    plt.figure(figsize=(10, 6))

    confidence_edges = np.linspace(0.5, 1.0, 11)

    for model_name, group in predictions_df.groupby("model"):
        bins = pd.cut(
            group["confidence"],
            bins=confidence_edges.tolist(),
            include_lowest=True,
        )

        summary = group.groupby(
            bins,
            observed=True,
        ).agg(
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            observations=("correct", "size"),
        )

        plt.plot(
            summary["mean_confidence"],
            summary["accuracy"],
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0.5, 1.0],
        [0.5, 1.0],
        linestyle="--",
        label="Reference line",
    )

    plt.xlabel("Mean prediction confidence")
    plt.ylabel("Observed prediction accuracy")
    plt.title("Confidence vs Actual Prediction Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confidence_vs_accuracy.png",
        dpi=150,
    )
    plt.close()


def main():
    # sourcery skip: merge-else-if-into-elif, reintroduce-else, swap-if-else-branches
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading previously evaluated test data...")
    X, y, dates = load_test_data()

    metrics_records = []
    prediction_frames = []
    calibration_frames = []
    probability_results = {}

    print("\nRunning probability diagnostics...")

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        print(f"Analyzing: {model_name}")

        model = load(model_path)

        model_features = getattr(model, "feature_names_in_", None)

        if model_features is not None:
            if missing_features := set(model_features) - set(X.columns):
                raise ValueError(
                    f"{model_name} is missing features: "
                    f"{sorted(missing_features)}"
                )

            model_X = X[list(model_features)]
        else:
            model_X = X

        metrics, predictions_df, bins_df = evaluate_model(
            model_name,
            model,
            model_X,
            y,
            dates,
        )

        metrics_records.append(metrics)
        prediction_frames.append(predictions_df)
        calibration_frames.append(bins_df)

        probability_results[model_name] = get_positive_probabilities(
            model,
            model_X,
        )

    metrics_df = pd.DataFrame(metrics_records)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    calibration_df = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "probability_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "probability_predictions.csv",
        index=False,
    )

    calibration_df.to_csv(
        OUTPUT_DIR / "calibration_bins.csv",
        index=False,
    )

    plot_reliability_curves(probability_results, y)
    plot_confidence_distributions(probability_results)
    plot_confidence_accuracy(predictions_df)

    print("\n========== PROBABILITY DIAGNOSTICS ==========")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nReports saved to:")
    print(OUTPUT_DIR)

    print("\nProbability diagnostics completed.")
    print("No model was trained or tuned.")
    print("Test results are descriptive only.")


if __name__ == "__main__":
    main()
