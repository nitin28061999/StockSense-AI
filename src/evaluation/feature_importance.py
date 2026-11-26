
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

TRAIN_PATH = Path("data/processed/splits/train.csv")
VALIDATION_PATH = Path("data/processed/splits/validation.csv")

OUTPUT_DIR = Path("reports/feature_importance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
TEST_SIZE = 395
N_REPEATS = 10
RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD PRE-TEST DATA
# ---------------------------------------------------------

def load_data():
    print("Loading pre-test historical data...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    date_col = "date" if "date" in df.columns else "Date"

    if date_col not in df.columns:
        raise ValueError("Date column not found.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if df[date_col].duplicated().any():
        raise ValueError("Duplicate dates found.")

    excluded_columns = {
        "date",
        "Date",
        "target_return",
        "target_direction",
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    if len(feature_columns) != 24:
        raise ValueError(
            f"Expected 24 features, found {len(feature_columns)}."
        )

    if df[feature_columns].isnull().any().any():
        raise ValueError("Missing values found in features.")

    if df["target_direction"].isnull().any():
        raise ValueError("Missing classification targets found.")

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in feature_columns
    ):
        raise ValueError("Non-numeric feature detected.")

    X = df[feature_columns]
    y = df["target_direction"].astype(int)

    print(f"Total observations: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Historical period: {df[date_col].min().date()} "
          f"to {df[date_col].max().date()}")

    return df, X, y, feature_columns, date_col


# ---------------------------------------------------------
# RANDOM FOREST CONFIGURATION
# ---------------------------------------------------------

def create_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------
# FEATURE IMPORTANCE ANALYSIS
# ---------------------------------------------------------

def run_feature_importance(
    df,
    X,
    y,
    feature_columns,
    date_col,
):  # sourcery skip: for-append-to-extend
    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
    )

    impurity_records = []
    permutation_records = []

    print("\nStarting expanding-window feature analysis...")

    for fold_number, (train_idx, eval_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_eval = X.iloc[eval_idx]

        y_train = y.iloc[train_idx]
        y_eval = y.iloc[eval_idx]

        train_dates = df.iloc[train_idx][date_col]
        eval_dates = df.iloc[eval_idx][date_col]

        print("\n" + "=" * 60)
        print(f"FOLD {fold_number}")
        print("=" * 60)

        print(f"Training records: {len(train_idx)}")
        print(f"Evaluation records: {len(eval_idx)}")
        print(
            f"Evaluation period: {eval_dates.min().date()} "
            f"to {eval_dates.max().date()}"
        )

        model = create_model()

        print("\nTraining Random Forest Classifier...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)

        score = balanced_accuracy_score(
            y_eval,
            predictions,
        )

        print(f"Balanced accuracy: {score:.4f}")

        # ---------------------------------------------
        # 1. Built-in impurity-based importance
        # ---------------------------------------------

        for feature, importance in zip(
            feature_columns,
            model.feature_importances_,
        ):
            impurity_records.append({
                "fold": fold_number,
                "feature": feature,
                "importance": importance,
                "train_start": train_dates.min().date(),
                "train_end": train_dates.max().date(),
                "eval_start": eval_dates.min().date(),
                "eval_end": eval_dates.max().date(),
                "balanced_accuracy": score,
            })

        # ---------------------------------------------
        # 2. Permutation importance
        # ---------------------------------------------

        print("Calculating permutation importance...")

        permutation = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring="balanced_accuracy",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

        for index, feature in enumerate(feature_columns):
            permutation_records.append({
                "fold": fold_number,
                "feature": feature,
                "importance_mean": permutation.importances_mean[index], # pyright: ignore[reportAttributeAccessIssue]
                "importance_std": permutation.importances_std[index], # pyright: ignore[reportAttributeAccessIssue]
                "train_start": train_dates.min().date(),
                "train_end": train_dates.max().date(),
                "eval_start": eval_dates.min().date(),
                "eval_end": eval_dates.max().date(),
                "balanced_accuracy": score,
            })

        print("Fold feature analysis completed.")

    return impurity_records, permutation_records


# ---------------------------------------------------------
# SUMMARIZE IMPORTANCE AND STABILITY
# ---------------------------------------------------------

def create_summaries(
    impurity_records,
    permutation_records,
):
    impurity_df = pd.DataFrame(impurity_records)
    permutation_df = pd.DataFrame(permutation_records)

    # Average impurity importance across folds
    impurity_summary = (
        impurity_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "importance_mean",
            "std": "importance_std",
        })
    )

    impurity_summary["mean_rank"] = (
        impurity_summary["importance_mean"]
        .rank(ascending=False, method="average")
    )

    # Average permutation importance across folds
    permutation_summary = (
        permutation_df.groupby("feature")["importance_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "permutation_mean",
            "std": "permutation_std",
        })
    )

    permutation_summary["mean_rank"] = (
        permutation_summary["permutation_mean"]
        .rank(ascending=False, method="average")
    )

    # Count folds where permutation importance is positive
    positive_folds = (
        permutation_df.assign(
            positive=permutation_df["importance_mean"] > 0
        )
        .groupby("feature")["positive"]
        .sum()
        .reset_index(name="positive_folds")
    )

    permutation_summary = permutation_summary.merge(
        positive_folds,
        on="feature",
        how="left",
    )

    permutation_summary["positive_fold_ratio"] = (
        permutation_summary["positive_folds"] / N_SPLITS
    )

    permutation_summary = permutation_summary.sort_values(
        "permutation_mean",
        ascending=False,
    )

    impurity_summary = impurity_summary.sort_values(
        "importance_mean",
        ascending=False,
    )

    return (
        impurity_df,
        permutation_df,
        impurity_summary,
        permutation_summary,
    )


# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------

def create_plots(
    impurity_summary,
    permutation_summary,
):  # sourcery skip: extract-duplicate-method
    # Built-in feature importance
    top_impurity = impurity_summary.head(15).sort_values(
        "importance_mean"
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_impurity["feature"],
        top_impurity["importance_mean"],
        xerr=top_impurity["importance_std"].fillna(0),
        capsize=3,
    )

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Mean impurity-based importance across folds")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "random_forest_impurity_importance.png",
        dpi=300,
    )
    plt.close()

    # Permutation feature importance
    top_permutation = permutation_summary.head(15).sort_values(
        "permutation_mean"
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_permutation["feature"],
        top_permutation["permutation_mean"],
        xerr=top_permutation["permutation_std"].fillna(0),
        capsize=3,
    )

    plt.axvline(0, linestyle="--", linewidth=1)

    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean balanced accuracy decrease across folds")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "permutation_importance.png",
        dpi=300,
    )
    plt.close()

    print("\nFeature importance charts saved.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():  # sourcery skip: extract-duplicate-method
    df, X, y, feature_columns, date_col = load_data()

    (
        impurity_records,
        permutation_records,
    ) = run_feature_importance(
        df,
        X,
        y,
        feature_columns,
        date_col,
    )

    (
        impurity_df,
        permutation_df,
        impurity_summary,
        permutation_summary,
    ) = create_summaries(
        impurity_records,
        permutation_records,
    )

    # Save detailed fold-level results
    impurity_df.to_csv(
        OUTPUT_DIR / "impurity_importance_by_fold.csv",
        index=False,
    )

    permutation_df.to_csv(
        OUTPUT_DIR / "permutation_importance_by_fold.csv",
        index=False,
    )

    # Save summary results
    impurity_summary.to_csv(
        OUTPUT_DIR / "impurity_importance_summary.csv",
        index=False,
    )

    permutation_summary.to_csv(
        OUTPUT_DIR / "permutation_importance_summary.csv",
        index=False,
    )

    create_plots(
        impurity_summary,
        permutation_summary,
    )

    print("\n" + "=" * 65)
    print("TOP 15 FEATURES — PERMUTATION IMPORTANCE")
    print("=" * 65)

    print(
        permutation_summary.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 65)
    print("TOP 15 FEATURES — IMPURITY IMPORTANCE")
    print("=" * 65)

    print(
        impurity_summary.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print(f"\nReports saved to: {OUTPUT_DIR.resolve()}")
    print("Feature importance analysis completed.")
    print("The untouched final test dataset was not loaded.")


if __name__ == "__main__":
    main()
