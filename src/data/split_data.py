
"""Create chronological train, validation, and test datasets."""

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT_DIR / "data" / "processed" / "nifty50_features.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "splits"

TARGET_COLUMNS = ["target_return", "target_direction"]

# Exclude raw OHLC prices and unreliable index volume.
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


def main() -> None:  # sourcery skip: use-named-expression
    print("Loading engineered features...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Validate required columns.
    required_columns = (
        ["Date"] + FEATURE_COLUMNS + TARGET_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not in chronological order.")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values detected in modeling data.")

    if not df["target_direction"].isin([0, 1]).all():
        raise ValueError("Invalid target_direction values.")

    # Retain only approved features, date, and targets.
    modeling_df = df[required_columns].copy()

    total_rows = len(modeling_df)

    # Chronological 70% / 15% / 15% split.
    train_end = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)

    train_df = modeling_df.iloc[:train_end].copy()
    validation_df = modeling_df.iloc[
        train_end:validation_end
    ].copy()
    test_df = modeling_df.iloc[validation_end:].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError("One or more data splits are empty.")

    # Confirm that the splits do not overlap.
    assert train_df["Date"].max() < validation_df["Date"].min()
    assert validation_df["Date"].max() < test_df["Date"].min()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv", index=False
    )
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Save an auditable summary.
    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "start_date": train_df["Date"].min().date(),
                "end_date": train_df["Date"].max().date(),
            },
            {
                "split": "validation",
                "rows": len(validation_df),
                "start_date": validation_df["Date"].min().date(),
                "end_date": validation_df["Date"].max().date(),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "start_date": test_df["Date"].min().date(),
                "end_date": test_df["Date"].max().date(),
            },
        ]
    )

    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    print("\n========== DATA SPLIT REPORT ==========")
    print(f"Total observations: {total_rows}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:
        percentage = len(split_df) / total_rows * 100

        print(f"\n{name}")
        print(f"Records: {len(split_df)} ({percentage:.2f}%)")
        print(f"Start: {split_df['Date'].min().date()}")
        print(f"End: {split_df['Date'].max().date()}")

        print("Direction distribution:")
        print(
            split_df["target_direction"]
            .value_counts(normalize=True)
            .sort_index()
            .round(4)
            .to_dict()
        )

    print("\nSaved datasets to:", OUTPUT_DIR)
    print("=======================================")


if __name__ == "__main__":
    main()
