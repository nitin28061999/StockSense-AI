
"""Build leakage-safe technical features for NIFTY 50."""

from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/raw/nifty50_historical.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_FILE = OUTPUT_DIR / "nifty50_features.csv"


FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "sma_10_distance",
    "sma_20_distance",
    "sma_50_distance",
    "ema_10_distance",
    "ema_20_distance",
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
    """Load and validate historical market data."""
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    required = ["Open", "High", "Low", "Close"]

    if df[required].isnull().any().any():
        raise ValueError("Missing OHLC values detected.")

    if (df[required] <= 0).any().any():
        raise ValueError("Non-positive OHLC prices detected.")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and prediction targets."""

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ------------------------------------------------
    # 1. Historical returns
    # ------------------------------------------------

    daily_return = close.pct_change(fill_method=None)

    df["return_1d"] = daily_return
    df["return_2d"] = close.pct_change(2, fill_method=None)
    df["return_3d"] = close.pct_change(3, fill_method=None)
    df["return_5d"] = close.pct_change(5, fill_method=None)
    df["return_10d"] = close.pct_change(10, fill_method=None)

    # ------------------------------------------------
    # 2. Moving averages
    # ------------------------------------------------

    for window in [10, 20, 50]:

        sma = close.rolling(window=window).mean()
        ema = close.ewm(
            span=window,
            adjust=False,
            min_periods=window,
        ).mean()

        df[f"sma_{window}_distance"] = close / sma - 1
        df[f"ema_{window}_distance"] = close / ema - 1

    # ------------------------------------------------
    # 3. Momentum indicators
    # ------------------------------------------------

    df["rsi_14"] = RSIIndicator(
        close=close,
        window=14,
    ).rsi()

    macd = MACD(
        close=close,
        window_fast=12,
        window_slow=26,
        window_sign=9,
    )

    df["macd_pct"] = macd.macd() / close
    df["macd_signal_pct"] = macd.macd_signal() / close
    df["macd_hist_pct"] = macd.macd_diff() / close

    # ------------------------------------------------
    # 4. Volatility indicators
    # ------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14_pct"] = (
        true_range.rolling(14, min_periods=14).mean()
        / close
    )

    bollinger = BollingerBands(
        close=close,
        window=20,
        window_dev=2,
    )

    upper = bollinger.bollinger_hband()
    lower = bollinger.bollinger_lband()
    middle = bollinger.bollinger_mavg()

    df["bb_upper_distance"] = upper / close - 1
    df["bb_lower_distance"] = lower / close - 1
    df["bb_width"] = (upper - lower) / middle

    # ------------------------------------------------
    # 5. Rolling historical volatility
    # ------------------------------------------------

    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = (
            daily_return.rolling(window).std()
        )

    # ------------------------------------------------
    # 6. Calendar features
    # ------------------------------------------------

    df["day_of_week"] = df["Date"].dt.day_of_week
    df["month"] = df["Date"].dt.month

    # ------------------------------------------------
    # 7. Prediction targets
    # ------------------------------------------------

    # Information available after today's close is used
    # to predict the next trading session.

    df["target_return"] = close.shift(-1) / close - 1

    df["target_direction"] = np.where(
        df["target_return"].notna(),
        (df["target_return"] > 0).astype(int),
        np.nan,
    )

    # ------------------------------------------------
    # 8. Remove warm-up rows and unavailable targets
    # ------------------------------------------------

    required_columns = FEATURE_COLUMNS + [
        "target_return",
        "target_direction",
    ]

    df = df.dropna(subset=required_columns).copy()

    df["target_direction"] = (
        df["target_direction"].astype("int8")
    )

    return df


def main():
    """Execute feature engineering pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading historical NIFTY 50 data...")

    raw_data = load_data()

    print("Building technical indicators...")

    features = build_features(raw_data)

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n========== FEATURE ENGINEERING REPORT ==========")

    print(f"Original records: {len(raw_data):,}")
    print(f"Processed records: {len(features):,}")
    print(f"Generated features: {len(FEATURE_COLUMNS)}")

    print(f"Start date: {features['Date'].min().date()}")
    print(f"End date: {features['Date'].max().date()}")

    print(
        f"Missing feature values: "
        f"{features[FEATURE_COLUMNS].isnull().sum().sum()}"
    )

    print(
        f"Missing targets: "
        f"{features[['target_return', 'target_direction']].isnull().sum().sum()}"
    )

    print("\nTarget direction distribution:")

    print(
        features["target_direction"]
        .value_counts(normalize=True)
        .sort_index()
        .rename(index={0: "Down/Flat", 1: "Up"})
        .mul(100)
    )

    print("\nFeature preview:")
    print(
        features[
            ["Date", "Close", "rsi_14", "macd_pct",
             "atr_14_pct", "target_return", "target_direction"]
        ].tail()
    )

    print(f"\nSaved dataset: {OUTPUT_FILE}")
    print("=================================================")


if __name__ == "__main__":
    main()
