
"""Inference engine for StockSense AI."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "data" / "raw" / "nifty50_historical.csv"
ARTIFACT_DIR = ROOT / "artifacts"

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

CLASSIFIER_PATHS = {
    "Logistic Regression": (
        ARTIFACT_DIR / "baselines" / "logistic_regression.joblib"
    ),
    "Random Forest": (
        ARTIFACT_DIR / "tree_models" / "random_forest_classifier.joblib"
    ),
    "XGBoost": (
        ARTIFACT_DIR / "tree_models" / "xgboost_classifier.joblib"
    ),
}

REGRESSOR_PATHS = {
    "Ridge Regression": (
        ARTIFACT_DIR / "baselines" / "ridge_regression.joblib"
    ),
    "Random Forest": (
        ARTIFACT_DIR / "tree_models" / "random_forest_regressor.joblib"
    ),
    "XGBoost": (
        ARTIFACT_DIR / "tree_models" / "xgboost_regressor.joblib"
    ),
}


def load_raw_data() -> pd.DataFrame:
    """Load and validate historical OHLC data."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw market data not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Date", "Open", "High", "Low", "Close"]

    if missing_columns := set(required) - set(df.columns):
        raise ValueError(
            f"Missing raw data columns: {sorted(missing_columns)}"
        )

    if df["Date"].isna().any():
        raise ValueError("Invalid dates detected in raw data.")

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate market dates detected.")

    if df[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Missing OHLC values detected.")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive OHLC prices detected.")

    invalid_ohlc = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )

    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC relationships detected.")

    return df


def build_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the latest 24 features without requiring a future target."""

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    daily_return = close.pct_change(fill_method=None)

    df["return_1d"] = daily_return
    df["return_2d"] = close.pct_change(2, fill_method=None)
    df["return_3d"] = close.pct_change(3, fill_method=None)
    df["return_5d"] = close.pct_change(5, fill_method=None)
    df["return_10d"] = close.pct_change(10, fill_method=None)

    for window in [10, 20, 50]:
        sma = close.rolling(window=window).mean()

        ema = close.ewm(
            span=window,
            adjust=False,
            min_periods=window,
        ).mean()

        df[f"sma_{window}_distance"] = close / sma - 1
        df[f"ema_{window}_distance"] = close / ema - 1

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

    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = (
            daily_return.rolling(window).std()
        )

    df["day_of_week"] = df["Date"].dt.day_of_week
    df["month"] = df["Date"].dt.month

    latest = df.dropna(subset=FEATURE_COLUMNS).tail(1)

    if latest.empty:
        raise ValueError(
            "Unable to calculate the latest features. "
            "More historical data may be required."
        )

    features = latest[FEATURE_COLUMNS].copy()

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in latest features.")

    return features


class StockSensePredictor:
    """Load saved models and generate next-session predictions."""

    def __init__(self):
        self.classifiers = {}
        self.regressors = {}

        self._load_models()

    def _load_models(self):
        """Load all trained classifier and regressor artifacts."""

        for name, path in CLASSIFIER_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Classifier artifact not found: {path}"
                )

            self.classifiers[name] = load(path)

        for name, path in REGRESSOR_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Regressor artifact not found: {path}"
                )

            self.regressors[name] = load(path)

    def get_latest_features(self):
        """Return latest market date, price, and feature values."""

        raw_data = load_raw_data()
        latest_features = build_latest_features(raw_data)

        latest_date = raw_data["Date"].iloc[-1]
        latest_close = float(raw_data["Close"].iloc[-1])

        return {
            "date": latest_date.date().isoformat(),
            "close": latest_close,
            "features": {
                key: float(value)
                for key, value in latest_features.iloc[0].items()
            },
        }

    def predict_latest(self):
        """Generate classification and regression predictions."""

        market_data = self.get_latest_features()

        X = pd.DataFrame(
            [market_data["features"]],
            columns=FEATURE_COLUMNS,
        )

        classification_results = {}

        for name, model in self.classifiers.items():
            probabilities = model.predict_proba(X)[0]

            classes = np.asarray(model.classes_)
            positive_indices = np.flatnonzero(classes == 1)

            if len(positive_indices) != 1:
                raise ValueError(
                    f"Positive class 1 unavailable for {name}."
                )

            probability_up = float(
                probabilities[positive_indices[0]]
            )

            predicted_class = int(
                model.predict(X)[0]
            )

            classification_results[name] = {
                "predicted_direction": (
                    "Up" if predicted_class == 1 else "Down/Flat"
                ),
                "probability_up": probability_up,
                "probability_down": 1.0 - probability_up,
            }

        regression_results = {}

        for name, model in self.regressors.items():
            predicted_return = float(model.predict(X)[0])

            regression_results[name] = {
                "predicted_return": predicted_return,
                "predicted_return_percent": predicted_return * 100,
            }

        return {
            "symbol": "NIFTY 50",
            "market_data_date": market_data["date"],
            "latest_close": market_data["close"],
            "forecast_horizon": "Next trading session",
            "classification": classification_results,
            "regression": regression_results,
            "disclaimer": (
                "Experimental machine learning predictions only. "
                "Models have not demonstrated reliable predictive "
                "performance on the held-out test period. "
                "Not investment advice."
            ),
        }
