
"""Download and validate historical NIFTY 50 market data."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TICKER = "^NSEI"
OUTPUT_FILE = RAW_DATA_DIR / "nifty50_historical.csv"

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def download_market_data(
    ticker: str = TICKER,
    period: str = "max",
) -> pd.DataFrame:
    """Download historical daily market data."""
    logger.info("Downloading %s historical data...", ticker)

    data = yf.Ticker(ticker).history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if data.empty:
        raise ValueError(
            f"No historical data received for {ticker}."
        )

    # Handle possible MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"
    data = data.reset_index()

    data["Date"] = pd.to_datetime(
        data["Date"],
        utc=True,
    ).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    data["Date"] = data["Date"].dt.normalize()

    data = data[
        ["Date"] + REQUIRED_COLUMNS
    ].copy()

    return data


def validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean historical market data."""
    logger.info("Validating market data...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    data = data.sort_values("Date").reset_index(drop=True)

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate trading dates detected.")

    if data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing OHLCV values detected.")

    if not data[["Open", "High", "Low", "Close"]].gt(0).all().all():
        raise ValueError("Invalid non-positive price detected.")

    invalid_prices = (
        (data["High"] < data["Low"])
        | (data["High"] < data["Open"])
        | (data["High"] < data["Close"])
        | (data["Low"] > data["Open"])
        | (data["Low"] > data["Close"])
    )

    if invalid_prices.any():
        raise ValueError("Invalid OHLC price relationships detected.")

    if (data["Volume"] < 0).any():
        raise ValueError("Negative trading volume detected.")

    return data


def save_market_data(data: pd.DataFrame) -> None:
    """Save validated data to the raw data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    logger.info("Saved historical data to %s", OUTPUT_FILE)


def main():
    """Execute the complete ingestion pipeline."""
    setup_logging()

    data = download_market_data()
    data = validate_market_data(data)

    save_market_data(data)

    print("\n========== STOCKSENSE AI DATA REPORT ==========")
    print(f"Ticker: {TICKER}")
    print(f"Total records: {len(data):,}")
    print(f"Start date: {data['Date'].min().date()}")
    print(f"End date: {data['Date'].max().date()}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate dates: {data['Date'].duplicated().sum()}")
    print(f"Saved file: {OUTPUT_FILE}")
    print("\nDataset preview:")
    print(data.tail())
    print("================================================")


if __name__ == "__main__":
    main()
