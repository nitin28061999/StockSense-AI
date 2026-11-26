
"""Perform a detailed quality audit of NIFTY 50 data."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nifty50_historical.csv"
)


def main():
    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    print("\n========== DATA QUALITY AUDIT ==========")

    print("\n1. Dataset shape:")
    print(df.shape)

    print("\n2. Data types:")
    print(df.dtypes)

    print("\n3. Missing values:")
    print(df.isnull().sum())

    print("\n4. Duplicate records:")
    print(df.duplicated().sum())

    print("\n5. Duplicate dates:")
    print(df["Date"].duplicated().sum())

    print("\n6. Zero volume analysis:")
    zero_volume = (df["Volume"] == 0).sum()

    print(f"Zero-volume records: {zero_volume}")
    print(
        f"Zero-volume percentage: "
        f"{zero_volume / len(df) * 100:.2f}%"
    )

    print("\n7. OHLC validation:")

    invalid = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )

    print(f"Invalid OHLC records: {invalid.sum()}")

    print("\n8. Date range:")
    print("Start:", df["Date"].min())
    print("End:", df["Date"].max())

    print("\n9. Daily percentage returns:")
    returns = df["Close"].pct_change() * 100

    print(returns.describe())

    print("\n10. Largest daily movements:")
    print(
        df.assign(Daily_Return=returns)
        .nlargest(5, "Daily_Return")[
            ["Date", "Close", "Daily_Return"]
        ]
    )

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
