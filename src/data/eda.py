
"""Exploratory data analysis for NIFTY 50."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data/raw/nifty50_historical.csv"
REPORT_DIR = PROJECT_ROOT / "reports/eda"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

sns.set_theme(style="whitegrid")


def load_data():
    """Load historical market data."""
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    return df


def create_features(df):
    """Calculate basic analytical metrics."""
    df = df.copy()

    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df["Cumulative_Return"] = (
        (1 + df["Daily_Return"].fillna(0)).cumprod() - 1
    )

    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
        * (252 ** 0.5)
        * 100
    )

    df["Running_Peak"] = df["Close"].cummax()

    df["Drawdown"] = (
        df["Close"] / df["Running_Peak"] - 1
    ) * 100

    return df


def plot_price_trend(df):
    """Plot historical closing prices."""
    plt.figure(figsize=(14, 6))

    plt.plot(df.index, df["Close"], linewidth=1.2)

    plt.title("NIFTY 50 Historical Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "price_trend.png", dpi=300)
    plt.close()


def plot_returns_distribution(df):
    """Plot daily return distribution."""
    returns = df["Daily_Return"].dropna() * 100

    plt.figure(figsize=(12, 6))

    sns.histplot(
        returns,
        bins=100,
        kde=True,
    )

    plt.title("NIFTY 50 Daily Returns Distribution")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "returns_distribution.png",
        dpi=300,
    )
    plt.close()


def plot_rolling_volatility(df):
    """Plot rolling annualized volatility."""
    plt.figure(figsize=(14, 6))

    plt.plot(
        df.index,
        df["Rolling_Volatility"],
        linewidth=1,
    )

    plt.title("20-Day Rolling Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "rolling_volatility.png",
        dpi=300,
    )
    plt.close()


def plot_drawdown(df):
    """Plot historical drawdowns."""
    plt.figure(figsize=(14, 6))

    plt.fill_between(
        df.index,
        df["Drawdown"],
        0,
        alpha=0.35,
    )

    plt.plot(
        df.index,
        df["Drawdown"],
        linewidth=0.8,
    )

    plt.title("NIFTY 50 Historical Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "drawdown.png", dpi=300)
    plt.close()


def plot_monthly_returns(df):
    """Plot monthly returns heatmap."""
    monthly = (
        (1 + df["Daily_Return"])
        .resample("ME")
        .prod()
        .sub(1)
        .mul(100)
    )

    monthly_df = monthly.to_frame("Return")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month

    heatmap_data = monthly_df.pivot(
        index="Year",
        columns="Month",
        values="Return",
    )

    plt.figure(figsize=(14, 9))

    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",
        center=0,
        annot=False,
        linewidths=0.3,
    )

    plt.title("NIFTY 50 Monthly Returns Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")

    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "monthly_returns_heatmap.png",
        dpi=300,
    )
    plt.close()

    return monthly


def generate_statistics(df):
    """Generate descriptive statistics."""
    returns = df["Daily_Return"].dropna()

    statistics = pd.DataFrame({
        "Metric": [
            "Total trading sessions",
            "Average daily return (%)",
            "Daily return volatility (%)",
            "Maximum daily return (%)",
            "Minimum daily return (%)",
            "Positive trading days (%)",
            "Negative trading days (%)",
            "Maximum drawdown (%)",
            "Average rolling volatility (%)",
        ],
        "Value": [
            len(df),
            returns.mean() * 100,
            returns.std() * 100,
            returns.max() * 100,
            returns.min() * 100,
            (returns > 0).mean() * 100,
            (returns < 0).mean() * 100,
            df["Drawdown"].min(),
            df["Rolling_Volatility"].mean(),
        ],
    })

    statistics.to_csv(
        REPORT_DIR / "eda_statistics.csv",
        index=False,
    )

    return statistics


def main():
    """Run the complete EDA pipeline."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NIFTY 50 historical data...")

    df = load_data()
    df = create_features(df)

    print("Generating visualizations...")

    plot_price_trend(df)
    plot_returns_distribution(df)
    plot_rolling_volatility(df)
    plot_drawdown(df)

    monthly = plot_monthly_returns(df)

    statistics = generate_statistics(df)

    df.to_csv(
        PROCESSED_DIR / "nifty50_eda_features.csv",
        index=True,
    )

    monthly.to_csv(
        PROCESSED_DIR / "monthly_returns.csv",
        header=["Monthly_Return"],
    )

    print("\n========== EDA SUMMARY ==========")
    print(statistics.to_string(index=False))

    print("\nGenerated reports:")
    for file in sorted(REPORT_DIR.glob("*.png")):
        print(file.name)

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()
