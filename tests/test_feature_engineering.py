
import numpy as np
import pandas as pd
import pytest

from src.features import build_features as feature_module


def make_sample_data(rows=130):
    """Create deterministic synthetic OHLC data for testing."""

    dates = pd.bdate_range(
        start="2025-01-01",
        periods=rows,
    )

    close = (
        10000
        + np.arange(rows) * 8
        + 80 * np.sin(np.arange(rows) / 4)
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 10,
            "High": close + 50,
            "Low": close - 50,
            "Close": close,
            "Volume": np.full(rows, 100000),
        }
    )


def test_feature_engineering_generates_all_features():
    """Check feature count, target columns and missing values."""

    raw_data = make_sample_data()

    features = feature_module.build_features(raw_data)

    assert len(feature_module.FEATURE_COLUMNS) == 24

# sourcery skip: no-loop-in-tests
# sourcery skip: no-loop-in-tests
# sourcery skip: no-loop-in-tests
    for column in feature_module.FEATURE_COLUMNS:
        assert column in features.columns

    assert "target_return" in features.columns
    assert "target_direction" in features.columns

    assert not features[
        feature_module.FEATURE_COLUMNS
    ].isnull().any().any()

    assert not features[
        ["target_return", "target_direction"]
    ].isnull().any().any()

    assert set(features["target_direction"].unique()).issubset(
        {0, 1}
    )


def test_target_matches_next_session_return():
    """Verify that targets use the next available trading session."""

    raw_data = make_sample_data()

    features = feature_module.build_features(raw_data)

    original_index = 75
    target_date = raw_data.loc[original_index, "Date"]

    result = features.loc[
        features["Date"] == target_date
    ].iloc[0]

    expected_return = (
        raw_data.loc[original_index + 1, "Close"]
        / raw_data.loc[original_index, "Close"] # pyright: ignore[reportOperatorIssue]
        - 1
    ) # pyright: ignore[reportOperatorIssue]

    expected_direction = int(expected_return > 0) # pyright: ignore[reportOperatorIssue]

    assert result["target_return"] == pytest.approx(
        expected_return
    )

    assert result["target_direction"] == expected_direction


def test_features_do_not_use_future_market_data():
    """Changing later prices must not change earlier features."""

    raw_data = make_sample_data()

    original_features = feature_module.build_features(
        raw_data
    )

    modified_data = raw_data.copy()

    # Change prices strictly after the observation being tested.
    modified_data.loc[90:, ["Open", "High", "Low", "Close"]] *= 1.5

    modified_features = feature_module.build_features(
        modified_data
    )

    target_date = raw_data.loc[80, "Date"]

    original_row = original_features.loc[
        original_features["Date"] == target_date
    ].iloc[0]

    modified_row = modified_features.loc[
        modified_features["Date"] == target_date
    ].iloc[0]

    np.testing.assert_allclose(
        original_row[feature_module.FEATURE_COLUMNS].to_numpy(
            dtype=float
        ),
        modified_row[feature_module.FEATURE_COLUMNS].to_numpy(
            dtype=float
        ),
    )


def test_feature_engineering_does_not_modify_input():
    """Ensure feature engineering preserves the original DataFrame."""

    raw_data = make_sample_data()
    original_data = raw_data.copy(deep=True)

    feature_module.build_features(raw_data)

    pd.testing.assert_frame_equal(
        raw_data,
        original_data,
    )


def test_load_data_rejects_duplicate_dates(
    tmp_path,
    monkeypatch,
):
    """Duplicate dates should raise a ValueError."""

    raw_data = make_sample_data(rows=5)

    raw_data.loc[4, "Date"] = raw_data.loc[3, "Date"]

    csv_path = tmp_path / "duplicate_dates.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        feature_module,
        "INPUT_FILE",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate dates detected",
    ):
        feature_module.load_data()


def test_load_data_rejects_non_positive_prices(
    tmp_path,
    monkeypatch,
):
    """Non-positive OHLC prices should raise a ValueError."""

    raw_data = make_sample_data(rows=5)
    raw_data.loc[2, "Close"] = 0

    csv_path = tmp_path / "invalid_prices.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        feature_module,
        "INPUT_FILE",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="Non-positive OHLC prices detected",
    ):
        feature_module.load_data()
