
import numpy as np
import pandas as pd
import pytest

from src.inference import predict as inference


def make_sample_data(rows=130):
    """Create deterministic synthetic OHLC data."""

    dates = pd.bdate_range("2025-01-01", periods=rows)

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
            "Volume": 100000,
        }
    )

def test_latest_features_have_expected_structure():
    """Verify that inference generates 24 finite features."""

    raw_data = make_sample_data()

    features = inference.build_latest_features(raw_data)

    assert features.shape == (1, 24)
    assert features.columns.tolist() == inference.FEATURE_COLUMNS

    assert np.isfinite(
        features.to_numpy(dtype=float)
    ).all()

def test_latest_features_reject_insufficient_history():
    """Insufficient historical data should raise a clear error."""

    raw_data = make_sample_data(rows=30)

    with pytest.raises(
        ValueError,
        match="More historical data may be required",
    ):
        inference.build_latest_features(raw_data)

def test_load_raw_data_rejects_duplicate_dates(
    tmp_path,
    monkeypatch,
):
    """Duplicate market dates should be rejected."""

    raw_data = make_sample_data(rows=10)
    raw_data.loc[9, "Date"] = raw_data.loc[8, "Date"]

    csv_path = tmp_path / "duplicate_dates.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        inference,
        "RAW_DATA_PATH",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate market dates detected",
    ):
        inference.load_raw_data()

def test_load_raw_data_rejects_invalid_ohlc(
    tmp_path,
    monkeypatch,
):
    """Invalid OHLC relationships should be rejected."""

    raw_data = make_sample_data(rows=10)

    raw_data.loc[5, "High"] = (
        raw_data.loc[5, "Low"] - 10
    ) # pyright: ignore[reportOperatorIssue]

    csv_path = tmp_path / "invalid_ohlc.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        inference,
        "RAW_DATA_PATH",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="Invalid OHLC relationships detected",
    ):
        inference.load_raw_data()

def test_load_raw_data_rejects_missing_columns(
    tmp_path,
    monkeypatch,
):
    """Missing required OHLC columns should be rejected."""

    raw_data = make_sample_data(rows=10)
    raw_data = raw_data.drop(columns=["High"])

    csv_path = tmp_path / "missing_column.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        inference,
        "RAW_DATA_PATH",
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match="Missing raw data columns",
    ):
        inference.load_raw_data()

class FakeClassifier:
    """Mock classifier for deterministic prediction tests."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])

    def predict(self, X):
        return np.array([1])

class FakeRegressor:
    """Mock regressor for deterministic prediction tests."""

    def predict(self, X):
        return np.array([0.001])

def test_predictor_generates_valid_predictions(
    tmp_path,
    monkeypatch,
):
    """Verify prediction structure using mock model artifacts."""

    raw_data = make_sample_data()

    csv_path = tmp_path / "market_data.csv"
    raw_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        inference,
        "RAW_DATA_PATH",
        csv_path,
    )

    artifact_dir = tmp_path / "artifacts"

    classifier_paths = {
        "Logistic Regression": (
            artifact_dir / "baselines" / "logistic_regression.joblib"
        ),
        "Random Forest": (
            artifact_dir / "tree_models" / "random_forest_classifier.joblib"
        ),
        "XGBoost": (
            artifact_dir / "tree_models" / "xgboost_classifier.joblib"
        ),
    }

    regressor_paths = {
        "Ridge Regression": (
            artifact_dir / "baselines" / "ridge_regression.joblib"
        ),
        "Random Forest": (
            artifact_dir / "tree_models" / "random_forest_regressor.joblib"
        ),
        "XGBoost": (
            artifact_dir / "tree_models" / "xgboost_regressor.joblib"
        ),
    }

# sourcery skip: no-loop-in-tests
    for path in (
        list(classifier_paths.values())
        + list(regressor_paths.values())
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setattr(
        inference,
        "CLASSIFIER_PATHS",
        classifier_paths,
    )

    monkeypatch.setattr(
        inference,
        "REGRESSOR_PATHS",
        regressor_paths,
    )

# sourcery skip: no-conditionals-in-tests
    def fake_load(path):
        if path in classifier_paths.values():
            return FakeClassifier()
        return FakeRegressor()

    monkeypatch.setattr(inference, "load", fake_load)

    predictor = inference.StockSensePredictor()
    result = predictor.predict_latest()

    assert result["symbol"] == "NIFTY 50"
    assert result["forecast_horizon"] == "Next trading session"

    assert len(result["classification"]) == 3
    assert len(result["regression"]) == 3

    for prediction in result["classification"].values():
        assert prediction["predicted_direction"] == "Up"
        assert prediction["probability_up"] == pytest.approx(0.6)
        assert prediction["probability_down"] == pytest.approx(0.4)

        assert (
            prediction["probability_up"]
            + prediction["probability_down"]
        ) == pytest.approx(1.0)

    for prediction in result["regression"].values():
# sourcery skip: no-loop-in-tests
        assert prediction["predicted_return"] == pytest.approx(0.001)
        assert prediction["predicted_return_percent"] == pytest.approx(0.1)
        assert np.isfinite(prediction["predicted_return"])

    assert "Not investment advice" in result["disclaimer"]
