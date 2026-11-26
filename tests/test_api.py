
import pytest
from fastapi.testclient import TestClient

from src.api import main as api


class FakePredictor:
    """Mock predictor for API endpoint tests."""

    classifiers = {  # noqa: RUF012
        "Logistic Regression": object(),
        "Random Forest": object(),
        "XGBoost": object(),
    }

    regressors = {  # noqa: RUF012
        "Ridge Regression": object(),
        "Random Forest": object(),
        "XGBoost": object(),
    }

    def get_latest_features(self):
        return {
            "date": "2026-11-25",
            "close": 25819.25,
            "features": {
                "return_1d": 0.002,
                "rsi_14": 55.0,
            },
        }

    def predict_latest(self):
        return {
            "symbol": "NIFTY 50",
            "market_data_date": "2026-11-25",
            "latest_close": 25819.25,
            "forecast_horizon": "Next trading session",
            "classification": {
                "Logistic Regression": {
                    "predicted_direction": "Up",
                    "probability_up": 0.60,
                    "probability_down": 0.40,
                }
            },
            "regression": {
                "Ridge Regression": {
                    "predicted_return": 0.001,
                    "predicted_return_percent": 0.1,
                }
            },
            "disclaimer": "Experimental predictions. Not investment advice.",
        }

@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)

def use_fake_predictor(monkeypatch):
    """Replace the real predictor with a mock."""

    monkeypatch.setattr(
        api,
        "get_predictor",
        lambda: FakePredictor(),
    )

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "StockSense AI"
    assert data["health"] == "/health"
    assert data["documentation"] == "/docs"

def test_health_endpoint(client, monkeypatch):
    use_fake_predictor(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "StockSense AI"

    assert len(data["models_loaded"]["classifiers"]) == 3
    assert len(data["models_loaded"]["regressors"]) == 3

    assert "timestamp" in data

def test_health_endpoint_handles_model_initialization_failure(
    client,
    monkeypatch,
):
    def failing_predictor():
        raise RuntimeError("Model loading failed")

    monkeypatch.setattr(
        api,
        "get_predictor",
        failing_predictor,
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert "Model initialization failed" in response.json()["detail"]

def test_latest_features_endpoint(client, monkeypatch):
    use_fake_predictor(monkeypatch)

    response = client.get("/features/latest")

    assert response.status_code == 200

    data = response.json()

    assert data["date"] == "2026-11-25"
    assert data["close"] == 25819.25
    assert "features" in data

def test_latest_features_endpoint_handles_failure(
    client,
    monkeypatch,
):
    class FailingPredictor(FakePredictor):
        def get_latest_features(self):
            raise RuntimeError("Feature calculation failed")

    monkeypatch.setattr(
        api,
        "get_predictor",
        lambda: FailingPredictor(),
    )

    response = client.get("/features/latest")

    assert response.status_code == 500
    assert "Feature retrieval failed" in response.json()["detail"]

def test_prediction_endpoint(client, monkeypatch):
    use_fake_predictor(monkeypatch)

    response = client.get("/predict/latest")

    assert response.status_code == 200

    data = response.json()

    assert data["symbol"] == "NIFTY 50"
    assert data["forecast_horizon"] == "Next trading session"

    assert "classification" in data
    assert "regression" in data
    assert "disclaimer" in data

def test_prediction_endpoint_handles_failure(
    client,
    monkeypatch,
):
    class FailingPredictor(FakePredictor):
        def predict_latest(self):
            raise RuntimeError("Prediction failed")

    monkeypatch.setattr(
        api,
        "get_predictor",
        lambda: FailingPredictor(),
    )

    response = client.get("/predict/latest")

    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]

def test_model_info_endpoint(client, monkeypatch):
    use_fake_predictor(monkeypatch)

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "StockSense AI"
    assert data["symbol"] == "NIFTY 50"
    assert data["features_count"] == 24
    assert len(data["classifiers"]) == 3
    assert len(data["regressors"]) == 3
    assert data["model_status"] == "Loaded"
    assert data["prediction_status"] == "Experimental"

def test_model_info_handles_unexpected_failure(
    client,
    monkeypatch,
):
    def failing_predictor():
        raise RuntimeError("Unexpected model error")

    monkeypatch.setattr(
        api,
        "get_predictor",
        failing_predictor,
    )

    response = client.get("/model-info")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An unexpected server error occurred."
    )
