
"""FastAPI application for StockSense AI."""

from datetime import UTC, datetime
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.inference.predict import StockSensePredictor

app = FastAPI(
    title="StockSense AI API",
    description=(
        "Experimental NIFTY 50 next-session forecasting API "
        "powered by machine learning."
    ),
    version="1.0.0",
    contact={
        "name": "StockSense AI",
    },
)


@lru_cache(maxsize=1)
def get_predictor():
    """Load the inference engine once and reuse it."""
    return StockSensePredictor()


@app.get("/", tags=["General"])
def root():
    """API welcome endpoint."""
    return {
        "project": "StockSense AI",
        "message": "NIFTY 50 forecasting API is running.",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check API and model availability."""

    try:
        predictor = get_predictor()

        return {
            "status": "healthy",
            "service": "StockSense AI",
            "models_loaded": {
                "classifiers": list(predictor.classifiers.keys()),
                "regressors": list(predictor.regressors.keys()),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization failed: {exc!s}",
        ) from exc


@app.get("/features/latest", tags=["Features"])
def latest_features():
    """Return the latest available market features."""

    try:
        predictor = get_predictor()
        return predictor.get_latest_features()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature retrieval failed: {exc!s}",
        ) from exc


@app.get("/predict/latest", tags=["Predictions"])
def predict_latest():
    """Generate next-session experimental predictions."""

    try:
        predictor = get_predictor()
        return predictor.predict_latest()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc!s}",
        ) from exc


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """Return model metadata."""

    predictor = get_predictor()

    return {
        "project": "StockSense AI",
        "symbol": "NIFTY 50",
        "forecast_horizon": "Next trading session",
        "features_count": 24,
        "classifiers": list(predictor.classifiers.keys()),
        "regressors": list(predictor.regressors.keys()),
        "model_status": "Loaded",
        "prediction_status": "Experimental",
    }


@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc):
    """Handle unexpected application errors."""

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
        },
    )
