"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.predict import PredictRequest, PredictResponse
from backend.services.prediction_service import PredictionPersistenceError, PredictionService

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the category for product text."""

    try:
        return PredictionService().predict(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except PredictionPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
