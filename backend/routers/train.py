"""Training endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.train import TrainRequest, TrainResponse
from backend.services.training_service import TrainingService

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    """Train models, select the best model, and save artifacts."""

    try:
        return TrainingService().train(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
