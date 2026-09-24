"""Model inventory endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.models import ModelsResponse
from backend.services.model_service import ModelService

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """Return local model artifacts and MLflow registered models."""

    return ModelService().list_models()
