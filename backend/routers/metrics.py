"""Metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.metrics import MetricsResponse
from backend.services.metrics_service import MetricsService

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Return the latest training metrics."""

    try:
        return MetricsService().get_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
