"""Metrics response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    metrics_path: str
    report_path: str | None = None
    best_model_name: str | None = None
    payload: dict[str, Any]
