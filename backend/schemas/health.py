"""Health response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    vectorizer_ready: bool
    database_ready: bool
    database_error: str | None = None
