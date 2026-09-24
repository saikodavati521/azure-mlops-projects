"""Health-check endpoints."""

from __future__ import annotations

import psycopg2
from fastapi import APIRouter

from backend.schemas.health import HealthResponse
from backend.utils.database import get_connection
from backend.utils.paths import MODEL_PATH, VECTORIZER_PATH

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service health and model artifact readiness."""

    model_ready = MODEL_PATH.exists()
    vectorizer_ready = VECTORIZER_PATH.exists()
    database_ready, database_error = _database_status()
    return HealthResponse(
        status="ok" if model_ready and vectorizer_ready and database_ready else "degraded",
        model_ready=model_ready,
        vectorizer_ready=vectorizer_ready,
        database_ready=database_ready,
        database_error=database_error,
    )


def _database_status() -> tuple[bool, str | None]:
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
    except (OSError, psycopg2.Error) as exc:
        return False, str(exc)
    return True, None
