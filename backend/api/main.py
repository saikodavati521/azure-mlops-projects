"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, Response

from backend.routers import health, metrics, models, predict, train
from backend.utils.database import initialize_database



async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize database tables when PostgreSQL is available."""

    try:
        initialize_database()
        app.state.database_startup_error = None
    except (OSError, psycopg2.Error) as exc:
        app.state.database_startup_error = str(exc)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="CategoryIQ API",
        description="Product classification API with training, metrics, and model registry views.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(train.router)
    app.include_router(metrics.router)
    app.include_router(models.router)

    @app.get("/", tags=["root"])
    def root() -> dict[str, str]:
        """Return a small landing response for browser checks."""

        return {
            "name": "CategoryIQ API",
            "status": "ok",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    return app


app = create_app()
