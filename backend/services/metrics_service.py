"""Metrics service for reading latest training output."""

from __future__ import annotations

import json

from backend.schemas.metrics import MetricsResponse
from backend.utils.paths import METRICS_PATH, REPORT_PATH


class MetricsService:
    """Read training metrics artifacts."""

    def get_metrics(self) -> MetricsResponse:
        if not METRICS_PATH.exists():
            raise FileNotFoundError(f"Metrics artifact not found: {METRICS_PATH}")

        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return MetricsResponse(
            metrics_path=str(METRICS_PATH),
            report_path=str(REPORT_PATH) if REPORT_PATH.exists() else None,
            best_model_name=payload.get("best_model_name"),
            payload=payload,
        )
