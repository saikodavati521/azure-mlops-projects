"""Model inventory service."""

from __future__ import annotations

from pathlib import Path

import mlflow
import psycopg2

from backend.schemas.models import DatabaseModelVersion, LocalArtifact, ModelsResponse, RegisteredModelVersion
from backend.services.postgres_service import CategoryIQPostgresService
from backend.utils.paths import METRICS_PATH, MLFLOW_TRACKING_URI, MODEL_PATH, VECTORIZER_PATH
SUPPORTED_MODEL_NAMES = (
    "Logistic Regression",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
)


class ModelService:
    """List local artifacts and MLflow registered models."""

    def list_models(self) -> ModelsResponse:
        return ModelsResponse(
            supported_models=list(SUPPORTED_MODEL_NAMES),
            local_artifacts=[
                self._artifact("model", MODEL_PATH),
                self._artifact("vectorizer", VECTORIZER_PATH),
                self._artifact("metrics", METRICS_PATH),
            ],
            registered_models=self._registered_models(),
            database_models=self._database_models(),
        )

    @staticmethod
    def _artifact(name: str, path: Path) -> LocalArtifact:
        return LocalArtifact(
            name=name,
            path=str(path),
            exists=path.exists(),
            size_bytes=path.stat().st_size if path.exists() else None,
        )

    @staticmethod
    def _registered_models() -> list[RegisteredModelVersion]:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            registered_models = client.search_registered_models()
        except Exception:
            return []

        versions: list[RegisteredModelVersion] = []
        for registered_model in registered_models:
            for version in registered_model.latest_versions:
                versions.append(
                    RegisteredModelVersion(
                        name=registered_model.name,
                        version=str(version.version),
                        current_stage=getattr(version, "current_stage", None),
                        run_id=getattr(version, "run_id", None),
                        source=getattr(version, "source", None),
                    )
                )
        return versions

    @staticmethod
    def _database_models() -> list[DatabaseModelVersion]:
        try:
            rows = CategoryIQPostgresService().list_models()
        except (OSError, psycopg2.Error):
            return []
        return [
            DatabaseModelVersion(
                version=str(row["version"]),
                accuracy=float(row["accuracy"]) if row.get("accuracy") is not None else None,
                created_at=row["created_at"].isoformat(),
            )
            for row in rows
        ]
