"""Register a successful MLflow run model into Azure ML Model Registry."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def model_version() -> str:
    return os.getenv("BUILD_BUILDID") or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def register_model(run_id: str, version: str | None = None):
    ml_client = get_ml_client()
    model = Model(
        name=CONFIG.model_name,
        version=version or model_version(),
        type=AssetTypes.MLFLOW_MODEL,
        path=f"azureml://jobs/{run_id}/outputs/artifacts/paths/model",
        description="MLflow sklearn Pipeline for steel energy consumption forecasting",
    )
    created = ml_client.models.create_or_update(model)
    print(f"Registered model: {created.name}:{created.version}")
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Azure ML job name or verified MLflow run output path owner")
    parser.add_argument("--version", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_model(args.run_id, args.version)
