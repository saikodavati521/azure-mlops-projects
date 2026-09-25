"""Validate local CSV data and register it as a versioned Azure ML data asset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from model_training.train_code.data_validation import validate_steel_energy_data
from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def version_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def register_data_asset(csv_path: str | Path, version: str | None = None) -> str:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Expected local dataset at {path}")

    data = pd.read_csv(path)
    validate_steel_energy_data(data)

    ml_client = get_ml_client()
    asset = Data(
        name=CONFIG.data_asset_name,
        version=version or version_stamp(),
        type=AssetTypes.URI_FILE,
        path=str(path),
        description="Validated UCI Steel Industry Energy Consumption CSV",
    )
    created = ml_client.data.create_or_update(asset)
    print(f"Data asset ready: azureml:{created.name}:{created.version}")
    return created.version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="data/Steel_industry_data.csv")
    parser.add_argument("--version", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_data_asset(args.csv_path, args.version)
