"""Submit the Azure ML command job for model training."""

from __future__ import annotations

import argparse

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def submit_training_job(data_asset_version: str | None = None):
    ml_client = get_ml_client()
    data_uri = f"azureml:{CONFIG.data_asset_name}:{data_asset_version}" if data_asset_version else f"azureml:{CONFIG.data_asset_name}@latest"
    job = command(
        code=".",
        command=(
            "python -m model_training.train_code.train "
            "--input-data ${{inputs.training_data}} "
            "--output-dir ${{outputs.output_dir}}"
        ),
        inputs={"training_data": Input(type=AssetTypes.URI_FILE, path=data_uri)},
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=f"{CONFIG.environment_name}@latest",
        compute=CONFIG.compute_name,
        experiment_name=CONFIG.experiment_name,
        display_name="steel-energy-random-forest-training",
    )
    created = ml_client.jobs.create_or_update(job)
    print(f"Submitted training job: {created.name}")
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-asset-version", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    submit_training_job(args.data_asset_version)
