"""Create or update the reusable Azure ML training environment."""

from __future__ import annotations

from azure.ai.ml.entities import Environment

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def main() -> None:
    ml_client = get_ml_client()
    environment = Environment(
        name=CONFIG.environment_name,
        description="Python 3.10 environment for Steel Energy forecasting",
        conda_file="create_env/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    created = ml_client.environments.create_or_update(environment)
    print(f"Environment ready: {created.name}:{created.version}")


if __name__ == "__main__":
    main()
