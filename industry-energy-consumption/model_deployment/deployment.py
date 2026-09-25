"""Deploy a registered Steel Energy model to the blue online deployment."""

from __future__ import annotations

import argparse

from azure.ai.ml.entities import CodeConfiguration, ManagedOnlineDeployment

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def deploy_blue(model_version: str, instance_type: str = "Standard_DS3_v2", instance_count: int = 1):
    ml_client = get_ml_client()
    model = ml_client.models.get(name=CONFIG.model_name, version=model_version)
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=CONFIG.endpoint_name,
        model=model,
        environment=f"{CONFIG.environment_name}@latest",
        code_configuration=CodeConfiguration(
            code="model_deployment/scoring_code",
            scoring_script="score.py",
        ),
        instance_type=instance_type,
        instance_count=instance_count,
    )
    created = ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Blue deployment ready for {CONFIG.model_name}:{model_version}")
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    parser.add_argument("--instance-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    deploy_blue(args.model_version, args.instance_type, args.instance_count)
