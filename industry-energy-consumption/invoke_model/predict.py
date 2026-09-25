"""Invoke the Azure ML managed online endpoint with a feature-complete request."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


REQUEST_SCHEMA = {
    "data": [
        "records containing raw exogenous fields plus precomputed calendar, lag, and rolling target features"
    ]
}


def invoke(request_file: str | Path = "invoke_model/sample.json", deployment_name: str = "blue"):
    ml_client = get_ml_client()
    response = ml_client.online_endpoints.invoke(
        endpoint_name=CONFIG.endpoint_name,
        deployment_name=deployment_name,
        request_file=str(request_file),
    )
    print(response)
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference expects precomputed lag and rolling features.")
    parser.add_argument("--request-file", default="invoke_model/sample.json")
    parser.add_argument("--deployment-name", default="blue")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    invoke(args.request_file, args.deployment_name)
