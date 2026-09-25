"""Create or update the managed online endpoint."""

from __future__ import annotations

from azure.ai.ml.entities import ManagedOnlineEndpoint

from src.config.azure_client import get_ml_client
from src.config.variables import CONFIG


def create_endpoint():
    ml_client = get_ml_client()
    endpoint = ManagedOnlineEndpoint(
        name=CONFIG.endpoint_name,
        auth_mode="key",
        description="Managed online endpoint for Steel Energy forecasting",
    )
    created = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint ready: {created.name}")
    return created


if __name__ == "__main__":
    create_endpoint()
