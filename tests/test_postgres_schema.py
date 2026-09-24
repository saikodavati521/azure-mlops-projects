from pathlib import Path

from backend.services.postgres_service import (
    CategoryIQPostgresService,
    ModelRecord,
    PredictionRecord,
    ProductRecord,
)
from backend.utils.database import DEFAULT_DATABASE_URL, DatabaseConfig


def test_postgres_schema_defines_required_tables_and_columns() -> None:
    schema = Path("database/schema.sql").read_text(encoding="utf-8").casefold()

    assert "create table if not exists products" in schema
    assert "id bigserial primary key" in schema
    assert "product_name text not null" in schema
    assert "description text" in schema
    assert "brand text" in schema
    assert "price numeric(12, 2)" in schema

    assert "create table if not exists predictions" in schema
    assert "prediction_id bigserial primary key" in schema
    assert "product_id bigint not null references products(id)" in schema
    assert "predicted_category text not null" in schema
    assert "predicted_sub_category text" in schema
    assert "confidence numeric(6, 5)" in schema
    assert "\"timestamp\" timestamptz not null default now()" in schema

    assert "create table if not exists models" in schema
    assert "version text primary key" in schema
    assert "accuracy numeric(6, 5)" in schema
    assert "created_at timestamptz not null default now()" in schema


def test_postgres_service_records_and_config_are_constructible() -> None:
    config = DatabaseConfig()
    service = CategoryIQPostgresService(config)

    assert config.database_url == DEFAULT_DATABASE_URL
    assert service.config == config
    assert ProductRecord(product_name="Phone", brand="Acme").product_name == "Phone"
    assert PredictionRecord(product_id=1, predicted_category="electronics").product_id == 1
    assert ModelRecord(version="202608110001", accuracy=0.95).version == "202608110001"
