"""PostgreSQL persistence service for products, predictions, and model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from backend.utils.database import DatabaseConfig, get_connection, initialize_database, row_to_dict


@dataclass(frozen=True)
class ProductRecord:
    product_name: str
    description: str | None = None
    brand: str | None = None
    price: Decimal | float | str | None = None


@dataclass(frozen=True)
class PredictionRecord:
    product_id: int
    predicted_category: str
    predicted_sub_category: str | None = None
    confidence: Decimal | float | str | None = None


@dataclass(frozen=True)
class ModelRecord:
    version: str
    accuracy: Decimal | float | str | None = None


class CategoryIQPostgresService:
    """Read and write CategoryIQ records in PostgreSQL."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig.from_env()

    def initialize(self) -> None:
        initialize_database(self.config)

    def create_product(self, product: ProductRecord) -> dict[str, Any]:
        query = """
            INSERT INTO products (product_name, description, brand, price)
            VALUES (%s, %s, %s, %s)
            RETURNING id, product_name, description, brand, price, created_at;
        """
        return self._fetch_one(
            query,
            (product.product_name, product.description, product.brand, product.price),
        )

    def get_product(self, product_id: int) -> dict[str, Any] | None:
        query = """
            SELECT id, product_name, description, brand, price, created_at
            FROM products
            WHERE id = %s;
        """
        return self._fetch_optional(query, (product_id,))

    def create_prediction(self, prediction: PredictionRecord) -> dict[str, Any]:
        query = """
            INSERT INTO predictions (
                product_id,
                predicted_category,
                predicted_sub_category,
                confidence
            )
            VALUES (%s, %s, %s, %s)
            RETURNING
                prediction_id,
                product_id,
                predicted_category,
                predicted_sub_category,
                confidence,
                "timestamp";
        """
        return self._fetch_one(
            query,
            (
                prediction.product_id,
                prediction.predicted_category,
                prediction.predicted_sub_category,
                prediction.confidence,
            ),
        )

    def create_product_with_prediction(
        self,
        product: ProductRecord,
        predicted_category: str,
        predicted_sub_category: str | None = None,
        confidence: Decimal | float | str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        product_query = """
            INSERT INTO products (product_name, description, brand, price)
            VALUES (%s, %s, %s, %s)
            RETURNING id, product_name, description, brand, price, created_at;
        """
        prediction_query = """
            INSERT INTO predictions (
                product_id,
                predicted_category,
                predicted_sub_category,
                confidence
            )
            VALUES (%s, %s, %s, %s)
            RETURNING
                prediction_id,
                product_id,
                predicted_category,
                predicted_sub_category,
                confidence,
                "timestamp";
        """
        with get_connection(self.config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    product_query,
                    (product.product_name, product.description, product.brand, product.price),
                )
                product_row = row_to_dict(cursor.fetchone())
                if product_row is None:
                    raise RuntimeError("PostgreSQL product insert did not return a row.")
                cursor.execute(
                    prediction_query,
                    (
                        product_row["id"],
                        predicted_category,
                        predicted_sub_category,
                        confidence,
                    ),
                )
                prediction_row = row_to_dict(cursor.fetchone())
                if prediction_row is None:
                    raise RuntimeError("PostgreSQL prediction insert did not return a row.")
            connection.commit()
        return product_row, prediction_row

    def create_model_version(self, model: ModelRecord) -> dict[str, Any]:
        query = """
            INSERT INTO models (version, accuracy)
            VALUES (%s, %s)
            ON CONFLICT (version)
            DO UPDATE SET accuracy = EXCLUDED.accuracy
            RETURNING version, accuracy, created_at;
        """
        return self._fetch_one(query, (model.version, model.accuracy))

    def list_models(self) -> list[dict[str, Any]]:
        query = """
            SELECT version, accuracy, created_at
            FROM models
            ORDER BY created_at DESC;
        """
        with get_connection(self.config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                return [dict(row) for row in cursor.fetchall()]

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any]:
        row = self._fetch_optional(query, params)
        if row is None:
            raise RuntimeError("PostgreSQL query did not return a row.")
        return row

    def _fetch_optional(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with get_connection(self.config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = row_to_dict(cursor.fetchone())
            connection.commit()
        return row
