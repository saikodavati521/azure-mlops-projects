"""PostgreSQL connection and schema initialization helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor

from backend.utils.paths import ROOT_DIR, SCHEMA_PATH

load_dotenv(ROOT_DIR / ".env")

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/categoryiq"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 3


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""

    database_url: str = DEFAULT_DATABASE_URL

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))


def get_connection(config: DatabaseConfig | None = None) -> PgConnection:
    """Open a PostgreSQL connection."""

    active_config = config or DatabaseConfig.from_env()
    return psycopg2.connect(
        active_config.database_url,
        cursor_factory=RealDictCursor,
        connect_timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS,
    )


def initialize_database(
    config: DatabaseConfig | None = None,
    schema_path: str | Path = SCHEMA_PATH,
) -> None:
    """Create the CategoryIQ PostgreSQL tables if they do not already exist."""

    schema = Path(schema_path).read_text(encoding="utf-8")
    with get_connection(config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema)
        connection.commit()


def row_to_dict(row: Any) -> dict[str, Any] | None:
    """Convert a psycopg row into a plain dict."""

    return dict(row) if row is not None else None


def main() -> int:
    initialize_database()
    print("PostgreSQL schema initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
