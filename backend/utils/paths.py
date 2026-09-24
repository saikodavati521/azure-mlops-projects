"""Shared backend paths."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models/model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models/vectorizer.pkl"
METRICS_PATH = ROOT_DIR / "artifacts/training/model_metrics.json"
REPORT_PATH = ROOT_DIR / "artifacts/training/model_metrics.md"
SCHEMA_PATH = ROOT_DIR / "database/schema.sql"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow/mlflow.db'}"
