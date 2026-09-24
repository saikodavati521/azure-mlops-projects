"""Training request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_path: str = Field(default="data/raw/Amazon-Products.csv")
    text_columns: list[str] = Field(default_factory=lambda: ["name"])
    category_column: str = Field(default="main_category")
    sub_category_column: str | None = Field(default="sub_category")
    sample_size: int | None = Field(default=None, ge=1)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    max_features: int = Field(default=50000, ge=1)
    min_df: int | float = Field(default=2)
    max_df: int | float = Field(default=0.95)
    skip_random_forest: bool = Field(default=False)
    skip_xgboost: bool = Field(default=False)
    enable_mlflow: bool = Field(default=True)
    registered_model_name: str = Field(default="CategoryIQBestProductClassifier")


class MetricItem(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    status: str
    message: str | None = None


class TrainResponse(BaseModel):
    best_model_name: str
    model_path: str
    vectorizer_path: str
    metrics_path: str
    report_path: str
    dataset_version: str
    model_version: str
    category_column: str
    sub_category_column: str | None = None
    mlflow_best_run_id: str | None = None
    mlflow_registered_model_name: str | None = None
    mlflow_registered_model_version: str | None = None
    metrics: list[MetricItem]
