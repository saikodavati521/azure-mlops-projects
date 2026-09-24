"""Training service used by the FastAPI layer."""

from __future__ import annotations

from pathlib import Path

import psycopg2

from backend.schemas.train import MetricItem, TrainRequest, TrainResponse
from backend.services.postgres_service import CategoryIQPostgresService, ModelRecord
from backend.utils.paths import METRICS_PATH, MLFLOW_TRACKING_URI, MODEL_PATH, REPORT_PATH, VECTORIZER_PATH
from src.training import TrainingConfig, train_and_select_best_model


class TrainingService:
    """Train models and convert training results into API responses."""

    def train(self, request: TrainRequest) -> TrainResponse:
        result = train_and_select_best_model(
            request.dataset_path,
            TrainingConfig(
                text_columns=tuple(request.text_columns),
                category_column=request.category_column,
                sub_category_column=request.sub_category_column,
                model_path=MODEL_PATH,
                vectorizer_path=VECTORIZER_PATH,
                metrics_path=METRICS_PATH,
                report_path=REPORT_PATH,
                test_size=request.test_size,
                random_state=request.random_state,
                max_features=request.max_features,
                min_df=request.min_df,
                max_df=request.max_df,
                sample_size=request.sample_size,
                include_random_forest=not request.skip_random_forest,
                include_xgboost=not request.skip_xgboost,
                enable_mlflow=request.enable_mlflow,
                mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                registered_model_name=request.registered_model_name,
            ),
        )
        self._persist_model_version(result.model_version, result.metrics)
        return TrainResponse(
            best_model_name=result.best_model_name,
            model_path=str(Path(result.model_path)),
            vectorizer_path=str(Path(result.vectorizer_path)),
            metrics_path=str(Path(result.metrics_path)),
            report_path=str(Path(result.report_path)),
            dataset_version=result.dataset_version,
            model_version=result.model_version,
            category_column=result.category_column,
            sub_category_column=result.sub_category_column,
            mlflow_best_run_id=result.mlflow_best_run_id,
            mlflow_registered_model_name=result.mlflow_registered_model_name,
            mlflow_registered_model_version=result.mlflow_registered_model_version,
            metrics=[
                MetricItem(
                    model_name=metric.model_name,
                    accuracy=metric.accuracy,
                    precision=metric.precision,
                    recall=metric.recall,
                    f1_score=metric.f1_score,
                    status=metric.status,
                    message=metric.message,
                )
                for metric in result.metrics
            ],
        )

    @staticmethod
    def _persist_model_version(model_version: str, metrics: object) -> None:
        best_accuracy = None
        trained_metrics = [metric for metric in metrics if getattr(metric, "status", None) == "trained"]
        if trained_metrics:
            best_metric = max(
                trained_metrics,
                key=lambda metric: (getattr(metric, "f1_score", 0), getattr(metric, "accuracy", 0)),
            )
            best_accuracy = getattr(best_metric, "accuracy", None)
        try:
            service = CategoryIQPostgresService()
            service.initialize()
            service.create_model_version(ModelRecord(version=model_version, accuracy=best_accuracy))
        except (OSError, psycopg2.Error):
            return
