"""Train and compare product category classification models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import PreprocessingConfig, TextPreprocessor

LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORY_COLUMN_CANDIDATES = ("category", "main_category", "label")
DEFAULT_SUB_CATEGORY_COLUMN_CANDIDATES = ("sub_category", "subcategory", "secondary_category")
LABEL_SEPARATOR = "|||"
LOGISTIC_REGRESSION_MODEL_NAME = "Logistic Regression"
NAIVE_BAYES_MODEL_NAME = "Naive Bayes"
RANDOM_FOREST_MODEL_NAME = "Random Forest"
XGBOOST_MODEL_NAME = "XGBoost"
SUPPORTED_MODEL_NAMES = (
    LOGISTIC_REGRESSION_MODEL_NAME,
    NAIVE_BAYES_MODEL_NAME,
    RANDOM_FOREST_MODEL_NAME,
    XGBOOST_MODEL_NAME,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training and comparison."""

    text_columns: tuple[str, ...] | None = None
    category_column: str | None = None
    category_column_candidates: tuple[str, ...] = DEFAULT_CATEGORY_COLUMN_CANDIDATES
    sub_category_column: str | None = None
    sub_category_column_candidates: tuple[str, ...] = DEFAULT_SUB_CATEGORY_COLUMN_CANDIDATES
    model_path: Path = Path("models/model.pkl")
    vectorizer_path: Path = Path("models/vectorizer.pkl")
    metrics_path: Path = Path("artifacts/training/model_metrics.json")
    report_path: Path = Path("artifacts/training/model_metrics.md")
    test_size: float = 0.2
    random_state: int = 42
    max_features: int | None = 50000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.95
    average: str = "weighted"
    sample_size: int | None = None
    include_xgboost: bool = True
    include_random_forest: bool = True
    logistic_regression_max_iter: int = 1000
    random_forest_estimators: int = 100
    xgboost_estimators: int = 200
    enable_mlflow: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlflow/mlflow.db"
    mlflow_experiment_name: str = "CategoryIQ Product Classification"
    registered_model_name: str = "CategoryIQBestProductClassifier"


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for one trained model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    status: str = "trained"
    message: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Training output metadata."""

    best_model_name: str
    model_path: Path
    vectorizer_path: Path
    metrics_path: Path
    report_path: Path
    row_count: int
    train_row_count: int
    test_row_count: int
    feature_count: int
    text_columns: tuple[str, ...]
    category_column: str
    sub_category_column: str | None
    dataset_version: str
    model_version: str
    mlflow_experiment_name: str | None = None
    mlflow_best_run_id: str | None = None
    mlflow_registered_model_name: str | None = None
    mlflow_registered_model_version: str | None = None
    metrics: list[ModelMetrics] = field(default_factory=list)


class DecodedLabelClassifier:
    """Wrap classifiers trained on encoded labels so inference returns labels."""

    def __init__(self, estimator: Any, label_encoder: LabelEncoder) -> None:
        self.estimator = estimator
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, features: Any) -> ArrayLike:
        encoded_predictions = self.estimator.predict(features)
        return self.label_encoder.inverse_transform(encoded_predictions.astype(int))

    def predict_proba(self, features: Any) -> Any:
        return self.estimator.predict_proba(features)


def train_and_select_best_model(
    input_path: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train multiple classifiers and save the best-performing model."""

    active_config = config or TrainingConfig()
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV input files are supported: {path}")

    LOGGER.info("Loading dataset for model training: %s", path)
    dataframe = pd.read_csv(path)
    return _train_and_select_best_model(
        dataframe=dataframe,
        config=active_config,
        dataset_path=path,
        dataset_version=_file_sha256(path),
    )


def train_and_select_best_model_from_dataframe(
    dataframe: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train multiple classifiers from an in-memory dataframe."""

    return _train_and_select_best_model(
        dataframe=dataframe,
        config=config or TrainingConfig(),
        dataset_path=None,
        dataset_version=_dataframe_version(dataframe),
    )


def _train_and_select_best_model(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
    dataset_path: Path | None,
    dataset_version: str,
) -> TrainingResult:
    """Shared implementation for CSV and in-memory training."""

    active_config = config or TrainingConfig()
    prepared = dataframe.copy()
    if active_config.sample_size is not None and len(prepared) > active_config.sample_size:
        prepared = prepared.sample(
            n=active_config.sample_size,
            random_state=active_config.random_state,
        )

    category_column = _resolve_category_column(prepared.columns, active_config)
    sub_category_column = _resolve_optional_sub_category_column(prepared.columns, active_config)
    label_columns = [category_column]
    if sub_category_column is not None:
        label_columns.append(sub_category_column)

    prepared = prepared.dropna(subset=label_columns).copy()
    prepared[category_column] = prepared[category_column].astype(str).str.strip()
    if sub_category_column is not None:
        prepared[sub_category_column] = prepared[sub_category_column].astype(str).str.strip()
    for label_column in label_columns:
        prepared = prepared[prepared[label_column] != ""]

    labels = _build_training_labels(prepared, category_column, sub_category_column)
    if labels.nunique() < 2:
        raise ValueError("Training requires at least two non-empty category values.")

    preprocessor = TextPreprocessor(PreprocessingConfig(text_columns=active_config.text_columns))
    processed = preprocessor.transform_dataframe(prepared)
    text_columns = preprocessor.resolve_text_columns(prepared.columns)
    cleaned_columns = tuple(f"{column}_cleaned" for column in text_columns)
    corpus = _combine_text_columns(processed, cleaned_columns)
    stratify = labels if _can_stratify(labels, active_config.test_size) else None
    x_train, x_test, y_train, y_test = train_test_split(
        corpus,
        labels,
        test_size=active_config.test_size,
        random_state=active_config.random_state,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(
        max_features=active_config.max_features,
        ngram_range=active_config.ngram_range,
        min_df=active_config.min_df,
        max_df=active_config.max_df,
        lowercase=False,
    )
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    label_encoder = LabelEncoder().fit(y_train)
    candidates = _build_model_candidates(active_config, label_encoder)
    metrics: list[ModelMetrics] = []
    trained_models: dict[str, Any] = {}
    encoded_label_models: set[str] = set()
    mlflow_run_ids: dict[str, str] = {}
    model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    if active_config.enable_mlflow:
        _configure_mlflow(active_config)

    for model_name, model, requires_encoded_labels, skip_message in candidates:
        if skip_message is not None:
            metrics.append(
                ModelMetrics(
                    model_name=model_name,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    status="skipped",
                    message=skip_message,
                )
            )
            continue

        train_labels = label_encoder.transform(y_train) if requires_encoded_labels else y_train
        model.fit(x_train_features, train_labels)
        predictions = model.predict(x_test_features)
        if requires_encoded_labels:
            predictions = label_encoder.inverse_transform(predictions.astype(int))

        model_metrics = _evaluate_model(model_name, y_test, predictions, active_config.average)
        metrics.append(model_metrics)
        trained_models[model_name] = model
        if requires_encoded_labels:
            encoded_label_models.add(model_name)
        if active_config.enable_mlflow:
            run_id = _log_mlflow_model_run(
                config=active_config,
                model_name=model_name,
                model=model,
                vectorizer=vectorizer,
                metrics=model_metrics,
                dataset_path=dataset_path,
                dataset_version=dataset_version,
                model_version=model_version,
                row_count=len(prepared),
                train_row_count=len(x_train),
                test_row_count=len(x_test),
                feature_count=x_train_features.shape[1],
                text_columns=text_columns,
                category_column=category_column,
                sub_category_column=sub_category_column,
                requires_encoded_labels=requires_encoded_labels,
            )
            mlflow_run_ids[model_name] = run_id

    trained_metrics = [metric for metric in metrics if metric.status == "trained"]
    if not trained_metrics:
        raise RuntimeError("No models were trained successfully.")

    best_metrics = max(trained_metrics, key=lambda item: (item.f1_score, item.accuracy))
    best_model = trained_models[best_metrics.model_name]
    if best_metrics.model_name in encoded_label_models:
        best_model = DecodedLabelClassifier(best_model, label_encoder)

    active_config.model_path.parent.mkdir(parents=True, exist_ok=True)
    active_config.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    active_config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    active_config.report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, active_config.model_path)
    joblib.dump(vectorizer, active_config.vectorizer_path)

    registered_model_version: str | None = None
    best_run_id = mlflow_run_ids.get(best_metrics.model_name)
    if active_config.enable_mlflow and best_run_id is not None:
        registered_model_version = _register_best_mlflow_model(
            run_id=best_run_id,
            registered_model_name=active_config.registered_model_name,
        )

    result = TrainingResult(
        best_model_name=best_metrics.model_name,
        model_path=active_config.model_path,
        vectorizer_path=active_config.vectorizer_path,
        metrics_path=active_config.metrics_path,
        report_path=active_config.report_path,
        row_count=int(len(prepared)),
        train_row_count=int(len(x_train)),
        test_row_count=int(len(x_test)),
        feature_count=int(x_train_features.shape[1]),
        text_columns=text_columns,
        category_column=category_column,
        sub_category_column=sub_category_column,
        dataset_version=dataset_version,
        model_version=model_version,
        mlflow_experiment_name=(
            active_config.mlflow_experiment_name if active_config.enable_mlflow else None
        ),
        mlflow_best_run_id=best_run_id,
        mlflow_registered_model_name=(
            active_config.registered_model_name if active_config.enable_mlflow else None
        ),
        mlflow_registered_model_version=registered_model_version,
        metrics=metrics,
    )
    _write_training_metrics(result)
    if active_config.enable_mlflow and best_run_id is not None:
        _log_best_run_artifacts(best_run_id, active_config)

    LOGGER.info("Best model: %s", result.best_model_name)
    LOGGER.info("Saved best model to: %s", active_config.model_path)
    LOGGER.info("Saved fitted vectorizer to: %s", active_config.vectorizer_path)
    return result


def _build_model_candidates(
    config: TrainingConfig,
    label_encoder: LabelEncoder,
) -> list[tuple[str, Any | None, bool, str | None]]:
    candidates: list[tuple[str, Any | None, bool, str | None]] = [
        (
            LOGISTIC_REGRESSION_MODEL_NAME,
            LogisticRegression(
                max_iter=config.logistic_regression_max_iter,
                class_weight="balanced",
                random_state=config.random_state,
            ),
            False,
            None,
        ),
        (NAIVE_BAYES_MODEL_NAME, MultinomialNB(), False, None),
    ]
    if config.include_random_forest:
        candidates.append(
            (
                RANDOM_FOREST_MODEL_NAME,
                RandomForestClassifier(
                    n_estimators=config.random_forest_estimators,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
                False,
                None,
            )
        )

    if not config.include_xgboost:
        return candidates

    try:
        from xgboost import XGBClassifier
    except ImportError:
        candidates.append(
            (
                XGBOOST_MODEL_NAME,
                None,
                True,
                "xgboost is not installed. Install requirements.txt to enable this model.",
            )
        )
        return candidates

    candidates.append(
        (
            XGBOOST_MODEL_NAME,
            XGBClassifier(
                n_estimators=config.xgboost_estimators,
                objective="multi:softmax",
                num_class=len(label_encoder.classes_),
                eval_metric="mlogloss",
                random_state=config.random_state,
                n_jobs=-1,
            ),
            True,
            None,
        )
    )
    return candidates


def _evaluate_model(
    model_name: str,
    actual: pd.Series,
    predicted: Any,
    average: str,
) -> ModelMetrics:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average=average,
        zero_division=0,
    )
    return ModelMetrics(
        model_name=model_name,
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
    )


def _resolve_category_column(columns: pd.Index, config: TrainingConfig) -> str:
    available_columns = tuple(str(column) for column in columns)
    normalized_lookup = {_normalize_column_name(column): column for column in available_columns}

    if config.category_column is not None:
        if config.category_column not in available_columns:
            raise ValueError(f"Configured category column not found: {config.category_column}")
        return config.category_column

    for candidate in config.category_column_candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    raise ValueError(
        "No category column found. Provide category_column or include one of: "
        f"{', '.join(config.category_column_candidates)}"
    )


def _resolve_optional_sub_category_column(columns: pd.Index, config: TrainingConfig) -> str | None:
    available_columns = tuple(str(column) for column in columns)
    normalized_lookup = {_normalize_column_name(column): column for column in available_columns}

    if config.sub_category_column is not None:
        if config.sub_category_column not in available_columns:
            raise ValueError(f"Configured sub-category column not found: {config.sub_category_column}")
        return config.sub_category_column

    for candidate in config.sub_category_column_candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]
    return None


def _build_training_labels(
    dataframe: pd.DataFrame,
    category_column: str,
    sub_category_column: str | None,
) -> pd.Series:
    category = dataframe[category_column].astype(str).str.strip()
    if sub_category_column is None:
        return category
    sub_category = dataframe[sub_category_column].astype(str).str.strip()
    return category + LABEL_SEPARATOR + sub_category


def _combine_text_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return dataframe.loc[:, columns].fillna("").agg(" ".join, axis=1).str.strip()


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    label_counts = labels.value_counts()
    if label_counts.min() < 2:
        return False

    class_count = labels.nunique()
    test_row_count = math.ceil(len(labels) * test_size)
    train_row_count = len(labels) - test_row_count
    return test_row_count >= class_count and train_row_count >= class_count


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().casefold()).strip("_")


def _coerce_document_frequency(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _configure_mlflow(config: TrainingConfig) -> None:
    Path("mlflow").mkdir(parents=True, exist_ok=True)
    if config.mlflow_tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)


def _log_mlflow_model_run(
    config: TrainingConfig,
    model_name: str,
    model: Any,
    vectorizer: TfidfVectorizer,
    metrics: ModelMetrics,
    dataset_path: Path | None,
    dataset_version: str,
    model_version: str,
    row_count: int,
    train_row_count: int,
    test_row_count: int,
    feature_count: int,
    text_columns: tuple[str, ...],
    category_column: str,
    sub_category_column: str | None,
    requires_encoded_labels: bool,
) -> str:
    run_name = _slugify(model_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "CategoryIQ",
                "stage": "training",
                "model_name": model_name,
                "dataset_version": dataset_version,
                "model_version": model_version,
            }
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_path": str(dataset_path) if dataset_path is not None else "in-memory",
                "dataset_version": dataset_version,
                "model_version": model_version,
                "text_columns": ",".join(text_columns),
                "category_column": category_column,
                "sub_category_column": sub_category_column or "",
                "test_size": config.test_size,
                "random_state": config.random_state,
                "max_features": config.max_features,
                "ngram_range": str(config.ngram_range),
                "min_df": config.min_df,
                "max_df": config.max_df,
                "average": config.average,
                "sample_size": config.sample_size,
                "requires_encoded_labels": requires_encoded_labels,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "row_count": row_count,
                "train_row_count": train_row_count,
                "test_row_count": test_row_count,
                "feature_count": feature_count,
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")
        _log_vectorizer_artifact(vectorizer)
        return run.info.run_id


def _log_vectorizer_artifact(vectorizer: TfidfVectorizer) -> None:
    artifact_dir = Path("artifacts/mlflow")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_artifact = artifact_dir / "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_artifact)
    mlflow.log_artifact(str(vectorizer_artifact), artifact_path="vectorizer")


def _register_best_mlflow_model(run_id: str, registered_model_name: str) -> str | None:
    model_uri = f"runs:/{run_id}/model"
    try:
        model_version = mlflow.register_model(model_uri, registered_model_name)
    except Exception as exc:  # pragma: no cover - depends on active MLflow backend
        LOGGER.warning("MLflow model registration failed: %s", exc)
        return None
    return str(model_version.version)


def _log_best_run_artifacts(run_id: str, config: TrainingConfig) -> None:
    client = mlflow.tracking.MlflowClient()
    artifact_paths = [
        config.model_path,
        config.vectorizer_path,
        config.metrics_path,
        config.report_path,
    ]
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            client.log_artifact(run_id, str(artifact_path), artifact_path="selected_model")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_version(dataframe: pd.DataFrame) -> str:
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _write_training_metrics(result: TrainingResult) -> None:
    payload = {
        **asdict(result),
        "model_path": str(result.model_path),
        "vectorizer_path": str(result.vectorizer_path),
        "metrics_path": str(result.metrics_path),
        "report_path": str(result.report_path),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    result.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result.report_path.write_text(_to_markdown(result), encoding="utf-8")


def _to_markdown(result: TrainingResult) -> str:
    lines = [
        "# Model Training Report",
        "",
        f"- Best model: {result.best_model_name}",
        f"- Rows: {result.row_count}",
        f"- Train rows: {result.train_row_count}",
        f"- Test rows: {result.test_row_count}",
        f"- Features: {result.feature_count}",
        f"- Text columns: {', '.join(result.text_columns)}",
        f"- Category column: {result.category_column}",
        f"- Sub-category column: {result.sub_category_column or 'not used'}",
        f"- Dataset version: {result.dataset_version}",
        f"- Model version: {result.model_version}",
        f"- Saved model: {result.model_path}",
        f"- Saved vectorizer: {result.vectorizer_path}",
        f"- MLflow experiment: {result.mlflow_experiment_name or 'disabled'}",
        f"- MLflow best run ID: {result.mlflow_best_run_id or 'not logged'}",
        f"- Registered model: {result.mlflow_registered_model_name or 'not registered'}",
        f"- Registered model version: {result.mlflow_registered_model_version or 'not registered'}",
        "",
        "## Metrics",
        "",
        "| Model | Status | Accuracy | Precision | Recall | F1-score | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in result.metrics:
        lines.append(
            "| "
            f"{metric.model_name} | "
            f"{metric.status} | "
            f"{metric.accuracy:.4f} | "
            f"{metric.precision:.4f} | "
            f"{metric.recall:.4f} | "
            f"{metric.f1_score:.4f} | "
            f"{metric.message or ''} |"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression, Naive Bayes, Random Forest, and XGBoost models."
    )
    parser.add_argument("input_path", help="Path to a raw or processed CSV dataset.")
    parser.add_argument(
        "--text-column",
        action="append",
        dest="text_columns",
        help="Text column to use. Pass more than once to combine multiple columns.",
    )
    parser.add_argument("--category-column", default=None, help="Target category column.")
    parser.add_argument(
        "--sub-category-column",
        default="sub_category",
        help="Optional target sub-category column.",
    )
    parser.add_argument("--model-path", default="models/model.pkl", help="Best model output path.")
    parser.add_argument(
        "--vectorizer-path",
        default="models/vectorizer.pkl",
        help="Fitted TF-IDF vectorizer output path.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/training/model_metrics.json",
        help="JSON metrics output path.",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/training/model_metrics.md",
        help="Markdown metrics report output path.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=float, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram size.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram size.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional training row sample.")
    parser.add_argument("--skip-random-forest", action="store_true", help="Skip Random Forest training.")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost training.")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking and model registration.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="sqlite:///mlflow/mlflow.db",
        help="MLflow tracking URI. Defaults to the local SQLite store at mlflow/mlflow.db.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="CategoryIQ Product Classification",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default="CategoryIQBestProductClassifier",
        help="MLflow registered model name for the best model.",
    )
    return parser


def main() -> int:
    """Run model training from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    config = TrainingConfig(
        text_columns=tuple(args.text_columns) if args.text_columns else None,
        category_column=args.category_column,
        sub_category_column=args.sub_category_column,
        model_path=Path(args.model_path),
        vectorizer_path=Path(args.vectorizer_path),
        metrics_path=Path(args.metrics_path),
        report_path=Path(args.report_path),
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        min_df=_coerce_document_frequency(args.min_df),
        max_df=_coerce_document_frequency(args.max_df),
        ngram_range=(args.ngram_min, args.ngram_max),
        sample_size=args.sample_size,
        include_random_forest=not args.skip_random_forest,
        include_xgboost=not args.skip_xgboost,
        enable_mlflow=not args.disable_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        registered_model_name=args.registered_model_name,
    )
    result = train_and_select_best_model(args.input_path, config)
    LOGGER.info("Training complete. Best model: %s", result.best_model_name)
    LOGGER.info("Metrics report: %s", result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
