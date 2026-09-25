"""Train the Steel Energy forecasting model with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.variables import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from model_training.train_code.data_validation import validate_steel_energy_data
from model_training.train_code.evaluate import regression_metrics, write_metrics
from model_training.train_code.feature_engineering import build_features

MODEL_SELECTION_METRIC = "validation_rmse"


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= len(data):
        raise ValueError("Not enough rows after feature engineering for 70/15/15 chronological split")
    return data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]


def create_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBRegressor
    except ImportError:
        pass
    else:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        pass
    else:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def create_pipeline(model: object | None = None) -> Pipeline:
    if model is None:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline(steps=[("preprocessor", create_preprocessor()), ("model", model)])


def evaluate_pipeline(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    pipeline.fit(x_train, y_train)
    validation_metrics = regression_metrics(y_validation, pipeline.predict(x_validation))
    test_metrics = regression_metrics(y_test, pipeline.predict(x_test))
    return {
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_r2": validation_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }


def train_model(input_data: str | Path, output_dir: str | Path = "outputs") -> dict[str, float]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.getenv("MLFLOW_TRACKING_URI") and not os.getenv("AZUREML_RUN_ID"):
        tracking_db = (Path.cwd() / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    raw_data = pd.read_csv(input_data)
    validate_steel_energy_data(raw_data)
    data = build_features(raw_data, dropna=True)
    train_df, validation_df, test_df = chronological_split(data)

    feature_columns = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    comparison: dict[str, dict[str, object]] = {}
    best_model_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] | None = None
    candidates = candidate_models()

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "target_column": TARGET_COLUMN,
                "candidate_models": ",".join(candidates.keys()),
                "model_selection_metric": MODEL_SELECTION_METRIC,
                "split_strategy": "chronological_70_15_15",
            }
        )

        for model_name, model in candidates.items():
            pipeline = create_pipeline(model)
            metrics = evaluate_pipeline(pipeline, x_train, y_train, x_validation, y_validation, x_test, y_test)
            comparison[model_name] = {"status": "trained", "metrics": metrics}
            mlflow.log_metrics({f"{model_name}_{key}": value for key, value in metrics.items()})

            if best_metrics is None or metrics[MODEL_SELECTION_METRIC] < best_metrics[MODEL_SELECTION_METRIC]:
                best_model_name = model_name
                best_pipeline = pipeline
                best_metrics = metrics

        skipped_models = {
            "xgboost": "Install xgboost to include this candidate.",
            "lightgbm": "Install lightgbm to include this candidate.",
        }
        for model_name, reason in skipped_models.items():
            if model_name not in comparison:
                comparison[model_name] = {"status": "skipped", "reason": reason}

        if best_model_name is None or best_pipeline is None or best_metrics is None:
            raise RuntimeError("No candidate models were trained")

        metrics = {"selected_model": best_model_name, **best_metrics}
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_metrics(best_metrics)
        metrics_file = output_path / "metrics.json"
        write_metrics(metrics, metrics_file)
        comparison_file = output_path / "model_comparison.json"
        comparison_file.write_text(
            json.dumps(
                {
                    "target": TARGET_COLUMN,
                    "selection_metric": MODEL_SELECTION_METRIC,
                    "selected_model": best_model_name,
                    "candidates": comparison,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(metrics_file))
        mlflow.log_artifact(str(comparison_file))
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        (output_path / "run_info.json").write_text(
            json.dumps({"mlflow_run_id": run.info.run_id}, indent=2),
            encoding="utf-8",
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True, help="Path to Steel_industry_data.csv")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.input_data, args.output_dir)
