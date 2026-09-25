from __future__ import annotations

from model_training.train_code.train import candidate_models, chronological_split, create_pipeline
from model_training.train_code.feature_engineering import build_features
from tests.test_data import valid_frame


def test_pipeline_contains_preprocessor_and_model():
    pipeline = create_pipeline()
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_candidate_models_include_required_sklearn_baselines():
    models = candidate_models()
    assert {"linear_regression", "random_forest", "gradient_boosting"}.issubset(models)


def test_chronological_split_preserves_order():
    featured = build_features(valid_frame(800))
    train_df, validation_df, test_df = chronological_split(featured)
    assert train_df["date"].max() < validation_df["date"].min()
    assert validation_df["date"].max() < test_df["date"].min()
