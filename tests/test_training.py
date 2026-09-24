from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

from src.training import (
    SUPPORTED_MODEL_NAMES,
    TrainingConfig,
    train_and_select_best_model,
    train_and_select_best_model_from_dataframe,
)
from src.training.train_models import DecodedLabelClassifier, LABEL_SEPARATOR, _build_model_candidates


def _training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "iphone smartphone charger camera", "main_category": "electronics"},
            {"name": "samsung galaxy android phone", "main_category": "electronics"},
            {"name": "wireless earbuds bluetooth audio", "main_category": "electronics"},
            {"name": "gaming laptop keyboard display", "main_category": "electronics"},
            {"name": "cotton shirt casual wear", "main_category": "fashion"},
            {"name": "denim jeans slim fit", "main_category": "fashion"},
            {"name": "running shoes sports footwear", "main_category": "fashion"},
            {"name": "leather wallet card holder", "main_category": "fashion"},
        ]
    )


def _subcategory_training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "iphone smartphone charger camera",
                "main_category": "electronics",
                "sub_category": "mobiles",
            },
            {
                "name": "samsung galaxy android phone",
                "main_category": "electronics",
                "sub_category": "mobiles",
            },
            {
                "name": "wireless earbuds bluetooth audio",
                "main_category": "electronics",
                "sub_category": "headphones",
            },
            {
                "name": "bluetooth over ear headphones",
                "main_category": "electronics",
                "sub_category": "headphones",
            },
            {
                "name": "cotton shirt casual wear",
                "main_category": "fashion",
                "sub_category": "shirts",
            },
            {
                "name": "formal office cotton shirt",
                "main_category": "fashion",
                "sub_category": "shirts",
            },
            {
                "name": "running shoes sports footwear",
                "main_category": "fashion",
                "sub_category": "shoes",
            },
            {
                "name": "leather sports running shoes",
                "main_category": "fashion",
                "sub_category": "shoes",
            },
        ]
    )


def test_train_and_select_best_model_from_dataframe_saves_artifacts(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        text_columns=("name",),
        category_column="main_category",
        model_path=tmp_path / "models" / "model.pkl",
        vectorizer_path=tmp_path / "models" / "vectorizer.pkl",
        metrics_path=tmp_path / "artifacts" / "model_metrics.json",
        report_path=tmp_path / "artifacts" / "model_metrics.md",
        min_df=1,
        max_df=1.0,
        test_size=0.25,
        include_xgboost=False,
        enable_mlflow=False,
        random_forest_estimators=5,
        logistic_regression_max_iter=200,
    )

    result = train_and_select_best_model_from_dataframe(_training_dataframe(), config)

    model = joblib.load(result.model_path)
    vectorizer = joblib.load(result.vectorizer_path)

    assert result.best_model_name in {"Logistic Regression", "Naive Bayes", "Random Forest"}
    assert result.row_count == 8
    assert result.train_row_count == 6
    assert result.test_row_count == 2
    assert result.feature_count == len(vectorizer.get_feature_names_out())
    assert result.model_path.exists()
    assert result.vectorizer_path.exists()
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    assert hasattr(model, "predict")
    assert all(metric.status == "trained" for metric in result.metrics)


def test_supported_model_roster_includes_requested_classifiers() -> None:
    label_encoder = LabelEncoder().fit(["electronics", "fashion"])

    candidates = _build_model_candidates(TrainingConfig(), label_encoder)

    assert SUPPORTED_MODEL_NAMES == (
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    )
    assert [candidate[0] for candidate in candidates] == list(SUPPORTED_MODEL_NAMES)


def test_train_and_select_best_model_preserves_subcategory_labels(tmp_path: Path) -> None:
    config = TrainingConfig(
        text_columns=("name",),
        category_column="main_category",
        sub_category_column="sub_category",
        model_path=tmp_path / "models" / "model.pkl",
        vectorizer_path=tmp_path / "models" / "vectorizer.pkl",
        metrics_path=tmp_path / "artifacts" / "model_metrics.json",
        report_path=tmp_path / "artifacts" / "model_metrics.md",
        min_df=1,
        max_df=1.0,
        test_size=0.25,
        include_xgboost=False,
        enable_mlflow=False,
        random_forest_estimators=5,
        logistic_regression_max_iter=200,
    )

    result = train_and_select_best_model_from_dataframe(_subcategory_training_dataframe(), config)
    model = joblib.load(result.model_path)
    vectorizer = joblib.load(result.vectorizer_path)
    prediction = str(model.predict(vectorizer.transform(["bluetooth headphones"]))[0])

    assert result.sub_category_column == "sub_category"
    assert LABEL_SEPARATOR in prediction


def test_decoded_label_classifier_returns_original_labels() -> None:
    label_encoder = LabelEncoder().fit(["electronics|||headphones", "fashion|||shoes"])
    features = [[0], [1]]
    estimator = DummyClassifier(strategy="constant", constant=0)
    estimator.fit(features, [0, 1])
    wrapped = DecodedLabelClassifier(estimator, label_encoder)

    assert wrapped.predict(features).tolist() == [
        "electronics|||headphones",
        "electronics|||headphones",
    ]


def test_train_and_select_best_model_loads_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "products.csv"
    _training_dataframe().to_csv(dataset_path, index=False)

    result = train_and_select_best_model(
        dataset_path,
        TrainingConfig(
            text_columns=("name",),
            model_path=tmp_path / "model.pkl",
            vectorizer_path=tmp_path / "vectorizer.pkl",
            metrics_path=tmp_path / "metrics.json",
            report_path=tmp_path / "metrics.md",
            min_df=1,
            max_df=1.0,
            include_xgboost=False,
            enable_mlflow=False,
            random_forest_estimators=5,
        ),
    )

    assert result.category_column == "main_category"
    assert result.model_path.exists()
    assert result.vectorizer_path.exists()


def test_train_and_select_best_model_logs_to_mlflow(tmp_path: Path) -> None:
    tracking_path = tmp_path / "mlflow.db"
    result = train_and_select_best_model_from_dataframe(
        _training_dataframe(),
        TrainingConfig(
            text_columns=("name",),
            category_column="main_category",
            model_path=tmp_path / "models" / "model.pkl",
            vectorizer_path=tmp_path / "models" / "vectorizer.pkl",
            metrics_path=tmp_path / "artifacts" / "metrics.json",
            report_path=tmp_path / "artifacts" / "metrics.md",
            min_df=1,
            max_df=1.0,
            include_xgboost=False,
            random_forest_estimators=5,
            mlflow_tracking_uri=f"sqlite:///{tracking_path.as_posix()}",
            mlflow_experiment_name="test-categoryiq",
            registered_model_name="TestCategoryIQBestModel",
        ),
    )

    assert result.mlflow_experiment_name == "test-categoryiq"
    assert result.mlflow_best_run_id is not None
    assert result.mlflow_registered_model_name == "TestCategoryIQBestModel"
    assert tracking_path.exists()
