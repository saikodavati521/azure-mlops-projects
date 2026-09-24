from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

from src.features import (
    TfidfFeatureConfig,
    build_tfidf_features,
    build_tfidf_features_from_dataframe,
)


def test_build_tfidf_features_from_dataframe_saves_vectorizer_and_matrix(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        [
            {"name": "Apple iPhone with fast charger", "main_category": "Electronics"},
            {"name": "Samsung phone with wireless charger", "main_category": "Electronics"},
            {"name": "Cotton running shoes for men", "main_category": "Shoes"},
        ]
    )
    vectorizer_path = tmp_path / "models" / "vectorizer.pkl"
    feature_matrix_path = tmp_path / "features" / "tfidf_features.npz"

    result = build_tfidf_features_from_dataframe(
        dataframe,
        TfidfFeatureConfig(
            text_columns=("name",),
            vectorizer_path=vectorizer_path,
            feature_matrix_path=feature_matrix_path,
            min_df=1,
            max_df=1.0,
        ),
    )

    vectorizer = joblib.load(vectorizer_path)
    feature_matrix = sparse.load_npz(feature_matrix_path)

    assert result.vectorizer_path == vectorizer_path
    assert result.feature_matrix_path == feature_matrix_path
    assert result.row_count == 3
    assert result.feature_count == len(vectorizer.get_feature_names_out())
    assert result.text_columns == ("name",)
    assert feature_matrix.shape == (3, result.feature_count)
    assert "iphone" in vectorizer.get_feature_names_out()


def test_build_tfidf_features_loads_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "products.csv"
    pd.DataFrame(
        [
            {"product_name": "Laptop stand", "description": "Adjustable desk stand"},
            {"product_name": "Gaming mouse", "description": "Wireless mouse"},
        ]
    ).to_csv(dataset_path, index=False)

    result = build_tfidf_features(
        dataset_path,
        TfidfFeatureConfig(
            vectorizer_path=tmp_path / "vectorizer.pkl",
            feature_matrix_path=tmp_path / "features.npz",
            min_df=1,
            max_df=1.0,
        ),
    )

    assert result.row_count == 2
    assert result.feature_count > 0
    assert result.vectorizer_path.exists()
    assert result.feature_matrix_path.exists()
