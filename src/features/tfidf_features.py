"""TF-IDF feature engineering for product classification datasets."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import PreprocessingConfig, TextPreprocessor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfidfFeatureConfig:
    """Configuration for building TF-IDF features."""

    text_columns: tuple[str, ...] | None = None
    vectorizer_path: Path = Path("models/vectorizer.pkl")
    feature_matrix_path: Path = Path("artifacts/features/tfidf_features.npz")
    cleaned_suffix: str = "_cleaned"
    max_features: int | None = 50000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.95
    lowercase: bool = False


@dataclass(frozen=True)
class TfidfFeatureResult:
    """Paths and metadata produced by TF-IDF feature generation."""

    vectorizer_path: Path
    feature_matrix_path: Path
    row_count: int
    feature_count: int
    text_columns: tuple[str, ...]


def build_tfidf_features(
    input_path: str | Path,
    config: TfidfFeatureConfig | None = None,
) -> TfidfFeatureResult:
    """Clean text, fit a TF-IDF vectorizer, and save the feature matrix."""

    active_config = config or TfidfFeatureConfig()
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV input files are supported: {path}")

    LOGGER.info("Loading dataset for TF-IDF feature engineering: %s", path)
    dataframe = pd.read_csv(path)
    return build_tfidf_features_from_dataframe(dataframe, active_config)


def build_tfidf_features_from_dataframe(
    dataframe: pd.DataFrame,
    config: TfidfFeatureConfig | None = None,
) -> TfidfFeatureResult:
    """Build TF-IDF features from an in-memory dataframe."""

    active_config = config or TfidfFeatureConfig()
    preprocessor = TextPreprocessor(
        PreprocessingConfig(
            text_columns=active_config.text_columns,
            cleaned_suffix=active_config.cleaned_suffix,
        )
    )
    processed = preprocessor.transform_dataframe(dataframe)
    text_columns = preprocessor.resolve_text_columns(dataframe.columns)
    cleaned_columns = tuple(f"{column}{active_config.cleaned_suffix}" for column in text_columns)
    corpus = _combine_text_columns(processed, cleaned_columns)

    vectorizer = TfidfVectorizer(
        max_features=active_config.max_features,
        ngram_range=active_config.ngram_range,
        min_df=active_config.min_df,
        max_df=active_config.max_df,
        lowercase=active_config.lowercase,
    )
    feature_matrix = vectorizer.fit_transform(corpus)

    active_config.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    active_config.feature_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, active_config.vectorizer_path)
    sparse.save_npz(active_config.feature_matrix_path, feature_matrix)

    LOGGER.info("Saved fitted TF-IDF vectorizer to: %s", active_config.vectorizer_path)
    LOGGER.info("Saved TF-IDF feature matrix to: %s", active_config.feature_matrix_path)
    return TfidfFeatureResult(
        vectorizer_path=active_config.vectorizer_path,
        feature_matrix_path=active_config.feature_matrix_path,
        row_count=int(feature_matrix.shape[0]),
        feature_count=int(feature_matrix.shape[1]),
        text_columns=text_columns,
    )


def _combine_text_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return dataframe.loc[:, columns].fillna("").agg(" ".join, axis=1).str.strip()


def _coerce_document_frequency(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean product text, fit TF-IDF, and save the vectorizer and feature matrix."
    )
    parser.add_argument("input_path", help="Path to a raw or processed CSV dataset.")
    parser.add_argument(
        "--text-column",
        action="append",
        dest="text_columns",
        help="Text column to use. Pass more than once to combine multiple columns.",
    )
    parser.add_argument(
        "--vectorizer-path",
        default="models/vectorizer.pkl",
        help="Path where the fitted TF-IDF vectorizer should be saved.",
    )
    parser.add_argument(
        "--feature-matrix-path",
        default="artifacts/features/tfidf_features.npz",
        help="Path where the sparse TF-IDF feature matrix should be saved.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument("--min-df", type=float, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram size.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram size.")
    return parser


def main() -> int:
    """Run TF-IDF feature engineering from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    config = TfidfFeatureConfig(
        text_columns=tuple(args.text_columns) if args.text_columns else None,
        vectorizer_path=Path(args.vectorizer_path),
        feature_matrix_path=Path(args.feature_matrix_path),
        max_features=args.max_features,
        min_df=_coerce_document_frequency(args.min_df),
        max_df=_coerce_document_frequency(args.max_df),
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    result = build_tfidf_features(args.input_path, config)
    LOGGER.info(
        "TF-IDF complete: %s rows, %s features, columns=%s",
        result.row_count,
        result.feature_count,
        ", ".join(result.text_columns),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
