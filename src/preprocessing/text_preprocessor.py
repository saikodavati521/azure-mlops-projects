"""Text cleaning pipeline for product classification datasets."""

from __future__ import annotations

import argparse
import html
import logging
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "without",
        "your",
    }
)

DEFAULT_TEXT_COLUMN_CANDIDATES = (
    "product_name",
    "description",
    "name",
    "title",
    "product_title",
    "product_description",
    "about_product",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    text_columns: tuple[str, ...] | None = None
    text_column_candidates: tuple[str, ...] = DEFAULT_TEXT_COLUMN_CANDIDATES
    output_dir: Path = Path("data/processed")
    output_filename: str | None = None
    cleaned_suffix: str = "_cleaned"
    remove_stop_words: bool = True
    lemmatize: bool = False
    stop_words: frozenset[str] = field(default_factory=lambda: DEFAULT_STOP_WORDS)


class TextPreprocessor:
    """Clean product text for model training and inference."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self._lemmatizer = self._load_lemmatizer() if self.config.lemmatize else None
        self._punctuation_translation = str.maketrans(
            {character: " " for character in string.punctuation}
        )

    def clean_text(self, value: object) -> str:
        """Clean a single text value.

        The cleaning steps are lowercase, remove HTML, remove URLs, remove punctuation,
        remove stop words, optional lemmatization, and whitespace normalization.
        """

        if pd.isna(value):
            return ""

        text = html.unescape(str(value))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = text.lower()
        text = text.translate(self._punctuation_translation)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()
        if self.config.remove_stop_words:
            tokens = [token for token in tokens if token not in self.config.stop_words]
        if self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(token) for token in tokens]

        return " ".join(tokens)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the dataset with cleaned text columns added."""

        text_columns = self.resolve_text_columns(dataframe.columns)
        if not text_columns:
            raise ValueError(
                "No text columns found for preprocessing. Provide text_columns in "
                "PreprocessingConfig or include one of: "
                f"{', '.join(self.config.text_column_candidates)}"
            )

        processed = dataframe.copy()
        for column in text_columns:
            output_column = f"{column}{self.config.cleaned_suffix}"
            LOGGER.info("Cleaning text column '%s' into '%s'", column, output_column)
            processed[output_column] = processed[column].map(self.clean_text)

        return processed

    def resolve_text_columns(self, columns: Iterable[str]) -> tuple[str, ...]:
        """Resolve which columns should be cleaned."""

        available_columns = tuple(str(column) for column in columns)
        normalized_lookup = {_normalize_column_name(column): column for column in available_columns}

        if self.config.text_columns is not None:
            missing = [
                column for column in self.config.text_columns if column not in available_columns
            ]
            if missing:
                raise ValueError(f"Configured text columns not found: {', '.join(missing)}")
            return self.config.text_columns

        resolved: list[str] = []
        for candidate in self.config.text_column_candidates:
            normalized_candidate = _normalize_column_name(candidate)
            if normalized_candidate in normalized_lookup:
                resolved.append(normalized_lookup[normalized_candidate])

        return tuple(dict.fromkeys(resolved))

    @staticmethod
    def _load_lemmatizer() -> object | None:
        try:
            from nltk.stem import WordNetLemmatizer

            return WordNetLemmatizer()
        except ImportError:
            LOGGER.warning("NLTK is not installed; lemmatization is disabled.")
            return None


def preprocess_dataset(
    input_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> Path:
    """Preprocess a CSV dataset and save the cleaned version to data/processed."""

    active_config = config or PreprocessingConfig()
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV input files are supported: {path}")

    LOGGER.info("Loading dataset for preprocessing: %s", path)
    dataframe = pd.read_csv(path)
    processed = TextPreprocessor(active_config).transform_dataframe(dataframe)

    active_config.output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = active_config.output_filename or f"{path.stem}_processed.csv"
    output_path = active_config.output_dir / output_filename
    processed.to_csv(output_path, index=False)
    LOGGER.info("Saved cleaned dataset to: %s", output_path)
    return output_path


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().casefold()).strip("_")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean product text and save a processed CSV.")
    parser.add_argument("input_path", help="Path to a raw or interim CSV dataset.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where the cleaned dataset should be written.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional output filename. Defaults to <input-name>_processed.csv.",
    )
    parser.add_argument(
        "--text-column",
        action="append",
        dest="text_columns",
        help="Text column to clean. Pass more than once to clean multiple columns.",
    )
    parser.add_argument(
        "--keep-stop-words",
        action="store_true",
        help="Keep stop words instead of removing them.",
    )
    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Enable optional NLTK WordNet lemmatization.",
    )
    return parser


def main() -> int:
    """Run preprocessing from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    config = PreprocessingConfig(
        text_columns=tuple(args.text_columns) if args.text_columns else None,
        output_dir=Path(args.output_dir),
        output_filename=args.output_filename,
        remove_stop_words=not args.keep_stop_words,
        lemmatize=args.lemmatize,
    )
    output_path = preprocess_dataset(args.input_path, config)
    LOGGER.info("Preprocessing complete: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

