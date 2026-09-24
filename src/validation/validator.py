"""Dataset validation for product category classification data."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

LOGGER = logging.getLogger(__name__)


CANONICAL_COLUMNS = ("product_name", "description", "brand", "price", "category")

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "product_name": ("product_name", "name", "title", "product_title"),
    "description": ("description", "product_description", "about_product", "name", "title"),
    "brand": ("brand", "manufacturer", "seller_name"),
    "price": ("price", "discount_price", "actual_price", "sale_price"),
    "category": ("category", "main_category", "sub_category", "label"),
}

DEFAULT_INVALID_CATEGORY_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "uncategorized",
    "not available",
    "not_applicable",
}


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for product dataset validation."""

    required_columns: tuple[str, ...] = CANONICAL_COLUMNS
    column_aliases: dict[str, tuple[str, ...]] = field(default_factory=lambda: DEFAULT_COLUMN_ALIASES)
    allowed_categories: tuple[str, ...] | None = None
    invalid_category_values: set[str] = field(
        default_factory=lambda: DEFAULT_INVALID_CATEGORY_VALUES.copy()
    )
    report_dir: Path = Path("artifacts/validation")
    max_sample_rows: int = 20


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result returned by the validator."""

    is_valid: bool
    dataset_path: str | None
    generated_at: str
    row_count: int
    column_count: int
    columns: list[str]
    column_mappings: dict[str, str | None]
    missing_required_columns: list[str]
    missing_values: dict[str, int]
    duplicate_row_count: int
    duplicate_row_indices: list[int]
    null_category_count: int
    null_category_indices: list[int]
    invalid_category_count: int
    invalid_category_indices: list[int]
    invalid_category_values: list[str]
    empty_description_count: int
    empty_description_indices: list[int]
    incorrect_data_types: dict[str, str]
    warnings: list[str]
    errors: list[str]
    report_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation result to JSON-serializable data."""

        return asdict(self)


def validate_dataset(
    dataset_path: str | Path,
    config: ValidationConfig | None = None,
    write_report: bool = True,
) -> ValidationResult:
    """Load and validate a CSV dataset.

    Args:
        dataset_path: Path to the CSV dataset.
        config: Optional validation configuration.
        write_report: Whether to write JSON and Markdown reports.

    Returns:
        A structured validation result.
    """

    active_config = config or ValidationConfig()
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV datasets are supported by this validator: {path}")

    LOGGER.info("Loading dataset for validation: %s", path)
    dataframe = pd.read_csv(path)
    return validate_dataframe(
        dataframe=dataframe,
        dataset_path=path,
        config=active_config,
        write_report=write_report,
    )


def validate_dataframe(
    dataframe: pd.DataFrame,
    dataset_path: str | Path | None = None,
    config: ValidationConfig | None = None,
    write_report: bool = True,
) -> ValidationResult:
    """Validate an in-memory product dataset."""

    active_config = config or ValidationConfig()
    normalized_columns = {_normalize_column_name(column): column for column in dataframe.columns}
    column_mappings = _resolve_column_mappings(active_config, normalized_columns)
    missing_required_columns = [
        column for column in active_config.required_columns if column_mappings.get(column) is None
    ]

    missing_values = _count_missing_values(dataframe, column_mappings)
    duplicate_indices = dataframe.index[dataframe.duplicated(keep=False)].astype(int).tolist()

    category_column = column_mappings.get("category")
    null_category_indices = _null_or_empty_indices(dataframe, category_column)
    invalid_category_indices, invalid_category_values = _invalid_category_details(
        dataframe=dataframe,
        category_column=category_column,
        config=active_config,
    )

    description_column = column_mappings.get("description")
    empty_description_indices = _null_or_empty_indices(dataframe, description_column)

    incorrect_data_types = _incorrect_data_types(dataframe, column_mappings)
    warnings = _build_warnings(column_mappings, active_config)

    errors: list[str] = []
    if missing_required_columns:
        errors.append(f"Missing required columns: {', '.join(missing_required_columns)}")
    if null_category_indices:
        errors.append("Dataset contains null or empty categories.")
    if invalid_category_indices:
        errors.append("Dataset contains invalid category values.")
    if empty_description_indices:
        errors.append("Dataset contains empty descriptions.")
    if incorrect_data_types:
        errors.append("Dataset contains incorrect data types.")

    generated_at = datetime.now(UTC).isoformat()
    result = ValidationResult(
        is_valid=not errors,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        generated_at=generated_at,
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        columns=[str(column) for column in dataframe.columns],
        column_mappings=column_mappings,
        missing_required_columns=missing_required_columns,
        missing_values=missing_values,
        duplicate_row_count=int(dataframe.duplicated().sum()),
        duplicate_row_indices=duplicate_indices[: active_config.max_sample_rows],
        null_category_count=len(null_category_indices),
        null_category_indices=null_category_indices[: active_config.max_sample_rows],
        invalid_category_count=len(invalid_category_indices),
        invalid_category_indices=invalid_category_indices[: active_config.max_sample_rows],
        invalid_category_values=invalid_category_values[: active_config.max_sample_rows],
        empty_description_count=len(empty_description_indices),
        empty_description_indices=empty_description_indices[: active_config.max_sample_rows],
        incorrect_data_types=incorrect_data_types,
        warnings=warnings,
        errors=errors,
    )

    if write_report:
        report_files = write_validation_report(result, active_config.report_dir)
        result = ValidationResult(**{**result.to_dict(), "report_files": report_files})

    return result


def write_validation_report(result: ValidationResult, report_dir: Path) -> dict[str, str]:
    """Write JSON and Markdown validation reports."""

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"validation_report_{timestamp}.json"
    markdown_path = report_dir / f"validation_report_{timestamp}.md"

    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(_to_markdown(result), encoding="utf-8")

    LOGGER.info("Validation reports written to %s", report_dir)
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _resolve_column_mappings(
    config: ValidationConfig,
    normalized_columns: dict[str, str],
) -> dict[str, str | None]:
    mappings: dict[str, str | None] = {}
    for canonical_column in config.required_columns:
        aliases = config.column_aliases.get(canonical_column, (canonical_column,))
        mappings[canonical_column] = next(
            (
                normalized_columns[_normalize_column_name(alias)]
                for alias in aliases
                if _normalize_column_name(alias) in normalized_columns
            ),
            None,
        )
    return mappings


def _count_missing_values(
    dataframe: pd.DataFrame,
    column_mappings: dict[str, str | None],
) -> dict[str, int]:
    missing_values: dict[str, int] = {}
    for canonical_column, actual_column in column_mappings.items():
        if actual_column is None:
            missing_values[canonical_column] = len(dataframe)
            continue
        missing_values[canonical_column] = int(dataframe[actual_column].isna().sum())
    return missing_values


def _null_or_empty_indices(dataframe: pd.DataFrame, column: str | None) -> list[int]:
    if column is None:
        return dataframe.index.astype(int).tolist()
    series = dataframe[column]
    mask = series.isna() | series.astype(str).str.strip().eq("")
    return dataframe.index[mask].astype(int).tolist()


def _invalid_category_details(
    dataframe: pd.DataFrame,
    category_column: str | None,
    config: ValidationConfig,
) -> tuple[list[int], list[str]]:
    if category_column is None:
        return dataframe.index.astype(int).tolist(), []

    categories = dataframe[category_column].astype("string").str.strip()
    invalid_mask = categories.isna() | categories.str.lower().isin(config.invalid_category_values)

    if config.allowed_categories is not None:
        allowed = {category.casefold().strip() for category in config.allowed_categories}
        invalid_mask = invalid_mask | ~categories.str.casefold().isin(allowed)
    else:
        invalid_mask = invalid_mask | ~categories.fillna("").map(_looks_like_category)

    invalid_indices = dataframe.index[invalid_mask.fillna(True)].astype(int).tolist()
    invalid_values = (
        categories[invalid_mask.fillna(True)]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return invalid_indices, invalid_values


def _incorrect_data_types(
    dataframe: pd.DataFrame,
    column_mappings: dict[str, str | None],
) -> dict[str, str]:
    issues: dict[str, str] = {}

    for canonical_column in ("product_name", "description", "brand", "category"):
        actual_column = column_mappings.get(canonical_column)
        if actual_column is None:
            continue
        if not (is_string_dtype(dataframe[actual_column]) or dataframe[actual_column].dtype == object):
            issues[canonical_column] = (
                f"Expected text-compatible column, got {dataframe[actual_column].dtype}."
            )

    price_column = column_mappings.get("price")
    if price_column is not None and not is_numeric_dtype(dataframe[price_column]):
        parsed_prices = dataframe[price_column].map(_parse_price)
        invalid_prices = parsed_prices.isna() & dataframe[price_column].notna()
        if invalid_prices.any():
            issues["price"] = (
                "Expected numeric or currency-formatted values; "
                f"{int(invalid_prices.sum())} non-null rows are not parseable."
            )

    return issues


def _build_warnings(
    column_mappings: dict[str, str | None],
    config: ValidationConfig,
) -> list[str]:
    warnings: list[str] = []
    for canonical_column, actual_column in column_mappings.items():
        if actual_column is None:
            continue
        if _normalize_column_name(actual_column) != _normalize_column_name(canonical_column):
            warnings.append(
                f"Using '{actual_column}' as '{canonical_column}' based on configured aliases."
            )
    if config.allowed_categories is None:
        warnings.append(
            "No allowed category list was provided; invalid category checks use format and "
            "placeholder-value rules."
        )
    return warnings


def _parse_price(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, int | float):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_category(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    if len(text) > 120:
        return False
    if text.isnumeric():
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().casefold()).strip("_")


def _to_markdown(result: ValidationResult) -> str:
    status = "PASS" if result.is_valid else "FAIL"
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Status: {status}",
        f"- Generated at: {result.generated_at}",
        f"- Dataset: {result.dataset_path or 'in-memory dataframe'}",
        f"- Rows: {result.row_count}",
        f"- Columns: {result.column_count}",
        "",
        "## Column Mappings",
        "",
    ]
    lines.extend(
        f"- {canonical}: {actual or 'MISSING'}"
        for canonical, actual in result.column_mappings.items()
    )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Missing required columns: {len(result.missing_required_columns)}",
            f"- Duplicate rows: {result.duplicate_row_count}",
            f"- Null categories: {result.null_category_count}",
            f"- Invalid categories: {result.invalid_category_count}",
            f"- Empty descriptions: {result.empty_description_count}",
            f"- Incorrect data types: {len(result.incorrect_data_types)}",
            "",
            "## Missing Values",
            "",
        ]
    )
    lines.extend(f"- {column}: {count}" for column, count in result.missing_values.items())
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {error}" for error in result.errors] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- None"])
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a product classification CSV dataset.")
    parser.add_argument("dataset_path", help="Path to the CSV dataset to validate.")
    parser.add_argument(
        "--report-dir",
        default="artifacts/validation",
        help="Directory where JSON and Markdown reports should be written.",
    )
    parser.add_argument(
        "--allowed-category",
        action="append",
        dest="allowed_categories",
        help="Allowed category value. Pass more than once to define a category whitelist.",
    )
    return parser


def main() -> int:
    """Run dataset validation from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()
    config = ValidationConfig(
        allowed_categories=tuple(args.allowed_categories) if args.allowed_categories else None,
        report_dir=Path(args.report_dir),
    )
    result = validate_dataset(args.dataset_path, config=config, write_report=True)
    LOGGER.info("Validation status: %s", "PASS" if result.is_valid else "FAIL")
    LOGGER.info("Report files: %s", result.report_files)
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

