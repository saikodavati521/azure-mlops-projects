"""Validation routines for the UCI steel industry energy dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.variables import (
    ALLOWED_CATEGORIES,
    DATE_COLUMN,
    DATE_FORMAT,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when input data fails validation."""


def _append(condition: bool, errors: list[str], message: str) -> None:
    if condition:
        errors.append(message)


def parse_steel_datetime(values: pd.Series) -> pd.Series:
    """Parse the dataset timestamp column, preferring its day-first CSV format."""

    parsed = pd.to_datetime(values, format=DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(values, format="mixed", errors="coerce", dayfirst=True)
        parsed = parsed.fillna(fallback)
    return parsed


def validate_steel_energy_data(
    data: pd.DataFrame,
    *,
    require_15_min_frequency: bool = True,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate steel energy consumption data and return validation errors."""

    errors: list[str] = []
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    _append(bool(missing_columns), errors, f"Missing required columns: {missing_columns}")

    if missing_columns:
        if raise_on_error:
            raise DataValidationError("; ".join(errors))
        return errors

    checked = data[REQUIRED_COLUMNS].copy()
    _append(checked.isna().any().any(), errors, "Dataset contains missing values")
    _append(checked.duplicated().any(), errors, "Dataset contains duplicate rows")

    parsed_dates = parse_steel_datetime(checked[DATE_COLUMN])
    _append(parsed_dates.isna().any(), errors, "Dataset contains invalid dates")
    _append(parsed_dates.duplicated().any(), errors, "Dataset contains duplicate timestamps")

    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(checked[column], errors="coerce")
        _append(numeric.isna().any(), errors, f"Column {column} must be numeric")
        _append(np.isinf(numeric.to_numpy(dtype=float)).any(), errors, f"Column {column} contains infinite values")

    usage = pd.to_numeric(checked[TARGET_COLUMN], errors="coerce")
    _append((usage < 0).any(), errors, "Usage_kWh cannot be negative")

    for column, allowed_values in ALLOWED_CATEGORIES.items():
        observed = set(checked[column].dropna().astype(str).unique())
        invalid = sorted(observed - allowed_values)
        _append(bool(invalid), errors, f"Column {column} has invalid values: {invalid}")

    if parsed_dates.notna().all() and not parsed_dates.empty:
        ordered_dates = parsed_dates.sort_values()
        _append(ordered_dates.iloc[-1] <= ordered_dates.iloc[0], errors, "Date range must span more than one timestamp")
        if require_15_min_frequency and len(ordered_dates) > 1:
            deltas = ordered_dates.diff().dropna()
            expected_delta = pd.Timedelta(minutes=15)
            bad_frequency = not (deltas == expected_delta).all()
            _append(bad_frequency, errors, "Timestamps must follow an uninterrupted 15-minute frequency")

    if errors and raise_on_error:
        raise DataValidationError("; ".join(errors))
    return errors
