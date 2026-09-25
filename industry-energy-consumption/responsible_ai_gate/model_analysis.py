"""Model quality checks for responsible industrial forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_predictions(predictions, *, max_reasonable_usage_kwh: float | None = None) -> dict[str, object]:
    values = pd.Series(predictions, dtype="float64")
    checks = {
        "finite_predictions": bool(np.isfinite(values).all()),
        "non_negative_predictions": bool((values >= 0).all()),
        "prediction_count": int(values.count()),
        "prediction_min": float(values.min()),
        "prediction_max": float(values.max()),
    }
    if max_reasonable_usage_kwh is not None:
        checks["within_configured_usage_range"] = bool((values <= max_reasonable_usage_kwh).all())
    return checks


def feature_distribution_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric = frame.select_dtypes(include=["number"])
    return {
        column: {
            "missing_rate": float(frame[column].isna().mean()),
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std(ddof=0)),
        }
        for column in numeric.columns
    }

