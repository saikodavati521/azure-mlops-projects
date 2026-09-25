"""Production monitoring metric helpers."""

from __future__ import annotations

import pandas as pd

from model_training.train_code.evaluate import regression_metrics


def production_performance(actuals, predictions) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 once delayed production labels arrive."""

    return regression_metrics(actuals, predictions)


def prediction_summary(predictions) -> dict[str, float]:
    series = pd.Series(predictions, dtype="float64")
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }

