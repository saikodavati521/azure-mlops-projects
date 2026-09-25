"""Feature engineering for 15-minute steel energy forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.variables import DATE_COLUMN, TARGET_COLUMN
from model_training.train_code.data_validation import parse_steel_datetime

LAG_FEATURES = {
    "usage_lag_1": 1,
    "usage_lag_4": 4,
    "usage_lag_96": 96,
    "usage_lag_672": 672,
}

ROLLING_FEATURES = {
    "usage_rolling_mean_4": ("mean", 4),
    "usage_rolling_mean_96": ("mean", 96),
    "usage_rolling_std_96": ("std", 96),
}


def build_features(data: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    """Create time, cyclical, lag, and leakage-safe rolling features."""

    featured = data.copy()
    featured[DATE_COLUMN] = parse_steel_datetime(featured[DATE_COLUMN])
    if featured[DATE_COLUMN].isna().any():
        raise ValueError(f"Column {DATE_COLUMN} contains invalid dates")
    featured = featured.sort_values(DATE_COLUMN).reset_index(drop=True)

    featured["year"] = featured[DATE_COLUMN].dt.year
    featured["month"] = featured[DATE_COLUMN].dt.month
    featured["day"] = featured[DATE_COLUMN].dt.day
    featured["day_of_year"] = featured[DATE_COLUMN].dt.dayofyear
    featured["hour"] = featured[DATE_COLUMN].dt.hour
    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)

    shifted_target = featured[TARGET_COLUMN].shift(1)
    for name, lag in LAG_FEATURES.items():
        featured[name] = featured[TARGET_COLUMN].shift(lag)

    for name, (aggregation, window) in ROLLING_FEATURES.items():
        rolling = shifted_target.rolling(window=window, min_periods=window)
        featured[name] = getattr(rolling, aggregation)()

    if dropna:
        featured = featured.dropna().reset_index(drop=True)
    return featured
