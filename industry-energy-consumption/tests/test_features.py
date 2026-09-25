from __future__ import annotations

import numpy as np

from model_training.train_code.feature_engineering import build_features
from tests.test_data import valid_frame


def test_feature_engineering_creates_lag_and_rolling_features_without_leakage():
    frame = valid_frame(700)
    featured = build_features(frame)
    assert "usage_lag_672" in featured.columns
    first = featured.iloc[0]
    assert first["usage_lag_1"] == frame.loc[671, "Usage_kWh"]
    expected_mean = frame.loc[668:671, "Usage_kWh"].mean()
    assert first["usage_rolling_mean_4"] == expected_mean
    assert np.isclose(first["hour_sin"], np.sin(2 * np.pi * first["hour"] / 24))

