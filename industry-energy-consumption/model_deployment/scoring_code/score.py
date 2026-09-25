"""Azure ML online endpoint scoring script.

Inference expects production features that already include historical lag and
rolling values. The serving layer does not invent usage_lag_672 from a single
timestamp; upstream feature services or batch preparation must provide it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

MODEL_NUMERIC_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "year",
    "month",
    "day",
    "day_of_year",
    "hour",
    "hour_sin",
    "hour_cos",
    "usage_lag_1",
    "usage_lag_4",
    "usage_lag_96",
    "usage_lag_672",
    "usage_rolling_mean_4",
    "usage_rolling_mean_96",
    "usage_rolling_std_96",
]

MODEL_CATEGORICAL_FEATURES = ["WeekStatus", "Day_of_week", "Load_Type"]

model = None


def init():
    global model
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "."))
    candidates = [model_dir / "model", model_dir]
    for candidate in candidates:
        if (candidate / "MLmodel").exists():
            model = mlflow.sklearn.load_model(str(candidate))
            return
    raise FileNotFoundError(f"Could not find MLflow model under {model_dir}")


def _records_from_payload(payload) -> list[dict]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Request body must be an object, a list of objects, or {'data': [...]}")
    return payload


def run(raw_data):
    if model is None:
        init()
    records = _records_from_payload(raw_data)
    frame = pd.DataFrame(records)
    required = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {missing}")
    predictions = model.predict(frame[required])
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return {"predictions": [float(value) for value in predictions]}
