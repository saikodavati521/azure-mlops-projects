from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from model_training.train_code.data_validation import (
    DataValidationError,
    parse_steel_datetime,
    validate_steel_energy_data,
)


def valid_frame(rows: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=rows, freq="15min")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "Usage_kWh": [10.0 + (idx % 24) for idx in range(rows)],
            "Lagging_Current_Reactive.Power_kVarh": 2.0,
            "Leading_Current_Reactive_Power_kVarh": 0.1,
            "CO2(tCO2)": 0.0,
            "Lagging_Current_Power_Factor": 90.0,
            "Leading_Current_Power_Factor": 100.0,
            "NSM": [(idx % 96) * 900 for idx in range(rows)],
            "WeekStatus": ["Weekend" if d.dayofweek >= 5 else "Weekday" for d in dates],
            "Day_of_week": [d.day_name() for d in dates],
            "Load_Type": "Light_Load",
        }
    )


def test_valid_data_passes_validation():
    assert validate_steel_energy_data(valid_frame()) == []


def test_negative_usage_fails_validation():
    frame = valid_frame()
    frame.loc[0, "Usage_kWh"] = -1
    with pytest.raises(DataValidationError, match="Usage_kWh cannot be negative"):
        validate_steel_energy_data(frame)


def test_dataset_date_parser_prefers_day_first_format():
    parsed = parse_steel_datetime(pd.Series(["01/02/2018 00:15", "13/02/2018 00:15"]))
    assert parsed.iloc[0] == pd.Timestamp("2018-02-01 00:15")
    assert parsed.iloc[1] == pd.Timestamp("2018-02-13 00:15")


def test_uploaded_dataset_matches_project_schema_when_present():
    dataset_path = Path("data/Steel_industry_data.csv")
    if not dataset_path.exists():
        pytest.skip("Local raw dataset is not present")

    frame = pd.read_csv(dataset_path)
    assert len(frame) == 35040
    assert validate_steel_energy_data(frame) == []
