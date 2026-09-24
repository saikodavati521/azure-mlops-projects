from pathlib import Path

import pandas as pd

from src.validation import ValidationConfig, validate_dataframe, validate_dataset


def test_validate_dataframe_detects_required_data_quality_issues(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "product_name": "Phone",
                "description": "",
                "brand": "Acme",
                "price": "100",
                "category": "Electronics",
            },
            {
                "product_name": "Phone",
                "description": "",
                "brand": "Acme",
                "price": "100",
                "category": "Electronics",
            },
            {
                "product_name": "Mystery",
                "description": "Useful item",
                "brand": None,
                "price": "not-a-price",
                "category": "unknown",
            },
            {
                "product_name": "Blank category",
                "description": "Missing category",
                "brand": "Acme",
                "price": "10",
                "category": None,
            },
        ]
    )

    result = validate_dataframe(
        dataframe,
        config=ValidationConfig(report_dir=tmp_path),
        write_report=True,
    )

    assert result.is_valid is False
    assert result.duplicate_row_count == 1
    assert result.null_category_count == 1
    assert result.invalid_category_count == 2
    assert result.empty_description_count == 2
    assert result.missing_values["brand"] == 1
    assert "price" in result.incorrect_data_types
    assert Path(result.report_files["json"]).exists()
    assert Path(result.report_files["markdown"]).exists()


def test_validate_dataframe_uses_common_column_aliases(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "name": "Running Shoes",
                "manufacturer": "Contoso",
                "discount_price": "₹1,999",
                "main_category": "Shoes",
            }
        ]
    )

    result = validate_dataframe(
        dataframe,
        config=ValidationConfig(report_dir=tmp_path),
        write_report=False,
    )

    assert result.column_mappings["product_name"] == "name"
    assert result.column_mappings["description"] == "name"
    assert result.column_mappings["brand"] == "manufacturer"
    assert result.column_mappings["price"] == "discount_price"
    assert result.column_mappings["category"] == "main_category"
    assert result.incorrect_data_types == {}


def test_validate_dataset_loads_csv_and_writes_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "products.csv"
    pd.DataFrame(
        [
            {
                "product_name": "Laptop",
                "description": "Business laptop",
                "brand": "Acme",
                "price": 1200.0,
                "category": "Electronics",
            }
        ]
    ).to_csv(dataset_path, index=False)

    result = validate_dataset(
        dataset_path,
        config=ValidationConfig(
            allowed_categories=("Electronics",),
            report_dir=tmp_path / "reports",
        ),
        write_report=True,
    )

    assert result.is_valid is True
    assert result.row_count == 1
    assert Path(result.report_files["json"]).exists()

