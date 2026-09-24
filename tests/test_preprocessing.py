from pathlib import Path

import pandas as pd

from src.preprocessing import PreprocessingConfig, TextPreprocessor, preprocess_dataset


def test_clean_text_applies_required_cleaning_steps() -> None:
    preprocessor = TextPreprocessor()

    cleaned = preprocessor.clean_text(
        "<b>The Apple iPhone!</b> Visit https://example.com with A18 chip."
    )

    assert cleaned == "apple iphone visit a18 chip"


def test_transform_dataframe_adds_cleaned_columns_for_canonical_fields() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "product_name": "THE Phone!!!",
                "description": "<p>Latest phone with charger.</p>",
                "brand": "Acme",
                "price": 100,
                "category": "Electronics",
            }
        ]
    )

    processed = TextPreprocessor().transform_dataframe(dataframe)

    assert processed.loc[0, "product_name_cleaned"] == "phone"
    assert processed.loc[0, "description_cleaned"] == "latest phone charger"
    assert processed.loc[0, "brand"] == "Acme"


def test_preprocess_dataset_saves_cleaned_csv_to_processed_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "raw_products.csv"
    output_dir = tmp_path / "processed"
    pd.DataFrame(
        [
            {
                "name": "Running Shoes, for Men",
                "main_category": "Shoes",
                "discount_price": "₹1,999",
            }
        ]
    ).to_csv(input_path, index=False)

    output_path = preprocess_dataset(
        input_path,
        PreprocessingConfig(output_dir=output_dir),
    )

    processed = pd.read_csv(output_path)
    assert output_path == output_dir / "raw_products_processed.csv"
    assert processed.loc[0, "name_cleaned"] == "running shoes men"

