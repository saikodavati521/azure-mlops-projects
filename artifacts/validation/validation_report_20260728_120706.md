# Dataset Validation Report

- Status: FAIL
- Generated at: 2026-07-28T12:07:04.995382+00:00
- Dataset: data\raw\Amazon-Products.csv
- Rows: 551585
- Columns: 10

## Column Mappings

- product_name: name
- description: name
- brand: MISSING
- price: discount_price
- category: main_category

## Checks

- Missing required columns: 1
- Duplicate rows: 0
- Null categories: 0
- Invalid categories: 0
- Empty descriptions: 0
- Incorrect data types: 0

## Missing Values

- product_name: 0
- description: 0
- brand: 551585
- price: 61163
- category: 0

## Errors

- Missing required columns: brand

## Warnings

- Using 'name' as 'product_name' based on configured aliases.
- Using 'name' as 'description' based on configured aliases.
- Using 'discount_price' as 'price' based on configured aliases.
- Using 'main_category' as 'category' based on configured aliases.
- No allowed category list was provided; invalid category checks use format and placeholder-value rules.
