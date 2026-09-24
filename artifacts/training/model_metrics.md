# Model Training Report

- Best model: Logistic Regression
- Rows: 10000
- Train rows: 8000
- Test rows: 2000
- Features: 14199
- Text columns: name
- Category column: main_category
- Sub-category column: sub_category
- Dataset version: bb770a5ce131bd466985ea262b53a0dec3a81f454e3e2870ca6c3139639648cf
- Model version: 20260904173330
- Saved model: models\model.pkl
- Saved vectorizer: models\vectorizer.pkl
- MLflow experiment: disabled
- MLflow best run ID: not logged
- Registered model: not registered
- Registered model version: not registered

## Metrics

| Model | Status | Accuracy | Precision | Recall | F1-score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | trained | 0.6165 | 0.6178 | 0.6165 | 0.5929 |  |
| Naive Bayes | trained | 0.5835 | 0.5133 | 0.5835 | 0.5137 |  |
| Random Forest | trained | 0.5915 | 0.6125 | 0.5915 | 0.5846 |  |
| XGBoost | trained | 0.6030 | 0.5912 | 0.6030 | 0.5893 |  |
