# CategoryIQ: AI-Powered Product Classification Platform with Azure MLOps

CategoryIQ is an end-to-end, Python-based machine-learning platform for automatically classifying e-commerce products into relevant categories and subcategories. It validates product datasets, cleans and preprocesses product text, generates TF-IDF features, trains and compares classification models, and saves the best-performing model for reuse. Users can submit product descriptions through the FastAPI backend or use the Streamlit dashboard to make predictions and view metrics. PostgreSQL stores product, prediction, and model information, while MLflow tracks experiments and model artifacts. The complete application is packaged as a Docker image and deployed on Microsoft Azure, with Azure ML configuration for cloud-based training and managed model-serving workflows.


ARCHITECTURE : 

https://github.com/saikodavati521/azure-mlops-projects/issues/1#issue-5574472107


The project works through the following steps:

1. Load product data: Product CSV files are loaded from the raw data directory for classification.
2. Validate the dataset: The validation pipeline checks required columns, missing values, duplicate records, empty descriptions, invalid categories, and incorrect price values.
3. Clean product text: Product names and descriptions are cleaned by removing HTML, URLs, punctuation, stop words, and unnecessary whitespace.
4. Generate features:Cleaned text is converted into numerical TF-IDF features so machine-learning models can process it.
5. Train and compare models: Logistic Regression, Naive Bayes, Random Forest, and XGBoost models are trained and evaluated using accuracy, precision, recall, and F1-score.
6. Save the best model: The best-performing classifier and TF-IDF vectorizer are saved as reusable model artifacts.
7. Serve predictions:The FastAPI backend accepts product text and returns the predicted category through API endpoints.
8. Display results: The Streamlit dashboard provides an interface for product prediction, model metrics, analytics, and prediction history.
9. Store application data: PostgreSQL stores product details, prediction results, and model version information.
10. Track experiments: MLflow records training parameters, evaluation metrics, datasets, and model artifacts.
11. Deploy to Azure:The complete application is packaged into a Docker image and deployed on Microsoft Azure, while the included Azure ML configuration supports cloud training and managed model-serving workflows.
