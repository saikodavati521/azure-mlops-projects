# CategoryIQ: AI-Powered Product Classification Platform with Azure MLOps

CategoryIQ is an end-to-end, Python-based machine-learning platform for automatically classifying e-commerce products into relevant categories and subcategories. It validates product datasets, cleans and preprocesses product text, generates TF-IDF features, trains and compares classification models, and saves the best-performing model for reuse. Users can submit product descriptions through the FastAPI backend or use the Streamlit dashboard to make predictions and view metrics. PostgreSQL stores product, prediction, and model information, while MLflow tracks experiments and model artifacts. The complete application is packaged as a Docker image and deployed on Microsoft Azure, with Azure ML configuration for cloud-based training and managed model-serving workflows.


ARCHITECTURE : 

<img width="1312" height="1199" alt="Image" src="https://github.com/user-attachments/assets/be791fc2-2bac-49f2-b6d9-cc8d29cd0a6a" />


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










[Steel industry Energy Consumption with azure MLOps] (https://github.com/saikodavati521/azure-mlops-projects/tree/main/industry-energy-consumption)

Dataset:

This project uses the Steel Industry Energy Consumption dataset, which contains chronological electricity-consumption records collected at 15-minute intervals from a steel manufacturing process.

The target variable is Usage_kWh, representing electricity consumption. The dataset also contains features such as timestamp, reactive power, power factor, CO2 emissions, NSM, weekday information, and load type. These features are used to develop regression models for predicting energy consumption.

Azure MLOps Training and Prediction:


<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/28dee2c9-c5b2-45a1-9330-660661a04ed2" />

The project implements an end-to-end Azure MLOps pipeline for training, evaluating, registering, deploying, and monitoring machine learning models.

GitHub stores the application source code, ML code, configuration files, and pipeline definitions.

Azure DevOps automates the CI/CD workflow when changes are pushed to the repository.

The dataset is uploaded to Azure Machine Learning as a versioned data asset, providing reproducibility and traceability.

The Azure DevOps pipeline runs automated tests, installs dependencies, prepares the Azure ML environment, and submits the training job.

The Azure Machine Learning training job validates the input data and performs feature engineering, including time-based and historical/lag features.

Multiple regression algorithms are trained and evaluated using chronological train, validation, and test datasets to avoid inappropriate random splitting of time-series data.

MLflow tracks experiments, training parameters, evaluation metrics, and model artifacts.

A model quality gate evaluates the trained models against predefined performance criteria before deployment.

The selected model is registered and versioned in the Azure Machine Learning model registry.

The approved model is deployed to an Azure Machine Learning managed online endpoint for real-time inference.

New feature data can be sent to the endpoint, which executes the scoring logic and returns the predicted Usage_kWh.

Azure Machine Learning, Azure Monitor, and Application Insights provide infrastructure and endpoint monitoring.

Production data and model behavior can be monitored for changes and potential data/model drift.

When performance degradation or significant drift is detected, the MLOps pipeline can be triggered to retrain, evaluate, register, and redeploy an updated model.

