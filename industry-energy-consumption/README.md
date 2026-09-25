Steel Energy Consumption MLOps
Dataset

This project uses the Steel Industry Energy Consumption dataset, which contains chronological electricity-consumption records collected at 15-minute intervals from a steel manufacturing process.

The target variable is Usage_kWh, representing electricity consumption. The dataset also contains features such as timestamp, reactive power, power factor, CO2 emissions, NSM, weekday information, and load type. These features are used to develop regression models for predicting energy consumption.

Azure MLOps Training and Prediction

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
