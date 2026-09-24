import json
from pathlib import Path

import joblib
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.api.main import create_app
from backend.services import metrics_service, prediction_service


def test_health_endpoint_returns_status() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}
    assert "model_ready" in response.json()
    assert "vectorizer_ready" in response.json()


def test_predict_endpoint_uses_saved_model_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(["phone charger", "running shoes"])
    model = DummyClassifier(strategy="constant", constant="electronics")
    model.fit(features, ["electronics", "fashion"])

    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    monkeypatch.setattr(prediction_service, "MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_service, "VECTORIZER_PATH", vectorizer_path)
    monkeypatch.setattr(
        prediction_service.PredictionService,
        "_persist_prediction",
        lambda self, request, raw_text, category, sub_category, confidence: (1, 1),
    )

    client = TestClient(create_app())
    response = client.post("/predict", json={"text": "New phone with charger"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "electronics"
    assert payload["cleaned_text"] == "new phone charger"
    assert payload["product_id"] == 1
    assert payload["prediction_id"] == 1


def test_metrics_endpoint_returns_latest_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metrics_path = tmp_path / "model_metrics.json"
    report_path = tmp_path / "model_metrics.md"
    metrics_path.write_text(json.dumps({"best_model_name": "Logistic Regression"}))
    report_path.write_text("# Report")
    monkeypatch.setattr(metrics_service, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(metrics_service, "REPORT_PATH", report_path)

    client = TestClient(create_app())
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_model_name"] == "Logistic Regression"
    assert payload["payload"]["best_model_name"] == "Logistic Regression"


def test_models_endpoint_lists_artifacts() -> None:
    client = TestClient(create_app())

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_models"] == [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
    assert "local_artifacts" in payload
    assert "registered_models" in payload


def test_train_endpoint_rejects_missing_dataset() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/train",
        json={
            "dataset_path": "missing.csv",
            "text_columns": ["name"],
            "category_column": "main_category",
            "enable_mlflow": False,
        },
    )

    assert response.status_code == 404
