"""Prediction service for saved CategoryIQ model artifacts."""

from __future__ import annotations

import joblib
import numpy as np
import psycopg2

from backend.schemas.predict import PredictRequest, PredictResponse
from backend.services.postgres_service import CategoryIQPostgresService, ProductRecord
from backend.utils.paths import MODEL_PATH, VECTORIZER_PATH
from src.training.train_models import LABEL_SEPARATOR
from src.preprocessing import TextPreprocessor


class PredictionService:
    """Load model artifacts and generate predictions."""

    def __init__(self, postgres_service: CategoryIQPostgresService | None = None) -> None:
        self.postgres_service = postgres_service or CategoryIQPostgresService()

    def predict(self, request: PredictRequest) -> PredictResponse:
        raw_text = self._resolve_text(request)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found: {VECTORIZER_PATH}")

        cleaned_text = TextPreprocessor().clean_text(raw_text)
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        features = vectorizer.transform([cleaned_text])
        prediction = str(model.predict(features)[0])
        category, sub_category = self._split_prediction_label(prediction)
        confidence = self._confidence(model, features)
        product_id, prediction_id = self._persist_prediction(
            request,
            raw_text,
            category,
            sub_category,
            confidence,
        )

        return PredictResponse(
            prediction=category,
            category=category,
            sub_category=sub_category,
            confidence=confidence,
            cleaned_text=cleaned_text,
            model_path=str(MODEL_PATH),
            vectorizer_path=str(VECTORIZER_PATH),
            product_id=product_id,
            prediction_id=prediction_id,
        )

    @staticmethod
    def _resolve_text(request: PredictRequest) -> str:
        parts = [
            request.text,
            request.product_name,
            request.description,
        ]
        text = " ".join(part.strip() for part in parts if part and part.strip()).strip()
        if not text:
            raise ValueError("Provide text, product_name, or description for prediction.")
        return text

    @staticmethod
    def _confidence(model: object, features: object) -> float | None:
        if not hasattr(model, "predict_proba"):
            return None
        probabilities = model.predict_proba(features)
        return float(np.max(probabilities))

    def _persist_prediction(
        self,
        request: PredictRequest,
        raw_text: str,
        category: str,
        sub_category: str | None,
        confidence: float | None,
    ) -> tuple[int | None, int | None]:
        product_name = request.product_name or raw_text
        try:
            self.postgres_service.initialize()
            product, prediction_row = self.postgres_service.create_product_with_prediction(
                ProductRecord(
                    product_name=product_name,
                    description=request.description,
                    brand=request.brand,
                    price=request.price,
                ),
                predicted_category=category,
                predicted_sub_category=sub_category,
                confidence=confidence,
            )
        except (OSError, psycopg2.Error) as exc:
            raise PredictionPersistenceError(
                "Prediction was generated, but it could not be saved to PostgreSQL. "
                "Check DATABASE_URL and make sure the database server is running."
            ) from exc
        return int(product["id"]), int(prediction_row["prediction_id"])

    @staticmethod
    def _split_prediction_label(label: str) -> tuple[str, str | None]:
        if LABEL_SEPARATOR not in label:
            return label, None
        category, sub_category = label.split(LABEL_SEPARATOR, 1)
        return category.strip(), sub_category.strip() or None


class PredictionPersistenceError(RuntimeError):
    """Raised when a prediction cannot be saved to PostgreSQL."""
