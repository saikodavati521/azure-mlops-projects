"""Prediction request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw product text to classify.")
    product_name: str | None = Field(default=None, description="Optional product name.")
    description: str | None = Field(default=None, description="Optional product description.")
    brand: str | None = Field(default=None, description="Optional product brand.")
    price: float | None = Field(default=None, ge=0, description="Optional product price.")


class PredictResponse(BaseModel):
    prediction: str
    category: str
    sub_category: str | None = None
    confidence: float | None = None
    cleaned_text: str
    model_path: str
    vectorizer_path: str
    product_id: int | None = None
    prediction_id: int | None = None
