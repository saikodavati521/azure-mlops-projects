"""Streamlit frontend for the CategoryIQ product classification platform."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.schemas.predict import PredictRequest  # noqa: E402
from backend.services.postgres_service import CategoryIQPostgresService  # noqa: E402
from backend.services.prediction_service import PredictionService  # noqa: E402
from backend.utils.paths import METRICS_PATH, MODEL_PATH, REPORT_PATH, VECTORIZER_PATH  # noqa: E402
PAGES = (
    "Predict Product",
    "Dashboard",
    "Prediction History",
    "Model Metrics",
    "Analytics",
    "Settings",
)

MODEL_COMPARISON_ORDER = (
    "Logistic Regression",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
)


def main() -> None:
    st.set_page_config(
        page_title="CategoryIQ",
        page_icon="C",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _initialize_state()

    with st.sidebar:
        st.title("CategoryIQ")
        st.markdown("")
        selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("")

    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Predict Product":
        render_predict_product()
    elif selected_page == "Prediction History":
        render_prediction_history()
    elif selected_page == "Model Metrics":
        render_model_metrics()
    elif selected_page == "Analytics":
        render_analytics()
    elif selected_page == "Settings":
        render_settings()


def render_dashboard() -> None:
    st.title("Dashboard")
    st.markdown("View model readiness, recent predictions, and session activity in one place.")
    history = _history_dataframe()
    metrics_payload = _load_metrics_payload()
    best_accuracy = _best_accuracy(metrics_payload)

    categories = history["predicted_category"].dropna().unique().tolist() if not history.empty else []
    sub_categories = history["predicted_sub_category"].dropna().unique().tolist() if not history.empty else []
    most_predicted = _most_common(history["predicted_category"].tolist()) if not history.empty else "No predictions"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", len(history))
    col2.metric("Accuracy", _format_percent(best_accuracy))
    col3.metric("Number of Categories", len(categories))
    col4.metric("Subcategories", len(sub_categories))
    st.caption(f"Most predicted category: {most_predicted}")

    st.subheader("Recent Activity")
    if history.empty:
        st.info("No predictions yet. Open Predict Product and classify your first item.")
    else:
        st.dataframe(
            history.sort_values("timestamp", ascending=False).head(10),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Model Status")
    st.caption("A prediction needs both the saved model and vectorizer artifacts.")
    status_columns = st.columns(3)
    status_columns[0].metric("Model Artifact", "Ready" if MODEL_PATH.exists() else "Missing")
    status_columns[1].metric("Vectorizer", "Ready" if VECTORIZER_PATH.exists() else "Missing")
    status_columns[2].metric("Metrics Report", "Ready" if METRICS_PATH.exists() else "Missing")


def render_predict_product() -> None:
    st.title("Predict Product")
    st.markdown("Enter a product name and description to predict the best matching category.")

    with st.form("predict-product-form", clear_on_submit=False):
        product_name = st.text_input("Product Name", placeholder="Example: Wireless Bluetooth Headphones")
        description = st.text_area(
            "Description",
            height=120,
            placeholder="Example: Noise cancelling over-ear headphones with long battery life",
        )
        brand = st.text_input("Brand", placeholder="Optional brand name")
        price = st.number_input(
            "Price",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            help="Optional product price.",
        )
        submitted = st.form_submit_button("Predict Category", type="primary")

    if not submitted:
        return

    if not product_name.strip() and not description.strip():
        st.error("Enter a product name or description before predicting.")
        return

    try:
        prediction = _predict(product_name, description, brand, float(price) if price else None)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "product_name": product_name.strip(),
        "description": description.strip(),
        "brand": brand.strip() or None,
        "price": float(price) if price else None,
        "predicted_category": prediction["category"],
        "predicted_sub_category": prediction.get("sub_category"),
        "confidence": prediction.get("confidence"),
        "source": prediction.get("source", "local"),
        "product_id": prediction.get("product_id"),
        "prediction_id": prediction.get("prediction_id"),
    }
    st.session_state.prediction_history.append(record)

    left, middle, right = st.columns([2, 2, 1])
    left.success(f"Predicted Category: {record['predicted_category']}")
    middle.success(f"Predicted Subcategory: {record['predicted_sub_category'] or 'N/A'}")
    right.metric("Confidence", _format_percent(record["confidence"]))
    st.caption(f"Saved to database as prediction #{record['prediction_id']}")
    st.caption(f"Cleaned text: {prediction.get('cleaned_text', '')}")


def render_prediction_history() -> None:
    st.title("Prediction History")
    st.markdown("Review predictions made during this Streamlit session.")
    history = _history_dataframe()

    if history.empty:
        st.info("Prediction history is empty for this Streamlit session.")
        return

    categories = ["All"] + sorted(history["predicted_category"].dropna().unique().tolist())
    selected_category = st.selectbox("Category", categories)
    filtered = history
    if selected_category != "All":
        filtered = filtered[filtered["predicted_category"] == selected_category]

    st.dataframe(
        filtered.sort_values("timestamp", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download History CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="categoryiq_prediction_history.csv",
        mime="text/csv",
    )

    if st.button("Clear Session History"):
        st.session_state.prediction_history = []
        st.rerun()


def render_model_metrics() -> None:
    st.title("Model Metrics")
    st.markdown("Compare trained models and inspect the latest saved training report.")
    payload = _load_metrics_payload()

    if not payload:
        st.warning(f"No metrics file found at {METRICS_PATH}. Train a model to generate metrics.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", payload.get("best_model_name", "Unknown"))
    col2.metric("Dataset Rows", payload.get("row_count", "Unknown"))
    col3.metric("Features", payload.get("feature_count", "Unknown"))
    col4.metric("Model Version", payload.get("model_version", "Unknown"))

    metrics = _model_comparison_dataframe(payload)
    if not metrics.empty:
        st.subheader("Model Comparison")
        st.dataframe(metrics, width="stretch", hide_index=True)

        chart_columns = [column for column in ("accuracy", "precision", "recall", "f1_score") if column in metrics]
        if chart_columns:
            st.bar_chart(metrics.set_index("model_name")[chart_columns])

    if REPORT_PATH.exists():
        st.subheader("Training Report")
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))


def render_analytics() -> None:
    st.title("Analytics")
    st.markdown("Explore prediction volume, confidence trends, and brand distribution.")
    history = _history_dataframe()

    if history.empty:
        st.info("Analytics will appear after predictions are made.")
        return

    category_counts = history["predicted_category"].value_counts()
    st.subheader("Predictions by Category")
    st.bar_chart(category_counts)

    if "predicted_sub_category" in history:
        sub_category_counts = history["predicted_sub_category"].dropna().value_counts()
        if not sub_category_counts.empty:
            st.subheader("Predictions by Subcategory")
            st.bar_chart(sub_category_counts)

    if "confidence" in history and history["confidence"].notna().any():
        st.subheader("Confidence Over Time")
        confidence_frame = history[["timestamp", "confidence"]].copy()
        confidence_frame["timestamp"] = pd.to_datetime(confidence_frame["timestamp"])
        confidence_frame = confidence_frame.sort_values("timestamp").set_index("timestamp")
        st.line_chart(confidence_frame)

    st.subheader("Brand Mix")
    brand_counts = history["brand"].fillna("Unknown").replace("", "Unknown").value_counts()
    st.bar_chart(brand_counts)


def render_settings() -> None:
    st.title("Settings")
    st.markdown("Configure prediction mode and verify local artifact paths.")

    st.session_state.api_base_url = st.text_input(
        "FastAPI Base URL",
        value=st.session_state.api_base_url,
        help="Used when API prediction mode is enabled.",
    )
    st.session_state.use_api = st.toggle("Use FastAPI for predictions", value=st.session_state.use_api)

    st.subheader("Local Artifact Paths")
    st.code(
        "\n".join(
            [
                f"Model: {MODEL_PATH}",
                f"Vectorizer: {VECTORIZER_PATH}",
                f"Metrics: {METRICS_PATH}",
                f"Report: {REPORT_PATH}",
            ]
        )
    )

    st.subheader("Readiness")
    database_ready, database_message = _database_status()
    readiness = pd.DataFrame(
        [
            {"artifact": "model", "path": str(MODEL_PATH), "exists": MODEL_PATH.exists()},
            {"artifact": "vectorizer", "path": str(VECTORIZER_PATH), "exists": VECTORIZER_PATH.exists()},
            {"artifact": "metrics", "path": str(METRICS_PATH), "exists": METRICS_PATH.exists()},
            {"artifact": "report", "path": str(REPORT_PATH), "exists": REPORT_PATH.exists()},
            {"artifact": "database", "path": st.session_state.get("database_url", "DATABASE_URL"), "exists": database_ready},
        ]
    )
    st.dataframe(readiness, width="stretch", hide_index=True)
    if database_ready:
        st.success(database_message)
    else:
        st.warning(database_message)


def _predict(
    product_name: str,
    description: str,
    brand: str | None = None,
    price: float | None = None,
) -> dict[str, Any]:
    if st.session_state.use_api:
        response = requests.post(
            f"{st.session_state.api_base_url.rstrip('/')}/predict",
            json={
                "product_name": product_name,
                "description": description,
                "brand": brand,
                "price": price,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        _require_database_ids(payload)
        payload["source"] = "api"
        return payload

    service = PredictionService()
    result = service.predict(
        PredictRequest(product_name=product_name, description=description, brand=brand, price=price)
    )
    payload = result.model_dump()
    _require_database_ids(payload)
    payload["source"] = "local"
    return payload


def _require_database_ids(payload: dict[str, Any]) -> None:
    if payload.get("product_id") is None or payload.get("prediction_id") is None:
        raise RuntimeError(
            "Prediction was generated, but the database did not return saved product and prediction IDs."
        )


def _initialize_state() -> None:
    st.session_state.setdefault("prediction_history", [])
    st.session_state.setdefault("api_base_url", os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000"))
    st.session_state.setdefault("use_api", False)
    st.session_state.setdefault("database_url", os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/categoryiq"))


def _history_dataframe() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("prediction_history", []))


def _load_metrics_payload() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _model_comparison_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    metrics_by_name = {
        str(metric.get("model_name")): metric
        for metric in payload.get("metrics", [])
        if metric.get("model_name")
    }
    rows = []
    for model_name in MODEL_COMPARISON_ORDER:
        metric = dict(metrics_by_name.get(model_name, {}))
        if not metric:
            metric = {
                "model_name": model_name,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "status": "not run",
                "message": "Retrain with this model enabled to populate metrics.",
            }
        rows.append(metric)

    extra_rows = [
        dict(metric)
        for model_name, metric in metrics_by_name.items()
        if model_name not in MODEL_COMPARISON_ORDER
    ]
    return pd.DataFrame(rows + extra_rows)


def _best_accuracy(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics", [])
    trained = [metric for metric in metrics if metric.get("status") == "trained"]
    if not trained:
        return None
    best = max(trained, key=lambda metric: (metric.get("f1_score", 0), metric.get("accuracy", 0)))
    return best.get("accuracy")


def _most_common(values: list[Any]) -> str:
    clean_values = [str(value) for value in values if value]
    if not clean_values:
        return "No predictions"
    return Counter(clean_values).most_common(1)[0][0]


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _database_status() -> tuple[bool, str]:
    try:
        CategoryIQPostgresService().initialize()
    except (OSError, psycopg2.Error) as exc:
        return False, f"PostgreSQL is not reachable with DATABASE_URL: {exc}"
    return True, "PostgreSQL connection is ready and schema initialization succeeded."


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f7fb;
            color: #111827;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp p,
        .stApp label,
        .stApp span {
            color: #111827;
        }
        section[data-testid="stSidebar"] {
            background: #14213d;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        div[data-baseweb="select"] {
            color: #0f172a;
            background: #ffffff;
            border-color: #94a3b8;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
