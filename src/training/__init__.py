"""Model training utilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.training.train_models import (
        ModelMetrics,
        SUPPORTED_MODEL_NAMES,
        TrainingConfig,
        TrainingResult,
        train_and_select_best_model,
        train_and_select_best_model_from_dataframe,
    )

__all__ = [
    "ModelMetrics",
    "SUPPORTED_MODEL_NAMES",
    "TrainingConfig",
    "TrainingResult",
    "train_and_select_best_model",
    "train_and_select_best_model_from_dataframe",
]


def __getattr__(name: str) -> Any:
    """Lazily expose training objects at the package level."""

    if name not in __all__:
        raise AttributeError(f"module 'src.training' has no attribute {name!r}")

    from src.training import train_models

    return getattr(train_models, name)
