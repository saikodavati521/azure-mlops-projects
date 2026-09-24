"""Text preprocessing utilities for product classification datasets.

Objects are loaded lazily so ``python -m src.preprocessing.text_preprocessor`` can run
without importing the module twice.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.preprocessing.text_preprocessor import (
        PreprocessingConfig,
        TextPreprocessor,
        preprocess_dataset,
    )

__all__ = ["PreprocessingConfig", "TextPreprocessor", "preprocess_dataset"]


def __getattr__(name: str) -> Any:
    """Lazily expose preprocessing objects at the package level."""

    if name not in __all__:
        raise AttributeError(f"module 'src.preprocessing' has no attribute {name!r}")

    from src.preprocessing import text_preprocessor

    return getattr(text_preprocessor, name)
