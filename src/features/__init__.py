"""Feature engineering utilities."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.features.tfidf_features import (
        TfidfFeatureConfig,
        TfidfFeatureResult,
        build_tfidf_features,
        build_tfidf_features_from_dataframe,
    )

__all__ = [
    "TfidfFeatureConfig",
    "TfidfFeatureResult",
    "build_tfidf_features",
    "build_tfidf_features_from_dataframe",
]


def __getattr__(name: str) -> Any:
    """Lazily expose feature engineering objects at the package level."""

    if name not in __all__:
        raise AttributeError(f"module 'src.features' has no attribute {name!r}")

    from src.features import tfidf_features

    return getattr(tfidf_features, name)
