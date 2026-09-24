"""Dataset validation utilities.

The public objects are loaded lazily so ``python -m src.validation.validator`` can run
without importing the module twice.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.validation.validator import (
        ValidationConfig,
        ValidationResult,
        validate_dataframe,
        validate_dataset,
    )

__all__ = [
    "ValidationConfig",
    "ValidationResult",
    "validate_dataframe",
    "validate_dataset",
]


def __getattr__(name: str) -> Any:
    """Lazily expose validator objects at the package level."""

    if name not in __all__:
        raise AttributeError(f"module 'src.validation' has no attribute {name!r}")

    from src.validation import validator

    return getattr(validator, name)
