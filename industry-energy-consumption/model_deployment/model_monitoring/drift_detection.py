"""Simple PSI-based data drift detection for production monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected, actual, buckets: int = 10) -> float:
    expected_series = pd.Series(expected, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    actual_series = pd.Series(actual, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if expected_series.empty or actual_series.empty:
        raise ValueError("PSI requires non-empty numeric expected and actual samples")

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected_series, quantiles))
    if len(breakpoints) < 3:
        breakpoints = np.linspace(expected_series.min(), expected_series.max() + 1e-6, buckets + 1)

    expected_counts, _ = np.histogram(expected_series, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_series, bins=breakpoints)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], threshold: float = 0.2) -> dict[str, dict]:
    """Return per-column PSI drift results.

    PSI under 0.1 is usually small, 0.1 to 0.2 deserves review, and over 0.2
    indicates material drift. These are heuristics, not business guarantees.
    """

    results: dict[str, dict] = {}
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        results[column] = {"psi": psi, "drifted": psi >= threshold, "threshold": threshold}
    return results

