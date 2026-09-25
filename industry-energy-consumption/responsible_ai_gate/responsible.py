"""Configurable model quality gate before registration or deployment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def evaluate_quality_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    max_allowed_mae = os.getenv("MAX_ALLOWED_MAE")
    min_allowed_r2 = os.getenv("MIN_ALLOWED_R2")
    if max_allowed_mae and metrics.get("validation_mae", float("inf")) > float(max_allowed_mae):
        failures.append(f"validation_mae exceeds MAX_ALLOWED_MAE={max_allowed_mae}")
    if min_allowed_r2 and metrics.get("validation_r2", float("-inf")) < float(min_allowed_r2):
        failures.append(f"validation_r2 is below MIN_ALLOWED_R2={min_allowed_r2}")
    return not failures, failures


def main(metrics_path: str | Path) -> None:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    passed, failures = evaluate_quality_gate(metrics)
    if not passed:
        raise SystemExit("Responsible AI quality gate failed: " + "; ".join(failures))
    print("Responsible AI quality gate passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="outputs/metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.metrics_path)

