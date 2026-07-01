#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.schemas import list_schemas, validate_payload


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "framework" / "fuzzyxai" / "fuzzyxai" / "schemas"


def main() -> int:
    for path in SCHEMA_DIR.glob("*.schema.json"):
        json.loads(path.read_text(encoding="utf-8"))
    sample = {
        "scenario_id": "external_wine_classification",
        "source_type": "tabular",
        "model_name": "SchemaSmokeModel",
        "dataset_name": "schema_smoke",
        "predicted_class": 0,
        "class_probability": 0.7,
        "feature_values": {"x": 1.0},
        "feature_importance": {"x": 0.6},
        "quality_metrics": {"missing_rate": 0.0},
    }
    result = validate_payload(sample, "classification")
    if not result.valid:
        raise SystemExit(result.errors)
    if {"classification", "regression", "signal", "image", "route", "proof_trace", "operator_trace"} - set(list_schemas()):
        raise SystemExit("missing schemas")
    print("fuzzyxai-schema-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
