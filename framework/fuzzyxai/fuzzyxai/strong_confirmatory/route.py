"""Controlled H5-A route-validity fault and guardrail evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


FAULT_TYPES = (
    "missing_provenance",
    "mismatched_model_version",
    "wrong_preprocessing",
    "wrong_explainer_model_pairing",
    "missing_calibration",
    "wrong_reference_population",
    "broken_transformation",
    "incompatible_dictionary",
    "excessive_reduction_loss",
    "corrupted_audit_hash",
    "forbidden_rule_conflict",
)


def evaluate_route_guardrails(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("route evaluation requires records")
    truth = np.asarray([record.get("fault_type") in FAULT_TYPES for record in records], dtype=bool)
    methods = {
        "confidence": lambda row: float(row["confidence"]) < 0.70,
        "data_quality_only": lambda row: float(row["data_quality"]) < 0.70,
        "provenance_presence_only": lambda row: not bool(row["provenance_present"]),
        "static_schema_validation": lambda row: not bool(row["schema_valid"]),
        "simple_or": lambda row: (
            float(row["confidence"]) < 0.70
            or float(row["data_quality"]) < 0.70
            or not bool(row["provenance_present"])
            or not bool(row["schema_valid"])
        ),
        "observer_without_route_features": lambda row: float(row["generic_risk"]) >= 0.65,
        "typed_route_validity": lambda row: row.get("detected_fault_type") in FAULT_TYPES,
    }
    results = []
    for name, decision in methods.items():
        predicted = np.asarray([decision(record) for record in records], dtype=bool)
        fault_count = max(1, int(truth.sum()))
        clean_count = max(1, int((~truth).sum()))
        typed = name == "typed_route_validity"
        localization = (
            float(
                np.mean(
                    [
                        record.get("fault_type") == record.get("detected_fault_type")
                        and record.get("fault_source") == record.get("detected_fault_source")
                        for record in records
                        if record.get("fault_type") in FAULT_TYPES
                    ]
                )
            )
            if typed
            else None
        )
        results.append(
            {
                "method": name,
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "invalid_action_recall": float(recall_score(truth, predicted, zero_division=0)),
                "false_certification": float(np.sum(truth & ~predicted) / fault_count),
                "false_block": float(np.sum(~truth & predicted) / clean_count),
                "fault_source_localization": localization,
            }
        )
    typed = next(row for row in results if row["method"] == "typed_route_validity")
    simple = [row for row in results if row["method"] != "typed_route_validity"]
    return {
        "phase": "formative_controlled",
        "fault_types": list(FAULT_TYPES),
        "n_records": len(records),
        "methods": results,
        "formative_target_met": bool(
            typed["f1"] >= 0.95
            and typed["false_certification"] <= 0.01
            and typed["fault_source_localization"] >= 0.90
            and typed["invalid_action_recall"] > max(row["invalid_action_recall"] for row in simple)
        ),
        "confirmatory_claim_allowed": False,
        "model_error_prediction_claim_allowed": False,
    }
