from __future__ import annotations

from dataclasses import replace

import numpy as np

from fuzzyxai.audit_certificate import build_action_certificate
from fuzzyxai.diagnostic_cut import graph_from_certificate, solve_exact
from fuzzyxai.practical_controller import DeploymentContext, ExplanationArtifact, PredictionArtifact, RouteArtifacts

from .common import ARTIFACTS, require_file, verify_protocol, write_json

DIGEST = "a" * 64


def _base():
    prediction = PredictionArtifact("object", "class-1", 0.9, (0.1, 0.9), "model-1")
    explanation = ExplanationArtifact(DIGEST, "explainer-1", "model-1", "plan-1", "dictionary-1", ("model", "explainer", "reference"))
    route = RouteArtifacts("prep-1", "cal-1", "population-1", "schema-1", DIGEST, ("model", "explainer", "reference"))
    context = DeploymentContext("model-1", "prep-1", "explainer-1", "cal-1", "population-1", "schema-1", "plan-1", "dictionary-1", DIGEST, ("model", "explainer", "reference"), 0.2, "remediation-v2")
    return prediction, explanation, route, context


def _inject(kind: str, prediction, explanation, route):
    if kind == "model_version":
        prediction = replace(prediction, model_version="model-stale")
    elif kind == "explainer_model_pair":
        explanation = replace(explanation, model_version="model-other")
    elif kind == "preprocessing":
        route = replace(route, preprocessing_version="prep-wrong")
    elif kind == "calibration":
        route = replace(route, calibration_version=None)
    elif kind == "reference_population":
        route = replace(route, reference_population="population-wrong")
    elif kind == "schema":
        route = replace(route, schema_version="schema-wrong")
    elif kind == "dictionary":
        explanation = replace(explanation, dictionary_version="dictionary-wrong")
    elif kind == "canonical":
        explanation = replace(explanation, canonical_sha256="b" * 64)
    elif kind == "provenance":
        route = replace(route, observed_provenance_channels=("model", "explainer"))
    elif kind == "reduction":
        explanation = replace(explanation, representation_loss=0.9)
    elif kind == "forbidden_rule":
        route = replace(route, forbidden_rule_conflict=True)
    elif kind == "data_quality":
        route = replace(route, critical_data_quality_fault=True)
    elif kind.startswith("registered_"):
        route = replace(route, route_fault_type=kind)
    elif kind.startswith("heldout_"):
        route = replace(route, natural_failure=kind)
    else:
        raise ValueError(kind)
    return prediction, explanation, route


def _simple_or(prediction, explanation, route, context) -> bool:
    return (
        prediction.confidence < 0.5
        or explanation.canonical_sha256 != context.expected_artifact_sha256
        or bool(set(context.mandatory_provenance_channels) - set(route.observed_provenance_channels))
    )


def main() -> None:
    require_file(ARTIFACTS / "lock" / "negative_remediation_lock.json", "H5 requires frozen remediation protocol")
    registered = (
        "model_version", "explainer_model_pair", "preprocessing", "calibration", "reference_population", "schema", "dictionary", "canonical", "provenance", "reduction", "forbidden_rule", "data_quality",
        *(f"registered_{index:02d}" for index in range(28)),
    )
    heldout = ("heldout_sampling_rate", "heldout_tokenizer_space", "heldout_post_reduction_link", "heldout_mixed_feature_space")
    records = []
    for index, kind in enumerate(registered):
        for composition in (1, 2, 3):
            prediction, explanation, route, context = _base()
            kinds = tuple(registered[(index + offset) % len(registered)] for offset in range(composition))
            for item in kinds:
                prediction, explanation, route = _inject(item, prediction, explanation, route)
            certificate = build_action_certificate(prediction, explanation, route, context)
            cut = solve_exact(graph_from_certificate(certificate))
            records.append(
                {
                    "group": "registered_single" if composition == 1 else "registered_composition",
                    "faults": kinds,
                    "detected": not certificate.certificate_exists,
                    "cut": cut.contracts,
                    "cut_exact": cut.exact,
                    "source_localized": bool(cut.fault_sources),
                    "simple_or_detected": _simple_or(prediction, explanation, route, context),
                }
            )
    for kind in heldout:
        prediction, explanation, route, context = _base()
        prediction, explanation, route = _inject(kind, prediction, explanation, route)
        certificate = build_action_certificate(prediction, explanation, route, context)
        cut = solve_exact(graph_from_certificate(certificate))
        records.append(
            {
                "group": "heldout_fault_type",
                "faults": (kind,),
                "detected": not certificate.certificate_exists,
                "cut": cut.contracts,
                "cut_exact": cut.exact,
                "source_localized": bool(cut.fault_sources),
                "simple_or_detected": _simple_or(prediction, explanation, route, context),
            }
        )
    registered_rows = [row for row in records if row["group"].startswith("registered")]
    heldout_rows = [row for row in records if row["group"] == "heldout_fault_type"]
    typed_recall = float(np.mean([row["detected"] for row in registered_rows]))
    simple_recall = float(np.mean([row["simple_or_detected"] for row in registered_rows]))
    source_localization = float(np.mean([row["source_localized"] for row in registered_rows]))
    exact_match = float(np.mean([row["cut_exact"] for row in registered_rows]))
    summary = {
        "phase": "controlled_confirmatory_after_protocol_lock",
        "protocol_sha256": verify_protocol(),
        "n_registered_templates": len(registered),
        "n_registered_cases": len(registered_rows),
        "n_heldout_types": len(heldout),
        "typed_validator": {
            "registered_recall": typed_recall,
            "source_localization": source_localization,
            "minimal_cut_solver_exact_rate": exact_match,
            "false_certification": 1.0 - typed_recall,
        },
        "simple_or": {"registered_recall": simple_recall, "false_certification": 1.0 - simple_recall},
        "heldout": {
            "generic_failure_detection": float(np.mean([row["detected"] for row in heldout_rows])),
            "specific_unknown_type_identification": 0.0,
            "arbitrary_unknown_failure_claim_allowed": False,
        },
        "H5-A2": "supported_registered_library_only" if source_localization - simple_recall >= 0.10 and 1.0 - typed_recall <= 0.01 and exact_match >= 0.80 else "not_supported",
        "H5-P2": "not_evaluated_independent_model_error_data_absent",
        "H5-P3": "not_evaluated_independent_model_error_data_absent",
    }
    write_json(ARTIFACTS / "h5" / "raw_cases.json", records)
    write_json(ARTIFACTS / "h5" / "summary.json", summary)
    print(f"PASS remediation-h5-confirmatory H5-A2={summary['H5-A2']} heldout_unknown_claim=false")


if __name__ == "__main__":
    main()
