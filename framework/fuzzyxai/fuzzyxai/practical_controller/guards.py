"""Hard structural contract checks that are independent of predictive confidence."""

from __future__ import annotations

from fuzzyxai.selective_observer.evidence_identity import LocalEvidenceStatus, observe_local_explanation

from .contracts import (
    DeploymentContext,
    ExplanationArtifact,
    GuardResult,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
)


BLOCKING_CHECKS = {
    "MODEL_VERSION_MISMATCH",
    "PREPROCESSING_VERSION_MISMATCH",
    "EXPLAINER_MODEL_MISMATCH",
    "SCHEMA_VERSION_MISMATCH",
    "EXPLAIN_PLAN_VERSION_MISMATCH",
    "DICTIONARY_VERSION_MISMATCH",
    "ARTIFACT_HASH_MISMATCH",
    "FORBIDDEN_RULE_CONFLICT",
    "CRITICAL_DATA_QUALITY_FAULT",
    "BROKEN_TRANSFORMATION",
    "CORRUPTED_AUDIT_HASH",
    "LOCAL_EXPLANATION_INCOMPATIBLE",
    "EXPLANATION_TARGET_MISMATCH",
}


def evaluate_hard_guard(
    prediction: PredictionArtifact,
    explanation: ExplanationArtifact,
    route: RouteArtifacts,
    context: DeploymentContext,
) -> GuardResult:
    reasons: list[str] = []
    missing: list[str] = []
    _mismatch(prediction.model_version, context.expected_model_version, "MODEL_VERSION_MISMATCH", reasons)
    _mismatch(explanation.model_version, context.expected_model_version, "EXPLAINER_MODEL_MISMATCH", reasons)
    _mismatch(route.preprocessing_version, context.expected_preprocessing_version, "PREPROCESSING_VERSION_MISMATCH", reasons)
    _mismatch(explanation.explainer_version, context.expected_explainer_version, "EXPLAINER_VERSION_MISMATCH", reasons)
    _mismatch(route.calibration_version, context.expected_calibration_version, "MISSING_OR_STALE_CALIBRATION", reasons)
    _mismatch(route.reference_population, context.expected_reference_population, "WRONG_REFERENCE_POPULATION", reasons)
    _mismatch(route.schema_version, context.expected_schema_version, "SCHEMA_VERSION_MISMATCH", reasons)
    _mismatch(explanation.explain_plan_version, context.expected_explain_plan_version, "EXPLAIN_PLAN_VERSION_MISMATCH", reasons)
    _mismatch(explanation.dictionary_version, context.expected_dictionary_version, "DICTIONARY_VERSION_MISMATCH", reasons)
    _mismatch(route.artifact_sha256, context.expected_artifact_sha256, "ARTIFACT_HASH_MISMATCH", reasons)
    _mismatch(explanation.canonical_sha256, context.expected_artifact_sha256, "CANONICAL_SOURCE_HASH_MISMATCH", reasons)
    missing.extend(sorted(set(context.mandatory_provenance_channels) - set(route.observed_provenance_channels)))
    if missing:
        reasons.append("MISSING_MANDATORY_PROVENANCE")
    if explanation.representation_loss > context.maximum_reduction_loss:
        reasons.append("EXCESSIVE_REDUCTION_LOSS")
    if context.require_local_evidence or explanation.local_evidence is not None:
        observation = observe_local_explanation(
            explanation.local_evidence,
            expected_object_id=prediction.object_id,
            expected_model_version=prediction.model_version,
            expected_target_class=prediction.prediction,
            expected_feature_names=prediction.feature_names,
            expected_source_sha256=explanation.canonical_sha256,
        )
        if observation.status is LocalEvidenceStatus.INCOMPATIBLE:
            reasons.append("LOCAL_EXPLANATION_INCOMPATIBLE")
            reasons.extend(f"LOCAL_EVIDENCE:{code}" for code in observation.contradictions)
        elif observation.status is LocalEvidenceStatus.INSUFFICIENT_EVIDENCE:
            reasons.append("LOCAL_EXPLANATION_INSUFFICIENT_EVIDENCE")
            missing.extend(f"LOCAL_EVIDENCE:{field}" for field in observation.missing_fields)
    if context.expected_explanation_target is not None and context.expected_explanation_target != prediction.prediction:
        reasons.append("EXPLANATION_TARGET_MISMATCH")
    if route.forbidden_rule_conflict:
        reasons.append("FORBIDDEN_RULE_CONFLICT")
    if route.critical_data_quality_fault:
        reasons.append("CRITICAL_DATA_QUALITY_FAULT")
    if route.route_fault_type:
        reasons.append(f"ROUTE_FAULT:{route.route_fault_type}")
    if route.natural_failure:
        reasons.append(f"NATURAL_FAILURE:{route.natural_failure}")

    blocking = any(reason.split(":", 1)[0] in BLOCKING_CHECKS for reason in reasons)
    if blocking:
        status = HardGuardStatus.BLOCKED
    elif reasons:
        status = HardGuardStatus.REVIEW_REQUIRED
    else:
        status = HardGuardStatus.CERTIFIED
    return GuardResult(
        status=status,
        reason_codes=tuple(dict.fromkeys(reasons)),
        missing_evidence=tuple(missing),
        fault_source=route.route_fault_source,
    )


def _mismatch(actual: str | None, expected: str, code: str, reasons: list[str]) -> None:
    if actual != expected:
        reasons.append(code)
