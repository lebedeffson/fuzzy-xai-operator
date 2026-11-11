"""Scientific regression: instance-bound local explanation evidence.

No planted labels, prediction correctness, or explanation faithfulness are
inferred from a successful structural check.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer.evidence_identity import (
    LocalEvidenceStatus,
    LocalExplanationEvidence,
    observe_local_explanation,
)


DIGEST = hashlib.sha256(b"source").hexdigest()
AXIS = ("age", "pressure", "temperature")


def prediction() -> PredictionArtifact:
    return PredictionArtifact(
        object_id="sample-7",
        prediction="positive",
        confidence=0.9,
        probabilities=(0.1, 0.9),
        model_version="model-1",
        feature_names=AXIS,
    )


def local() -> LocalExplanationEvidence:
    return LocalExplanationEvidence(
        object_id="sample-7",
        model_version="model-1",
        target_class="positive",
        scope="local",
        feature_names=AXIS,
        contributions=(0.2, -0.1, 0.3),
        source_sha256=DIGEST,
    )


def explanation(evidence: LocalExplanationEvidence | None) -> ExplanationArtifact:
    return ExplanationArtifact(
        canonical_sha256=DIGEST,
        explainer_version="explainer-1",
        model_version="model-1",
        explain_plan_version="plan-1",
        dictionary_version="dict-1",
        available_channels=("prediction", "attribution", "provenance"),
        local_evidence=evidence,
    )


def route() -> RouteArtifacts:
    return RouteArtifacts(
        preprocessing_version="prep-1",
        calibration_version="cal-1",
        reference_population="reference-1",
        schema_version="schema-1",
        artifact_sha256=DIGEST,
        observed_provenance_channels=("prediction", "attribution", "provenance"),
    )


def context(*, strict: bool = True) -> DeploymentContext:
    return DeploymentContext(
        expected_model_version="model-1",
        expected_preprocessing_version="prep-1",
        expected_explainer_version="explainer-1",
        expected_calibration_version="cal-1",
        expected_reference_population="reference-1",
        expected_schema_version="schema-1",
        expected_explain_plan_version="plan-1",
        expected_dictionary_version="dict-1",
        expected_artifact_sha256=DIGEST,
        mandatory_provenance_channels=("prediction", "attribution", "provenance"),
        maximum_reduction_loss=0.2,
        policy_version="policy-1",
        require_local_evidence=strict,
    )


def test_certified_local_vector_binds_one_object_one_target_one_feature_axis() -> None:
    result = evaluate_hard_guard(prediction(), explanation(local()), route(), context())
    assert result.status is HardGuardStatus.CERTIFIED
    assert not result.reason_codes


@pytest.mark.parametrize(
    ("change", "expected_code"),
    [
        ({"object_id": "sample-8"}, "object_id_mismatch"),
        ({"model_version": "model-2"}, "model_version_mismatch"),
        ({"target_class": "negative"}, "target_class_mismatch"),
        ({"scope": "global"}, "non_local_scope"),
        ({"feature_names": ("pressure", "age", "temperature")}, "feature_axis_mismatch"),
        ({"feature_names": ("age", "age", "temperature")}, "invalid_feature_axis"),
        ({"contributions": (0.2, 0.3)}, "contribution_width_mismatch"),
        ({"contributions": (0.2, float("nan"), 0.3)}, "invalid_contribution_values"),
        ({"contributions": (0.2, float("inf"), 0.3)}, "invalid_contribution_values"),
        ({"source_sha256": "f" * 64}, "source_sha256_mismatch"),
        ({"source_sha256": "invalid"}, "invalid_source_sha256"),
        ({"contributions": (0.2,) * 9}, "contribution_width_mismatch"),
    ],
)
def test_incompatible_evidence_blocks(change: dict[str, object], expected_code: str) -> None:
    result = evaluate_hard_guard(prediction(), explanation(replace(local(), **change)), route(), context())
    assert result.status is HardGuardStatus.BLOCKED
    assert "LOCAL_EXPLANATION_INCOMPATIBLE" in result.reason_codes
    assert f"LOCAL_EVIDENCE:{expected_code}" in result.reason_codes


@pytest.mark.parametrize("change", [{"target_class": None}, {"scope": None}, {"feature_names": None}, {"contributions": None}, {"source_sha256": None}])
def test_missing_evidence_forces_review(change: dict[str, object]) -> None:
    result = evaluate_hard_guard(prediction(), explanation(replace(local(), **change)), route(), context())
    assert result.status is HardGuardStatus.REVIEW_REQUIRED
    assert "LOCAL_EXPLANATION_INSUFFICIENT_EVIDENCE" in result.reason_codes
    assert result.missing_evidence


def test_absent_evidence_is_not_certified_in_strict_mode() -> None:
    result = evaluate_hard_guard(prediction(), explanation(None), route(), context())
    assert result.status is HardGuardStatus.REVIEW_REQUIRED
    assert "LOCAL_EVIDENCE:local_explanation_evidence" in result.missing_evidence


def test_old_callers_remain_compatible_when_strict_gate_disabled() -> None:
    result = evaluate_hard_guard(prediction(), explanation(None), route(), context(strict=False))
    assert result.status is HardGuardStatus.CERTIFIED


def test_declared_local_artifact_is_validated_even_in_legacy_mode() -> None:
    result = evaluate_hard_guard(
        prediction(), explanation(replace(local(), object_id="another")), route(), context(strict=False)
    )
    assert result.status is HardGuardStatus.BLOCKED


def test_target_override_cannot_be_different_from_prediction_for_auto_action() -> None:
    result = evaluate_hard_guard(
        prediction(),
        explanation(local()),
        route(),
        replace(context(), expected_explanation_target="negative"),
    )
    assert result.status is HardGuardStatus.BLOCKED
    assert "EXPLANATION_TARGET_MISMATCH" in result.reason_codes


def test_missing_prediction_axis_is_explicit_insufficient_evidence() -> None:
    result = evaluate_hard_guard(
        replace(prediction(), feature_names=None), explanation(local()), route(), context()
    )
    assert result.status is HardGuardStatus.REVIEW_REQUIRED
    assert "LOCAL_EVIDENCE:expected_feature_names" in result.missing_evidence


def test_observer_makes_no_predictive_correctness_claim() -> None:
    result = observe_local_explanation(
        local(),
        expected_object_id="sample-7",
        expected_model_version="model-1",
        expected_target_class="positive",
        expected_feature_names=AXIS,
        expected_source_sha256=DIGEST,
    )
    assert result.status is LocalEvidenceStatus.CERTIFIED
    assert result.certified
    assert not hasattr(result, "predictive_accuracy")
