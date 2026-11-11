"""Additive model-score evidence: positive and adversarial regression tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    PracticalPolicy,
    ReviewBudget,
    RouteArtifacts,
    assess_action,
    cost_profile,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer import SelectiveAction
from fuzzyxai.selective_observer.evidence_identity import LocalEvidenceStatus, LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import (
    SUPPORTED_PROTOCOL,
    LocalEffectEvidence,
    verify_additive_local_effects,
)

DIGEST = hashlib.sha256(b"fixed-source-evidence").hexdigest()


def _local() -> LocalExplanationEvidence:
    return LocalExplanationEvidence(
        object_id="obj-1",
        model_version="model-v1",
        target_class="positive",
        scope="local",
        feature_names=("a", "b"),
        contributions=(0.2, -0.1),
        source_sha256=DIGEST,
    )


def _effects() -> LocalEffectEvidence:
    return LocalEffectEvidence(
        protocol=SUPPORTED_PROTOCOL,
        object_id="obj-1",
        model_version="model-v1",
        target_class="positive",
        feature_names=("a", "b"),
        effect_on_score=(0.2, -0.1),
        reference_score=0.5,
        explained_score=0.6,
    )


def _artifacts():
    prediction = PredictionArtifact(
        object_id="obj-1",
        prediction="positive",
        model_version="model-v1",
        confidence=0.95,
        probabilities=(0.05, 0.95),
        feature_names=("a", "b"),
    )
    explanation = ExplanationArtifact(
        canonical_sha256=DIGEST,
        explainer_version="linear-logit-v1",
        model_version="model-v1",
        explain_plan_version="plan-v1",
        dictionary_version="dict-v1",
        available_channels=("prediction", "attribution", "provenance"),
        local_evidence=_local(),
    )
    route = RouteArtifacts(
        preprocessing_version="prep-v1",
        calibration_version="cal-v1",
        reference_population="population-v1",
        schema_version="schema-v1",
        artifact_sha256=DIGEST,
        observed_provenance_channels=("prediction", "attribution", "provenance"),
        observed_local_effects=_effects(),
    )
    context = DeploymentContext(
        expected_model_version="model-v1",
        expected_preprocessing_version="prep-v1",
        expected_explainer_version="linear-logit-v1",
        expected_calibration_version="cal-v1",
        expected_reference_population="population-v1",
        expected_schema_version="schema-v1",
        expected_explain_plan_version="plan-v1",
        expected_dictionary_version="dict-v1",
        expected_artifact_sha256=DIGEST,
        mandatory_provenance_channels=("prediction", "attribution", "provenance"),
        maximum_reduction_loss=0.2,
        policy_version="policy-v1",
        require_local_evidence=True,
        require_local_effect_verification=True,
    )
    return prediction, explanation, route, context


def test_effect_oracle_matches_linear_logit_contributions() -> None:
    result = verify_additive_local_effects(_local(), _effects())
    assert result.status is LocalEvidenceStatus.CERTIFIED
    assert not result.contradictions
    assert evaluate_hard_guard(*_artifacts()).status is HardGuardStatus.CERTIFIED


def test_value_swap_passes_shape_check_but_fails_model_effects() -> None:
    prediction, explanation, route, context = _artifacts()
    swapped = replace(_local(), contributions=(-0.1, 0.2))
    broken_explanation = replace(explanation, local_evidence=swapped)
    result = evaluate_hard_guard(prediction, broken_explanation, route, context)
    assert result.status is HardGuardStatus.BLOCKED
    assert "LOCAL_EFFECT:feature_effect_mismatch" in result.reason_codes
    shape_only = evaluate_hard_guard(
        prediction,
        broken_explanation,
        replace(route, observed_local_effects=None),
        replace(context, require_local_effect_verification=False),
    )
    assert shape_only.status is HardGuardStatus.CERTIFIED


def test_missing_effects_require_review_not_automatic_acceptance() -> None:
    prediction, explanation, route, context = _artifacts()
    result = evaluate_hard_guard(
        prediction, explanation, replace(route, observed_local_effects=None), context
    )
    assert result.status is HardGuardStatus.REVIEW_REQUIRED
    assert "LOCAL_EFFECT:observed_local_effects" in result.missing_evidence


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        ({"object_id": "another"}, "object_id_mismatch"),
        ({"target_class": "negative"}, "target_class_mismatch"),
        ({"model_version": "stale"}, "model_version_mismatch"),
        ({"feature_names": ("b", "a")}, "feature_axis_mismatch"),
        ({"effect_on_score": (0.1, 0.2)}, "feature_effect_mismatch"),
        ({"effect_on_score": (0.1,)}, "effect_width_mismatch"),
        ({"explained_score": 0.7}, "score_reconstruction_mismatch"),
        ({"effect_on_score": (float("nan"), -0.1)}, "nonfinite_or_invalid_effect"),
    ],
)
def test_effect_inconsistency_never_certifies(change: dict[str, object], expected_reason: str) -> None:
    result = verify_additive_local_effects(_local(), replace(_effects(), **change))
    assert result.status is LocalEvidenceStatus.INCOMPATIBLE
    assert expected_reason in result.contradictions


def test_unsupported_protocol_is_insufficient_not_fake_certification() -> None:
    result = verify_additive_local_effects(_local(), replace(_effects(), protocol="generic_shap"))
    assert result.status is LocalEvidenceStatus.INSUFFICIENT_EVIDENCE
    assert "supported_effect_protocol" in result.missing_fields


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_invalid_tolerance_cannot_disable_fidelity_check(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        verify_additive_local_effects(_local(), _effects(), absolute_tolerance=tolerance)



def test_effect_check_reaches_public_action_and_cannot_be_overridden_by_budget() -> None:
    prediction, explanation, route, context = _artifacts()
    policy = PracticalPolicy(
        schema_version="1.0",
        policy_version="policy-v1",
        predictive_weights=(1.0,) * 8,
        predictive_intercept=-5.0,
        route_weights=(1.0,) * 10,
        route_intercept=-5.0,
        accept_max_risk=0.2,
        short_review_max_risk=0.6,
        full_review_max_risk=0.9,
        calibration_method="temperature",
        calibration_parameters=(1.0,),
        development_sha256="a" * 64,
        selected_without_test=True,
    )
    swapped = replace(explanation, local_evidence=replace(_local(), contributions=(-0.1, 0.2)))
    blocked = assess_action(
        prediction, swapped, route, context, ReviewBudget(0.05),
        cost_profile("review_expensive"), policy=policy,
    )
    assert blocked.hard_guard_status is HardGuardStatus.BLOCKED
    assert blocked.action is SelectiveAction.BLOCK
    assert "LOCAL_EFFECTS_INCOMPATIBLE" in blocked.reason_codes

    missing = assess_action(
        prediction, explanation, replace(route, observed_local_effects=None),
        context, ReviewBudget(0.05), cost_profile("review_expensive"), policy=policy,
    )
    assert missing.hard_guard_status is HardGuardStatus.REVIEW_REQUIRED
    assert missing.action is not SelectiveAction.ACCEPT
