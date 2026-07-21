from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import fuzzyxai
from fuzzyxai.practical_controller import (
    CanonicalExplanation,
    CanonicalReason,
    CanaryPolicy,
    CostProfileName,
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PracticalDevelopmentExample,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
    ShadowCanaryMonitor,
    assess_action,
    assess_batch,
    assess_stream,
    allocate_score_budget,
    cost_profile,
    fit_practical_policy,
    project_explanation,
    projection_metrics,
    verify_replay,
)
from fuzzyxai.selective_observer import SelectiveAction


DIGEST = hashlib.sha256(b"explanation").hexdigest()


def test_practical_controller_is_part_of_public_sdk() -> None:
    assert fuzzyxai.assess_action is assess_action
    assert fuzzyxai.ActionAssessment is not None


def _policy() -> PracticalPolicy:
    return PracticalPolicy(
        schema_version="1.0",
        policy_version="practical-v1",
        predictive_weights=(2.0,) * 8,
        predictive_intercept=-5.0,
        route_weights=(2.0,) * 10,
        route_intercept=-5.0,
        accept_max_risk=0.20,
        short_review_max_risk=0.60,
        full_review_max_risk=0.90,
        calibration_method="temperature",
        calibration_parameters=(1.0,),
        development_sha256="a" * 64,
        selected_without_test=True,
    )


def _prediction(index: int = 0, *, confidence: float = 0.95) -> PredictionArtifact:
    return PredictionArtifact(
        object_id=f"object-{index}",
        prediction="positive",
        confidence=confidence,
        probabilities=(1.0 - confidence, confidence),
        model_version="model-v1",
        entropy=1.0 - confidence,
        prediction_margin=confidence,
        boundary_distance=confidence,
    )


def _explanation(**changes) -> ExplanationArtifact:
    values = {
        "canonical_sha256": DIGEST,
        "explainer_version": "explainer-v1",
        "model_version": "model-v1",
        "explain_plan_version": "plan-v1",
        "dictionary_version": "dictionary-v1",
        "available_channels": ("prediction", "attribution", "provenance"),
    }
    values.update(changes)
    return ExplanationArtifact(**values)


def _route(**changes) -> RouteArtifacts:
    values = {
        "preprocessing_version": "prep-v1",
        "calibration_version": "cal-v1",
        "reference_population": "population-v1",
        "schema_version": "schema-v1",
        "artifact_sha256": DIGEST,
        "observed_provenance_channels": ("prediction", "attribution", "provenance"),
    }
    values.update(changes)
    return RouteArtifacts(**values)


def _context() -> DeploymentContext:
    return DeploymentContext(
        expected_model_version="model-v1",
        expected_preprocessing_version="prep-v1",
        expected_explainer_version="explainer-v1",
        expected_calibration_version="cal-v1",
        expected_reference_population="population-v1",
        expected_schema_version="schema-v1",
        expected_explain_plan_version="plan-v1",
        expected_dictionary_version="dictionary-v1",
        expected_artifact_sha256=DIGEST,
        mandatory_provenance_channels=("prediction", "attribution", "provenance"),
        maximum_reduction_loss=0.20,
        policy_version="practical-v1",
    )


def test_hard_guard_blocks_version_mismatch_but_not_low_confidence() -> None:
    blocked = assess_action(
        _prediction(),
        _explanation(),
        _route(preprocessing_version="wrong"),
        _context(),
        ReviewBudget(0.20),
        cost_profile(CostProfileName.BALANCED),
        policy=_policy(),
    )
    assert blocked.action is SelectiveAction.BLOCK
    assert blocked.hard_guard_status is HardGuardStatus.BLOCKED

    uncertain = assess_action(
        _prediction(confidence=0.51),
        _explanation(),
        _route(),
        _context(),
        ReviewBudget(0.20),
        cost_profile(CostProfileName.BALANCED),
        policy=_policy(),
    )
    assert uncertain.action in {SelectiveAction.SHORT_REVIEW, SelectiveAction.FULL_REVIEW}
    assert uncertain.action is not SelectiveAction.BLOCK


def test_missing_provenance_requires_review_and_never_silent_accept() -> None:
    assessment = assess_action(
        _prediction(),
        _explanation(),
        _route(observed_provenance_channels=("prediction",)),
        _context(),
        ReviewBudget(0.05),
        cost_profile("balanced"),
        policy=_policy(),
    )
    assert assessment.action is SelectiveAction.FULL_REVIEW
    assert assessment.budget_feasible is False
    assert "MISSING_MANDATORY_PROVENANCE" in assessment.reason_codes


def test_batch_budget_is_deterministic_and_explicit() -> None:
    predictions = tuple(_prediction(index, confidence=0.55 + index * 0.03) for index in range(10))
    explanations = tuple(
        _explanation(
            seed_instability=index / 10,
            bootstrap_instability=index / 10,
            perturbation_instability=index / 10,
        )
        for index in range(10)
    )
    routes = tuple(_route() for _ in range(10))
    first = assess_batch(
        predictions,
        explanations,
        routes,
        _context(),
        ReviewBudget(0.20),
        cost_profile("review_expensive"),
        policy=_policy(),
    )
    second = assess_batch(
        predictions,
        explanations,
        routes,
        _context(),
        ReviewBudget(0.20),
        cost_profile("review_expensive"),
        policy=_policy(),
    )
    assert first.audit_sha256 == second.audit_sha256
    assert first.realized_review_fraction == 0.20
    assert sum("BUDGET_CONSTRAINED_ACCEPT" in item.reason_codes for item in first.assessments) > 0
    assert all(verify_replay(item) for item in first.assessments)


def test_budget_baseline_preserves_typed_actions_and_partition() -> None:
    actions = allocate_score_budget((0.1, 0.4, 0.9, 0.2), review_budget=0.25)
    assert actions == [
        SelectiveAction.ACCEPT,
        SelectiveAction.ACCEPT,
        SelectiveAction.FULL_REVIEW,
        SelectiveAction.ACCEPT,
    ]
    assert sum(action is SelectiveAction.ACCEPT for action in actions) / len(actions) == 0.75


def test_stream_matches_batch_order() -> None:
    items = [(_prediction(index), _explanation(), _route()) for index in range(5)]
    output = list(
        assess_stream(
            items,
            _context(),
            ReviewBudget(0.20),
            cost_profile("balanced"),
            policy=_policy(),
            batch_size=2,
        )
    )
    assert [item.model_version for item in output] == ["model-v1"] * 5
    assert len({item.trace_id for item in output}) == 5


def test_policy_training_requires_oof_and_selects_development_calibration() -> None:
    rng = np.random.default_rng(7)
    examples = []
    for index in range(120):
        predictive = tuple(float(value) for value in rng.uniform(size=8))
        route = tuple(float(value) for value in rng.uniform(size=10))
        outcome = predictive[0] + route[4] + route[5] > 1.4
        examples.append(
            PracticalDevelopmentExample(
                object_id=f"dev-{index}",
                group_id=f"group-{index // 2}",
                predictive_features=predictive,
                route_features=route,
                operationally_invalid_action=outcome,
                partition="train" if index % 5 else "validation",
                source_features_are_oof=True,
            )
        )
    policy, report = fit_practical_policy(examples, policy_version="trained-v1")
    assert policy.selected_without_test is True
    assert report["confirmatory_test_used"] is False
    assert report["calibration"]["selected_method"] in {"platt", "isotonic", "temperature", "conformal_selective"}
    with pytest.raises(ValueError, match="out-of-fold"):
        PracticalDevelopmentExample(
            object_id="bad",
            group_id="bad",
            predictive_features=(0.0,) * 8,
            route_features=(0.0,) * 10,
            operationally_invalid_action=False,
            partition="train",
            source_features_are_oof=False,
        )


def test_canonical_payload_is_exact_and_projection_is_separate() -> None:
    payload = json.dumps({"raw": [0.7, -0.2, 0.1]}, separators=(",", ":")).encode()
    canonical = CanonicalExplanation.from_source(
        payload,
        source_media_type="application/json",
        explainer_parameters={"method": "native"},
        background_identity="train:v1",
        reasons=(
            CanonicalReason("r1", "fracture_density", 1, 0.7, 1, "feature"),
            CanonicalReason("r2", "distance", -1, -0.2, 2, "feature"),
            CanonicalReason("r3", "water", 1, 0.1, 3, "feature"),
        ),
    )
    projection = project_explanation(canonical, labels={"fracture_density": "Трещиноватость"}, top_k=2)
    metrics = projection_metrics(canonical, projection)
    assert canonical.verify_exact_source(payload)
    assert metrics["canonical_hash_preserved"] is True
    assert metrics["sign_preserved"] is True
    assert len(projection.omitted_reason_ids) == 1
    with pytest.raises(ValueError, match="hash differs"):
        CanonicalExplanation(
            source_payload=b"changed",
            source_sha256=canonical.source_sha256,
            source_media_type="application/json",
            explainer_parameters_json="{}",
            background_identity="train:v1",
            reasons=canonical.reasons,
        )


def test_shadow_canary_rolls_back_on_frozen_ceiling() -> None:
    assessment = assess_action(
        _prediction(),
        _explanation(),
        _route(),
        _context(),
        ReviewBudget(0.20),
        cost_profile("balanced"),
        policy=_policy(),
    )
    monitor = ShadowCanaryMonitor(
        CanaryPolicy("practical-v1", 0.25, 0.01, 100.0, 0.50, 0.20, 0.05),
        active_policy_version="practical-v1",
        rollback_policy_version="practical-v0",
    )
    for index in range(10):
        monitor.record_shadow(
            f"shadow-{index}",
            assessment,
            actual_action=SelectiveAction.ACCEPT,
            latency_ms=10.0,
            calibration_residual=0.01,
            route_fault_observed=False,
        )
        monitor.attach_delayed_label(f"shadow-{index}", invalid_outcome=False, false_block=index == 0)
    snapshot = monitor.snapshot()
    assert snapshot.rollback_required is True
    assert "FALSE_BLOCK_CEILING_EXCEEDED" in snapshot.rollback_reason_codes
    assert monitor.rollback({"practical-v0": _policy()}) == "practical-v0"
