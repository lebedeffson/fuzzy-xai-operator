from __future__ import annotations

from dataclasses import replace

import pytest

from fuzzyxai.audit_certificate import build_action_certificate
from fuzzyxai.diagnostic_cut import graph_from_certificate, solve_exact
from fuzzyxai.practical_controller import DeploymentContext, ExplanationArtifact, PredictionArtifact, RouteArtifacts
from fuzzyxai.practical_controller_v2 import (
    EXPLANATION_FEATURES,
    PREDICTIVE_FEATURES,
    ROUTE_FEATURES,
    SHIFT_FEATURES,
    ActionCostProfile,
    CalibratedRiskHead,
    ControllerV2Policy,
    RiskHeadTrainingRow,
    assess_actions_v2,
)


DIGEST = "a" * 64


def inputs():
    prediction = PredictionArtifact("obj-1", "class-a", 0.9, (0.9, 0.1), "model-1", entropy=0.2, prediction_margin=0.8)
    explanation = ExplanationArtifact(DIGEST, "explainer-1", "model-1", "plan-1", "dictionary-1", ("model", "explainer"))
    route = RouteArtifacts("prep-1", "cal-1", "population-1", "schema-1", DIGEST, ("model", "explainer"))
    context = DeploymentContext(
        "model-1",
        "prep-1",
        "explainer-1",
        "cal-1",
        "population-1",
        "schema-1",
        "plan-1",
        "dictionary-1",
        DIGEST,
        ("model", "explainer"),
        0.2,
        "policy-v2",
    )
    return prediction, explanation, route, context


def head(target: str, names: tuple[str, ...]) -> CalibratedRiskHead:
    return CalibratedRiskHead(target, names, (0.0,) * len(names), 0.0, (0.0,) * len(names), (1.0,) * len(names), "identity", (1.0,), "global", DIGEST)


def policy() -> ControllerV2Policy:
    return ControllerV2Policy(
        "policy-v2",
        head("model_error", PREDICTIVE_FEATURES),
        head("route_not_certifiable", ROUTE_FEATURES),
        head("explanation_unstable_or_incomplete", EXPLANATION_FEATURES),
        head("outside_deployment_envelope", SHIFT_FEATURES),
        ActionCostProfile(),
        True,
    )


def test_clean_route_has_action_conditioned_certificate() -> None:
    prediction, explanation, route, context = inputs()
    certificate = build_action_certificate(prediction, explanation, route, context)
    assert certificate.certificate_exists
    assert certificate.sha256
    assert certificate.unsatisfied_contracts == ()


def test_missing_provenance_has_exact_cut_and_repair() -> None:
    prediction, explanation, route, context = inputs()
    route = replace(route, observed_provenance_channels=("model",))
    certificate = build_action_certificate(prediction, explanation, route, context)
    cut = solve_exact(graph_from_certificate(certificate))
    assert not certificate.certificate_exists
    assert cut.contracts == ("provenance:explainer",)
    assert cut.exact


def test_controller_uses_global_budget_and_blocks_only_hard_faults() -> None:
    prediction, explanation, route, context = inputs()
    predictions = tuple(replace(prediction, object_id=f"obj-{index}") for index in range(10))
    explanations = tuple(explanation for _ in predictions)
    routes = tuple(route for _ in predictions)
    assessments = assess_actions_v2(predictions, explanations, routes, context, 0.2, ActionCostProfile(), policy=policy())
    assert sum(item.action.value.endswith("review") for item in assessments) == 2
    assert all(item.action.value != "block" for item in assessments)
    broken = replace(route, preprocessing_version="wrong")
    assessments = assess_actions_v2((prediction,), (explanation,), (broken,), context, 0.2, ActionCostProfile(), policy=policy())
    assert assessments[0].action.value == "block"


def test_four_targets_cannot_be_merged() -> None:
    valid = policy()
    with pytest.raises(ValueError, match="four distinct"):
        ControllerV2Policy(
            "bad",
            valid.predictive_head,
            replace(valid.route_head, target_name="model_error"),
            valid.explanation_head,
            valid.shift_head,
            valid.costs,
            True,
        )


def test_test_or_non_oof_rows_fail_closed() -> None:
    with pytest.raises(ValueError):
        RiskHeadTrainingRow("x", "g", {"a": 1.0}, False, partition="test")
    with pytest.raises(ValueError):
        RiskHeadTrainingRow("x", "g", {"a": 1.0}, False, source_features_are_oof=False)


def test_composed_route_faults_remain_individually_localizable() -> None:
    prediction, explanation, route, context = inputs()
    route = replace(route, route_fault_type="fault-a|fault-b")
    certificate = build_action_certificate(prediction, explanation, route, context)
    cut = solve_exact(graph_from_certificate(certificate))
    assert cut.contracts == ("route_fault:fault-a", "route_fault:fault-b")
