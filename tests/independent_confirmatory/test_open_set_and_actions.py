from __future__ import annotations

from dataclasses import replace

import numpy as np

from fuzzyxai.open_set_validator import OpenSetOutcome, OpenSetTrainingRow, StructuralObservation, assess_open_set, fit_open_set_validator
from fuzzyxai.practical_controller import DeploymentContext, ExplanationArtifact, PredictionArtifact, RouteArtifacts
from fuzzyxai.practical_controller_v2 import (
    EXPLANATION_FEATURES,
    PREDICTIVE_FEATURES,
    ROUTE_FEATURES,
    SHIFT_FEATURES,
    ActionCostProfile,
    CalibratedRiskHead,
    ControllerV2Policy,
    assess_actions_v2,
)

DIGEST = "a" * 64
FEATURES = ("a", "b", "c")
REGIONS = {"a": "schema", "b": "graph", "c": "certificate"}


def _observation(index: int, values: tuple[float, ...], *, partition: str = "development", missing=()) -> StructuralObservation:
    return StructuralObservation(f"obs-{index}", dict(zip(FEATURES, values, strict=True)), REGIONS, tuple(missing), partition != "test", partition)


def test_open_set_abstains_on_unknown_family_and_missing_evidence() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(30):
        rows.append(OpenSetTrainingRow(_observation(index, tuple(rng.normal(0.0, 0.04, 3))), "valid_route"))
    for index in range(30, 60):
        rows.append(OpenSetTrainingRow(_observation(index, tuple(rng.normal((2.0, 0.0, 0.0), 0.05))), "schema_fault"))
    for index in range(60, 90):
        rows.append(OpenSetTrainingRow(_observation(index, tuple(rng.normal((0.0, 2.0, 0.0), 0.05))), "graph_fault"))
    spec = fit_open_set_validator(rows)
    known = assess_open_set(spec, _observation(100, (2.0, 0.0, 0.0), partition="test"))
    unknown = assess_open_set(spec, _observation(101, (0.0, 0.0, 4.0), partition="test"))
    insufficient = assess_open_set(spec, _observation(102, (0.0, 0.0, 0.0), partition="test", missing=("graph_embedding",)))
    assert known.outcome is OpenSetOutcome.KNOWN_FAULT_TYPE
    assert unknown.outcome is OpenSetOutcome.UNKNOWN_STRUCTURAL_FAULT
    assert unknown.suspected_regions[0] == "certificate"
    assert insufficient.outcome is OpenSetOutcome.INSUFFICIENT_EVIDENCE


def _head(target: str, names: tuple[str, ...]) -> CalibratedRiskHead:
    return CalibratedRiskHead(target, names, (0.0,) * len(names), 0.0, (0.0,) * len(names), (1.0,) * len(names), "identity", (1.0,), "global", DIGEST)


def _inputs():
    prediction = PredictionArtifact("obj", "yes", 0.9, (0.1, 0.9), "model-1")
    explanation = ExplanationArtifact(DIGEST, "explainer-1", "model-1", "plan-1", "dict-1", ("model", "explainer"))
    route = RouteArtifacts("prep-1", "cal-1", "population-1", "schema-1", DIGEST, ("model", "explainer"))
    context = DeploymentContext("model-1", "prep-1", "explainer-1", "cal-1", "population-1", "schema-1", "plan-1", "dict-1", DIGEST, ("model", "explainer"), 0.2, "policy")
    policy = ControllerV2Policy(
        "policy",
        _head("model_error", PREDICTIVE_FEATURES),
        _head("route_not_certifiable", ROUTE_FEATURES),
        _head("explanation_unstable_or_incomplete", EXPLANATION_FEATURES),
        _head("outside_deployment_envelope", SHIFT_FEATURES),
        ActionCostProfile(),
        True,
    )
    return prediction, explanation, route, context, policy


def test_repairable_blocking_fault_becomes_repair_then_retry() -> None:
    prediction, explanation, route, context, policy = _inputs()
    route = replace(route, preprocessing_version="wrong")
    result = assess_actions_v2((prediction,), (explanation,), (route,), context, 0.2, policy.costs, policy=policy)[0]
    assert result.action.value == "repair_then_retry"
    assert "preprocessing_version" in result.minimal_repair_set


def test_irreparable_fault_remains_blocked() -> None:
    prediction, explanation, route, context, policy = _inputs()
    route = replace(route, critical_data_quality_fault=True)
    result = assess_actions_v2((prediction,), (explanation,), (route,), context, 0.2, policy.costs, policy=policy)[0]
    assert result.action.value == "block"


def test_action_costs_include_repair_then_retry() -> None:
    costs = ActionCostProfile()
    assert costs.repair_then_retry > 0.0
    assert costs.repair_residual <= costs.full_review_residual
