from __future__ import annotations

from fuzzyxai.operational_audit import (
    AuditAction,
    LexicographicController,
    PredictiveSelector,
    RouteArtifact,
    TypedRouteGuard,
    mutate_route_artifact,
)


def artifact() -> RouteArtifact:
    return RouteArtifact(
        "artifact-1",
        "model-v1",
        "model-v1",
        "model-v1",
        ("01-load", "02-normalize", "03-predict"),
        ("a", "b"),
        ("a", "b"),
        "a" * 64,
        "a" * 64,
        "source-1",
        "source-1",
        "reference-1",
        "reference-1",
        ("model", "explainer", "reference"),
        ("model", "explainer", "reference"),
        "dictionary-v1",
        "dictionary-v1",
    )


def test_lexicographic_priority_and_repair_recertification() -> None:
    guard = TypedRouteGuard()
    controller = LexicographicController(PredictiveSelector(0.70))
    valid = guard.assess(artifact())
    assert controller.decide(valid, 0.20).action == AuditAction.ACCEPT
    assert controller.decide(valid, 0.80).action == AuditAction.REVIEW
    fault = guard.assess(mutate_route_artifact(artifact(), "stale_calibration", "subtle"))
    decision = controller.decide(fault, 0.99)
    assert decision.action == AuditAction.REPAIR_THEN_RETRY
    assert decision.repair_plan.requires_recertification
    assert "refresh_calibration_artifact" in decision.repair_plan.candidate_actions


def test_trace_is_byte_identical() -> None:
    guard = TypedRouteGuard()
    controller = LexicographicController(PredictiveSelector(0.70))
    route = guard.assess(mutate_route_artifact(artifact(), "reference_population_substitution", "moderate"))
    assert controller.decide(route, 0.4).audit_trace == controller.decide(route, 0.4).audit_trace


def test_composition_abstains_instead_of_forcing_family() -> None:
    guard = TypedRouteGuard()
    value = mutate_route_artifact(artifact(), "checksum_corruption", "moderate")
    value = mutate_route_artifact(value, "cross_model_artifact_mix", "moderate")
    assessment = guard.assess(value)
    assert assessment.family is None
    assert assessment.irreparable_fault
