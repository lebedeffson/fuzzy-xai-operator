from __future__ import annotations

import json

import fuzzyxai
from fuzzyxai.audit.operators_manifest import validate_manifest
from fuzzyxai.operators import (
    AlignmentInput,
    ReductionInput,
    RiskInput,
    compute_alignment,
    compute_reduction,
    observe_risk,
)


class ProbabilityModel:
    classes_ = [0, 1]

    def predict_proba(self, inputs):
        return [[0.12, 0.88] for _ in inputs]

    def predict(self, inputs):
        return [1 for _ in inputs]


def test_public_operators_delegate_to_typed_core() -> None:
    alignment = compute_alignment(
        AlignmentInput(
            components={"d_mu": 0.2, "d_R": 0.4},
            weights={"d_mu": 0.5, "d_R": 0.5},
            gamma_max=0.35,
        )
    )
    reduction = compute_reduction(
        ReductionInput(
            components={"term_loss": 0.1},
            weights={"term_loss": 1.0},
            delta_max=0.2,
            kappa_delta=2.0,
        )
    )
    risk = observe_risk(
        RiskInput(
            components={"uncertainty": 0.2},
            weights={"uncertainty": 1.0},
            thresholds={"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
        )
    )

    assert alignment.gamma == 0.3
    assert alignment.certified is True
    assert reduction.delta == 0.1
    assert reduction.r_delta == 0.2
    assert risk.rho == 0.2
    assert risk.action == "lower_confidence"


def test_model_independent_wrap_is_honest_without_operator_evidence(tmp_path) -> None:
    result = fuzzyxai.FuzzyXAI.wrap(ProbabilityModel()).explain([[1.0, 2.0]])

    assert result.prediction.primary_score() == 0.88
    assert result.action == "review"
    assert {item["code"] for item in result.view_model.diagnostics} == {
        "D_k_alignment_missing",
        "D_k_reduction_missing",
        "D_k_risk_missing",
    }
    output = result.export_json(tmp_path / "explanation.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trace"]["adapter_id"] == "predict_proba"
    assert payload["risk"]["rho"] is None


def test_model_independent_wrap_computes_supplied_operator_evidence(tmp_path) -> None:
    result = fuzzyxai.FuzzyXAI.wrap(ProbabilityModel(), adapter="sklearn").explain(
        [[1.0, 2.0]],
        evidence={
            "contributions": {"feature_a": 0.31, "feature_b": -0.06},
            "memberships": {"medium": 0.34, "high": 0.71},
            "alignment": {
                "components": {"d_mu": 0.2, "d_R": 0.4},
                "weights": {"d_mu": 0.5, "d_R": 0.5},
                "gamma_max": 0.35,
            },
            "reduction": {
                "components": {"term_loss": 0.1},
                "weights": {"term_loss": 1.0},
                "delta_max": 0.2,
            },
            "risk": {
                "components": {"uncertainty": 0.2},
                "weights": {"uncertainty": 1.0},
                "thresholds": {"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
            },
        },
    )

    assert result.view_model.disagreement["gamma"] == 0.3
    assert result.view_model.disagreement["delta"] == 0.1
    assert result.view_model.risk["rho"] == 0.2
    assert result.action == "lower_confidence"
    dashboard = result.plot(tmp_path / "dashboard.png")
    assert dashboard.exists() and dashboard.stat().st_size > 0


def test_structural_failure_cannot_be_hidden_by_low_risk() -> None:
    result = fuzzyxai.FuzzyXAI.wrap(ProbabilityModel()).explain(
        [[1.0, 2.0]],
        evidence={
            "alignment": {
                "components": {"d_mu": 0.9},
                "weights": {"d_mu": 1.0},
                "gamma_max": 0.2,
            },
            "reduction": {
                "components": {"term_loss": 0.8},
                "weights": {"term_loss": 1.0},
                "delta_max": 0.2,
            },
            "risk": {
                "components": {"uncertainty": 0.01},
                "weights": {"uncertainty": 1.0},
                "thresholds": {"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
            },
        },
    )

    assert result.action == "review"
    assert {item["code"] for item in result.view_model.diagnostics} == {"D_ij_alignment", "D_reduction"}


def test_operator_manifest_is_complete_and_resolvable() -> None:
    report = validate_manifest()
    assert report["status"] == "PASS", report["errors"]
    assert report["operator_count"] == 39
    assert {
        "h10_c5.natural_incident_route_audit",
        "h10_c6.cut_robustness",
        "multimodal.interpretable_route_validation",
        "h9.end_to_end_latency",
    }.issubset(report["operator_ids"])


def test_public_exports_are_not_overwritten() -> None:
    import fuzzyxai.adapters as adapters
    import fuzzyxai.operators as operators

    assert {"compute_alignment", "compute_reduction", "observe_risk", "list_operators"} <= set(operators.__all__)
    assert {"MedicalImageToExplanationAdapter", "ModelAdapter", "SklearnAdapter", "list_adapters"} <= set(adapters.__all__)
