from __future__ import annotations

import json

import fuzzyxai
import pytest
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
            components={
                "rho_p": 0.1,
                "u_M": 0.2,
                "one_minus_I_pre": 0.1,
                "Delta": 0.0,
                "chi_R": 0.0,
            },
            weights={
                "rho_p": 0.3,
                "u_M": 0.25,
                "one_minus_I_pre": 0.2,
                "Delta": 0.15,
                "chi_R": 0.1,
            },
            thresholds={"theta_1": 0.1, "theta_2": 0.3, "theta_3": 0.6, "theta_4": 0.8},
        )
    )

    assert alignment.gamma == 0.3
    assert alignment.certified is True
    assert reduction.delta == 0.1
    assert reduction.r_delta == 0.2
    assert risk.rho == pytest.approx(0.1)
    assert risk.action == "lower_confidence"


def test_model_independent_wrap_is_honest_without_operator_evidence(tmp_path) -> None:
    """Without manual `evidence={"alignment"/"reduction"/"risk": ...}`, the
    automatic Γ/Δ/ρ layer (P16/P17) still only uses *real* available
    signal — a real predict_proba score gives a real predicted_risk
    component of ρ (predicted_risk=1-score=0.12, diagnostic=0, since no
    rupture was detected), but there is no local-contribution channel for
    this bare predict_proba model, so I_pre stays unmeasured. Per P18, a
    default ExplainPlan never declares alignment/reduction applicable for a
    single-channel model, so those are honestly not_applicable (not
    "missing") — but interpretability_gap IS always an expected component of
    the canonical 5-term formula, and it has no source here, so the risk
    interface is incomplete and disclosed as such rather than silently
    renormalized into a confident "accept"."""

    result = fuzzyxai.FuzzyXAI.wrap(ProbabilityModel()).explain([[1.0, 2.0]])

    assert result.prediction.primary_score() == 0.88
    assert result.action == "insufficient_evidence"
    assert {item["code"] for item in result.view_model.diagnostics} == {
        "D_k_alignment_not_applicable",
        "D_k_reduction_not_applicable",
        "D_risk_incomplete_interface",
    }
    output = result.export_json(tmp_path / "explanation.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["trace"]["adapter_id"] == "predict_proba"
    assert payload["risk"]["components"] == {"predicted_risk": pytest.approx(0.12), "diagnostic": pytest.approx(0.0)}
    assert payload["risk"]["status"] == "incomplete"
    # P18 item 2: an incomplete interface never reports its renormalized
    # weighted average as the real, complete rho.
    assert payload["risk"]["rho"] is None
    # partial_risk_score = (0.30*0.12 + 0.10*0.0) / (0.30+0.10) under DEFAULT_RISK_WEIGHTS, renormalized over the two available components.
    assert payload["risk"]["partial_risk_score"] == pytest.approx(0.09)
    assert payload["disagreement"]["gamma"] is None  # no second explanatory channel for this bare model
    assert payload["disagreement"]["pre_interpretability"] is None  # no local-contribution channel to build an explanatory object from


def test_model_independent_wrap_rejects_untransformed_gamma_evidence(tmp_path) -> None:
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

    assert result.view_model.disagreement["gamma"] is None
    assert result.view_model.disagreement["alignment_status"] == "missing"
    assert result.view_model.disagreement["delta"] == 0.1
    assert result.view_model.risk["rho"] == 0.2
    assert result.action == "insufficient_evidence"
    dashboard = result.plot(tmp_path / "dashboard.png")
    assert dashboard.exists() and dashboard.stat().st_size > 0


def test_untransformed_structural_evidence_is_insufficient_not_a_fake_gamma() -> None:
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

    assert result.action == "insufficient_evidence"
    assert {item["code"] for item in result.view_model.diagnostics} == {"D_k_alignment_missing", "D_reduction"}


def test_operator_manifest_is_complete_and_resolvable() -> None:
    report = validate_manifest()
    assert report["status"] == "PASS", report["errors"]
    assert report["operator_count"] == 41
    assert {
        "h10_c5.natural_incident_route_audit",
        "h10_c6.cut_robustness",
        "multimodal.interpretable_route_validation",
        "h9.end_to_end_latency",
    }.issubset(report["operator_ids"])


def test_public_exports_are_not_overwritten() -> None:
    from fuzzyxai import adapters, operators

    assert {"compute_alignment", "compute_reduction", "observe_risk", "list_operators"} <= set(operators.__all__)
    assert {"MedicalImageToExplanationAdapter", "ModelAdapter", "SklearnAdapter", "list_adapters"} <= set(adapters.__all__)
