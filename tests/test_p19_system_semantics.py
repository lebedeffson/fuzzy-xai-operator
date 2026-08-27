from __future__ import annotations

from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.evidence import ExplanationEvidence
from fuzzyxai.evidence.graph import build_explanation_graph
from fuzzyxai.hierarchy.interval import IntervalFS
from fuzzyxai.system_semantics import aggregate_uncertainty, compute_strict_rho, reduce_interval_representation


def test_p19_uncertainty_is_explicit_three_channel_aggregation() -> None:
    evidence = aggregate_uncertainty(
        u_model=0.2, u_rules=0.1, u_trace=0.3, plan=ExplainPlan(),
        required={"model": True, "rules": True, "trace": True},
    )
    assert evidence.status == "measured"
    assert evidence.u_m == 0.19


def test_p19_missing_required_uncertainty_keeps_u_m_none() -> None:
    evidence = aggregate_uncertainty(
        u_model=0.2, u_rules=None, u_trace=0.3, plan=ExplainPlan(),
        required={"model": True, "rules": True, "trace": True},
    )
    assert evidence.status == "incomplete"
    assert evidence.u_m is None


def test_p19_delta_is_measured_from_interval_reduction_and_inverse_embedding() -> None:
    source = IntervalFS(lambda _x: 0.2, lambda _x: 0.6)
    _, reduction = reduce_interval_representation(source, delta_threshold=0.5)
    assert reduction.operation == "Pi_interval_midpoint"
    assert reduction.inverse_embedding == "iota_F0_to_interval_diagonal"
    assert reduction.delta == 0.2


def test_p19_strict_rho_does_not_rename_partial_score() -> None:
    result = compute_strict_rho(
        rho_p=0.1, u_m=None, i_pre=0.9, delta=0.1, chi_r=0.0,
        weights={"w_p": 0.3, "w_u": 0.25, "w_I": 0.2, "w_Delta": 0.15, "w_R": 0.1},
        thresholds={"theta_1": 0.35, "theta_2": 0.6, "theta_3": 0.85, "theta_4": 0.95}, critical=0,
    )
    assert result.status == "incomplete"
    assert result.rho is None
    assert result.partial_risk_score is not None


def test_p19_critical_rupture_cannot_auto_accept() -> None:
    result = compute_strict_rho(
        rho_p=0.0, u_m=0.0, i_pre=1.0, delta=0.0, chi_r=0.0,
        weights={"w_p": 0.3, "w_u": 0.25, "w_I": 0.2, "w_Delta": 0.15, "w_R": 0.1},
        thresholds={"theta_1": 0.35, "theta_2": 0.6, "theta_3": 0.85, "theta_4": 0.95}, critical=1,
    )
    assert result.status == "complete"
    assert result.rho == 0.0
    assert result.candidate_action == "accept"
    assert result.critical_override is True
    assert result.action == "block"


def test_p19_zero_weight_missing_component_is_not_required() -> None:
    result = compute_strict_rho(
        rho_p=0.1, u_m=None, i_pre=0.9, delta=0.1, chi_r=0.0,
        weights={"w_p": 0.4, "w_u": 0.0, "w_I": 0.3, "w_Delta": 0.2, "w_R": 0.1},
        thresholds={"theta_1": 0.35, "theta_2": 0.6, "theta_3": 0.85, "theta_4": 0.95}, critical=0,
    )
    assert result.status == "complete"
    assert result.rho == 0.09
    assert result.components["u_M"] is None


def test_p19_four_threshold_action_regions_are_distinct() -> None:
    weights = {"w_p": 1.0, "w_u": 0.0, "w_I": 0.0, "w_Delta": 0.0, "w_R": 0.0}
    thresholds = {"theta_1": 0.2, "theta_2": 0.4, "theta_3": 0.6, "theta_4": 0.8}
    values = ((0.1, "accept"), (0.3, "lower_confidence"), (0.5, "request_more_data"), (0.7, "defer_to_human"), (0.9, "block"))
    for rho_p, expected in values:
        result = compute_strict_rho(
            rho_p=rho_p, u_m=None, i_pre=None, delta=None, chi_r=None,
            weights=weights, thresholds=thresholds, critical=0,
        )
        assert result.status == "complete"
        assert result.candidate_action == expected
        assert result.action == expected
        assert result.candidate_action_reason
        assert result.thresholds == thresholds


def test_p19_graph_separates_five_term_rho_candidate_and_override() -> None:
    system = {
        "E_model": {}, "alignment_transform": {}, "aligned_E_model": {}, "E_target": {}, "gamma": {},
        "uncertainty": {}, "representation": "F_int", "reduction": {}, "E_pre": {}, "i_pre": {"value": 0.9},
        "risk": {
            "rho": 0.18,
            "components": {"rho_p": 0.0, "u_M": 0.2, "one_minus_I_pre": 0.1, "Delta": 0.0, "chi_R": 1.0},
            "candidate_action": "accept",
            "critical_override": True,
            "action": "block",
        },
    }
    graph = build_explanation_graph(
        ExplanationEvidence(), prediction={"predictions": [1]}, diagnostics=(), action="block", system_evidence=system,
    )
    pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert {source for source, target in pairs if target == "system:rho"} == {
        "system:rho_p", "system:u_M", "system:one_minus_I_pre", "system:Delta", "system:chi_R",
    }
    assert ("system:I_pre", "system:rho") not in pairs
    assert ("system:rho", "system:threshold_policy") in pairs
    assert ("system:threshold_policy", "system:candidate_action") in pairs
    assert ("system:critical", "system:critical_override") in pairs
    assert ("system:candidate_action", "system:policy_resolution") in pairs
    assert ("system:critical_override", "system:policy_resolution") in pairs
    assert ("system:policy_resolution", "action") in pairs
    assert ("system:rho", "action") not in pairs
