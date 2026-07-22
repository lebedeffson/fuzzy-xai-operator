"""Public hierarchical practical-controller v2 API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from fuzzyxai.audit_certificate import ActionConditionedAuditCertificate, build_action_certificate
from fuzzyxai.diagnostic_cut import MinimalDiagnosticCut, build_repair_set, graph_from_certificate, solve_approximate, solve_exact
from fuzzyxai.practical_controller import DeploymentContext, ExplanationArtifact, HardGuardStatus, PredictionArtifact, RouteArtifacts
from fuzzyxai.practical_controller.guards import evaluate_hard_guard
from fuzzyxai.selective_observer import SelectiveAction

from .budget_optimizer import BudgetCandidate, optimize_review_budget
from .calibration import CalibratedRiskHead
from .expected_loss import ActionCostProfile, ExpectedActionLosses, expected_action_losses
from .explanation_head import estimate_explanation_risk
from .predictive_head import estimate_predictive_risk
from .route_head import estimate_route_risk
from .shift_head import estimate_shift_risk


@dataclass(frozen=True)
class ControllerV2Policy:
    policy_version: str
    predictive_head: CalibratedRiskHead
    route_head: CalibratedRiskHead
    explanation_head: CalibratedRiskHead
    shift_head: CalibratedRiskHead
    costs: ActionCostProfile
    selected_without_test: bool
    false_block_ceiling: float = 0.01
    hard_fault_recall_minimum: float = 0.95

    def __post_init__(self) -> None:
        targets = (
            self.predictive_head.target_name,
            self.route_head.target_name,
            self.explanation_head.target_name,
            self.shift_head.target_name,
        )
        expected = ("model_error", "route_not_certifiable", "explanation_unstable_or_incomplete", "outside_deployment_envelope")
        if targets != expected:
            raise ValueError("controller heads must retain four distinct frozen targets")
        if not self.selected_without_test:
            raise ValueError("controller policy must be selected without confirmatory test")


@dataclass(frozen=True)
class ActionAssessmentV2:
    action: SelectiveAction
    predictive_risk: float
    route_risk: float
    explanation_risk: float
    shift_risk: float
    certificate_status: str
    certificate: ActionConditionedAuditCertificate
    diagnostic_cut: MinimalDiagnosticCut
    minimal_repair_set: tuple[str, ...]
    expected_action_losses: ExpectedActionLosses
    review_priority: float
    reason_codes: tuple[str, ...]
    trace_id: str
    policy_version: str
    budget_feasible: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["action"] = self.action.value
        return result


def assess_action_v2(
    prediction_artifact: PredictionArtifact,
    explanation_artifact: ExplanationArtifact,
    route_artifacts: RouteArtifacts,
    deployment_context: DeploymentContext,
    review_budget: float,
    cost_profile: ActionCostProfile,
    *,
    policy: ControllerV2Policy,
) -> ActionAssessmentV2:
    return assess_actions_v2(
        (prediction_artifact,),
        (explanation_artifact,),
        (route_artifacts,),
        deployment_context,
        review_budget,
        cost_profile,
        policy=policy,
    )[0]


def assess_actions_v2(
    predictions: Sequence[PredictionArtifact],
    explanations: Sequence[ExplanationArtifact],
    routes: Sequence[RouteArtifacts],
    deployment_context: DeploymentContext,
    review_budget: float,
    cost_profile: ActionCostProfile,
    *,
    policy: ControllerV2Policy,
) -> tuple[ActionAssessmentV2, ...]:
    if not predictions or not (len(predictions) == len(explanations) == len(routes)):
        raise ValueError("assessment inputs must be non-empty and aligned")
    pending = []
    candidates = []
    for index, (prediction, explanation, route) in enumerate(zip(predictions, explanations, routes, strict=True)):
        guard = evaluate_hard_guard(prediction, explanation, route, deployment_context)
        certificate = build_action_certificate(prediction, explanation, route, deployment_context)
        graph = graph_from_certificate(certificate)
        cut = solve_exact(graph) if len(graph.contracts) <= 22 else solve_approximate(graph)
        features = certificate.features(minimal_cut_size=len(cut.contracts), minimal_repair_cost=cut.total_repair_cost)
        risks = (
            estimate_predictive_risk(policy.predictive_head, prediction),
            estimate_route_risk(policy.route_head, features),
            estimate_explanation_risk(policy.explanation_head, explanation),
            estimate_shift_risk(policy.shift_head, prediction),
        )
        hard_probability = 1.0 if guard.status is HardGuardStatus.BLOCKED else 0.0
        losses = expected_action_losses(*risks, hard_fault_probability=hard_probability, costs=cost_profile)
        pending.append((guard, certificate, cut, risks, losses))
        candidates.append(BudgetCandidate(index=index, losses=losses, hard_guard_status=guard.status))
    actions, feasible = optimize_review_budget(candidates, review_budget=review_budget)
    result = []
    for index, (guard, certificate, cut, risks, losses) in enumerate(pending):
        repairs = build_repair_set(cut)
        action = actions[index]
        reason_codes = tuple(dict.fromkeys((*guard.reason_codes, f"ACTION:{action.value}", *(f"REPAIR:{item.contract_id}" for item in repairs))))
        payload = {
            "object_id": predictions[index].object_id,
            "certificate_sha256": certificate.sha256,
            "risks": risks,
            "losses": losses.as_dict(),
            "action": action.value,
            "policy_version": policy.policy_version,
        }
        trace_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append(
            ActionAssessmentV2(
                action=action,
                predictive_risk=risks[0],
                route_risk=risks[1],
                explanation_risk=risks[2],
                shift_risk=risks[3],
                certificate_status="certified" if certificate.certificate_exists else "not_certified",
                certificate=certificate,
                diagnostic_cut=cut,
                minimal_repair_set=tuple(item.contract_id for item in repairs),
                expected_action_losses=losses,
                review_priority=candidates[index].marginal_review_benefit,
                reason_codes=reason_codes,
                trace_id=trace_id,
                policy_version=policy.policy_version,
                budget_feasible=feasible,
            )
        )
    return tuple(result)
