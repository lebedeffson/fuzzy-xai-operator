"""Public practical-controller API, batch allocation, streaming and replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Iterable, Iterator, Sequence

from fuzzyxai.selective_observer import SelectiveAction

from .contracts import (
    ActionAssessment,
    BatchAssessment,
    CostProfile,
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
    ensure_unique_object_ids,
)
from .guards import evaluate_hard_guard
from .optimizer import ActionCandidate, allocate_actions
from .risk import combine_operational_risk, estimate_predictive_risk, estimate_route_risk, risk_interval


def assess_action(
    prediction_artifact: PredictionArtifact,
    explanation_artifact: ExplanationArtifact,
    route_artifacts: RouteArtifacts,
    deployment_context: DeploymentContext,
    review_budget: ReviewBudget,
    cost_profile: CostProfile,
    *,
    policy: PracticalPolicy,
) -> ActionAssessment:
    batch = assess_batch(
        (prediction_artifact,),
        (explanation_artifact,),
        (route_artifacts,),
        deployment_context,
        review_budget,
        cost_profile,
        policy=policy,
    )
    assessment = batch.assessments[0]
    budget_available = review_budget.current_review_fraction < review_budget.fraction
    if (
        assessment.hard_guard_status is HardGuardStatus.CERTIFIED
        and assessment.operational_risk > policy.accept_max_risk
        and budget_available
    ):
        action = (
            SelectiveAction.FULL_REVIEW
            if assessment.operational_risk > policy.short_review_max_risk
            else SelectiveAction.SHORT_REVIEW
        )
        reasons = tuple(code for code in assessment.reason_codes if code != "BUDGET_CONSTRAINED_ACCEPT") + (
            "SINGLE_REQUEST_REVIEW_BUDGET_AVAILABLE",
        )
        assessment = replace(assessment, action=action, reason_codes=reasons, budget_feasible=True, deterministic_replay_sha256="")
        assessment = replace(
            assessment,
            deterministic_replay_sha256=hashlib.sha256(_canonical_json(_assessment_replay_payload(assessment))).hexdigest(),
        )
    return assessment


def assess_batch(
    predictions: Sequence[PredictionArtifact],
    explanations: Sequence[ExplanationArtifact],
    routes: Sequence[RouteArtifacts],
    deployment_context: DeploymentContext,
    review_budget: ReviewBudget,
    cost_profile: CostProfile,
    *,
    policy: PracticalPolicy,
) -> BatchAssessment:
    if not predictions or not (len(predictions) == len(explanations) == len(routes)):
        raise ValueError("assessment inputs must be non-empty and aligned")
    ensure_unique_object_ids(predictions)
    preliminary = []
    candidates = []
    for index, (prediction, explanation, route) in enumerate(zip(predictions, explanations, routes, strict=True)):
        guard = evaluate_hard_guard(prediction, explanation, route, deployment_context)
        predictive = estimate_predictive_risk(policy, prediction)
        route_risk = estimate_route_risk(
            policy,
            explanation,
            route,
            required_channels=deployment_context.mandatory_provenance_channels,
        )
        operational = combine_operational_risk(predictive, route_risk)
        preliminary.append((guard, predictive, route_risk, operational))
        candidates.append(ActionCandidate(index=index, risk=operational, guard=guard))
    available_budget = max(0.0, review_budget.fraction - review_budget.current_review_fraction)
    actions, allocation_reasons, feasible = allocate_actions(
        candidates,
        review_budget=available_budget,
        cost_profile=cost_profile,
        policy=policy,
    )
    assessments = []
    for index, (prediction, explanation, route) in enumerate(zip(predictions, explanations, routes, strict=True)):
        guard, predictive, route_risk, operational = preliminary[index]
        reasons = tuple(dict.fromkeys((*guard.reason_codes, *allocation_reasons[index])))
        trace = _trace_id(prediction, explanation, route, deployment_context, review_budget, cost_profile, policy)
        assessment = ActionAssessment(
            action=actions[index],
            operational_risk=operational,
            predictive_risk=predictive,
            route_risk=route_risk,
            hard_guard_status=guard.status,
            reason_codes=reasons,
            missing_evidence=guard.missing_evidence,
            review_priority=operational,
            confidence_interval=risk_interval(operational),
            trace_id=trace,
            model_version=prediction.model_version,
            explain_plan_version=explanation.explain_plan_version,
            policy_version=policy.policy_version,
            budget_feasible=feasible,
            deterministic_replay_sha256="",
            monitoring_events=deployment_context.monitoring_hooks,
        )
        replay_hash = hashlib.sha256(_canonical_json(_assessment_replay_payload(assessment))).hexdigest()
        assessments.append(replace(assessment, deterministic_replay_sha256=replay_hash))
    review_count = sum(item.action in {SelectiveAction.SHORT_REVIEW, SelectiveAction.FULL_REVIEW} for item in assessments)
    audit_payload = [item.to_dict() for item in assessments]
    return BatchAssessment(
        assessments=tuple(assessments),
        review_budget=review_budget.fraction,
        realized_review_fraction=review_count / len(assessments),
        budget_feasible=feasible,
        policy_version=policy.policy_version,
        audit_sha256=hashlib.sha256(_canonical_json(audit_payload)).hexdigest(),
    )


def assess_stream(
    items: Iterable[tuple[PredictionArtifact, ExplanationArtifact, RouteArtifacts]],
    deployment_context: DeploymentContext,
    review_budget: ReviewBudget,
    cost_profile: CostProfile,
    *,
    policy: PracticalPolicy,
    batch_size: int = 1024,
) -> Iterator[ActionAssessment]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pending: list[tuple[PredictionArtifact, ExplanationArtifact, RouteArtifacts]] = []
    for item in items:
        pending.append(item)
        if len(pending) >= batch_size:
            yield from _assess_pending(pending, deployment_context, review_budget, cost_profile, policy)
            pending = []
    if pending:
        yield from _assess_pending(pending, deployment_context, review_budget, cost_profile, policy)


def verify_replay(assessment: ActionAssessment) -> bool:
    return hashlib.sha256(_canonical_json(_assessment_replay_payload(assessment))).hexdigest() == assessment.deterministic_replay_sha256


def _assess_pending(
    pending: Sequence[tuple[PredictionArtifact, ExplanationArtifact, RouteArtifacts]],
    context: DeploymentContext,
    budget: ReviewBudget,
    costs: CostProfile,
    policy: PracticalPolicy,
) -> tuple[ActionAssessment, ...]:
    predictions, explanations, routes = zip(*pending, strict=True)
    return assess_batch(predictions, explanations, routes, context, budget, costs, policy=policy).assessments


def _trace_id(
    prediction: PredictionArtifact,
    explanation: ExplanationArtifact,
    route: RouteArtifacts,
    context: DeploymentContext,
    budget: ReviewBudget,
    costs: CostProfile,
    policy: PracticalPolicy,
) -> str:
    payload = {
        "prediction": prediction.__dict__,
        "explanation": explanation.__dict__,
        "route": route.__dict__,
        "context": {**context.__dict__, "mode": context.mode.value},
        "budget": budget.__dict__,
        "cost_profile": {**costs.__dict__, "name": costs.name.value},
        "policy": policy.__dict__,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()


def _assessment_replay_payload(assessment: ActionAssessment) -> dict[str, object]:
    payload = assessment.to_dict()
    payload.pop("deterministic_replay_sha256", None)
    return payload
