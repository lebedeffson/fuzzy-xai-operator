from __future__ import annotations

import json
from dataclasses import asdict

from .contracts import AuditAction, OperationalDecision, PredictiveState, RepairPlan, RouteAssessment, RouteOutcome


class PredictiveSelector:
    def __init__(self, review_threshold: float) -> None:
        if not 0.0 <= review_threshold <= 1.0:
            raise ValueError("review threshold must be in [0, 1]")
        self.review_threshold = review_threshold

    def assess(self, risk: float) -> PredictiveState:
        if not 0.0 <= risk <= 1.0:
            raise ValueError("predictive risk must be in [0, 1]")
        return PredictiveState(risk, risk >= self.review_threshold, self.review_threshold)


class RepairPlanner:
    _REPAIRS = {
        "model_explainer_mismatch": "reload_matching_explainer",
        "stale_calibration": "refresh_calibration_artifact",
        "preprocessing_order_change": "restore_preprocessing_order",
        "feature_schema_incompatibility": "restore_feature_mapping",
        "cross_model_artifact_mix": "reload_single_model_artifact_set",
        "checksum_corruption": "restore_canonical_artifact",
        "reduction_link_loss": "restore_reduction_source_link",
        "reference_population_substitution": "restore_reference_population",
        "partial_provenance_deletion": "collect_missing_provenance",
        "dictionary_or_tokenizer_version_change": "restore_dictionary_or_tokenizer",
    }

    def plan(self, route: RouteAssessment) -> RepairPlan:
        actions = tuple(self._REPAIRS[item] for item in route.violations if item in self._REPAIRS)
        if route.outcome == RouteOutcome.INSUFFICIENT:
            actions = (*actions, "collect_required_evidence")
        return RepairPlan(tuple(dict.fromkeys(actions)), bool(actions) and not route.irreparable_fault)


class LexicographicController:
    def __init__(self, selector: PredictiveSelector, planner: RepairPlanner | None = None) -> None:
        self.selector = selector
        self.planner = planner or RepairPlanner()

    def decide(self, route: RouteAssessment, predictive_risk: float) -> OperationalDecision:
        predictive = self.selector.assess(predictive_risk)
        repair = self.planner.plan(route)
        if route.irreparable_fault:
            action = AuditAction.BLOCK
            reasons = ("IRREPARABLE_CONTRACT_FAULT", *route.reason_codes)
        elif route.repairable_fault or route.outcome in {RouteOutcome.UNKNOWN_FAULT, RouteOutcome.INSUFFICIENT}:
            action = AuditAction.REPAIR_THEN_RETRY
            reasons = ("REPAIR_AND_RECERTIFY", *route.reason_codes)
        elif predictive.requires_review:
            action = AuditAction.REVIEW
            reasons = ("PREDICTIVE_RISK_REVIEW",)
        else:
            action = AuditAction.ACCEPT
            reasons = ("ROUTE_CERTIFIED_AND_RISK_ACCEPTABLE",)
        payload = {
            "action": action.value,
            "predictive": asdict(predictive),
            "reason_codes": reasons,
            "repair_plan": asdict(repair),
            "route": {**asdict(route), "outcome": route.outcome.value},
            "trace_schema": "operational-audit-v16",
        }
        trace = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return OperationalDecision(action, reasons, route, predictive, repair, trace)
