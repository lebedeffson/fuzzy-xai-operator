"""P19 system-operator contracts: uncertainty, reduction, E_pre, and rho."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fuzzyxai.core.explain_plan import ExplainPlan, MembershipPolicy
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.hierarchy.interval import IntervalFS
from fuzzyxai.hierarchy.reductions import reduce_to_f0
from fuzzyxai.scientific_alignment import AlignmentTransform, compute_real_alignment
from fuzzyxai.trust.trust_evaluator import (
    compute_interpretability_index,
    compute_interpretability_loss,
    entropy_component,
    rule_complexity_component,
    rule_contradiction_component,
    term_overlap_component,
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class UncertaintyEvidence:
    u_model: float | None
    u_rules: float | None
    u_trace: float | None
    u_m: float | None
    status: str
    weights: dict[str, float]
    sources: dict[str, Any]


@dataclass(frozen=True)
class SystemSourceEvidence:
    """Factual, domain-neutral source object consumed by the system operator."""

    source_interface_id: str
    terms: tuple[str, ...]
    representation_value: float
    representation_label: str
    rules: tuple[Rule, ...]
    activations: dict[str, float]
    model_uncertainty_inputs: dict[str, Any]
    trace: Trace
    source_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_interface_id": self.source_interface_id,
            "terms": list(self.terms),
            "representation_value": self.representation_value,
            "representation_label": self.representation_label,
            "rules": [rule.signature() for rule in self.rules],
            "activations": dict(self.activations),
            "model_uncertainty_inputs": dict(self.model_uncertainty_inputs),
            "trace": self.trace.as_dict(),
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SystemObservation:
    """Factual inputs for one declared system-operator route.

    This belongs in :class:`ObservationContext`: it records the target
    interface and observed trace/rule facts for this run.  Thresholds,
    weights, and whether the operators are allowed remain in ExplainPlan.
    """

    alignment_transform: AlignmentTransform
    risk_membership_policy: MembershipPolicy
    risk_class: Any
    trace: Trace
    model_trace: Trace | None = None
    trace_complete: bool = True
    trace_verification_source: str = "ObservationContext externally verified trace status"
    rule_conflict: float = 0.0
    rules_applicable: bool = True
    source_refs: tuple[str, ...] = ()
    rupture_present: bool | None = None
    critical_rupture: bool | None = None
    rupture_code: str = "registered_system_rupture"
    rupture_source_refs: tuple[str, ...] = ()


def _object_dict(value: ExplanationObject) -> dict[str, Any]:
    return {
        "terms": sorted(value.terms),
        "representation": getattr(value.representation, "label", type(value.representation).__name__),
        "rules": [rule.signature() for rule in value.rules],
        "activations": dict(value.activations),
        "uncertainty": value.uncertainty,
        "trace": value.trace.as_dict(),
        "reduction_loss": value.reduction_loss,
        "metadata": dict(value.metadata),
    }


@dataclass(frozen=True)
class SystemEvidence:
    """Canonical persisted result of one executed system-operator route."""

    source_evidence: SystemSourceEvidence
    e_model: ExplanationObject
    alignment_transform: Any
    aligned_explanation: ExplanationObject
    e_target: ExplanationObject
    alignment: dict[str, Any]
    uncertainty: UncertaintyEvidence
    representation: Any
    representation_policy: dict[str, Any]
    reduction: ReductionEvidence | None
    reduction_status: str
    e_pre: ExplanationObject
    i_pre: float
    risk: SystemRisk
    diagnostics: tuple[dict[str, Any], ...]

    def audit_dict(self) -> dict[str, Any]:
        gamma = {key: value for key, value in self.alignment.items() if key != "aligned"}
        if "d_L" in gamma.get("components", {}):
            gamma["d_L_aggregation_status"] = (
                "diagnostic_only_not_included_in_Gamma_under_ExplainPlan.beta"
            )
        return {
            "system_source": self.source_evidence.to_dict(),
            "E_model": _object_dict(self.e_model),
            "alignment_transform": self.alignment_transform.to_dict(),
            "aligned_E_model": _object_dict(self.aligned_explanation),
            "E_target": _object_dict(self.e_target),
            "gamma": gamma,
            "uncertainty": self.uncertainty.__dict__,
            "representation": {
                "type": "F_int" if isinstance(self.representation, IntervalFS) else getattr(self.representation, "class_name", type(self.representation).__name__),
                "policy": dict(self.representation_policy),
            },
            "reduction": (
                {"status": self.reduction_status, "delta": 0.0}
                if self.reduction is None
                else self.reduction.__dict__
            ),
            "E_pre": _object_dict(self.e_pre),
            "i_pre": {"value": self.i_pre, "formula": "exp(-L(E_pre))", "weights": self.e_pre.metadata.get("interpretability_weights", {}), "components": self.e_pre.metadata.get("interpretability_components", {}), "input_refs": ["E_pre", "ExplainPlan.lambda"]},
            "risk": dict(self.risk.__dict__),
            "diagnostics": list(self.diagnostics),
        }


def build_system_evidence(
    *,
    object_id: str,
    model_fingerprint: str,
    source: SystemSourceEvidence,
    plan: ExplainPlan,
    observation: SystemObservation,
) -> SystemEvidence:
    """Run the P19 operator chain once from native evidence and factual context."""

    if not plan.alignment_policy.applicable:
        raise ValueError("ExplainPlan alignment_policy is not applicable; T_ij was not executed")
    if not plan.alignment_policy.transform:
        raise ValueError("ExplainPlan alignment_policy is applicable but has no registered transform")
    plan_transform = AlignmentTransform.from_dict(plan.alignment_policy.transform)
    if plan_transform.to_dict() != observation.alignment_transform.to_dict():
        raise ValueError("ObservationContext transform conflicts with ExplainPlan alignment transform")
    transform = plan_transform

    plan_membership = (
        plan.membership_policies.get("system_risk")
        or plan.membership_policies.get(observation.risk_membership_policy.variable)
    )
    if plan_membership is None:
        raise ValueError("ExplainPlan must own the system risk membership policy")
    if plan_membership.to_dict() != observation.risk_membership_policy.to_dict():
        raise ValueError("ObservationContext membership policy conflicts with ExplainPlan")
    membership_policy = plan_membership

    if source.source_interface_id != transform.source_interface:
        raise ValueError("system source interface conflicts with ExplainPlan alignment transform")
    source_value = _clip01(source.representation_value)
    u_model, uncertainty_method, uncertainty_inputs, uncertainty_formula = _system_model_uncertainty(
        plan=plan,
        source=source,
    )
    if u_model is None:
        raise ValueError(
            f"ExplainPlan uncertainty method {plan.uncertainty_policy.method!r} "
            "is not applicable or lacks required evidence"
        )
    e_model = ExplanationObject(
        terms=set(source.terms),
        representation=F0(lambda _x, value=source_value: value, source.representation_label),
        rules=list(source.rules),
        activations=dict(source.activations),
        uncertainty=u_model, trace=source.trace,
        metadata={"interface": source.source_interface_id, "system_source": source.to_dict(), **dict(source.metadata or {})},
    )
    memberships = membership_policy.evaluate(source_value)
    strongest = max(memberships.values())
    target_terms = {f"risk:{label}" for label in memberships}
    target_consequents = dict(plan.metadata.get("system_target_consequents", {}))
    e_target = ExplanationObject(
        terms=target_terms,
        representation=F0(lambda _x, value=strongest: value, "risk_membership_partition"),
        rules=[
            Rule(f"target_{label}", {membership_policy.variable: label}, str(target_consequents.get(label, f"target_state:{label}")))
            for label in memberships
        ],
        activations={f"target_{label}": value for label, value in memberships.items()},
        uncertainty=_clip01(1.0 - strongest), trace=observation.trace,
        metadata={"interface": transform.target_interface, "membership_policy": membership_policy.to_dict(), "memberships": memberships},
    )
    alignment = compute_real_alignment(e_model, e_target, plan=plan, transform=transform)
    if alignment.get("gamma") is None or not isinstance(alignment.get("aligned"), ExplanationObject):
        raise ValueError(f"system alignment is incomplete: {alignment.get('missing_components') or alignment.get('reason')}")
    aligned = alignment["aligned"]
    u_rules = None if not observation.rules_applicable else _clip01(max(float(observation.rule_conflict), 1.0 - strongest))
    required_trace_fields = ("id", "version", "timestamp", "source", "checksum")
    trace_values = observation.trace.as_dict()
    missing_trace_fields = [name for name in required_trace_fields if not trace_values.get(name)]
    externally_verified = bool(observation.trace_complete)
    u_trace = 0.0 if externally_verified and not missing_trace_fields else 1.0
    uncertainty = aggregate_uncertainty(
        u_model=u_model, u_rules=u_rules, u_trace=u_trace, plan=plan,
        required={"model": True, "rules": observation.rules_applicable, "trace": True},
    )
    if uncertainty.u_m is None:
        raise ValueError("system route has incomplete uncertainty profile")
    representation_policy = plan.uncertainty_representation_policy
    if not representation_policy.applicable:
        raise ValueError("ExplainPlan uncertainty representation policy is not applicable")
    low_clip, high_clip = representation_policy.clip
    width = representation_policy.scale * u_model
    interval = IntervalFS(
        lambda _x, value=source_value, width=width, low=low_clip: max(low, value - width),
        lambda _x, value=source_value, width=width, high=high_clip: min(high, value + width),
    )
    if plan.reduction_policy.applicable:
        if plan.reduction_policy.method != "F_int_to_F0_midpoint":
            raise ValueError(f"unsupported ExplainPlan reduction method: {plan.reduction_policy.method}")
        _, reduction = reduce_interval_representation(interval, delta_threshold=plan.delta_critical)
        reduction_status = reduction.status
        delta_for_risk = reduction.delta
    else:
        reduction = None
        reduction_status = "not_applied"
        delta_for_risk = 0.0
    uncertainty_values = {"model": u_model, "rules": u_rules, "trace": u_trace}
    aggregation_terms: dict[str, float] = {}
    for key, weight in uncertainty.weights.items():
        component = uncertainty_values[key]
        if component is not None:
            aggregation_terms[key] = weight * component
    uncertainty = UncertaintyEvidence(
        uncertainty.u_model, uncertainty.u_rules, uncertainty.u_trace, uncertainty.u_m, uncertainty.status, uncertainty.weights,
        {"U_model": {"value": u_model, "status": "measured", "method": uncertainty_method, "inputs": {**uncertainty_inputs, "risk_coordinate": source_value, "expected_range": [0.0, 0.5] if uncertainty_method == "ensemble_vote_standard_deviation" else [0.0, 1.0]}, "formula": uncertainty_formula, "source_refs": list(source.source_refs) or [plan.uncertainty_policy.source or uncertainty_method]},
         "U_rules": {"value": u_rules, "status": "measured" if observation.rules_applicable else "not_applicable", "coverage_component": 1.0 - strongest, "conflict_component": observation.rule_conflict, "applicable_rule_ids": [f"target_{label}" for label in memberships], "activations": memberships, "formula": "max(rule_conflict, 1-max(rule_activations))", "source_refs": list(observation.source_refs)},
         "U_trace": {"value": u_trace, "status": "externally_verified" if externally_verified and not missing_trace_fields else "externally_reported_incomplete", "method": "required_field_check_plus_external_verification_status", "required_fields": list(required_trace_fields), "missing_fields": missing_trace_fields if missing_trace_fields else ([] if externally_verified else ["externally_verified_status"]), "invalid_fields": [], "externally_verified": externally_verified, "verification_source": observation.trace_verification_source, "formula": "0 iff required fields exist and external verifier reports complete; otherwise 1", "source_refs": [observation.trace.source, observation.trace_verification_source]},
         "aggregation": {"eta": uncertainty.weights, "terms": aggregation_terms, "u_M": uncertainty.u_m, "formula": "sum(eta_k * U_k)"}},
    )
    e_pre = compose_pre_explanation(aligned_model=aligned, target=e_target, uncertainty=uncertainty, reduction=reduction)
    components = {"H": entropy_component(e_pre), "C": rule_complexity_component(e_pre), "O": term_overlap_component(e_pre), "K": rule_contradiction_component(e_pre), "U": e_pre.uncertainty}
    e_pre.metadata.update({"interpretability_components": components, "interpretability_weights": dict(plan.lambda_), "interpretability_loss": compute_interpretability_loss(e_pre, plan.lambda_)})
    i_pre = interpretability_pre(e_pre, plan)
    risk_weights = dict(plan.metadata.get("system_risk_weights", {"w_p": 0.30, "w_u": 0.25, "w_I": 0.20, "w_Delta": 0.15, "w_R": 0.10}))
    thresholds = dict(plan.metadata.get("system_risk_thresholds", {"theta_1": plan.rho_accept, "theta_2": plan.rho_warning, "theta_3": plan.rho_audit, "theta_4": plan.rho_critical}))
    detected_critical = bool(not alignment["certified"] or u_trace >= 1.0)
    critical_rupture = bool(detected_critical or observation.critical_rupture)
    rupture_present = bool(detected_critical or observation.rupture_present or critical_rupture)
    risk = compute_strict_rho(
        rho_p=source_value, u_m=uncertainty.u_m, i_pre=i_pre,
        delta=delta_for_risk, chi_r=float(rupture_present),
        weights=risk_weights, thresholds=thresholds,
        critical=int(critical_rupture),
        action_policy=plan.metadata.get("system_action_policy"),
    )
    diagnostics = []
    if rupture_present:
        diagnostics.append({
            "code": observation.rupture_code,
            "reason": (
                "critical structural rupture overrides numeric action"
                if critical_rupture else
                "non-critical structural rupture contributes to chi_R without forcing block"
            ),
            "severity": "critical" if critical_rupture else "warning",
            "rupture_present": True,
            "critical_rupture": critical_rupture,
            "source_refs": list(observation.rupture_source_refs or observation.source_refs),
        })
    return SystemEvidence(
        source_evidence=source, e_model=e_model, alignment_transform=transform, aligned_explanation=aligned,
        e_target=e_target, alignment={key: value for key, value in alignment.items() if key != "aligned"},
        uncertainty=uncertainty, representation=interval,
        representation_policy=representation_policy.to_dict(), reduction=reduction,
        reduction_status=reduction_status, e_pre=e_pre, i_pre=i_pre,
        risk=risk, diagnostics=tuple(diagnostics),
    )


def _system_model_uncertainty(
    *,
    plan: ExplainPlan,
    source: SystemSourceEvidence,
) -> tuple[float | None, str, dict[str, Any], str]:
    method = plan.uncertainty_policy.method
    if method == "none":
        return None, "none", {}, "not applicable"
    if method in {"ensemble_disagreement", "ensemble_vote_standard_deviation"}:
        value = source.model_uncertainty_inputs.get("ensemble_vote_standard_deviation")
        measured = float(value) if isinstance(value, (float, int)) else None
        return measured, "ensemble_vote_standard_deviation", dict(source.model_uncertainty_inputs), "std(binary per-tree votes)"
    probabilities = source.model_uncertainty_inputs.get("probabilities")
    values = [float(value) for value in probabilities] if isinstance(probabilities, list) else []
    if not values:
        return None, method, {}, f"{method} requires class probabilities"
    total = sum(values)
    if total <= 0:
        return None, method, {"probabilities": values}, "probabilities must have positive sum"
    normalized = [value / total for value in values]
    if method == "entropy":
        denominator = math.log(len(normalized)) if len(normalized) > 1 else 1.0
        value = -sum(probability * math.log(probability) for probability in normalized if probability > 0) / denominator
        return _clip01(value), method, {"probabilities": normalized}, "-sum(p_k*ln(p_k))/ln(K)"
    if method == "margin":
        ordered = sorted(normalized, reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        value = 1.0 - (ordered[0] - second)
        return _clip01(value), method, {"probabilities": normalized, "top": ordered[0], "second": second}, "1-(p_top-p_second)"
    if method == "calibrated_interval":
        width = source.model_uncertainty_inputs.get("calibrated_interval_width")
        measured = float(width) if isinstance(width, (float, int)) else None
        return measured, method, {"calibrated_interval_width": width}, "calibrated interval half-width"
    return None, method, {}, "unsupported uncertainty policy"


def aggregate_uncertainty(*, u_model: float | None, u_rules: float | None, u_trace: float | None, plan: ExplainPlan, required: Mapping[str, bool]) -> UncertaintyEvidence:
    values = {"model": u_model, "rules": u_rules, "trace": u_trace}
    missing = [name for name, enabled in required.items() if enabled and values[name] is None]
    if missing:
        return UncertaintyEvidence(u_model, u_rules, u_trace, None, "incomplete", dict(plan.eta), {"missing": ",".join(missing)})
    used = {name: _clip01(value) for name, value in values.items() if required.get(name, False) and value is not None}
    weights = {name: float(plan.eta[name]) for name in used}
    if not used:
        return UncertaintyEvidence(u_model, u_rules, u_trace, None, "not_applicable", weights, {})
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        return UncertaintyEvidence(u_model, u_rules, u_trace, None, "incomplete", weights, {"reason": "ExplainPlan eta must sum to one over required uncertainty channels"})
    return UncertaintyEvidence(u_model, u_rules, u_trace, sum(weights[key] * used[key] for key in used), "measured", weights, {})


@dataclass(frozen=True)
class ReductionEvidence:
    source_representation: str
    target_representation: str
    operation: str
    inverse_embedding: str
    distance_method: str
    delta: float
    delta_threshold: float
    status: str
    source_interval: tuple[float, float]
    reduced_scalar: float
    reconstructed_interval: tuple[float, float]
    distance_terms: dict[str, float]


def reduce_interval_representation(source: IntervalFS, *, delta_threshold: float) -> tuple[F0, ReductionEvidence]:
    """Perform F_int -> F0 -> iota(F0) and measure D_F, not presentation loss."""
    reduced, declared_delta = reduce_to_f0(source)
    inverse = IntervalFS(reduced.mu, reduced.mu)
    measured = float(source.distance(inverse))
    low, high = (float(value) for value in source.membership(0.0))
    scalar = float(reduced.membership(0.0))
    return reduced, ReductionEvidence(
        source_representation="F_int", target_representation="F0", operation="Pi_interval_midpoint",
        inverse_embedding="iota_F0_to_interval_diagonal", distance_method="IntervalFS.sup_grid_distance",
        delta=measured, delta_threshold=float(delta_threshold), status="measured" if measured == declared_delta else "inconsistent",
        source_interval=(low, high), reduced_scalar=scalar, reconstructed_interval=(scalar, scalar),
        distance_terms={"lower_abs": abs(low - scalar), "upper_abs": abs(high - scalar), "D_F": measured},
    )


def compose_pre_explanation(*, aligned_model: ExplanationObject, target: ExplanationObject, uncertainty: UncertaintyEvidence, reduction: ReductionEvidence | None) -> ExplanationObject:
    if uncertainty.u_m is None:
        raise ValueError("E_pre requires a complete u_M")
    metadata: dict[str, Any] = {"interface": "system_pre", "composition": ["aligned_model", "target", "uncertainty", "reduction"]}
    if reduction is not None:
        metadata["reduction"] = reduction.__dict__
    return ExplanationObject(
        terms=set(aligned_model.terms) | set(target.terms), representation=target.representation,
        rules=list(aligned_model.rules) + list(target.rules),
        activations={**aligned_model.activations, **target.activations}, uncertainty=uncertainty.u_m,
        trace=target.trace, reduction_loss=reduction.delta if reduction is not None else 0.0, metadata=metadata,
    )


@dataclass(frozen=True)
class SystemRisk:
    rho: float | None
    partial_risk_score: float | None
    status: str
    components: dict[str, float | None]
    weights: dict[str, float]
    action: str
    candidate_action: str
    critical_override: bool
    chi_R_critical: int
    candidate_action_reason: str
    final_action_reason: str
    thresholds: dict[str, float]


def compute_strict_rho(*, rho_p: float | None, u_m: float | None, i_pre: float | None, delta: float | None, chi_r: float | None, weights: Mapping[str, float], thresholds: Mapping[str, float], critical: int, action_policy: Any = None) -> SystemRisk:
    components = {"rho_p": rho_p, "u_M": u_m, "one_minus_I_pre": None if i_pre is None else 1.0 - i_pre, "Delta": delta, "chi_R": chi_r}
    expected = {"rho_p": "w_p", "u_M": "w_u", "one_minus_I_pre": "w_I", "Delta": "w_Delta", "chi_R": "w_R"}
    mapped = {key: float(weights[weight]) for key, weight in expected.items()}
    if abs(sum(mapped.values()) - 1.0) > 1e-9:
        raise ValueError("P19 rho weights must sum to one")
    if any(weight < 0 for weight in mapped.values()):
        raise ValueError("P19 rho weights must be non-negative")
    threshold_values = {f"theta_{index}": float(thresholds[f"theta_{index}"]) for index in range(1, 5)}
    theta_1, theta_2, theta_3, theta_4 = (threshold_values[f"theta_{index}"] for index in range(1, 5))
    if not 0.0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1.0:
        raise ValueError("P19 action thresholds must satisfy 0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1")
    missing = [key for key, value in components.items() if value is None and mapped[key] > 0.0]
    available = {key: float(value) for key, value in components.items() if value is not None}
    partial = sum(mapped[key] * _clip01(value) for key, value in available.items()) if available else None
    if missing:
        critical_override = int(critical) == 1
        return SystemRisk(
            rho=None,
            partial_risk_score=partial,
            status="incomplete",
            components=components,
            weights=mapped,
            action="block" if critical_override else "request_more_data",
            candidate_action="request_more_data",
            critical_override=critical_override,
            chi_R_critical=int(critical_override),
            candidate_action_reason=f"missing positive-weight risk components: {', '.join(missing)}",
            final_action_reason="critical override" if critical_override else "required risk evidence is incomplete",
            thresholds=threshold_values,
        )
    complete_components = {key: 0.0 if value is None else float(value) for key, value in components.items()}
    rho = sum(mapped[key] * _clip01(value) for key, value in complete_components.items())
    policy = dict(action_policy) if isinstance(action_policy, Mapping) else {}
    middle_action = str(policy.get("theta_2_to_theta_3", "request_more_data"))
    upper_action = str(policy.get("theta_3_to_theta_4", "defer_to_human"))
    if middle_action not in {"request_more_data", "review"}:
        raise ValueError("theta_2_to_theta_3 action must be request_more_data or review")
    if upper_action not in {"defer_to_human", "review"}:
        raise ValueError("theta_3_to_theta_4 action must be defer_to_human or review")
    if rho < theta_1:
        candidate_action = "accept"
        candidate_reason = "rho is below theta_1"
    elif rho < theta_2:
        candidate_action = "lower_confidence"
        candidate_reason = "rho is between theta_1 and theta_2"
    elif rho < theta_3:
        candidate_action = middle_action
        candidate_reason = f"rho is between theta_2 and theta_3; ExplainPlan selected {middle_action}"
    elif rho < theta_4:
        candidate_action = upper_action
        candidate_reason = f"rho is between theta_3 and theta_4; ExplainPlan selected {upper_action}"
    else:
        candidate_action = "block"
        candidate_reason = "rho is at or above theta_4"
    critical_override = int(critical) == 1
    return SystemRisk(
        rho=rho,
        partial_risk_score=None,
        status="complete",
        components=components,
        weights=mapped,
        action="block" if critical_override else candidate_action,
        candidate_action=candidate_action,
        critical_override=critical_override,
        chi_R_critical=int(critical_override),
        candidate_action_reason=candidate_reason,
        final_action_reason="critical structural override has priority" if critical_override else candidate_reason,
        thresholds=threshold_values,
    )


def interpretability_pre(explanation_pre: ExplanationObject, plan: ExplainPlan) -> float:
    return float(compute_interpretability_index(explanation_pre, plan.lambda_))
