"""Strict research interface for TISU explanation composition.

This module checks *declared* identities and transformations. It does not
verify the truth or authenticity of a label, probability or provenance record.
It is intentionally opt-in; legacy compose() remains backward compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping

from .composition import probabilistic_t_conorm
from .diagnostics import DiagnosticState
from .explanation_object import ExplanationObject
from .trust_evaluator import jaccard_distance, trace_distance


@dataclass(frozen=True)
class RegisteredAlignment:
    source_event: str
    target_event: str
    transform: str  # "identity" or "complement"
    term_map: Mapping[str, str] | None = None
    probability_tolerance: float = 1e-9


def _refuse(reason: str, **data: object) -> DiagnosticState:
    return DiagnosticState("D_ij", reason, "critical", data)


def _interface(obj: ExplanationObject) -> Mapping[str, object] | None:
    value = obj.metadata.get("interface")
    if not isinstance(value, dict):
        return None
    required = (
        "sample_id", "model_fingerprint", "feature_schema",
        "event_id", "probability", "provenance_ref",
    )
    if any(value.get(key) in (None, "") for key in required):
        return None
    return value


def _rules_well_defined(obj: ExplanationObject) -> bool:
    by_id: dict[str, str] = {}
    for rule in obj.rules:
        signature = rule.signature()
        if rule.name in by_id and by_id[rule.name] != signature:
            return False
        by_id[rule.name] = signature
    return True


def conditional_explanation_distance(
    left: ExplanationObject,
    right: ExplanationObject,
    weights: Mapping[str, float],
    activation_vocabulary: tuple[str, ...],
    representation_pseudometric: Callable[[object, object], float] | None = None,
) -> float:
    """Pseudometric if all declared component-distance axioms hold.

    The same activation_vocabulary must be used for every compared pair.
    A registered representation metric must itself satisfy the axioms.
    """
    if len(set(activation_vocabulary)) != len(activation_vocabulary):
        raise ValueError("duplicate activation coordinate")
    universe = set(activation_vocabulary)
    if (set(left.activations) | set(right.activations)) - universe:
        raise ValueError("unregistered activation coordinate")
    terms = ("repr", "rules", "activations", "uncertainty", "trace", "reduction")
    if any(
        not isinstance(weights.get(k, 0), (int, float))
        or not isfinite(weights.get(k, 0))
        or weights.get(k, 0) < 0
        for k in terms
    ):
        raise ValueError("distance weights must be nonnegative finite numbers")
    if weights.get("repr", 0) and representation_pseudometric is None:
        raise ValueError("no registered representation pseudometric")
    rule_left = {r.signature() for r in left.rules if r.name in left.active_rules}
    rule_right = {r.signature() for r in right.rules if r.name in right.active_rules}
    activation_delta = (
        sum(abs(left.activations.get(k, 0) - right.activations.get(k, 0)) for k in activation_vocabulary)
        / len(activation_vocabulary)
        if activation_vocabulary else 0.0
    )
    repr_delta = (
        float(representation_pseudometric(left.representation, right.representation))
        if weights.get("repr", 0) else 0.0
    )
    if not isfinite(repr_delta) or repr_delta < 0:
        raise ValueError("invalid representation distance")
    d = (
        weights.get("repr", 0) * repr_delta
        + weights.get("rules", 0) * jaccard_distance(rule_left, rule_right)
        + weights.get("activations", 0) * activation_delta
        + weights.get("uncertainty", 0) * abs(left.uncertainty - right.uncertainty)
        + weights.get("trace", 0) * trace_distance(left.trace.as_dict(), right.trace.as_dict())
        + weights.get("reduction", 0) * abs(left.reduction_loss - right.reduction_loss)
    )
    return min(1.0, d)


def compose_with_registered_alignment(
    left: ExplanationObject,
    right: ExplanationObject,
    weights: Mapping[str, float],
    activation_vocabulary: tuple[str, ...],
    alignment: RegisteredAlignment | None = None,
    representation_pseudometric: Callable[[object, object], float] | None = None,
) -> ExplanationObject | DiagnosticState:
    """Fail closed on an incompatible or insufficiently declared interface."""
    li, ri = _interface(left), _interface(right)
    if li is None or ri is None:
        return _refuse("required interface declaration or provenance is missing")
    for field in ("sample_id", "model_fingerprint", "feature_schema"):
        if li[field] != ri[field]:
            return _refuse("incompatible interface identities", field=field, left=li[field], right=ri[field])
    if not _rules_well_defined(left) or not _rules_well_defined(right):
        return _refuse("conflicting rule IDs within one input")
    seen = {r.name: r.signature() for r in left.rules}
    for rule in right.rules:
        if rule.name in seen and seen[rule.name] != rule.signature():
            return _refuse("rule identifier has incompatible signatures", rule_id=rule.name)
    if alignment is None:
        if li["event_id"] != ri["event_id"]:
            return _refuse("no registered event mapping")
        alignment = RegisteredAlignment(str(li["event_id"]), str(ri["event_id"]), "identity")
    if (li["event_id"], ri["event_id"]) != (alignment.source_event, alignment.target_event):
        return _refuse("event mapping does not match declared interfaces")
    if alignment.transform not in ("identity", "complement"):
        return _refuse("unsupported event transform")
    if alignment.transform == "identity" and li["event_id"] != ri["event_id"]:
        return _refuse("identity transform cannot change event")
    if alignment.transform == "complement" and li["event_id"] == ri["event_id"]:
        return _refuse("complement transform needs distinct opposite events")
    p, q = li["probability"], ri["probability"]
    if any(not isinstance(x, (int, float)) or not isfinite(x) or not 0 <= x <= 1 for x in (p, q)):
        return _refuse("event probability is non-finite or outside [0,1]")
    if not isfinite(alignment.probability_tolerance) or alignment.probability_tolerance < 0:
        return _refuse("invalid registered tolerance")
    expected = float(p) if alignment.transform == "identity" else 1 - float(p)
    if abs(expected - float(q)) > alignment.probability_tolerance:
        return _refuse("probability incompatible with registered event mapping", expected=expected, observed=q)
    if left.terms != right.terms:
        if alignment.term_map is None:
            return _refuse("linguistic term mapping absent")
        translated = [alignment.term_map.get(t) for t in left.terms]
        if None in translated or len(set(translated)) != len(translated) or set(translated) != right.terms:
            return _refuse("linguistic mapping incomplete or non-bijective")
    try:
        gamma = conditional_explanation_distance(
            left, right, weights, activation_vocabulary, representation_pseudometric
        )
    except (TypeError, ValueError) as error:
        return _refuse("metric prerequisites unmet", error=str(error))
    threshold = weights.get("gamma_max")
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or not isfinite(threshold) or threshold < 0:
            return _refuse("invalid registered distance threshold")
        if gamma > threshold:
            return _refuse("explanation disagreement exceeds threshold", gamma=gamma, threshold=threshold)
    merged = list(left.rules)
    for rule in right.rules:
        if rule.name not in seen:
            merged.append(rule)
    activations = dict(left.activations)
    for key, value in right.activations.items():
        activations[key] = max(activations.get(key, 0), value)
    return ExplanationObject(
        terms=set(right.terms),
        representation=right.representation,
        rules=merged,
        activations=activations,
        uncertainty=probabilistic_t_conorm(left.uncertainty, right.uncertainty, gamma),
        trace=right.trace,
        reduction_loss=min(1.0, left.reduction_loss + right.reduction_loss),
        metadata={
            "interface": dict(ri),
            "gamma": gamma,
            "left_trace": left.trace.as_dict(),
            "right_trace": right.trace.as_dict(),
            "semantic_truth": "UNVERIFIED_EXTERNAL",
            "metric_scope": "fixed-activation-vocabulary and registered representation pseudometric",
        },
    )
