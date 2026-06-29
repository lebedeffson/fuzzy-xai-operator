from __future__ import annotations

import math
from typing import Mapping

from fuzzyxai.core.explanation_object import ExplanationObject
from fuzzyxai.core.trust_evaluator import entropy_from_memberships

DEFAULT_LAMBDA = {'H': 0.20, 'C': 0.20, 'O': 0.20, 'K': 0.20, 'U': 0.20}


def entropy_component(explanation: ExplanationObject) -> float:
    values = list(explanation.activations.values())
    return float(entropy_from_memberships(values)) if values else 0.0


def rule_complexity_component(explanation: ExplanationObject) -> float:
    if not explanation.rules:
        return 0.0
    avg_conditions = sum(len(r.conditions) for r in explanation.rules) / len(explanation.rules)
    norm = max(1.0, float(len(explanation.terms)))
    return max(0.0, min(1.0, avg_conditions / norm))


def term_overlap_component(explanation: ExplanationObject) -> float:
    if not explanation.rules:
        return 0.0
    used = []
    for rule in explanation.rules:
        used.extend(str(v) for v in rule.conditions.values())
        used.append(str(rule.conclusion))
    unique = len(set(used))
    if unique == 0:
        return 0.0
    overlap = 1.0 - (unique / max(1.0, float(len(used))))
    return max(0.0, min(1.0, overlap))


def rule_contradiction_component(explanation: ExplanationObject) -> float:
    if len(explanation.rules) <= 1:
        return 0.0
    by_body: dict[tuple[tuple[str, str], ...], set[str]] = {}
    for rule in explanation.rules:
        key = tuple(sorted((str(k), str(v)) for k, v in rule.conditions.items()))
        by_body.setdefault(key, set()).add(str(rule.conclusion))
    contradictory = sum(1 for conclusions in by_body.values() if len(conclusions) > 1)
    return max(0.0, min(1.0, contradictory / max(1.0, float(len(by_body)))))


def compute_interpretability_loss(
    explanation: ExplanationObject,
    lambda_weights: Mapping[str, float] | None = None,
    *,
    lambda_delta: float = 0.0,
) -> float:
    w = dict(DEFAULT_LAMBDA)
    if lambda_weights:
        w.update({k: float(v) for k, v in lambda_weights.items()})
    h = entropy_component(explanation)
    c = rule_complexity_component(explanation)
    o = term_overlap_component(explanation)
    k = rule_contradiction_component(explanation)
    u = max(0.0, min(1.0, float(explanation.uncertainty)))
    loss = (
        w.get('H', 0.0) * h
        + w.get('C', 0.0) * c
        + w.get('O', 0.0) * o
        + w.get('K', 0.0) * k
        + w.get('U', 0.0) * u
        + float(lambda_delta) * max(0.0, min(1.0, float(explanation.reduction_loss)))
    )
    return max(0.0, float(loss))


def compute_interpretability_index(
    explanation: ExplanationObject,
    lambda_weights: Mapping[str, float] | None = None,
    *,
    lambda_delta: float = 0.0,
) -> float:
    """Compute I_pre = exp(-L(E)) for a concrete explanation object."""
    return float(math.exp(-compute_interpretability_loss(explanation, lambda_weights, lambda_delta=lambda_delta)))
