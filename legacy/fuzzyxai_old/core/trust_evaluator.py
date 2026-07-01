import math
from typing import Mapping, Sequence
import numpy as np
from .explanation_object import ExplanationObject

def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)

def trace_distance(a: Mapping, b: Mapping) -> float:
    fields = ['id', 'version', 'timestamp', 'params', 'source', 'checksum']
    return sum(1 for f in fields if a.get(f) != b.get(f)) / len(fields)

def activation_distance(a: ExplanationObject, b: ExplanationObject) -> float:
    keys = set(a.activations) | set(b.activations)
    if not keys:
        return 0.0
    return sum(abs(a.activations.get(k, 0.0) - b.activations.get(k, 0.0)) for k in keys) / len(keys)

def representation_distance(a: ExplanationObject, b: ExplanationObject) -> float:
    try:
        return float(a.representation.distance(b.representation))
    except Exception:
        return 1.0

def semantic_disagreement(a: ExplanationObject, b: ExplanationObject, beta: Mapping[str, float]) -> float:
    d = (
        beta.get('repr', 0) * representation_distance(a, b) +
        beta.get('rules', 0) * jaccard_distance(a.active_rules, b.active_rules) +
        beta.get('activations', 0) * activation_distance(a, b) +
        beta.get('uncertainty', 0) * abs(a.uncertainty - b.uncertainty) +
        beta.get('trace', 0) * trace_distance(a.trace.as_dict(), b.trace.as_dict()) +
        beta.get('reduction', 0) * min(1.0, a.reduction_loss + b.reduction_loss)
    )
    return max(0.0, min(1.0, float(d)))

def entropy_from_memberships(mu_values: Sequence[float], epsilon: float = 1e-3) -> float:
    values = np.asarray([max(0, min(1, float(v))) for v in mu_values], dtype=float) + epsilon
    p = values / values.sum()
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(p)))

def interpretability_loss(H: float, C: float, O: float, K: float, U: float, weights: Mapping[str, float], reduction_loss: float = 0.0, lambda_delta: float = 0.0) -> float:
    return max(0.0, float(weights.get('H',0)*H + weights.get('C',0)*C + weights.get('O',0)*O + weights.get('K',0)*K + weights.get('U',0)*U + lambda_delta * reduction_loss))

def interpretability_index(loss: float) -> float:
    return float(math.exp(-max(0.0, float(loss))))
