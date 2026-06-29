from typing import Mapping, Union
from .diagnostics import DiagnosticState
from .explanation_object import ExplanationObject
from .trust_evaluator import semantic_disagreement

def probabilistic_t_conorm(*values: float) -> float:
    product = 1.0
    for value in values:
        value = max(0.0, min(1.0, float(value)))
        product *= (1.0 - value)
    return 1.0 - product

def compose(left: ExplanationObject, right: ExplanationObject, beta: Mapping[str, float], allow_missing_terms: bool = False) -> Union[ExplanationObject, DiagnosticState]:
    common = left.terms & right.terms
    if not common and not allow_missing_terms:
        return DiagnosticState('D_ij', 'no common linguistic terms for composition', 'critical', {
            'left_terms': sorted(left.terms), 'right_terms': sorted(right.terms)
        })
    gamma = semantic_disagreement(left, right, beta)
    gamma_max = beta.get('gamma_max')
    if gamma_max is not None and gamma > float(gamma_max):
        return DiagnosticState('D_ij', 'semantic disagreement exceeds gamma_max', 'critical', {
            'gamma': gamma, 'gamma_max': float(gamma_max)
        })
    uncertainty = probabilistic_t_conorm(left.uncertainty, right.uncertainty, gamma)
    rule_names = {r.name for r in left.rules}
    rules = list(left.rules) + [r for r in right.rules if r.name not in rule_names]
    activations = dict(left.activations)
    for k, v in right.activations.items():
        activations[k] = max(activations.get(k, 0.0), v)
    return ExplanationObject(
        terms=set(left.terms) | set(right.terms), representation=right.representation,
        rules=rules, activations=activations, uncertainty=uncertainty, trace=right.trace,
        reduction_loss=min(1.0, left.reduction_loss + right.reduction_loss),
        metadata={'gamma': gamma, 'left_trace': left.trace.as_dict(), 'right_trace': right.trace.as_dict(), 'composition': 'left_then_right'}
    )
