from typing import Dict, List
from .explain_plan import ExplainPlan
from .explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.hierarchy.f0 import F0

class SystemOperator:
    """Omega = Trace o Unc o Rule o Fuzz o Norm for a scalar-risk demo interface."""
    def __init__(self, plan: ExplainPlan):
        plan.validate()
        self.plan = plan

    def explain_scalar_risk(self, risk: float, rules: List[Rule], trace: Trace, model_uncertainty: float = 0.08, trace_uncertainty: float = 0.02) -> ExplanationObject:
        z = max(0.0, min(1.0, float(risk)))
        def low(x): return max(0.0, min(1.0, (0.5 - x) / 0.5))
        def medium(x):
            if 0.25 <= x <= 0.5: return 4*x - 1
            if 0.5 < x <= 0.75: return 3 - 4*x
            return 0.0
        def high(x):
            if 0.5 <= x <= 0.75: return 4*x - 2
            if 0.75 < x <= 1.0: return 1.0
            return 0.0
        mu = {'low': low(z), 'medium': medium(z), 'high': high(z)}
        activations: Dict[str, float] = {}
        for r in rules:
            activations[r.name] = mu.get(r.conditions.get('risk', ''), 0.0)
        u_rules = 1.0 - max(activations.values() or [0.0])
        eta = self.plan.eta
        u = eta['model'] * model_uncertainty + eta['rules'] * u_rules + eta['trace'] * trace_uncertainty
        return ExplanationObject(set(mu), F0(lambda x: high(x), 'high-risk'), rules, activations, max(0.0, min(1.0, u)), trace,
                                 metadata={'risk': z, 'memberships': mu, 'activation_threshold': self.plan.activation_threshold})
