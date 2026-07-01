from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .core.explain_plan import ExplainPlan
from .core.explanation_object import ExplanationObject, Rule, Trace
from .core.system_operator import SystemOperator
from .core.composition import compose
from .core.trust_evaluator import interpretability_index, interpretability_loss, semantic_disagreement
from .selection.profile_builder import build_profile
from .selection.pareto_selector import Candidate, select_minimal_sufficient
from .visual.composition_graph import edge_report, composition_graph_dot


@dataclass
class ExplanationResult:
    object: ExplanationObject
    profile: set[str]
    selected_class: Any
    plan: ExplainPlan
    report: Dict[str, Any]

    def save_report(self, path: str) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.report, indent=2, ensure_ascii=False), encoding='utf-8')


class FuzzyXAIPipeline:
    """High-level facade for the chapter 2/3 reproducible contour."""

    def __init__(self, plan: ExplainPlan, *, mode: str = 'audit'):
        plan.validate()
        self.plan = plan
        self.mode = mode
        self.operator = SystemOperator(plan)

    @classmethod
    def from_data(cls, X, y=None, *, target: str | None = None, mode: str = 'audit') -> 'FuzzyXAIPipeline':
        plan = ExplainPlan.from_data(X, y, target=target, mode=mode)
        return cls(plan, mode=mode)

    def default_candidates(self) -> list[Candidate]:
        return [
            Candidate('F0', {'u_num', 'u_ling'}, 0.10, 0.03, 0.42),
            Candidate('FI', {'u_num', 'u_ling', 'u_int'}, 0.22, 0.05, 0.30),
            Candidate('FH', {'u_num', 'u_ling', 'u_exp'}, 0.30, 0.81, 0.25),
            Candidate('FNsrc', {'u_num', 'u_ling', 'u_if', 'u_conf', 'u_trace'}, 0.42, 0.09, 0.18),
            Candidate('FML-user', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_multi'}, 0.60, 0.99, 0.08),
            Candidate('FML-audit', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}, 0.68, 1.00, 0.04),
        ]

    def explain_scalar_risk(self, risk: float, *, metadata: Mapping[str, Any] | None = None, component_id: str = 'component') -> ExplanationResult:
        metadata = dict(metadata or {})
        profile = build_profile(metadata)
        selected = select_minimal_sufficient(profile, self.default_candidates(), mode=self.mode)
        rules = [
            Rule('r_high_check', {'risk': 'high'}, 'additional_check'),
            Rule('r_medium_watch', {'risk': 'medium'}, 'watch'),
        ]
        trace = Trace(component_id, 'v1', metadata.get('timestamp', 'demo-time'), source=metadata.get('source', 'demo'), checksum=metadata.get('checksum', component_id))
        obj = self.operator.explain_scalar_risk(float(risk), rules, trace)
        report = {
            'risk': float(risk),
            'profile': sorted(profile),
            'selected_class': getattr(selected, 'name', getattr(selected, 'code', str(selected))),
            'memberships': obj.metadata.get('memberships', {}),
            'uncertainty': obj.uncertainty,
        }
        return ExplanationResult(obj, profile, selected, self.plan, report)

    def compose(self, named_objects: Sequence[tuple[str, ExplanationObject]]) -> Dict[str, Any]:
        if len(named_objects) < 2:
            raise ValueError('at least two objects are required')
        edges = []
        current_name, current_obj = named_objects[0]
        diagnostics = []
        for next_name, next_obj in named_objects[1:]:
            edges.append((current_name, current_obj, next_name, next_obj))
            comp = compose(current_obj, next_obj, self.plan.beta)
            if hasattr(comp, 'code'):
                diagnostics.append(comp)
                break
            current_obj = comp
            current_name = f'{current_name}_{next_name}'
        report = edge_report(edges, self.plan.beta)
        return {
            'edges': report,
            'dot': composition_graph_dot(edges, self.plan.beta),
            'diagnostics': [getattr(d, '__dict__', {}) for d in diagnostics],
            'result': current_obj,
        }
