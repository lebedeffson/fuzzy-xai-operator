from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from fuzzyxai import (
    ExplainPlan,
    HesitantFS,
    IntervalFS,
    MultiLevelFS,
    NeutrosophicFS,
    Rule,
    SystemOperator,
    Trace,
    build_profile,
    compose,
    interpretability_index,
    interpretability_loss,
    select_minimal_sufficient,
)
from fuzzyxai.core.plan_builder import build_explain_plan_from_dataframe
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from fuzzyxai.selection.pareto_selector import Candidate
from fuzzyxai.text.explanation_generator import generate_explanation_with_optional_llm


def sample_dataframe() -> pd.DataFrame:
    """Small deterministic medical-like dataset for first-run GUI demos."""
    return pd.DataFrame({
        'age': [45, 51, 60, 39, 72, 66, 58, 49, 54, 63, 57, 70],
        'risk_score': [0.32, 0.47, 0.72, 0.25, 0.88, 0.81, 0.69, 0.44, 0.59, 0.76, 0.64, 0.91],
        'marker': [1.2, 2.1, 4.8, 1.0, 6.1, 5.5, 4.2, 2.0, 3.4, 4.9, 3.9, 6.5],
        'pressure': [118, 124, 147, 112, 165, 158, 142, 126, 134, 151, 139, 168],
        'target': [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    })


def default_plan() -> ExplainPlan:
    return build_explain_plan_from_dataframe(sample_dataframe(), target='target', mode='audit').with_reduction_weight(0.10)


def default_candidates() -> List[Candidate]:
    return [
        Candidate('F0', {'u_num', 'u_ling'}, 0.10, 0.03, 0.42),
        Candidate('FI', {'u_num', 'u_ling', 'u_int'}, 0.22, 0.05, 0.30),
        Candidate('FH', {'u_num', 'u_ling', 'u_exp'}, 0.30, 0.81, 0.25),
        Candidate('FNsrc', {'u_num', 'u_ling', 'u_if', 'u_conf', 'u_trace'}, 0.42, 0.09, 0.18),
        Candidate('FML-user', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_multi'}, 0.60, 0.99, 0.08),
        Candidate('FML-audit', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}, 0.68, 1.00, 0.04),
    ]


def metadata_for_demo(*, intervals: bool = True, experts: int = 2, conflict: bool = True, audit: bool = True, cf: bool = False) -> Dict[str, Any]:
    return {
        'has_intervals': intervals,
        'num_experts': experts,
        'source_conflict': conflict,
        'requires_audit': audit,
        'counterfactual_instability': cf,
        'multi_level': True,
        'source': 'synthetic-gui-demo',
    }


def build_demo_representation(risk: float, *, audit: bool = True) -> MultiLevelFS:
    risk = max(0.0, min(1.0, float(risk)))
    interval = IntervalFS(lambda x: max(0.0, risk - 0.04), lambda x: min(1.0, risk + 0.04))
    hesitant = HesitantFS(lambda x: [max(0.0, risk - 0.11), min(1.0, risk + 0.06)])
    neutro = NeutrosophicFS(
        lambda x: risk,
        lambda x: 0.20,
        lambda x: max(0.0, 1.0 - risk - 0.08),
        source_t='model',
        source_f='expert',
    )
    if audit:
        return MultiLevelFS(
            [interval, hesitant, neutro],
            gamma={('risk', 'experts', 'same_case'), ('neutro', 'trace', 'source_link')},
            weights=[0.25, 0.25, 0.50],
        )
    return MultiLevelFS(
        [interval, hesitant, neutro],
        gamma={('risk', 'experts', 'same_case')},
        weights=[0.30, 0.30, 0.40],
    )


def build_demo_explanation(
    risk: float = 0.72,
    *,
    plan: ExplainPlan | None = None,
    metadata: Mapping[str, Any] | None = None,
    component_id: str = 'risk-model',
    audience: str = 'doctor',
    use_llm: bool = False,
) -> Dict[str, Any]:
    plan = plan or default_plan()
    metadata = dict(metadata or metadata_for_demo())
    audit = bool(metadata.get('requires_audit', True))
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, default_candidates(), mode='audit' if audit else 'user')

    op = SystemOperator(plan)
    rules = [
        Rule('r_high_check', {'risk': 'high'}, 'additional_check'),
        Rule('r_medium_watch', {'risk': 'medium'}, 'watch'),
    ]
    obj = op.explain_scalar_risk(
        risk,
        rules,
        Trace(component_id, 'v1', 'demo-time', source=metadata.get('source', 'synthetic-demo'), checksum=component_id),
    )
    representation = build_demo_representation(risk, audit=audit)
    reducer = MetaReducer(goal='audit' if audit else 'user')
    reduction = reducer.reduce(representation)
    delta = reduction.delta
    policy = reduction.policy
    obj.representation = representation
    obj.reduction_loss = delta
    report = {
        'risk': float(risk),
        'profile': sorted(profile),
        'selected_class': getattr(selected, 'name', getattr(selected, 'code', str(selected))),
        'memberships': obj.metadata.get('memberships', {}),
        'uncertainty': obj.uncertainty,
        'reduction_loss': delta,
        'reduction_policy': policy,
        'representation': representation.class_name,
        'diagnostic': getattr(selected, 'code', None),
    }
    text, text_trace = generate_explanation_with_optional_llm(report, audience=audience, use_llm=use_llm)
    report['text_generation'] = dict(text_trace)
    return {
        'object': obj,
        'representation': representation,
        'profile': profile,
        'selected': selected,
        'report': report,
        'text': text,
        'plan': plan,
    }


def build_demo_composition(
    risk_a: float = 0.72,
    risk_b: float = 0.74,
    *,
    plan: ExplainPlan | None = None,
    conflict: bool = False,
) -> Dict[str, Any]:
    plan = plan or default_plan()
    left = build_demo_explanation(risk_a, plan=plan, component_id='risk-model')['object']

    op = SystemOperator(plan)
    rules2 = [Rule('r_decision_high', {'risk': 'high'}, 'send_to_check')]
    right = op.explain_scalar_risk(
        risk_b,
        rules2,
        Trace('decision-module', 'v1', 'demo-time', source='synthetic-demo', checksum='decision'),
        model_uncertainty=0.10,
        trace_uncertainty=0.01,
    )
    if conflict:
        right.terms = {'allow', 'deny'}
    comp = compose(left, right, plan.beta)
    edges = [('Risk model', left, 'Decision module', right)]
    loss = None
    index = None
    if not hasattr(comp, 'code'):
        loss = interpretability_loss(0.34, 0.40, 0.18, 0.05, comp.uncertainty, plan.lambda_, comp.reduction_loss, lambda_delta=0.10)
        index = interpretability_index(loss)
    return {
        'plan': plan,
        'e1': left,
        'e2': right,
        'comp': comp,
        'edges': edges,
        'loss': loss,
        'index': index,
        'conflict': conflict,
    }
