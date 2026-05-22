from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai import (
    ExplainPlan, Rule, SystemOperator, Trace, compose, build_profile,
    Candidate, select_minimal_sufficient,
    IntervalFS, HesitantFS, NeutrosophicFS, MultiLevelFS,
)
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from fuzzyxai.core.trust_evaluator import semantic_disagreement, interpretability_loss, interpretability_index
from fuzzyxai.selection.compatibility import synthesize_levels

OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def candidates():
    return [
        Candidate('F0', {'u_num', 'u_ling'}, 0.10, 0.03, 0.42),
        Candidate('FI', {'u_num', 'u_ling', 'u_int'}, 0.22, 0.05, 0.30),
        Candidate('FH', {'u_num', 'u_ling', 'u_exp'}, 0.30, 0.81, 0.25),
        Candidate('FNsrc', {'u_num', 'u_ling', 'u_if', 'u_conf', 'u_trace'}, 0.42, 0.09, 0.18),
        Candidate('FML-user', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_multi'}, 0.60, 0.99, 0.08),
        Candidate('FML-audit', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}, 0.68, 1.00, 0.04),
    ]


def main():
    metadata = {
        'has_intervals': True,
        'num_experts': 2,
        'source_conflict': True,
        'requires_audit': True,
        'multi_level': True,
    }
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, candidates(), mode='audit')

    interval = IntervalFS(lambda x: 0.68, lambda x: 0.76, policy='mid')
    hesitant = HesitantFS(lambda x: [0.61, 0.78])
    neutro = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64, source_t='model', source_f='expert')
    multi = MultiLevelFS([interval, hesitant, neutro], gamma={('interval', 'hesitant', 'same_case'), ('neutro', 'trace', 'source_link')}, weights=[0.25, 0.25, 0.50])

    reducer = MetaReducer(goal='audit')
    reduction = reducer.reduce(multi)
    reduced, delta, policy = reduction.reduced, reduction.delta, reduction.policy

    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)
    rules_a = [Rule('r_high_check', {'risk': 'high'}, 'additional_check')]
    e_ext = op.explain_scalar_risk(0.72, rules_a, Trace('risk', 'v1', 'demo', source='demo', checksum='r'))
    e_ext.representation = multi
    e_ext.reduction_loss = delta

    rules_b = [Rule('r_decision_high', {'risk': 'high'}, 'send_to_check')]
    e_dec = op.explain_scalar_risk(0.74, rules_b, Trace('decision', 'v1', 'demo', source='demo', checksum='d'))

    d_ext = semantic_disagreement(e_ext, e_dec, plan.beta)
    composed = compose(e_ext, e_dec, plan.beta)
    loss = interpretability_loss(0.34, 0.40, 0.18, 0.05, getattr(composed, 'uncertainty', 1.0), plan.lambda_, getattr(composed, 'reduction_loss', 0.0), 0.10)
    index = interpretability_index(loss)

    profile_bad = set(profile) | {'u_ethical'}
    d_choice = select_minimal_sufficient(profile_bad, candidates(), mode='audit')
    temporal_cf_levels = synthesize_levels({'u_num', 'u_ling', 'u_time', 'u_cf', 'u_trace'})

    report = {
        'chapter': 3,
        'route': 'metadata -> P_sit -> Pareto -> F* -> A_k^F -> Pi/Delta -> E_k^ext -> d_E^ext -> composition',
        'profile': sorted(profile),
        'selected_class': getattr(selected, 'name', None),
        'reduction_policy': policy,
        'Delta_k': round(delta, 6),
        'd_E_ext': round(d_ext, 6),
        'composition_result': 'ExplanationObject' if hasattr(composed, 'uncertainty') else getattr(composed, 'code', 'unknown'),
        'L_ext': round(loss, 6),
        'I_ext': round(index, 6),
        'diagnostic_choice': getattr(d_choice, 'code', None),
        'diagnostic_missing': getattr(d_choice, 'context', {}).get('missing'),
        'temporal_cf_levels': [sorted(level) for level in temporal_cf_levels],
        'candidates': [c.__dict__ | {'coverage': sorted(c.coverage)} for c in candidates()],
    }
    (OUT / 'chapter3_end_to_end_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        '# Chapter 3 end-to-end proof',
        '',
        f"Selected class: **{report['selected_class']}**",
        f"Reduction policy: **{policy}**, Delta = **{report['Delta_k']}**",
        f"d_E_ext = **{report['d_E_ext']}**, L_ext = **{report['L_ext']}**, I_ext = **{report['I_ext']}**",
        f"Diagnostic for unknown type: **{report['diagnostic_choice']}**, missing = `{report['diagnostic_missing']}`",
    ]
    (OUT / 'chapter3_end_to_end.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
