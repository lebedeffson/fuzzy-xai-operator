from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import StudioExplainPlan


def build_defense_case_payload(case_state: dict[str, Any], plan: StudioExplainPlan) -> dict[str, Any]:
    risk = case_state.get('risk', {})
    model = case_state.get('model', {})
    uncertainty = case_state.get('uncertainty', {})
    explanation = case_state.get('explanation', {})
    payload = {
        'dataset': case_state.get('dataset', {}),
        'scenario': case_state.get('scenario'),
        'input': case_state.get('input', {}),
        'model': {
            'prediction': model.get('prediction'),
            'true_y': model.get('true_y'),
            'predicted_risk': model.get('predicted_risk', model.get('p_malignant')),
            'uncertainty': model.get('uncertainty'),
        },
        'explanation': {
            'E_model': explanation.get('E_model', {}),
            'I_pre': explanation.get('I_pre'),
        },
        'representation': {
            'selected_class': uncertainty.get('selected_class'),
            'reduction_loss': uncertainty.get('delta'),
        },
        'topos': {
            'chi_Auto': case_state.get('contexts', {}).get('AutoAccept', {}).get('E_action'),
            'RiskContext': case_state.get('contexts', {}).get('RiskContext', {}).get('E_action'),
            'omega_evidence': 'finite sieve enumeration',
        },
        'observer': {
            'rho': risk.get('rho'),
            'chi_R': risk.get('chi_R'),
            'chi_R_crit': risk.get('chi_R_crit'),
            'action': risk.get('action'),
            'reason': risk.get('reason'),
            'breakdown': risk.get('breakdown', {}),
            'weights': risk.get('weights', {}),
            'thresholds': risk.get('thresholds', ()),
        },
        'explain_plan_version': {
            'mode': plan.mode,
            'baseline_access': plan.baseline_access,
            'reduction_strategy': plan.reduction_strategy,
        },
    }
    return payload


def export_defense_case(case_state: dict[str, Any], plan: StudioExplainPlan, *, out_dir: str | Path = 'reports/layered_demo', stem: str = 'defense_case') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_defense_case_payload(case_state, plan)

    json_path = out / f'{stem}.json'
    md_path = out / f'{stem}.md'
    tex_path = out / f'{stem}.tex'

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    obs = payload['observer']
    rep = payload['representation']
    model = payload['model']
    md = [
        '# Defense case report',
        '',
        f"- object: `{payload['input'].get('sample_id')}`",
        f"- predicted_risk: `{model.get('predicted_risk')}`",
        f"- selected_representation: `{rep.get('selected_class')}`",
        f"- reduction_loss: `{rep.get('reduction_loss')}`",
        f"- I_pre: `{payload['explanation'].get('I_pre')}`",
        f"- rho: `{obs.get('rho')}`",
        f"- chi_R: `{obs.get('chi_R')}`",
        f"- chi_R_crit: `{obs.get('chi_R_crit')}`",
        f"- chi_Auto: `{payload['topos'].get('chi_Auto')}`",
        f"- action: `{obs.get('action')}`",
        f"- reason: `{obs.get('reason')}`",
    ]
    md_path.write_text('\n'.join(md), encoding='utf-8')

    tex = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{FuzzyXAI defense case}',
        r'\begin{tabular}{|l|l|}',
        r'\hline',
        f"object & {payload['input'].get('sample_id')} \\\\ \\hline",
        f"predicted\\\\_risk & {model.get('predicted_risk')} \\\\ \\hline",
        f"selected\\\\_representation & {rep.get('selected_class')} \\\\ \\hline",
        f"reduction\\\\_loss & {rep.get('reduction_loss')} \\\\ \\hline",
        f"I\\\\_pre & {payload['explanation'].get('I_pre')} \\\\ \\hline",
        f"rho & {obs.get('rho')} \\\\ \\hline",
        f"chi\\\\_R & {obs.get('chi_R')} \\\\ \\hline",
        f"chi\\\\_R\\\\_crit & {obs.get('chi_R_crit')} \\\\ \\hline",
        f"chi\\\\_Auto & {payload['topos'].get('chi_Auto')} \\\\ \\hline",
        f"action & {obs.get('action')} \\\\ \\hline",
        r'\end{tabular}',
        r'\end{table}',
    ]
    tex_path.write_text('\n'.join(tex), encoding='utf-8')

    return {'json': str(json_path), 'md': str(md_path), 'tex': str(tex_path)}
