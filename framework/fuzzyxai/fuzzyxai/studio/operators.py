from __future__ import annotations

from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f'{v:.6f}'
    if isinstance(v, (list, tuple)):
        return ', '.join(str(x) for x in v[:6]) + (' ...' if len(v) > 6 else '')
    if isinstance(v, dict):
        keys = list(v.keys())
        preview = ', '.join(str(k) for k in keys[:6])
        return '{' + preview + (' ...' if len(keys) > 6 else '') + '}'
    return str(v)


def _severity_for_row(operator: str, status: str, details: dict[str, Any]) -> tuple[str, str]:
    st = str(status).lower()
    op = str(operator)
    if 'rupture' in st or 'block' in st or st == 'blocked':
        return 'critical', 'rupture/block state'
    if op == 'ExplComposition':
        if int(details.get('chi_R_crit', 0)) == 1:
            return 'critical', 'critical rupture detected'
        if int(details.get('chi_R', 0)) == 1:
            return 'warning', 'non-critical rupture detected'
    if op == 'ContextTopos' and details.get('chi_Auto_E_action') in {False, 'False', 0, '0'}:
        return 'warning', 'auto-accept forbidden by context'
    if op == 'RiskObserver':
        action = str(details.get('action', '')).lower()
        if action == 'block':
            return 'critical', 'observer blocked action'
        if action in {'defer_to_human', 'request_more_data', 'lower_confidence'}:
            return 'warning', f'observer action={action}'
    if 'restricted' in st or 'pending' in st:
        return 'warning', 'restricted/pending state'
    return 'ok', 'nominal'


def build_operator_trace(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    model = case_state.get('model', {})
    explanation = case_state.get('explanation', {})
    composition = case_state.get('composition', {})
    uncertainty = case_state.get('uncertainty', {})
    contexts = case_state.get('contexts', {})
    risk = case_state.get('risk', {})
    route = case_state.get('route_header', {})

    rows = [
        {
            'operator': 'DatasetAdapter',
            'takes_from': 'dataset_case.features, dataset_case.target',
            'uses': _fmt(case_state.get('dataset_case', {}).get('features', {})),
            'outputs': 'input.features, model.true_y',
            'status': route.get('Input', 'built'),
            'details': {
                'dataset': case_state.get('dataset', {}),
                'sample_id': case_state.get('input', {}).get('sample_id'),
                'feature_count': len(case_state.get('input', {}).get('features', {})),
            },
        },
        {
            'operator': 'ModelPredictor',
            'takes_from': 'input.features',
            'uses': _fmt(case_state.get('input', {}).get('features', {})),
            'outputs': 'model.prediction, model.predicted_risk, model.uncertainty',
            'status': route.get('Model', 'built'),
            'details': {
                'prediction': model.get('prediction'),
                'predicted_risk': model.get('predicted_risk', model.get('p_malignant')),
                'uncertainty': model.get('uncertainty'),
            },
        },
        {
            'operator': 'OmegaOperator',
            'takes_from': 'model.predicted_risk, explain_plan',
            'uses': _fmt(
                {
                    'predicted_risk': model.get('predicted_risk', model.get('p_malignant')),
                    'plan': case_state.get('plan', {}),
                }
            ),
            'outputs': 'explanation.E_model, explanation.I_pre',
            'status': route.get('Omega', 'built'),
            'details': {
                'E_model': explanation.get('E_model', {}),
                'I_pre': explanation.get('I_pre'),
            },
        },
        {
            'operator': 'ExplComposition',
            'takes_from': 'explanation.E_model -> composition.edges',
            'uses': _fmt(composition.get('edges', [])),
            'outputs': 'composition.ruptures, risk.chi_R, risk.chi_R_crit',
            'status': route.get('Expl', 'built'),
            'details': {
                'edges': composition.get('edges', []),
                'ruptures': composition.get('ruptures', []),
                'chi_R': risk.get('chi_R', 0),
                'chi_R_crit': risk.get('chi_R_crit', 0),
            },
        },
        {
            'operator': 'RepresentationSelector',
            'takes_from': 'uncertainty.profile, reduction_graph',
            'uses': _fmt(uncertainty.get('profile', {})),
            'outputs': 'uncertainty.selected_class, uncertainty.delta',
            'status': route.get('Fuzzy', 'built'),
            'details': {
                'selected_class': uncertainty.get('selected_class'),
                'delta': uncertainty.get('delta'),
                'classes': uncertainty.get('classes', []),
            },
        },
        {
            'operator': 'ContextTopos',
            'takes_from': 'contexts.RiskContext, contexts.AutoAccept',
            'uses': _fmt(contexts),
            'outputs': 'chi_Auto(E_action)',
            'status': route.get('Topos', 'built'),
            'details': {
                'RiskContext': contexts.get('RiskContext', {}),
                'AutoAccept': contexts.get('AutoAccept', {}),
                'chi_Auto_E_action': contexts.get('AutoAccept', {}).get('E_action'),
            },
        },
        {
            'operator': 'RiskObserver',
            'takes_from': 'predicted_risk, uncertainty, I_pre, delta, chi_R, chi_R_crit, chi_Auto',
            'uses': _fmt(
                {
                    'predicted_risk': model.get('predicted_risk', model.get('p_malignant')),
                    'uncertainty': model.get('uncertainty'),
                    'I_pre': explanation.get('I_pre'),
                    'delta': uncertainty.get('delta'),
                    'chi_R': risk.get('chi_R', 0),
                    'chi_R_crit': risk.get('chi_R_crit', 0),
                    'chi_Auto': contexts.get('AutoAccept', {}).get('E_action'),
                }
            ),
            'outputs': 'risk.rho, risk.action, risk.reason',
            'status': route.get('Observer', 'built'),
            'details': {
                'rho': risk.get('rho'),
                'action': risk.get('action'),
                'reason': risk.get('reason'),
                'components': risk.get('components', {}),
                'breakdown': risk.get('breakdown', {}),
            },
        },
        {
            'operator': 'ActionPolicy',
            'takes_from': 'risk.action, thresholds, diagnostics',
            'uses': _fmt(
                {
                    'action': risk.get('action'),
                    'thresholds': risk.get('thresholds', []),
                    'chi_R_crit': risk.get('chi_R_crit', 0),
                }
            ),
            'outputs': 'final route_header.Action',
            'status': route.get('Action', risk.get('action')),
            'details': {
                'final_action': risk.get('action'),
                'thresholds': risk.get('thresholds', []),
            },
        },
    ]
    for row in rows:
        severity, signal = _severity_for_row(
            str(row.get('operator', '')),
            str(row.get('status', '')),
            dict(row.get('details', {})),
        )
        row['severity'] = severity
        row['signal'] = signal
    return rows
