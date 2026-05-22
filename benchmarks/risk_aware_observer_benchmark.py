from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def _action_counts(actions: list[str]) -> dict[str, int]:
    return {action: actions.count(action) for action in sorted(set(actions))}


def build_risk_aware_observer_report(*, write: bool = True) -> dict[str, Any]:
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from fuzzyxai import ExplainPlan
    from fuzzyxai.risk import RiskAwareModel, RiskPolicy
    from fuzzyxai.risk.metrics import (
        accepted_accuracy,
        block_rate,
        coverage,
        defer_rate,
        diagnostic_rate,
        expected_cost_after,
        expected_cost_before,
        mean_interpretability,
        request_rate,
        risk_reduction,
    )

    data = load_breast_cancer(as_frame=True)
    x = data.data
    y = (data.target == 0).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    base_model = RandomForestClassifier(n_estimators=140, max_depth=5, random_state=42)
    base_model.fit(x_train, y_train)
    base_pred = base_model.predict(x_test)
    base_proba = base_model.predict_proba(x_test)[:, 1]

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    policy = RiskPolicy(theta_mid=0.34, theta_high=0.62)
    observer = RiskAwareModel(base_model, plan=plan, policy=policy, positive_class=1)
    observed = observer.predict_with_risk(x_test)

    actions = [row['action'] for row in observed]
    y_pred = [row['raw_prediction'] for row in observed]
    indices = [row['interpretability_index'] for row in observed]
    diagnostics = [row['diagnostics'] for row in observed]

    cost_matrix = [[0.0, 1.0], [5.0, 0.0]]
    cost_before = expected_cost_before(y_test, y_pred, cost_matrix)
    cost_after = expected_cost_after(y_test, y_pred, actions, cost_matrix, defer_cost=0.04, request_cost=0.03, block_cost=0.05)
    conflict_action = observer.predict_with_risk(x_test.iloc[:1], metadata={'force_diagnostic': True})[0]['action']

    report = {
        'benchmark': 'risk-aware observer on sklearn breast_cancer',
        'dataset': {'name': 'sklearn breast_cancer', 'samples': int(len(x)), 'features': int(x.shape[1])},
        'model': {
            'name': 'RandomForestClassifier',
            'accuracy_base': round(float(accuracy_score(y_test, base_pred)), 6),
            'roc_auc': round(float(roc_auc_score(y_test, base_proba)), 6),
        },
        'policy': {
            'theta_mid': policy.theta_mid,
            'theta_high': policy.theta_high,
            'block_on_diagnostic': policy.block_on_diagnostic,
            'risk_weights': dict(policy.risk_weights),
        },
        'observer_metrics': {
            'accepted_accuracy': None if accepted_accuracy(y_test, y_pred, actions) is None else round(float(accepted_accuracy(y_test, y_pred, actions)), 6),
            'coverage': round(float(coverage(actions)), 6),
            'defer_rate': round(float(defer_rate(actions)), 6),
            'request_rate': round(float(request_rate(actions)), 6),
            'block_rate': round(float(block_rate(actions)), 6),
            'expected_cost_before': round(float(cost_before), 6),
            'expected_cost_after': round(float(cost_after), 6),
            'risk_reduction': round(float(risk_reduction(cost_before, cost_after)), 6),
            'mean_interpretability': round(float(mean_interpretability(indices)), 6),
            'diagnostic_rate': round(float(diagnostic_rate(diagnostics)), 6),
            'action_counts': _action_counts(actions),
            'forced_conflict_action': conflict_action,
        },
        'sample_cases': [
            {
                'predicted_risk': round(row['predicted_risk'], 6),
                'uncertainty': round(row['uncertainty'], 6),
                'risk_score': round(row['risk_score'], 6),
                'action': row['action'],
                'corrected_confidence': round(row['corrected_confidence'], 6),
                'interpretability_index': round(row['interpretability_index'], 6),
                'reason': row['reason'],
            }
            for row in observed[:5]
        ],
        'success_criteria': {
            'risk_reduction_positive': risk_reduction(cost_before, cost_after) > 0,
            'forced_conflict_safe_action': conflict_action in {'block', 'defer_to_human'},
        },
        'conclusion': 'Наблюдатель не меняет RandomForest, а управляет допустимостью автоматического применения прогноза через риск, неопределённость, интерпретируемость и D_ij.',
    }
    if write:
        (OUT / 'risk_aware_observer_benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'risk_aware_observer_benchmark.md').write_text(render_markdown(report), encoding='utf-8')
    return report


def render_markdown(report: dict[str, Any]) -> str:
    model = report['model']
    metrics = report['observer_metrics']
    lines = [
        '# Risk-Aware Observer benchmark',
        '',
        f"Датасет: `{report['dataset']['name']}` ({report['dataset']['samples']} объектов, {report['dataset']['features']} признаков).",
        f"Модель: `{model['name']}`.",
        '',
        '## Базовая модель',
        '',
        f"- Accuracy: `{model['accuracy_base']}`",
        f"- ROC-AUC: `{model['roc_auc']}`",
        '',
        '## Наблюдатель',
        '',
        f"- Accepted accuracy: `{metrics['accepted_accuracy']}`",
        f"- Coverage: `{metrics['coverage']}`",
        f"- Defer rate: `{metrics['defer_rate']}`",
        f"- Request rate: `{metrics['request_rate']}`",
        f"- Expected cost before: `{metrics['expected_cost_before']}`",
        f"- Expected cost after: `{metrics['expected_cost_after']}`",
        f"- Risk reduction: `{metrics['risk_reduction']}`",
        f"- Mean I(E): `{metrics['mean_interpretability']}`",
        f"- Forced conflict action: `{metrics['forced_conflict_action']}`",
        '',
        '## Распределение действий',
        '',
    ]
    for action, count in metrics['action_counts'].items():
        lines.append(f'- `{action}`: {count}')
    lines.extend(['', '## Итог', '', report['conclusion'], ''])
    return '\n'.join(lines)


def main() -> None:
    report = build_risk_aware_observer_report(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
