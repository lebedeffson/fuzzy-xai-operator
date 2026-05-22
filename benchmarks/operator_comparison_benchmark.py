from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def build_operator_comparison_report(*, write: bool = True) -> dict[str, Any]:
    """Compare a normal ML explanation baseline with the FuzzyXAI system operator.

    Baseline: model risk + feature importance.
    FuzzyXAI: the same model risk plus composition diagnostics: gamma, I(E_G), D_ij.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose
    from fuzzyxai.core.trust_evaluator import interpretability_index, interpretability_loss

    data = load_breast_cancer(as_frame=True)
    x = data.data
    # sklearn target: 0 = malignant, 1 = benign. For a risk story we model malignant probability.
    y = (data.target == 0).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=140, max_depth=5, random_state=42)
    model.fit(x_train, y_train)
    risk_scores = model.predict_proba(x_test)[:, 1]
    y_pred = (risk_scores >= 0.5).astype(int)

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    operator = SystemOperator(plan)
    model_rules = [
        Rule('rf_high_malignancy', {'risk': 'high'}, 'urgent_review'),
        Rule('rf_medium_malignancy', {'risk': 'medium'}, 'additional_screening'),
    ]
    decision_rules = [Rule('decision_high_risk', {'risk': 'high'}, 'send_to_oncologist')]

    normal_indices: list[float] = []
    normal_gammas: list[float] = []
    conflict_diagnostics = 0
    checked = 0
    for idx, risk in enumerate(risk_scores[:50]):
        checked += 1
        risk = float(risk)
        e_model = operator.explain_scalar_risk(
            risk,
            model_rules,
            Trace(f'case-{idx}-rf-model', 'v1', 'benchmark', source='sklearn_breast_cancer', checksum='rf'),
            model_uncertainty=1.0 - max(risk, 1.0 - risk),
            trace_uncertainty=0.01,
        )
        e_decision = operator.explain_scalar_risk(
            min(1.0, risk + 0.03),
            decision_rules,
            Trace(f'case-{idx}-decision', 'v1', 'benchmark', source='clinical-protocol-demo', checksum='decision'),
            model_uncertainty=0.08,
            trace_uncertainty=0.01,
        )
        composed = compose(e_model, e_decision, plan.beta)
        if not hasattr(composed, 'code'):
            gamma = float(composed.metadata.get('gamma', 0.0))
            normal_gammas.append(gamma)
            loss = interpretability_loss(
                0.30, 0.33, 0.16, 0.03,
                composed.uncertainty,
                plan.lambda_,
                composed.reduction_loss,
                0.10,
            )
            normal_indices.append(float(interpretability_index(loss)))

        broken_decision = e_decision.copy_with(terms={'approve', 'deny'})
        diagnostic = compose(e_model, broken_decision, plan.beta)
        if getattr(diagnostic, 'code', None) == 'D_ij':
            conflict_diagnostics += 1

    top_importances = sorted(
        [
            {'feature': str(feature), 'importance': round(float(importance), 6)}
            for feature, importance in zip(x.columns, model.feature_importances_)
        ],
        key=lambda row: row['importance'],
        reverse=True,
    )[:10]

    report = {
        'benchmark': 'operator comparison on sklearn breast_cancer',
        'dataset': {
            'name': 'sklearn breast_cancer',
            'samples': int(len(x)),
            'features': int(x.shape[1]),
            'target': 'malignant risk, derived from sklearn target == 0',
        },
        'model': {
            'name': 'RandomForestClassifier',
            'n_estimators': 140,
            'max_depth': 5,
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 6),
            'roc_auc': round(float(roc_auc_score(y_test, risk_scores)), 6),
            'top_feature_importances': top_importances,
        },
        'without_operator': {
            'available': ['risk_score', 'accuracy', 'roc_auc', 'feature_importance'],
            'missing': ['semantic_gamma', 'interpretability_index', 'D_ij_conflict_detection'],
            'claim': 'Модель объясняет риск локально, но не проверяет согласованность цепочки model -> decision.',
        },
        'with_operator': {
            'available': ['risk_score', 'E_k', 'A_k^F', 'semantic_gamma', 'interpretability_index', 'D_ij_conflict_detection'],
            'checked_cases': checked,
            'mean_gamma': round(sum(normal_gammas) / len(normal_gammas), 6) if normal_gammas else None,
            'mean_interpretability_index': round(sum(normal_indices) / len(normal_indices), 6) if normal_indices else None,
            'conflict_detection_rate': round(conflict_diagnostics / checked, 6) if checked else None,
            'claim': 'Оператор добавляет проверку семантической совместимости и диагностирует разрыв терминов.',
        },
        'conclusion': (
            'FuzzyXAI не повышает accuracy модели напрямую. Его вклад — проверка объяснительной цепочки: '
            'gamma, I(E_G), D_ij и воспроизводимый trace.'
        ),
    }

    if write:
        json_path = OUT / 'operator_comparison_benchmark.json'
        md_path = OUT / 'operator_comparison_benchmark.md'
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        md_path.write_text(render_markdown(report), encoding='utf-8')
    return report


def render_markdown(report: dict[str, Any]) -> str:
    model = report['model']
    with_operator = report['with_operator']
    without_operator = report['without_operator']
    lines = [
        '# Benchmark: без оператора / с оператором',
        '',
        f"Датасет: `{report['dataset']['name']}` ({report['dataset']['samples']} объектов, {report['dataset']['features']} признаков).",
        f"Модель: `{model['name']}`.",
        '',
        '## Качество модели',
        '',
        f"- Accuracy: `{model['accuracy']}`",
        f"- ROC-AUC: `{model['roc_auc']}`",
        '',
        '## Что есть без оператора',
        '',
        f"- Доступно: {', '.join(without_operator['available'])}.",
        f"- Отсутствует: {', '.join(without_operator['missing'])}.",
        f"- Вывод: {without_operator['claim']}",
        '',
        '## Что добавляет оператор',
        '',
        f"- Проверено кейсов: `{with_operator['checked_cases']}`",
        f"- Среднее `gamma`: `{with_operator['mean_gamma']}`",
        f"- Средний `I(E_G)`: `{with_operator['mean_interpretability_index']}`",
        f"- Доля обнаружения искусственного конфликта `D_ij`: `{with_operator['conflict_detection_rate']}`",
        f"- Вывод: {with_operator['claim']}",
        '',
        '## Топ признаков модели',
        '',
    ]
    for row in model['top_feature_importances']:
        lines.append(f"- `{row['feature']}`: {row['importance']}")
    lines.extend(['', '## Итог', '', report['conclusion'], ''])
    return '\n'.join(lines)


def main() -> None:
    report = build_operator_comparison_report(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
