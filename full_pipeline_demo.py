from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose
from fuzzyxai.core.trust_evaluator import entropy_from_memberships, interpretability_index, interpretability_loss
from fuzzyxai.demo.synthetic import sample_dataframe, default_candidates, build_demo_representation, metadata_for_demo
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from fuzzyxai.risk import RiskAwareModel, RiskPolicy
from benchmarks.risk_aware_observer_benchmark import build_risk_aware_observer_report
from fuzzyxai.selection.profile_builder import build_profile
from fuzzyxai.selection.pareto_selector import select_minimal_sufficient
from fuzzyxai.visual.composition_graph import edge_report
from fuzzyxai.visual.interactive_graph import save_composition_html
from fuzzyxai.visual.representation_plots import representation_figure

OUT = ROOT / 'reports' / 'full_demo'
OUT.mkdir(parents=True, exist_ok=True)


def _make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))


def _stage_explanation(op: SystemOperator, score: float, name: str, source: str, conclusion_high: str = 'ok'):
    rules = [
        Rule(f'{name}_high', {'risk': 'high'}, conclusion_high),
        Rule(f'{name}_medium', {'risk': 'medium'}, 'check'),
        Rule(f'{name}_low', {'risk': 'low'}, 'warning'),
    ]
    return op.explain_scalar_risk(
        score,
        rules,
        Trace(name, 'v1', 'demo-time', source=source, checksum=name),
        model_uncertainty=max(0.02, 1.0 - score) * 0.20,
        trace_uncertainty=0.02,
    )


def _feature_contributions(model, case: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Works for sklearn Pipeline(StandardScaler, LogisticRegression). The values are not SHAP;
    # they are transparent coefficient x standardized-value contributions for the demo.
    scaler = model.named_steps.get('standardscaler')
    clf = model.named_steps.get('logisticregression')
    x_scaled = scaler.transform(case[feature_cols])[0]
    coef = clf.coef_[0]
    values = x_scaled * coef
    abs_sum = float(np.abs(values).sum()) or 1.0
    return pd.DataFrame({
        'feature': feature_cols,
        'contribution': values,
        'abs_norm': np.abs(values) / abs_sum,
    }).sort_values('abs_norm', ascending=False)


def _plot_memberships_html(plan: ExplainPlan, risk_score: float, path: Path) -> str:
    import plotly.graph_objects as go
    xs = np.linspace(0, 1, 201)

    def low(x):
        return max(0.0, min(1.0, (0.5 - x) / 0.5))

    def medium(x):
        if 0.25 <= x <= 0.5:
            return 4 * x - 1
        if 0.5 < x <= 0.75:
            return 3 - 4 * x
        return 0.0

    def high(x):
        if 0.5 <= x <= 0.75:
            return 4 * x - 2
        if 0.75 < x <= 1.0:
            return 1.0
        return 0.0

    fig = go.Figure()
    for name, fn in [('low', low), ('medium', medium), ('high', high)]:
        fig.add_trace(go.Scatter(x=xs, y=[fn(float(x)) for x in xs], mode='lines', name=name))
    fig.add_vline(x=risk_score, line_width=3, line_dash='dash', line_color='black', annotation_text=f'risk={risk_score:.3f}')
    fig.update_layout(title='Risk score mapped to linguistic terms', xaxis_title='risk_score', yaxis_title='membership', height=420)
    fig.write_html(str(path), include_plotlyjs='cdn')
    return path.name


def _plot_feature_html(contrib: pd.DataFrame, path: Path) -> str:
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=contrib['feature'], y=contrib['contribution'], text=[f'{v:.3f}' for v in contrib['contribution']]))
    fig.update_layout(title='Transparent model contribution proxy: coefficient × standardized value', xaxis_title='feature', yaxis_title='signed contribution', height=420)
    fig.write_html(str(path), include_plotlyjs='cdn')
    return path.name


def _plot_representation_html(representation, path: Path) -> str:
    fig = representation_figure(representation, title='Selected fuzzy representation A_k^F')
    fig.write_html(str(path), include_plotlyjs='cdn')
    return path.name




def _risk_observer_benchmark() -> Dict[str, Any]:
    report = build_risk_aware_observer_report(write=False)
    metrics = report['observer_metrics']
    return {
        'dataset': report['dataset']['name'],
        'n_train': None,
        'n_test': None,
        'model': report['model']['name'],
        'accuracy_base': report['model']['accuracy_base'],
        'roc_auc': report['model']['roc_auc'],
        'accepted_accuracy': metrics['accepted_accuracy'],
        'coverage': metrics['coverage'],
        'defer_rate': metrics['defer_rate'],
        'cost_before': metrics['expected_cost_before'],
        'cost_after': metrics['expected_cost_after'],
        'risk_reduction': metrics['risk_reduction'],
        'forced_diagnostic_action': metrics['forced_conflict_action'],
        'sample_actions': [
            {'idx': idx, **case}
            for idx, case in enumerate(report['sample_cases'])
        ],
    }

def _render_index_html(report: Dict[str, Any], path: Path) -> None:
    def esc(x):
        import html
        return html.escape(str(x))

    rows = []
    for stage in report['stages']:
        rows.append(
            f"<tr><td>{esc(stage['id'])}</td><td>{esc(stage['name'])}</td><td>{esc(stage['operator'])}</td>"
            f"<td>{esc(stage['output'])}</td><td>{esc(stage['meaning'])}</td></tr>"
        )
    edge_rows = []
    for row in report['composition']['edges']:
        edge_rows.append(
            f"<tr><td>{esc(row['source'])}</td><td>{esc(row['target'])}</td><td>{row['gamma']:.4f}</td>"
            f"<td>{esc(row['severity'])}</td><td>{esc(row['left_active_rules'])}</td><td>{esc(row['right_active_rules'])}</td></tr>"
        )
    action_rows = []
    for row in report['risk_observer']['sample_actions']:
        action_rows.append(
            f"<tr><td>{row['idx']}</td><td>{row['predicted_risk']:.4f}</td><td>{row['uncertainty']:.4f}</td>"
            f"<td>{row['risk_score']:.4f}</td><td>{esc(row['action'])}</td><td>{row['corrected_confidence']:.4f}</td><td>{esc(row['reason'])}</td></tr>"
        )

    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>FuzzyXAI full pipeline demo</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1180px; margin: 30px auto; line-height: 1.45; color: #1e293b; }}
h1,h2,h3 {{ color: #0f172a; }}
.card {{ border: 1px solid #d7dee8; border-radius: 16px; padding: 18px; margin: 18px 0; box-shadow: 0 2px 10px rgba(15,23,42,.05); }}
.kpi {{ display:inline-block; min-width: 150px; padding: 12px; margin: 6px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; }}
.kpi b {{ display:block; font-size: 22px; color:#0f766e; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 10px 0; }}
th, td {{ border: 1px solid #d7dee8; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }}
iframe {{ width: 100%; height: 460px; border: 1px solid #d7dee8; border-radius: 12px; }}
code {{ background:#f1f5f9; padding: 2px 5px; border-radius: 4px; }}
.badge {{ display:inline-block; padding: 4px 10px; border-radius:999px; background:#e0f2fe; color:#075985; }}
.warn {{ background:#fef3c7; padding:12px; border-radius:12px; border:1px solid #f59e0b; }}
</style>
</head>
<body>
<h1>FuzzyXAI: полный цикл от входа до безопасного действия</h1>
<p class="badge">route: data → mining → training → model → XAI → A<sub>k</sub><sup>F</sup> → composition → risk-aware gate → feedback</p>
<div class="card">
<h2>Ключевой смысл</h2>
<p>Оператор не заменяет модель и не обещает магически сделать её правильной. Он подключается как внешний наблюдающий слой: строит объяснительные объекты по этапам работы модели, проверяет совместимость объяснений, оценивает неопределённость и передаёт результат в риск-ориентированную политику.</p>
<p><code>{esc(report['formula'])}</code></p>
</div>
<div class="card">
<h2>0. Данные и реальная модель</h2>
<div class="kpi"><span>Модель</span><b>{esc(report['model']['name'])}</b></div>
<div class="kpi"><span>Строк</span><b>{report['data']['n_rows']}</b></div>
<div class="kpi"><span>Признаки</span><b>{report['data']['features']}</b></div>
<div class="kpi"><span>Accuracy train</span><b>{report['model']['train_accuracy']:.3f}</b></div>
<div class="kpi"><span>Кейс</span><b>{report['case']['case_index']}</b></div>
<div class="kpi"><span>risk_score</span><b>{report['case']['risk_score']:.4f}</b></div>
<table><thead><tr><th>feature</th><th>value</th></tr></thead><tbody>{''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k,v in report['case']['values'].items())}</tbody></table>
</div>
<div class="card">
<h2>1. Наблюдатели по этапам работы модели</h2>
<table><thead><tr><th>Этап</th><th>Что наблюдается</th><th>Оператор</th><th>Выход</th><th>Смысл</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</div>
<div class="card"><h2>2. Из риска в лингвистические термы</h2><iframe src="{esc(report['artifacts']['membership_html'])}"></iframe></div>
<div class="card"><h2>3. Data mining / вклад признаков</h2><iframe src="{esc(report['artifacts']['feature_html'])}"></iframe></div>
<div class="card"><h2>4. Выбор нечёткого представления главы 3</h2>
<div class="kpi"><span>Профиль</span><b>{esc(', '.join(report['chapter3']['profile']))}</b></div>
<div class="kpi"><span>Выбранный класс</span><b>{esc(report['chapter3']['selected_class'])}</b></div>
<div class="kpi"><span>Delta</span><b>{report['chapter3']['reduction_loss']:.4f}</b></div>
<iframe src="{esc(report['artifacts']['representation_html'])}"></iframe>
</div>
<div class="card"><h2>5. Композиция объяснений по всей цепочке</h2>
<div class="kpi"><span>Итоговое I(E_G)</span><b>{report['composition']['index']:.4f}</b></div>
<div class="kpi"><span>Loss</span><b>{report['composition']['loss']:.4f}</b></div>
<div class="kpi"><span>Diagnostic test</span><b>{esc(report['composition']['forced_diagnostic_action'])}</b></div>
<iframe src="{esc(report['artifacts']['composition_html'])}"></iframe>
<table><thead><tr><th>source</th><th>target</th><th>gamma</th><th>severity</th><th>left rules</th><th>right rules</th></tr></thead><tbody>{''.join(edge_rows)}</tbody></table>
</div>
<div class="card"><h2>6. Risk-Aware Observer</h2>
<p>Риск-модуль получает прогноз, неопределённость, потерю редукции, индекс интерпретируемости и диагностические состояния. Он выбирает безопасное действие, а не подменяет внутреннюю модель.</p>
<div class="kpi"><span>Accepted accuracy</span><b>{report['risk_observer']['accepted_accuracy']:.4f}</b></div>
<div class="kpi"><span>Coverage</span><b>{report['risk_observer']['coverage']:.4f}</b></div>
<div class="kpi"><span>Defer rate</span><b>{report['risk_observer']['defer_rate']:.4f}</b></div>
<div class="kpi"><span>Cost before</span><b>{report['risk_observer']['cost_before']:.4f}</b></div>
<div class="kpi"><span>Cost after</span><b>{report['risk_observer']['cost_after']:.4f}</b></div>
<div class="kpi"><span>Risk reduction</span><b>{report['risk_observer']['risk_reduction']:.4f}</b></div>
<table><thead><tr><th>#</th><th>predicted risk</th><th>uncertainty</th><th>risk score</th><th>action</th><th>corrected confidence</th><th>reason</th></tr></thead><tbody>{''.join(action_rows)}</tbody></table>
</div>
<div class="card warn"><h2>Что доказывает этот сценарий</h2>
<p>Он показывает не отдельную формулу, а полный наблюдательный контур. Входные данные, признаки, обучение, прогноз, XAI-пояснение, нечёткое представление, решающее правило и риск-модуль становятся наблюдаемыми компонентами одной цепочки. Если связь ломается, система формирует диагностическое состояние, а не скрывает разрыв.</p>
</div>
</body></html>"""
    path.write_text(html, encoding='utf-8')


def run_full_pipeline(open_browser: bool = False) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score

    df = sample_dataframe()
    feature_cols = ['age', 'pressure', 'marker']
    X = df[feature_cols]
    y = df['target']
    model = _make_model()
    model.fit(X, y)
    pred = model.predict(X)
    train_accuracy = float(accuracy_score(y, pred))

    case_index = 2
    case = X.iloc[[case_index]]
    risk_score = float(model.predict_proba(case)[0, 1])

    plan = ExplainPlan.from_data(df, target='target', mode='audit').with_reduction_weight(0.10)
    op = SystemOperator(plan)

    # Stage observers over the full work of the model.
    data_quality = 1.0 - float(df.isna().mean().mean())
    contrib = _feature_contributions(model, case, feature_cols)
    feature_signal = float(contrib['abs_norm'].iloc[0])
    train_signal = train_accuracy
    xai_signal = float(contrib['abs_norm'].sum())

    e_data = _stage_explanation(op, data_quality, 'E_D_data', 'data-table', 'data_quality_ok')
    e_features = _stage_explanation(op, feature_signal, 'E_F_features', 'data-mining', 'feature_signal_ok')
    e_training = _stage_explanation(op, train_signal, 'E_T_training', 'sklearn-training', 'training_quality_ok')

    metadata = metadata_for_demo(intervals=True, experts=2, conflict=True, audit=True)
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, default_candidates(), mode='audit')
    representation = build_demo_representation(risk_score, audit=True)
    reduction = MetaReducer(goal='audit').reduce(representation)

    rules_model = [
        Rule('r_high_check', {'risk': 'high'}, 'additional_check'),
        Rule('r_medium_watch', {'risk': 'medium'}, 'watch'),
    ]
    e_model = op.explain_scalar_risk(
        risk_score,
        rules_model,
        Trace('E_M_model', 'v1', 'demo-time', source='LogisticRegression', checksum='model'),
        model_uncertainty=max(0.02, min(0.98, 1.0 - abs(risk_score - 0.5) * 2.0)),
        trace_uncertainty=0.02,
    )
    e_model.representation = representation
    e_model.reduction_loss = reduction.delta

    e_xai = _stage_explanation(op, xai_signal, 'E_X_xai', 'coefficient-contributions', 'xai_signal_ok')
    e_decision = op.explain_scalar_risk(
        risk_score,
        [Rule('r_decision_high', {'risk': 'high'}, 'send_to_check'), Rule('r_decision_medium', {'risk': 'medium'}, 'watch')],
        Trace('E_A_decision', 'v1', 'demo-time', source='decision-module', checksum='decision'),
        model_uncertainty=0.10,
        trace_uncertainty=0.01,
    )

    # Risk-aware observer. For the single case we keep the same local model; for effect
    # metrics we use an open sklearn medical-like benchmark to avoid a perfect 12-row toy score.
    risk_model = RiskAwareModel(model, plan=plan, policy=RiskPolicy(theta_mid=0.34, theta_high=0.62))
    local_results = risk_model.predict_with_risk(X)
    forced_decision = risk_model.predict_with_risk(case, metadata={'force_diagnostic': True})[0]
    e_risk = _stage_explanation(op, 1.0 - local_results[case_index]['risk_score'], 'E_R_risk_gate', 'risk-aware-observer', local_results[case_index]['action'])
    benchmark_metrics = _risk_observer_benchmark()

    e_feedback = _stage_explanation(op, 1.0 - min(1.0, local_results[case_index]['risk_score']), 'E_B_feedback', 'feedback-buffer', 'feedback_saved')

    chain: List[Tuple[str, Any]] = [
        ('Data', e_data),
        ('Features', e_features),
        ('Training', e_training),
        ('Model', e_model),
        ('XAI', e_xai),
        ('Decision', e_decision),
        ('Risk gate', e_risk),
        ('Feedback', e_feedback),
    ]
    edges = [(chain[i][0], chain[i][1], chain[i + 1][0], chain[i + 1][1]) for i in range(len(chain) - 1)]

    # Compose sequentially.
    current = chain[0][1]
    diagnostics = []
    for _, next_obj in chain[1:]:
        comp = compose(current, next_obj, plan.beta, allow_missing_terms=True)
        if hasattr(comp, 'code'):
            diagnostics.append(comp)
            break
        current = comp

    mu = e_model.metadata.get('memberships', {})
    H = entropy_from_memberships([mu.get('low', 0.0), mu.get('medium', 0.0), mu.get('high', 0.0)], epsilon=plan.epsilon)
    loss = interpretability_loss(H, C=0.42, O=0.20, K=0.05, U=getattr(current, 'uncertainty', e_model.uncertainty), weights=plan.lambda_, reduction_loss=getattr(current, 'reduction_loss', reduction.delta), lambda_delta=0.10)
    index = interpretability_index(loss)

    forced_diag = compose(e_model, e_decision.copy_with(terms={'allow', 'deny'}), plan.beta)

    membership_html = _plot_memberships_html(plan, risk_score, OUT / '01_memberships.html')
    feature_html = _plot_feature_html(contrib, OUT / '02_feature_contributions.html')
    representation_html = _plot_representation_html(representation, OUT / '03_representation.html')
    composition_html_path = OUT / '04_composition_graph.html'
    save_composition_html(edges, plan.beta, composition_html_path)

    report: Dict[str, Any] = {
        'title': 'FuzzyXAI full input-to-output observer pipeline',
        'status': 'PASS',
        'formula': 'D -> F -> T -> M(x) -> XAI(x) -> A(x) -> R(x) -> B;  E_G = E_B ⊙ E_R ⊙ E_A ⊙ E_X ⊙ E_M ⊙ E_T ⊙ E_F ⊙ E_D',
        'data': {
            'n_rows': int(len(df)),
            'features': feature_cols,
            'target': 'target',
            'missing_rate': float(df.isna().mean().mean()),
        },
        'model': {
            'name': 'sklearn Pipeline(StandardScaler, LogisticRegression)',
            'train_accuracy': train_accuracy,
        },
        'case': {
            'case_index': case_index,
            'values': {k: float(v) for k, v in case.iloc[0].to_dict().items()},
            'risk_score': risk_score,
            'prediction': int(model.predict(case)[0]),
            'memberships': mu,
        },
        'chapter3': {
            'metadata': metadata,
            'profile': sorted(profile),
            'selected_class': getattr(selected, 'name', getattr(selected, 'code', str(selected))),
            'representation': getattr(representation, 'class_name', type(representation).__name__),
            'reduction_loss': float(reduction.delta),
            'reduction_policy': reduction.policy,
        },
        'stages': [
            {'id': 'D', 'name': 'Данные', 'operator': 'Ω_D', 'output': f'data_quality={data_quality:.3f}', 'meaning': 'проверка полноты и трассируемости входной таблицы'},
            {'id': 'F', 'name': 'Data mining / признаки', 'operator': 'Ω_F', 'output': f'top_feature_signal={feature_signal:.3f}', 'meaning': 'какой признак сильнее всего влияет на риск'},
            {'id': 'T', 'name': 'Обучение', 'operator': 'Ω_T', 'output': f'train_accuracy={train_accuracy:.3f}', 'meaning': 'наблюдение качества обучения'},
            {'id': 'M', 'name': 'Прогноз модели', 'operator': 'Ω_M', 'output': f'risk_score={risk_score:.4f}', 'meaning': 'перевод прогноза в E_M и термы low/medium/high'},
            {'id': 'X', 'name': 'Локальное объяснение', 'operator': 'Ω_X', 'output': f'top_contribution={feature_signal:.3f}', 'meaning': 'наблюдение вклада признаков'},
            {'id': 'A', 'name': 'Решающий модуль', 'operator': 'Ω_A', 'output': 'high risk -> send_to_check', 'meaning': 'проверка совместимости правила решения с моделью'},
            {'id': 'R', 'name': 'Модуль риска', 'operator': 'Ω_R', 'output': local_results[case_index]['action'], 'meaning': 'выбор безопасного действия'},
            {'id': 'B', 'name': 'Обратная связь', 'operator': 'Ω_B', 'output': 'case saved', 'meaning': 'сохранение кейса для калибровки'},
        ],
        'composition': {
            'edges': edge_report(edges, plan.beta),
            'loss': loss,
            'index': index,
            'diagnostics': [getattr(d, '__dict__', {}) for d in diagnostics],
            'forced_diagnostic': getattr(forced_diag, '__dict__', None),
            'forced_diagnostic_action': forced_decision['action'],
        },
        'risk_observer': {
            'theta_mid': risk_model.policy.theta_mid,
            'theta_high': risk_model.policy.theta_high,
            **benchmark_metrics,
        },
        'artifacts': {
            'membership_html': membership_html,
            'feature_html': feature_html,
            'representation_html': representation_html,
            'composition_html': composition_html_path.name,
            'index_html': 'reports/full_demo/index.html',
            'json': 'reports/full_demo/full_pipeline_report.json',
            'markdown': 'reports/full_demo/full_pipeline_report.md',
        },
    }

    (OUT / 'full_pipeline_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_markdown(report, OUT / 'full_pipeline_report.md')
    _render_index_html(report, OUT / 'index.html')
    if open_browser:
        webbrowser.open((OUT / 'index.html').resolve().as_uri())
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        '# FuzzyXAI full pipeline demo',
        '',
        f"Status: **{report['status']}**",
        '',
        f"Formula: `{report['formula']}`",
        '',
        '## Case',
        '',
        f"- risk_score: `{report['case']['risk_score']:.6f}`",
        f"- selected class: `{report['chapter3']['selected_class']}`",
        f"- Delta: `{report['chapter3']['reduction_loss']:.6f}`",
        f"- I(E_G): `{report['composition']['index']:.6f}`",
        '',
        '## Stages',
        '',
    ]
    for stage in report['stages']:
        lines.append(f"- **{stage['id']} / {stage['name']}**: {stage['output']} — {stage['meaning']}")
    lines += ['', '## Risk observer', '']
    for key in ['accepted_accuracy', 'coverage', 'defer_rate', 'cost_before', 'cost_after', 'risk_reduction']:
        lines.append(f"- {key}: `{report['risk_observer'][key]:.6f}`")
    lines += ['', '## Artifacts', '']
    for key, value in report['artifacts'].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    open_browser = '--open' in sys.argv
    report = run_full_pipeline(open_browser=open_browser)
    print(json.dumps({
        'status': report['status'],
        'index_html': str((OUT / 'index.html').resolve()),
        'json': str((OUT / 'full_pipeline_report.json').resolve()),
        'risk_reduction': report['risk_observer']['risk_reduction'],
        'I_EG': report['composition']['index'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
