"""Focused defense demo for dissertation chapters 2 and 3.

Run from repository root:
    python apps/defense_demo.py --port 8085
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import plotly.graph_objects as go
    from nicegui import ui
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit('Install requirements.txt first: pandas, plotly, nicegui, scikit-learn are required.') from exc

from fuzzyxai.core.plan_builder import build_explain_plan_from_dataframe
from fuzzyxai import Rule, SystemOperator, Trace, compose, interpretability_index, interpretability_loss
from fuzzyxai.core.trust_evaluator import activation_distance, jaccard_distance, representation_distance, trace_distance
from fuzzyxai.demo.synthetic import (
    build_demo_composition,
    build_demo_explanation,
    default_candidates,
    metadata_for_demo,
    sample_dataframe,
)
from fuzzyxai.selection.pareto_selector import pareto_front, select_minimal_sufficient
from fuzzyxai.visual.composition_graph import edge_report
from fuzzyxai.visual.representation_plots import explainplan_membership_figure, representation_figure
from benchmarks.risk_aware_observer_benchmark import build_risk_aware_observer_report

REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)


STATE: Dict[str, Any] = {
    'df': sample_dataframe(),
    'case_index': 2,
    'risk': 0.72,
    'feature': 'risk_score',
    'conflict': False,
    'mode': 'audit',
    'presentation': False,
    'model': None,
    'model_report': None,
    'plan': None,
    'explanation': None,
    'composition': None,
    'operator_benchmark': None,
    'risk_observer_benchmark': None,
    'full_pipeline_report': None,
}

MODEL_FEATURES = ['age', 'pressure', 'marker']


def fit_model_from_state() -> None:
    """Train a small real sklearn model and write its risk back into the demo table."""
    df = STATE['df'].copy()
    features = [name for name in MODEL_FEATURES if name in df.columns]
    target = 'target' if 'target' in df.columns else df.columns[-1]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(df[features], df[target])
    risks = model.predict_proba(df[features])[:, 1]
    df['risk_score'] = risks
    pred = (risks >= 0.5).astype(int)
    case_index = int(max(0, min(int(STATE.get('case_index', 0)), len(df) - 1)))
    STATE['case_index'] = case_index
    STATE['risk'] = float(risks[case_index])
    STATE['df'] = df
    STATE['model'] = model
    STATE['model_report'] = {
        'features': features,
        'target': target,
        'accuracy': float(accuracy_score(df[target], pred)),
        'case_index': case_index,
        'risk': float(risks[case_index]),
        'prediction': int(pred[case_index]),
    }


def build_plan_from_state() -> None:
    fit_model_from_state()
    df = STATE['df']
    target = 'target' if 'target' in df.columns else df.columns[-1]
    STATE['plan'] = build_explain_plan_from_dataframe(df, target=target, mode=STATE['mode']).with_reduction_weight(0.10)
    numeric = STATE['plan'].metadata.get('numeric_features', [])
    if STATE.get('feature') not in numeric and numeric:
        STATE['feature'] = numeric[0]


def recompute() -> None:
    if STATE.get('plan') is None:
        build_plan_from_state()
    metadata = metadata_for_demo(intervals=True, experts=2, conflict=True, audit=STATE['mode'] == 'audit')
    STATE['explanation'] = build_demo_explanation(
        float(STATE['risk']),
        plan=STATE['plan'],
        metadata=metadata,
        audience='doctor',
        use_llm=False,
    )
    STATE['composition'] = build_demo_composition(
        float(STATE['risk']),
        min(1.0, float(STATE['risk']) + 0.02),
        plan=STATE['plan'],
        conflict=bool(STATE['conflict']),
    )


def safe(fn, *, where: str) -> None:
    try:
        fn()
    except Exception as exc:  # pragma: no cover - visible GUI safety net
        ui.notify(f'{where}: {exc}', type='negative', timeout=7000)
        (REPORTS / 'defense_demo_last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')


def apply_style() -> None:
    ui.add_head_html(
        """
        <style>
        :root {
          --bg: #f6f7f9;
          --panel: #ffffff;
          --text: #16202a;
          --muted: #637083;
          --line: #d9dee7;
          --blue: #2563eb;
          --green: #0f9f6e;
          --amber: #c47b00;
          --red: #d83a3a;
        }
        body { background: var(--bg); color: var(--text); }
        .fx-shell { max-width: 1440px; margin: 0 auto; }
        .fx-topbar {
          background: rgba(255,255,255,0.96); border-bottom: 1px solid var(--line);
          position: sticky; top: 0; z-index: 20;
          backdrop-filter: blur(10px);
        }
        .fx-panel {
          background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 18px;
          box-shadow: 0 10px 26px rgba(22, 32, 42, 0.06);
        }
        .fx-metric {
          background: #f9fafb; border: 1px solid #e5e7eb;
          border-radius: 8px; padding: 12px; min-width: 150px; flex: 1;
        }
        .fx-muted { color: var(--muted); }
        .fx-title { font-weight: 760; letter-spacing: 0; }
        .fx-chip {
          display: inline-block; border-radius: 999px; padding: 3px 8px;
          background: #eef2ff; color: #1f3a8a; font-size: 12px; font-weight: 650;
        }
        .fx-chip-green { background: #e7f8f1; color: #08684a; }
        .fx-chip-red { background: #fdecec; color: #9f2323; }
        .fx-note {
          background: #f8fafc; border: 1px solid #e2e8f0;
          border-radius: 8px; padding: 10px; color: #334155;
        }
        .fx-status-ok {
          background: #e7f8f1; border: 1px solid #a6e3c8;
          border-radius: 8px; padding: 12px; color: #08684a;
        }
        .fx-status-bad {
          background: #fdecec; border: 1px solid #f4b4b4;
          border-radius: 8px; padding: 12px; color: #9f2323;
        }
        .fx-step {
          border-left: 4px solid var(--blue); padding-left: 10px;
        }
        .fx-route {
          display: grid; grid-template-columns: repeat(8, minmax(0, 1fr));
          gap: 8px;
        }
        .fx-route-item {
          background: #ffffff; border: 1px solid var(--line);
          border-radius: 8px; padding: 10px;
        }
        .fx-route-item strong { display: block; font-size: 13px; }
        .fx-route-item span { display: block; font-size: 12px; color: var(--muted); }
        .fx-case {
          background: linear-gradient(135deg, #f8fafc, #eef6ff);
          border: 1px solid #cfe0f7; border-radius: 8px; padding: 12px;
        }
        .fx-presentation .fx-panel { padding: 24px; }
        .fx-presentation .fx-title { font-size: 1.08em; }
        .fx-presentation .fx-muted,
        .fx-presentation .text-sm,
        .fx-presentation .text-xs { font-size: 15px; }
        .q-table th, .q-table td { font-size: 12px; }
        .js-plotly-plot .plotly .main-svg { border-radius: 8px; }
        @media print {
          .fx-topbar { position: static; }
          .q-btn, .q-field, .q-toggle { display: none !important; }
          body { background: #ffffff; }
          .fx-panel { break-inside: avoid; box-shadow: none; }
        }
        </style>
        """
    )


def metric(title: str, value: Any, caption: str = '') -> None:
    with ui.element('div').classes('fx-metric'):
        ui.label(title).classes('text-xs fx-muted')
        ui.label(str(value)).classes('text-xl fx-title')
        if caption:
            ui.label(caption).classes('text-xs fx-muted')


def row_metrics(items) -> None:
    with ui.row().classes('w-full gap-2'):
        for item in items:
            metric(*item)


def feature_options() -> list[str]:
    plan = STATE['plan']
    if plan is None:
        return []
    return list(plan.metadata.get('numeric_features', []))


def quantile_rows() -> list[dict[str, Any]]:
    plan = STATE['plan']
    feature = STATE['feature']
    terms = ((plan.metadata or {}).get('feature_terms', {}).get(feature, {}) if plan else {})
    q = terms.get('quantiles', {})
    term_specs = terms.get('terms', {})
    rows = []
    for name, spec in term_specs.items():
        rows.append({
            'term': name,
            'shape': spec.get('kind', ''),
            'left': round(float(spec.get('a', q.get('min', 0.0))), 4),
            'center': round(float(spec.get('b', q.get('median', 0.0))), 4),
            'right': round(float(spec.get('c', q.get('max', 0.0))), 4) if 'c' in spec else '',
        })
    return rows


def current_case_row() -> dict[str, Any]:
    df = STATE['df']
    idx = int(max(0, min(int(STATE.get('case_index', 0)), len(df) - 1)))
    return df.iloc[idx].to_dict()


def model_contributions() -> list[dict[str, Any]]:
    model = STATE.get('model')
    report = STATE.get('model_report') or {}
    if model is None:
        return []
    features = list(report.get('features', MODEL_FEATURES))
    case = current_case_row()
    x = pd.DataFrame([{feature: case[feature] for feature in features}])
    scaler = model.named_steps['standardscaler']
    clf = model.named_steps['logisticregression']
    z = scaler.transform(x)[0]
    coefs = clf.coef_[0]
    rows = []
    for feature, raw, scaled, coef in zip(features, [case[f] for f in features], z, coefs):
        rows.append({
            'feature': feature,
            'value': float(raw),
            'coef': float(coef),
            'contribution': float(coef * scaled),
        })
    rows.sort(key=lambda row: abs(row['contribution']), reverse=True)
    return rows


def model_contribution_figure() -> go.Figure:
    rows = model_contributions()
    colors = ['#d83a3a' if row['contribution'] >= 0 else '#0f9f6e' for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row['contribution'] for row in rows],
        y=[row['feature'] for row in rows],
        orientation='h',
        marker_color=colors,
        text=[f"{row['contribution']:+.2f}" for row in rows],
        textposition='outside',
        hovertemplate='%{y}<br>вклад=%{x:.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1, line_color='#64748b')
    fig.update_layout(
        title='Как модель получила риск для выбранного кейса',
        xaxis_title='вклад в logit: вправо риск выше, влево ниже',
        yaxis_title='признак',
        height=300,
        margin=dict(l=86, r=28, t=56, b=42),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor='#edf1f6', zeroline=False)
    fig.update_yaxes(showgrid=False, autorange='reversed')
    return fig


def model_pipeline_figure() -> go.Figure:
    report = STATE.get('model_report') or {}
    risk = float(report.get('risk', STATE['risk']))
    fig = go.Figure()
    boxes = [
        (0.05, 0.24, 'Признаки пациента', '<br>'.join(report.get('features', MODEL_FEATURES))),
        (0.38, 0.57, 'LogisticRegression', f"accuracy={report.get('accuracy', 0):.2f}"),
        (0.72, 0.94, 'risk_score', f'{risk:.3f}'),
    ]
    for x0, x1, title, caption in boxes:
        fig.add_shape(type='rect', x0=x0, y0=0.30, x1=x1, y1=0.78, line=dict(color='#cbd5e1'), fillcolor='#ffffff')
        fig.add_annotation(x=(x0 + x1) / 2, y=0.64, text=f'<b>{title}</b>', showarrow=False, font=dict(size=15))
        fig.add_annotation(x=(x0 + x1) / 2, y=0.47, text=caption, showarrow=False, font=dict(size=12, color='#637083'))
    for x0, x1 in [(0.25, 0.37), (0.58, 0.71)]:
        fig.add_annotation(x=x1, y=0.54, ax=x0, ay=0.54, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=3, arrowwidth=4, arrowcolor='#2563eb')
    fig.add_annotation(x=0.83, y=0.17, text='Этот риск дальше идёт в ExplainPlan, E_k, A_k^F и оператор композиции.', showarrow=False, font=dict(size=12, color='#334155'))
    fig.update_layout(
        title='Сквозной контур: модель -> риск -> оператор',
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=260,
        margin=dict(l=16, r=16, t=54, b=16),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    return fig


def operator_benchmark_report() -> dict[str, Any]:
    """Open medical benchmark: model alone vs model plus chapter-2 operator."""
    cached = STATE.get('operator_benchmark')
    if cached is not None:
        return cached

    data = load_breast_cancer(as_frame=True)
    x = data.data
    y_malignant = (data.target == 0).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_malignant, test_size=0.25, random_state=42, stratify=y_malignant
    )
    model = RandomForestClassifier(n_estimators=140, max_depth=5, random_state=42)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    pred = (scores >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, pred))
    auc = float(roc_auc_score(y_test, scores))

    case_pos = int(min(range(len(scores)), key=lambda idx: abs(float(scores[idx]) - 0.72)))
    case_risk = float(scores[case_pos])
    case_values = x_test.iloc[case_pos].to_dict()
    importances = sorted(
        [
            {'feature': str(feature), 'importance': float(value), 'value': float(case_values[feature])}
            for feature, value in zip(x.columns, model.feature_importances_)
        ],
        key=lambda row: row['importance'],
        reverse=True,
    )[:8]

    plan_df = x_train.copy()
    plan_df['target'] = list(y_train)
    plan = build_explain_plan_from_dataframe(plan_df, target='target', mode='audit').with_reduction_weight(0.10)
    op = SystemOperator(plan)
    model_rules = [
        Rule('rf_high_malignancy', {'risk': 'high'}, 'urgent_review'),
        Rule('rf_medium_malignancy', {'risk': 'medium'}, 'additional_screening'),
    ]
    decision_rules = [Rule('decision_high_risk', {'risk': 'high'}, 'send_to_oncologist')]
    e_model = op.explain_scalar_risk(
        case_risk,
        model_rules,
        Trace('breast-cancer-rf-model', 'v1', '2026-05-22', source='sklearn_breast_cancer', checksum='rf'),
        model_uncertainty=1.0 - max(case_risk, 1.0 - case_risk),
        trace_uncertainty=0.01,
    )
    e_decision = op.explain_scalar_risk(
        min(1.0, case_risk + 0.03),
        decision_rules,
        Trace('breast-cancer-decision', 'v1', '2026-05-22', source='clinical-protocol-demo', checksum='decision'),
        model_uncertainty=0.08,
        trace_uncertainty=0.01,
    )
    composed = compose(e_model, e_decision, plan.beta)
    gamma = None
    index = None
    if not hasattr(composed, 'code'):
        gamma = float(composed.metadata.get('gamma', 0.0))
        loss = interpretability_loss(0.30, 0.33, 0.16, 0.03, composed.uncertainty, plan.lambda_, composed.reduction_loss, 0.10)
        index = float(interpretability_index(loss))
    diagnostic = compose(e_model, e_decision.copy_with(terms={'approve', 'deny'}), plan.beta)

    report = {
        'dataset': 'sklearn breast_cancer',
        'n_samples': int(len(x)),
        'n_features': int(x.shape[1]),
        'model': 'RandomForestClassifier',
        'accuracy': accuracy,
        'roc_auc': auc,
        'case_risk': case_risk,
        'case_true_malignant': int(y_test.iloc[case_pos]),
        'top_importances': importances,
        'without_operator': {
            'shows_risk': True,
            'shows_feature_importance': True,
            'semantic_gamma': None,
            'interpretability_index': None,
            'detects_term_conflict': False,
            'verdict': 'локальная модель есть, проверки цепочки нет',
        },
        'with_operator': {
            'shows_risk': True,
            'shows_feature_importance': True,
            'semantic_gamma': gamma,
            'interpretability_index': index,
            'detects_term_conflict': getattr(diagnostic, 'code', None) == 'D_ij',
            'diagnostic': getattr(diagnostic, 'code', None),
            'verdict': 'появляется композиция объяснений и диагностика D_ij',
        },
    }
    STATE['operator_benchmark'] = report
    return report


def benchmark_quality_figure() -> go.Figure:
    report = operator_benchmark_report()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['accuracy', 'ROC-AUC'],
        y=[report['accuracy'], report['roc_auc']],
        marker_color=['#2563eb', '#0f9f6e'],
        text=[f"{report['accuracy']:.3f}", f"{report['roc_auc']:.3f}"],
        textposition='outside',
    ))
    fig.update_layout(
        title='Нормальная модель на открытом медицинском датасете',
        yaxis=dict(range=[0, 1.08], title='score', showgrid=True, gridcolor='#edf1f6'),
        xaxis=dict(showgrid=False),
        height=280,
        margin=dict(l=42, r=20, t=56, b=40),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    return fig


def benchmark_importance_figure() -> go.Figure:
    rows = operator_benchmark_report()['top_importances']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row['importance'] for row in rows],
        y=[row['feature'] for row in rows],
        orientation='h',
        marker_color='#7aa2f7',
        text=[f"{row['importance']:.3f}" for row in rows],
        textposition='outside',
    ))
    fig.update_layout(
        title='Что показывает модель без оператора',
        xaxis_title='global feature importance',
        yaxis_title='признак',
        height=330,
        margin=dict(l=150, r=24, t=56, b=42),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor='#edf1f6', zeroline=False)
    fig.update_yaxes(showgrid=False, autorange='reversed')
    return fig


def operator_added_value_figure() -> go.Figure:
    report = operator_benchmark_report()
    without = report['without_operator']
    with_op = report['with_operator']
    fig = go.Figure()
    labels = ['risk', 'feature importance', 'gamma', 'I(E_G)', 'D_ij conflict']
    fig.add_trace(go.Bar(
        name='без оператора',
        x=labels,
        y=[1, 1, 0, 0, 0],
        marker_color='#94a3b8',
        text=['есть', 'есть', 'нет', 'нет', 'нет'],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='с оператором',
        x=labels,
        y=[
            1,
            1,
            1 if with_op['semantic_gamma'] is not None else 0,
            1 if with_op['interpretability_index'] is not None else 0,
            1 if with_op['detects_term_conflict'] else 0,
        ],
        marker_color='#0f9f6e',
        text=[
            'есть',
            'есть',
            f"{with_op['semantic_gamma']:.3f}" if with_op['semantic_gamma'] is not None else 'нет',
            f"{with_op['interpretability_index']:.3f}" if with_op['interpretability_index'] is not None else 'нет',
            with_op['diagnostic'] or 'нет',
        ],
        textposition='outside',
    ))
    fig.update_layout(
        title='Что добавляет нечёткий системный оператор',
        yaxis=dict(range=[0, 1.18], visible=False),
        xaxis=dict(showgrid=False),
        barmode='group',
        height=330,
        margin=dict(l=24, r=24, t=56, b=60),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.30, x=0),
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f"Без оператора: {without['verdict']}<br>С оператором: {with_op['verdict']}",
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#d9dee7',
        font=dict(size=12, color='#334155'),
    )
    return fig


def risk_observer_report() -> dict[str, Any]:
    cached = STATE.get('risk_observer_benchmark')
    if cached is None:
        cached = build_risk_aware_observer_report(write=False)
        STATE['risk_observer_benchmark'] = cached
    return cached


def risk_observer_cost_figure() -> go.Figure:
    metrics = risk_observer_report()['observer_metrics']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['до наблюдателя', 'после наблюдателя'],
        y=[metrics['expected_cost_before'], metrics['expected_cost_after']],
        marker_color=['#d83a3a', '#0f9f6e'],
        text=[f"{metrics['expected_cost_before']:.3f}", f"{metrics['expected_cost_after']:.3f}"],
        textposition='outside',
    ))
    fig.update_layout(
        title='Ожидаемая стоимость решения',
        yaxis=dict(title='expected cost', rangemode='tozero', showgrid=True, gridcolor='#edf1f6'),
        xaxis=dict(showgrid=False),
        height=300,
        margin=dict(l=52, r=24, t=56, b=42),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    return fig


def risk_observer_actions_figure() -> go.Figure:
    counts = risk_observer_report()['observer_metrics']['action_counts']
    colors = {
        'accept': '#0f9f6e',
        'lower_confidence': '#7aa2f7',
        'request_more_data': '#c47b00',
        'defer_to_human': '#d83a3a',
        'block': '#7f1d1d',
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=[colors.get(key, '#94a3b8') for key in counts],
        text=[str(v) for v in counts.values()],
        textposition='outside',
    ))
    fig.update_layout(
        title='Какие действия выбрал наблюдатель',
        xaxis_title='действие',
        yaxis_title='кейсов',
        height=300,
        margin=dict(l=52, r=24, t=56, b=42),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor='#edf1f6', rangemode='tozero')
    return fig


def selected_feature_value() -> float | None:
    case = current_case_row()
    value = case.get(STATE['feature'])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plan_membership_demo_figure() -> go.Figure:
    """ExplainPlan curves plus the current case marker."""
    fig = explainplan_membership_figure(STATE['plan'], STATE['feature'], df=STATE['df'])
    value = selected_feature_value()
    if value is None:
        return fig
    fig.add_vline(x=value, line_width=3, line_dash='dash', line_color='#16202a')
    fig.add_annotation(
        x=value,
        y=1.03,
        xref='x',
        yref='paper',
        text=f'текущий кейс: {value:.3g}',
        showarrow=True,
        arrowhead=2,
        ax=44,
        ay=-38,
        bgcolor='#ffffff',
        bordercolor='#16202a',
        font=dict(color='#16202a', size=12),
    )
    return fig


def target_distribution_figure() -> go.Figure:
    df = STATE['df']
    fig = go.Figure()
    if 'target' in df.columns:
        counts = df['target'].value_counts().sort_index()
        labels = [f'class {idx}' for idx in counts.index]
        fig.add_trace(go.Bar(
            x=labels,
            y=counts.values,
            marker_color=['#7aa2f7', '#0f9f6e'][:len(counts)],
            text=[str(v) for v in counts.values],
            textposition='outside',
        ))
    fig.update_layout(
        title='Баланс классов в demo-данных',
        xaxis_title='класс',
        yaxis_title='объектов',
        height=250,
        margin=dict(l=42, r=16, t=54, b=38),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor='#edf1f6', rangemode='tozero')
    fig.update_xaxes(showgrid=False)
    return fig


def membership_bar() -> go.Figure:
    expl = STATE['explanation']
    memberships = expl['object'].metadata.get('memberships', {}) if expl else {}
    colors = {'low': '#0f9f6e', 'medium': '#c47b00', 'high': '#d83a3a'}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(memberships.keys()),
        y=list(memberships.values()),
        marker_color=[colors.get(k, '#2563eb') for k in memberships.keys()],
        text=[f'{v:.2f}' for v in memberships.values()],
        textposition='outside',
    ))
    fig.update_layout(
        title='Как этот риск попал в термы',
        xaxis_title='терм',
        yaxis_title='степень принадлежности',
        yaxis=dict(range=[0, 1.12], showgrid=True, gridcolor='#edf1f6'),
        xaxis=dict(showgrid=False),
        height=300,
        margin=dict(l=45, r=18, t=56, b=42),
        showlegend=False,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    return fig


def representation_layers_figure(expl: Mapping[str, Any]) -> go.Figure:
    """Show chapter-3 representation as visible uncertainty layers."""
    rep = expl['representation']
    risk = float(expl['report']['risk'])
    if not hasattr(rep, 'levels'):
        return representation_figure(rep, title='A_k^F: представление неопределенности')

    fig = go.Figure()
    level_names = ['Интервал', 'Эксперты', 'Конфликт']
    for idx, level in enumerate(rep.levels):
        label = level_names[idx] if idx < len(level_names) else f'Уровень {idx + 1}'
        try:
            value = level.membership(risk)
        except Exception:
            continue
        cls_name = getattr(level, 'class_name', type(level).__name__)
        if cls_name == 'FI':
            lo, hi = float(value[0]), float(value[1])
            fig.add_trace(go.Scatter(
                x=[lo, hi],
                y=[label, label],
                mode='lines+markers+text',
                text=[f'{lo:.2f}', f'{hi:.2f}'],
                textposition='top center',
                name='интервал уверенности',
                line=dict(width=12, color='#2563eb'),
                marker=dict(size=13, color='#2563eb', line=dict(color='white', width=1)),
                hovertemplate='интервал: %{x:.3f}<extra></extra>',
            ))
        elif cls_name == 'FH':
            values = [float(v) for v in value]
            fig.add_trace(go.Scatter(
                x=values,
                y=[label] * len(values),
                mode='markers+text',
                text=[f'эксперт {i + 1}' for i in range(len(values))],
                textposition='top center',
                name='экспертные оценки',
                marker=dict(size=17, color='#c47b00', line=dict(color='white', width=1)),
                hovertemplate='оценка=%{x:.3f}<extra></extra>',
            ))
        elif cls_name in {'FNsrc', 'FN'}:
            t, ind, f = (float(v) for v in value)
            fig.add_trace(go.Bar(
                x=[t, ind, f],
                y=['T: поддержка', 'I: неопределенность', 'F: возражение'],
                orientation='h',
                name='нейтрософский слой',
                marker_color=['#0f9f6e', '#c47b00', '#d83a3a'],
                text=[f'{v:.2f}' for v in (t, ind, f)],
                textposition='outside',
                hovertemplate='%{y}: %{x:.3f}<extra></extra>',
            ))

    fig.update_layout(
        title='A_k^F: из чего состоит выбранное представление',
        xaxis_title='степень / оценка',
        yaxis_title='слой',
        xaxis=dict(range=[0, 1.08], showgrid=True, gridcolor='#edf1f6', zeroline=False),
        yaxis=dict(showgrid=False),
        height=330,
        margin=dict(l=126, r=24, t=56, b=40),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.5,
        y=-0.22,
        xref='paper',
        yref='paper',
        text='Интервал = разброс риска; эксперты = разные мнения; T/I/F = поддержка, неопределенность и конфликт.',
        showarrow=False,
        font=dict(size=12, color='#637083'),
    )
    return fig


def selection_figure() -> go.Figure:
    """Visual explanation of the chapter-3 class selection."""
    expl = STATE['explanation']
    profile = set(expl['profile'])
    selected_name = expl['report']['selected_class']
    candidates = default_candidates()
    front = {c.name for c in pareto_front([c for c in candidates if c.covers(profile)])}

    fig = go.Figure()
    for candidate in candidates:
        covers = candidate.covers(profile)
        selected = candidate.name == selected_name
        color = '#0f9f6e' if selected else ('#2563eb' if covers else '#94a3b8')
        symbol = 'star' if selected else ('circle' if candidate.name in front else 'circle-open')
        fig.add_trace(go.Scatter(
            x=[candidate.cognitive_complexity],
            y=[candidate.expected_reduction_loss],
            mode='markers+text',
            text=[candidate.name],
            textposition='top center',
            name=candidate.name,
            marker=dict(
                size=22 + 18 * float(candidate.computational_complexity),
                color=color,
                symbol=symbol,
                line=dict(color='white', width=1.5),
            ),
            hovertemplate=(
                f'{candidate.name}<br>'
                f'покрывает профиль: {covers}<br>'
                'C_cog=%{x}<br>Delta=%{y}<extra></extra>'
            ),
            showlegend=False,
        ))
    fig.update_layout(
        title='Почему выбран именно этот класс из главы 3',
        xaxis_title='когнитивная сложность: проще левее',
        yaxis_title='ожидаемая потеря: лучше ниже',
        height=320,
        margin=dict(l=58, r=18, t=56, b=50),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#edf1f6', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#edf1f6', zeroline=False)
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text='Зеленая звезда = выбранный минимум: достаточно выразительно, но не перегружено.',
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.88)',
        bordercolor='#d9dee7',
        font=dict(size=12, color='#334155'),
    )
    return fig


def readable_class_name(name: str) -> str:
    labels = {
        'F0': 'простая нечёткость',
        'FI': 'интервалы',
        'FH': 'несколько экспертных оценок',
        'FNsrc': 'конфликт источников',
        'FML-audit': 'многоуровневое представление для аудита',
        'FML-user': 'многоуровневое представление для пользователя',
    }
    return labels.get(name, name)


def agreement_level(gamma: float) -> tuple[str, str, str]:
    if gamma < 0.20:
        return 'низкое', '#0f9f6e', 'компоненты почти не спорят'
    if gamma < 0.45:
        return 'среднее', '#c47b00', 'есть заметное расхождение'
    return 'высокое', '#d83a3a', 'нужна проверка'


def composition_story_figure(comp: Mapping[str, Any], rows: list[dict[str, Any]]) -> go.Figure:
    """Human-readable diagram for model-to-decision agreement."""
    comp_obj = comp['comp']
    diagnostic = getattr(comp_obj, 'code', None)
    gamma = float(rows[0]['gamma']) if rows else 0.0
    level, color, caption = agreement_level(gamma)
    left_terms = ', '.join(rows[0].get('left_terms', [])) if rows else 'n/a'
    right_terms = ', '.join(rows[0].get('right_terms', [])) if rows else 'n/a'
    status = 'РАЗРЫВ ОБНАРУЖЕН' if diagnostic else 'СОГЛАСОВАНО'
    status_color = '#d83a3a' if diagnostic else color

    fig = go.Figure()
    fig.add_shape(type='rect', x0=0.04, y0=0.30, x1=0.35, y1=0.82, line=dict(color='#cbd5e1', width=1.5), fillcolor='#ffffff')
    fig.add_shape(type='rect', x0=0.65, y0=0.30, x1=0.96, y1=0.82, line=dict(color='#cbd5e1', width=1.5), fillcolor='#ffffff')
    fig.add_shape(type='rect', x0=0.40, y0=0.43, x1=0.60, y1=0.61, line=dict(color=status_color, width=1), fillcolor='rgba(255,255,255,0.88)')
    fig.add_annotation(x=0.63, y=0.52, ax=0.37, ay=0.52, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=5, arrowcolor=status_color)
    fig.add_annotation(x=0.50, y=0.56, text=f'{gamma:.3f}', showarrow=False, font=dict(color=status_color, size=20))
    fig.add_annotation(x=0.50, y=0.47, text='расхождение', showarrow=False, font=dict(color='#637083', size=11))
    fig.add_annotation(x=0.195, y=0.69, text='<b>Модель риска</b>', showarrow=False, font=dict(size=16, color='#16202a'))
    fig.add_annotation(x=0.195, y=0.53, text=f'термы<br>{left_terms}', showarrow=False, font=dict(size=12, color='#637083'))
    fig.add_annotation(x=0.805, y=0.69, text='<b>Модуль решения</b>', showarrow=False, font=dict(size=16, color='#16202a'))
    fig.add_annotation(x=0.805, y=0.53, text=f'термы<br>{right_terms}', showarrow=False, font=dict(size=12, color='#637083'))
    fig.add_annotation(x=0.50, y=0.30, text=caption, showarrow=False, font=dict(color='#637083', size=13))
    fig.add_annotation(
        x=0.50,
        y=0.13,
        text=f'<b>{status}</b>' if not diagnostic else f'<b>{status}</b>: {diagnostic}',
        showarrow=False,
        font=dict(color=status_color, size=20),
    )
    fig.update_layout(
        title='Проверка цепочки объяснения',
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=340,
        margin=dict(l=16, r=16, t=54, b=16),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    return fig


def composition_breakdown_rows(comp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explain which parts of gamma made the model-to-decision edge risky."""
    if not comp.get('edges'):
        return []
    _, left, _, right = comp['edges'][0]
    beta = comp['plan'].beta
    pieces = [
        ('repr', 'представления A^F', representation_distance(left, right), 'похожи ли выбранные нечёткие представления'),
        ('rules', 'активные правила', jaccard_distance(left.active_rules, right.active_rules), 'совпадает ли логика правил'),
        ('activations', 'активации правил', activation_distance(left, right), 'насколько похожи численные степени срабатывания'),
        ('uncertainty', 'неопределённость', abs(left.uncertainty - right.uncertainty), 'одинаково ли уверены компоненты'),
        ('trace', 'след tau', trace_distance(left.trace.as_dict(), right.trace.as_dict()), 'один ли источник, версия и параметры'),
        ('reduction', 'потеря редукции', min(1.0, left.reduction_loss + right.reduction_loss), 'сколько потеряно при упрощении'),
    ]
    rows = []
    for key, label, raw, meaning in pieces:
        weight = float(beta.get(key, 0.0))
        rows.append({
            'part': label,
            'weight': round(weight, 4),
            'distance': round(float(raw), 4),
            'contribution': round(weight * float(raw), 4),
            'meaning': meaning,
        })
    rows.sort(key=lambda row: row['contribution'], reverse=True)
    return rows


def composition_breakdown_figure(comp: Mapping[str, Any]) -> go.Figure:
    rows = composition_breakdown_rows(comp)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row['contribution'] for row in rows],
        y=[row['part'] for row in rows],
        orientation='h',
        marker_color=['#d83a3a' if row['contribution'] >= 0.12 else '#c47b00' if row['contribution'] >= 0.05 else '#0f9f6e' for row in rows],
        text=[f"{row['contribution']:.3f}" for row in rows],
        textposition='outside',
        hovertemplate='%{y}<br>вклад=%{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title='Из чего складывается рассогласование gamma',
        xaxis_title='beta_i * distance_i',
        yaxis_title='компонент',
        height=320,
        margin=dict(l=132, r=24, t=56, b=42),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor='#edf1f6', zeroline=False)
    fig.update_yaxes(showgrid=False, autorange='reversed')
    return fig


def summary_report() -> dict[str, Any]:
    expl = STATE['explanation']
    comp = STATE['composition']
    comp_obj = comp['comp']
    edge_rows = edge_report(comp['edges'], comp['plan'].beta)
    return {
        'risk': STATE['risk'],
        'mode': STATE['mode'],
        'model': STATE.get('model_report'),
        'case': current_case_row(),
        'selected_class': expl['report']['selected_class'],
        'reduction_loss_delta': expl['report']['reduction_loss'],
        'uncertainty': expl['report']['uncertainty'],
        'composition': {
            'diagnostic': getattr(comp_obj, 'code', None),
            'gamma': edge_rows[0]['gamma'] if edge_rows else None,
            'loss': comp.get('loss'),
            'index': comp.get('index'),
        },
    }


def download_report() -> None:
    payload = summary_report()
    path = REPORTS / 'defense_demo_report.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    ui.download(path.read_bytes(), 'defense_demo_report.json')


def print_report() -> None:
    ui.run_javascript('window.print()')


def tour_dialog() -> None:
    with ui.dialog() as dialog, ui.card().classes('w-[760px] max-w-full'):
        ui.label('Экскурсия по демо').classes('text-xl fx-title')
        ui.markdown(
            '1. **Данные -> термы**: числовой признак превращается в low / medium / high.\n\n'
            '2. **Кейс -> объяснение**: система выбирает класс представления из главы 3 и строит `E_k`.\n\n'
            '3. **Модель -> решение**: оператор из главы 2 проверяет, согласованы ли компоненты.\n\n'
            '4. Включи **показать конфликт**: появится `D_ij`, то есть найдено место разрыва объяснимости.'
        )
        with ui.row().classes('justify-end w-full'):
            ui.button('Закрыть', on_click=dialog.close).props('color=primary')
    dialog.open()


def route_strip() -> None:
    with ui.element('div').classes('fx-route w-full'):
        for title, caption in [
            ('Модель', 'age/pressure/marker -> risk'),
            ('Данные', 'sample medical table'),
            ('ExplainPlan', 'термы и веса'),
            ('E_k', 'объяснение кейса'),
            ('A_k^F', 'класс главы 3'),
            ('D_ij / I(E_G)', 'проверка цепочки'),
            ('Benchmark', 'with / without operator'),
            ('Observer', 'risk-aware gate'),
        ]:
            with ui.element('div').classes('fx-route-item'):
                ui.html(f'<strong>{title}</strong><span>{caption}</span>')


def controls(on_change) -> None:
    df = STATE['df']
    with ui.element('div').classes('fx-topbar w-full'):
        with ui.row().classes('fx-shell w-full items-center gap-3 p-3'):
            ui.label('FuzzyXAI: демонстрация глав 2-3').classes('text-lg fx-title')
            ui.label('Главы 2-3').classes('fx-chip')
            ui.label('кейс').classes('text-sm fx-muted')
            case = ui.slider(min=0, max=max(0, len(df) - 1), value=STATE['case_index'], step=1).props('label-always').classes('w-44')
            mode = ui.select(['audit', 'user'], value=STATE['mode'], label='режим').classes('w-36')
            feature = ui.select(feature_options(), value=STATE['feature'], label='признак').classes('w-48')
            conflict = ui.switch('показать конфликт', value=STATE['conflict'])
            presentation = ui.switch('презентация', value=STATE['presentation'])

            def sync() -> None:
                STATE['case_index'] = int(case.value)
                STATE['mode'] = str(mode.value)
                STATE['feature'] = str(feature.value)
                STATE['conflict'] = bool(conflict.value)
                STATE['presentation'] = bool(presentation.value)
                build_plan_from_state()
                recompute()
                feature.options = feature_options()
                feature.update()
                on_change()

            for control in (case, mode, feature, conflict, presentation):
                control.on_value_change(lambda e: safe(sync, where='recompute'))
            ui.button(icon='refresh', on_click=lambda: safe(sync, where='recompute')).props('flat round color=primary')
            ui.button(icon='help_outline', on_click=tour_dialog).props('flat round')
            ui.button(icon='print', on_click=print_report).props('flat round')
            ui.button(icon='download', on_click=download_report).props('flat round')
            ui.label(f'{len(df)} rows').classes('text-sm fx-muted ml-auto')


def model_section() -> None:
    report = STATE.get('model_report') or {}
    case = current_case_row()
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('0. Реальная модель, а не абстракция').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Сначала обучается sklearn LogisticRegression. Она берёт age, pressure, marker и выдаёт risk_score. Уже этот риск разбирается оператором глав 2-3.').classes('text-sm')
        row_metrics([
            ('Модель', 'LogisticRegression', 'sklearn pipeline'),
            ('Кейс', report.get('case_index'), 'выбирается сверху'),
            ('Риск модели', round(float(report.get('risk', STATE['risk'])), 4), 'predict_proba'),
            ('Класс', report.get('prediction'), 'threshold 0.5'),
            ('Accuracy', round(float(report.get('accuracy', 0.0)), 3), 'на demo-таблице'),
        ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(model_pipeline_figure()).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(model_contribution_figure()).classes('w-full')
        rows = [
            {
                'feature': row['feature'],
                'value': round(row['value'], 4),
                'model_coef': round(row['coef'], 4),
                'contribution': round(row['contribution'], 4),
            }
            for row in model_contributions()
        ]
        ui.table(
            columns=[
                {'name': 'feature', 'label': 'признак', 'field': 'feature'},
                {'name': 'value', 'label': 'значение кейса', 'field': 'value'},
                {'name': 'model_coef', 'label': 'коэф. модели', 'field': 'model_coef'},
                {'name': 'contribution', 'label': 'вклад', 'field': 'contribution'},
            ],
            rows=rows,
        ).classes('w-full q-table')


def plan_section() -> None:
    plan = STATE['plan']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('1. Из данных в термы').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Здесь числовой признак превращается в понятные термы: низкий, средний, высокий. Это вход для главы 2.').classes('text-sm')
        row_metrics([
            ('Строк', plan.metadata.get('n_rows'), 'demo-таблица'),
            ('Числовых признаков', len(plan.metadata.get('numeric_features', [])), ''),
            ('Сейчас показан', STATE['feature'], ''),
        ])
        ui.plotly(plan_membership_demo_figure()).classes('w-full')
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(target_distribution_figure()).classes('w-full')
            with ui.column().classes('w-full'):
                rows = quantile_rows()
                if rows:
                    ui.table(
                        columns=[
                            {'name': 'term', 'label': 'терм', 'field': 'term'},
                            {'name': 'shape', 'label': 'форма', 'field': 'shape'},
                            {'name': 'left', 'label': 'лево', 'field': 'left'},
                            {'name': 'center', 'label': 'центр', 'field': 'center'},
                            {'name': 'right', 'label': 'право', 'field': 'right'},
                        ],
                        rows=rows,
                    ).classes('w-full q-table')


def explanation_section() -> None:
    expl = STATE['explanation']
    obj = expl['object']
    report = expl['report']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('2. Объяснение одного кейса').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Берём risk_score из модели выше. Затем оператор строит E_k: термы, правила, неопределённость, потеря редукции и класс представления из главы 3.').classes('text-sm')
        case = current_case_row()
        with ui.element('div').classes('fx-case w-full'):
            ui.label('Входной кейс').classes('text-sm fx-muted')
            with ui.row().classes('w-full gap-2'):
                for key in ['age', 'pressure', 'marker', 'risk_score']:
                    if key in case:
                        metric(key, round(float(case[key]), 3), '')
        row_metrics([
            ('Представление', report['selected_class'], readable_class_name(report['selected_class'])),
            ('Неопределённость', round(obj.uncertainty, 4), 'меньше значит увереннее'),
            ('Потеря упрощения', round(obj.reduction_loss, 4), report['reduction_policy']),
        ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(membership_bar()).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(representation_layers_figure(expl)).classes('w-full')
        ui.plotly(selection_figure()).classes('w-full')
        rules = [
            {
                'rule': r.name,
                'condition': ', '.join(f'{k}={v}' for k, v in r.conditions.items()),
                'activation': round(obj.activations.get(r.name, 0.0), 4),
                'result': r.conclusion,
            }
            for r in obj.rules
        ]
        ui.table(
            columns=[
                {'name': 'rule', 'label': 'правило', 'field': 'rule'},
                {'name': 'condition', 'label': 'условие', 'field': 'condition'},
                {'name': 'activation', 'label': 'активация', 'field': 'activation'},
                {'name': 'result', 'label': 'вывод', 'field': 'result'},
            ],
            rows=rules,
        ).classes('w-full')
        ui.label(expl['text']).classes('text-sm leading-relaxed')


def composition_section() -> None:
    comp = STATE['composition']
    comp_obj = comp['comp']
    rows = edge_report(comp['edges'], comp['plan'].beta)
    gamma = rows[0]['gamma'] if rows else 'n/a'
    diagnostic = getattr(comp_obj, 'code', None)
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('3. Проверка цепочки: модель -> решение').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.markdown(
                '**Что здесь проверяется.** Модель риска выдаёт объяснение `E_model`: термы, правила, активации, неопределённость и след `tau`. '
                'Модуль решения строит своё объяснение `E_decision`. Оператор главы 2 не спрашивает “правильный ли прогноз”, '
                'а проверяет другое: можно ли безопасно склеить эти два объяснения в одну цепочку.'
            ).classes('text-sm')
            ui.markdown(
                '`gamma` показывает семантическое рассогласование ребра `model -> decision`: 0 значит компоненты говорят на одном языке, 1 значит объяснения плохо совместимы. '
                '`I(E_G)` показывает итоговую интерпретируемость всей цепочки. `D_ij` появляется, если общий язык разрушен, например термы слева и справа несовместимы.'
            ).classes('text-sm')
        if diagnostic:
            with ui.element('div').classes('fx-status-bad w-full'):
                ui.label('Остановлено: компоненты используют несовместимые термы.').classes('text-base fx-title')
                ui.label('Так система показывает место, где объяснимость ломается, вместо красивого, но неверного отчёта.').classes('text-sm')
            row_metrics([
                ('Диагностика', diagnostic, getattr(comp_obj, 'reason', '')),
                ('Расхождение', gamma, '0 хорошо, 1 плохо'),
                ('Итог', 'остановлено', 'нужен аудит'),
            ])
        else:
            level, _, caption = agreement_level(float(gamma))
            with ui.element('div').classes('fx-status-ok w-full'):
                ui.label('Проверка пройдена: объяснения можно связать в цепочку.').classes('text-base fx-title')
                ui.label(f'Уровень расхождения: {level}; {caption}.').classes('text-sm')
            row_metrics([
                ('Расхождение', round(float(gamma), 4), '0 хорошо, 1 плохо'),
                ('Потеря', round(float(comp['loss']), 4), 'интерпретируемость'),
                ('Индекс', round(float(comp['index']), 4), 'чем выше, тем лучше'),
            ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(composition_story_figure(comp, rows)).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(composition_breakdown_figure(comp)).classes('w-full')
        breakdown = composition_breakdown_rows(comp)
        if breakdown:
            ui.table(
                columns=[
                    {'name': 'part', 'label': 'что сравниваем', 'field': 'part'},
                    {'name': 'weight', 'label': 'вес beta', 'field': 'weight'},
                    {'name': 'distance', 'label': 'расстояние', 'field': 'distance'},
                    {'name': 'contribution', 'label': 'вклад в gamma', 'field': 'contribution'},
                    {'name': 'meaning', 'label': 'смысл', 'field': 'meaning'},
                ],
                rows=breakdown,
            ).classes('w-full q-table')
        safe_rows = [{k: ', '.join(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows]
        if safe_rows:
            ui.table(
                columns=[
                    {'name': 'source', 'label': 'откуда', 'field': 'source'},
                    {'name': 'target', 'label': 'куда', 'field': 'target'},
                    {'name': 'gamma', 'label': 'расхождение', 'field': 'gamma'},
                    {'name': 'severity', 'label': 'уровень', 'field': 'severity'},
                    {'name': 'left_terms', 'label': 'термы слева', 'field': 'left_terms'},
                    {'name': 'right_terms', 'label': 'термы справа', 'field': 'right_terms'},
                ],
                rows=safe_rows,
            ).classes('w-full')


def benchmark_section() -> None:
    report = operator_benchmark_report()
    without = report['without_operator']
    with_op = report['with_operator']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('4. Проверка на нормальном датасете: без оператора и с оператором').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Датасет: sklearn breast_cancer. Модель: RandomForestClassifier. Без оператора есть риск и важности признаков; с оператором добавляются gamma, I(E_G) и диагностика D_ij при разрыве терминов.').classes('text-sm')
        row_metrics([
            ('Датасет', report['dataset'], f"{report['n_samples']} объектов, {report['n_features']} признаков"),
            ('Модель', report['model'], 'RandomForest'),
            ('Accuracy', round(report['accuracy'], 4), 'test split'),
            ('ROC-AUC', round(report['roc_auc'], 4), 'test split'),
            ('Риск кейса', round(report['case_risk'], 4), 'malignancy probability'),
        ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(benchmark_quality_figure()).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(benchmark_importance_figure()).classes('w-full')
        ui.plotly(operator_added_value_figure()).classes('w-full')
        rows = [
            {
                'mode': 'без оператора',
                'risk': 'есть',
                'feature_importance': 'есть',
                'gamma': 'нет',
                'I_EG': 'нет',
                'D_ij': 'нет',
                'meaning': without['verdict'],
            },
            {
                'mode': 'с оператором',
                'risk': 'есть',
                'feature_importance': 'есть',
                'gamma': None if with_op['semantic_gamma'] is None else round(with_op['semantic_gamma'], 4),
                'I_EG': None if with_op['interpretability_index'] is None else round(with_op['interpretability_index'], 4),
                'D_ij': with_op['diagnostic'],
                'meaning': with_op['verdict'],
            },
        ]
        ui.table(
            columns=[
                {'name': 'mode', 'label': 'режим', 'field': 'mode'},
                {'name': 'risk', 'label': 'risk', 'field': 'risk'},
                {'name': 'feature_importance', 'label': 'важности', 'field': 'feature_importance'},
                {'name': 'gamma', 'label': 'gamma', 'field': 'gamma'},
                {'name': 'I_EG', 'label': 'I(E_G)', 'field': 'I_EG'},
                {'name': 'D_ij', 'label': 'D_ij', 'field': 'D_ij'},
                {'name': 'meaning', 'label': 'смысл', 'field': 'meaning'},
            ],
            rows=rows,
        ).classes('w-full q-table')


def risk_observer_section() -> None:
    report = risk_observer_report()
    metrics = report['observer_metrics']
    policy = report['policy']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('5. Risk-Aware Observer: управление применением прогноза').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Наблюдатель не меняет модель. Он берёт прогноз, неопределённость, I(E), D_ij и выбирает безопасное действие: accept, lower_confidence, request_more_data, defer_to_human или block.').classes('text-sm')
        row_metrics([
            ('theta_mid', policy['theta_mid'], 'средний риск'),
            ('theta_high', policy['theta_high'], 'высокий риск'),
            ('Accepted accuracy', metrics['accepted_accuracy'], 'только auto-accept'),
            ('Coverage', metrics['coverage'], 'доля auto-accept'),
            ('Risk reduction', metrics['risk_reduction'], 'стоимость до минус после'),
            ('Forced conflict', metrics['forced_conflict_action'], 'D_ij -> safe action'),
        ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(risk_observer_cost_figure()).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(risk_observer_actions_figure()).classes('w-full')
        rows = [
            {
                'case': idx + 1,
                'risk': case['predicted_risk'],
                'uncertainty': case['uncertainty'],
                'rho': case['risk_score'],
                'action': case['action'],
                'corrected_confidence': case['corrected_confidence'],
                'reason': case['reason'],
            }
            for idx, case in enumerate(report['sample_cases'])
        ]
        ui.table(
            columns=[
                {'name': 'case', 'label': 'кейс', 'field': 'case'},
                {'name': 'risk', 'label': 'risk', 'field': 'risk'},
                {'name': 'uncertainty', 'label': 'u', 'field': 'uncertainty'},
                {'name': 'rho', 'label': 'rho', 'field': 'rho'},
                {'name': 'action', 'label': 'действие', 'field': 'action'},
                {'name': 'corrected_confidence', 'label': 'conf_corr', 'field': 'corrected_confidence'},
                {'name': 'reason', 'label': 'причина', 'field': 'reason'},
            ],
            rows=rows,
        ).classes('w-full q-table')


def full_pipeline_section() -> None:
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('6. Полная демонстрация: вход -> модель -> оператор -> наблюдатель').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Этот блок собирает один отчёт по всей цепочке: данные, обучение, прогноз, глава 2, глава 3, композиция, Risk-Aware Observer и итоговое действие.').classes('text-sm')

        result_area = ui.column().classes('w-full gap-2')

        def generate() -> None:
            try:
                from full_pipeline_demo import OUT, run_full_pipeline

                report = run_full_pipeline(open_browser=False)
                STATE['full_pipeline_report'] = report
                result_area.clear()
                with result_area:
                    row_metrics([
                        ('Status', report['status'], 'сквозной сценарий'),
                        ('I(E_G)', round(report['composition']['index'], 4), 'индекс цепочки'),
                        ('Risk reduction', report['risk_observer']['risk_reduction'], 'benchmark'),
                        ('HTML', 'index.html', 'reports/full_demo'),
                    ])
                    with ui.row().classes('gap-2'):
                        ui.button(
                            'Скачать HTML-отчёт',
                            on_click=lambda: ui.download((OUT / 'index.html').read_bytes(), 'full_pipeline_demo.html'),
                        )
                        ui.button(
                            'Скачать JSON',
                            on_click=lambda: ui.download((OUT / 'full_pipeline_report.json').read_bytes(), 'full_pipeline_report.json'),
                        ).props('outline')
                ui.notify('Полная демонстрация собрана: reports/full_demo/index.html', type='positive')
            except Exception as exc:  # pragma: no cover
                ui.notify(f'Full demo failed: {exc}', type='negative', timeout=7000)
                (REPORTS / 'defense_demo_last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')

        ui.button('Собрать полный отчёт', on_click=generate).props('color=primary')
        cached = STATE.get('full_pipeline_report')
        if cached:
            with result_area:
                row_metrics([
                    ('Status', cached['status'], 'сквозной сценарий'),
                    ('I(E_G)', round(cached['composition']['index'], 4), 'индекс цепочки'),
                    ('Risk reduction', cached['risk_observer']['risk_reduction'], 'benchmark'),
                ])


def advanced_section() -> None:
    with ui.expansion('Технические детали и отчёт').classes('w-full'):
        profile = set(STATE['explanation']['profile'])
        candidates = default_candidates()
        selected = select_minimal_sufficient(profile, candidates, mode=STATE['mode'])
        front = {c.name for c in pareto_front([c for c in candidates if c.covers(profile)])}
        rows = [
            {
                'class': c.name,
                'covers': c.covers(profile),
                'pareto': c.name in front,
                'C_cog': c.cognitive_complexity,
                'C_comp': c.computational_complexity,
                'Delta_hat': c.expected_reduction_loss,
            }
            for c in candidates
        ]
        ui.label(f"Selected: {getattr(selected, 'name', getattr(selected, 'code', selected))}").classes('fx-title')
        ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
        ui.json_editor({'content': summary_report()}).classes('w-full h-72')


@ui.page('/')
def page() -> None:
    build_plan_from_state()
    recompute()
    apply_style()
    body = ui.column().classes('fx-shell w-full gap-4 pb-6')

    def redraw() -> None:
        body.clear()
        body.classes(replace='fx-shell w-full gap-4 pb-6' + (' fx-presentation' if STATE.get('presentation') else ''))
        with body:
            controls(redraw)
            route_strip()
            model_section()
            with ui.row().classes('w-full gap-4 items-start'):
                with ui.column().classes('w-full xl:w-[49%] gap-4'):
                    plan_section()
                with ui.column().classes('w-full xl:w-[49%] gap-4'):
                    explanation_section()
            composition_section()
            benchmark_section()
            risk_observer_section()
            full_pipeline_section()
            advanced_section()

    redraw()


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    ui.run(title='FuzzyXAI Defense Demo', port=args.port, reload=False, show=True)


if __name__ in {'__main__', '__mp_main__'}:
    run()
