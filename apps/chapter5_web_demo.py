"""Interactive chapter 5 web demo: model -> explanation -> I_pre -> rho -> action."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fuzzyxai.category import ExplanationCategory, RiskContext, auto_accept_subpresheaf
from fuzzyxai.core.composition import compose
from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.data.breast_cancer_adapter import build_explanation_for_prediction
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.risk import RiskAwareObserver
from fuzzyxai.trust import compute_interpretability_index


@dataclass
class DemoBackend:
    feature_names: list[str]
    x_test: pd.DataFrame
    y_test: pd.Series
    model: RandomForestClassifier
    plan: ExplainPlan
    observer: RiskAwareObserver


def _risk_module_explanation(sample_id: str, predicted_risk: float, uncertainty: float) -> ExplanationObject:
    p = max(0.0, min(1.0, float(predicted_risk)))
    rules = [
        Rule('risk_low', {'predicted_risk': 'low'}, 'accept'),
        Rule('risk_medium', {'predicted_risk': 'medium'}, 'lower_confidence'),
        Rule('risk_high', {'predicted_risk': 'high'}, 'defer_to_human'),
    ]
    return ExplanationObject(
        terms={'low_risk', 'medium_risk', 'high_risk'},
        representation=F0(lambda _x, val=p: val, label='risk_module'),
        rules=rules,
        activations={
            'risk_low': float(max(0.0, 1.0 - p * 1.4)),
            'risk_medium': float(max(0.0, 1.0 - abs(p - 0.5) * 4.0)),
            'risk_high': float(p),
        },
        uncertainty=float(uncertainty),
        trace=Trace(
            id=f'{sample_id}:risk',
            version='risk_module_v1',
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='risk_module',
            checksum=f'{sample_id}:risk',
        ),
        metadata={'activation_threshold': 0.05},
    )


def _action_explanation(sample_id: str, action: str, rho: float) -> ExplanationObject:
    return ExplanationObject(
        terms={'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
        representation=F0(lambda _x, val=rho: val, label='action_module'),
        rules=[Rule(f'action_{action}', {'rho': action}, action)],
        activations={f'action_{action}': 1.0},
        uncertainty=0.0,
        trace=Trace(
            id=f'{sample_id}:action',
            version='action_module_v1',
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='observer_action',
            checksum=f'{sample_id}:action:{action}',
        ),
        metadata={'activation_threshold': 0.05},
    )


def build_backend(seed: int = 42, test_size: float = 0.25) -> DemoBackend:
    ds = load_breast_cancer(as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(
        ds.data,
        ds.target,
        test_size=test_size,
        random_state=seed,
        stratify=ds.target,
    )
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)
    model.fit(x_train, y_train)
    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    observer = RiskAwareObserver(plan=plan)
    return DemoBackend(
        feature_names=list(ds.feature_names),
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        model=model,
        plan=plan,
        observer=observer,
    )


def evaluate_vector(backend: DemoBackend, vector: np.ndarray, sample_id: str = 'manual') -> dict[str, Any]:
    row = pd.DataFrame([vector], columns=backend.feature_names)
    proba = backend.model.predict_proba(row)[0]
    pred = int(backend.model.predict(row)[0])
    p_malignant = float(proba[0])
    uncertainty = float(1.0 - abs(float(proba[1]) - float(proba[0])))

    e_model = build_explanation_for_prediction(
        p_malignant,
        sample_id=sample_id,
        model_version='rf_breast_cancer_v1',
        dataset_id='sklearn_breast_cancer',
    )
    i_pre = compute_interpretability_index(e_model, lambda_weights=backend.plan.lambda_, lambda_delta=0.10)
    e_risk = _risk_module_explanation(sample_id, p_malignant, uncertainty)

    beta = dict(backend.plan.beta)
    beta['gamma_max'] = 0.45
    diagnostics: list[str] = []
    gamma_mr = None
    gamma_ra = None

    comp_mr = compose(e_model, e_risk, beta, allow_missing_terms=True)
    has_rupture = isinstance(comp_mr, DiagnosticState)
    if has_rupture:
        diagnostics.append(comp_mr.code)
    else:
        gamma_mr = float(comp_mr.metadata.get('gamma', 0.0))

    decision = backend.observer.evaluate(
        predicted_risk=p_malignant,
        uncertainty=uncertainty,
        pre_interpretability=float(i_pre),
        reduction_loss=float(e_model.reduction_loss),
        diagnostics=diagnostics,
    )
    e_action = _action_explanation(sample_id, decision.action, decision.rho)
    comp_ra = compose(e_risk, e_action, beta, allow_missing_terms=True)
    if isinstance(comp_ra, DiagnosticState):
        has_rupture = True
        diagnostics.append(comp_ra.code)
    else:
        gamma_ra = float(comp_ra.metadata.get('gamma', 0.0))

    cat = ExplanationCategory(gamma_max=0.45)
    o_model = cat.object('E_model', e_model)
    o_risk = cat.object('E_risk', e_risk)
    o_action = cat.object('E_action', e_action)
    risk_context = RiskContext(
        cat,
        {
            o_model: {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
            o_risk: {'lower_confidence', 'request_more_data', 'defer_to_human', 'block'},
            o_action: {decision.action},
        },
    )
    auto_accept = auto_accept_subpresheaf(risk_context)
    context_rows = [
        {
            'object': obj.key,
            'RiskContext': ', '.join(sorted(risk_context(obj))),
            'AutoAccept': ', '.join(sorted(auto_accept(obj))),
        }
        for obj in (o_model, o_risk, o_action)
    ]

    return {
        'prediction': pred,
        'prob_malignant': p_malignant,
        'prob_benign': float(proba[1]),
        'uncertainty': uncertainty,
        'I_pre': float(i_pre),
        'rho': float(decision.rho),
        'action': str(decision.action),
        'rupture': bool(has_rupture),
        'diagnostics': diagnostics,
        'gamma_model_risk': gamma_mr,
        'gamma_risk_action': gamma_ra,
        'risk_weights': dict(backend.observer.weights),
        'thresholds': tuple(float(v) for v in backend.observer.thresholds),
        'contexts': context_rows,
    }


def run_ui(port: int = 8086) -> None:  # pragma: no cover - interactive UI
    from nicegui import ui

    backend = build_backend()
    current_vector = backend.x_test.iloc[0].to_numpy(dtype=float)

    ui.add_head_html(
        """
        <style>
        body { background: linear-gradient(160deg,#f2f6ff,#fbfdff 40%,#f5faf7); }
        .shell { max-width: 1500px; margin: 0 auto; }
        .hero { background: linear-gradient(135deg,#0b132b,#1c2541,#3a506b); color:white; border-radius:16px; padding:18px; }
        .card { background:white; border:1px solid #dfe6ef; border-radius:14px; padding:14px; box-shadow:0 10px 24px rgba(18,42,66,.07); }
        .kpi { font-size: 1.35rem; font-weight: 700; }
        </style>
        """
    )

    with ui.column().classes('shell w-full gap-4 p-4'):
        with ui.row().classes('hero w-full items-center justify-between'):
            with ui.column():
                ui.label('Risk-Aware Breast Cancer Web Demo').classes('text-2xl font-bold')
                ui.label('model -> explanation -> I_pre -> rho -> action -> contexts').classes('opacity-80')
            ui.label(f'weights: {backend.observer.weights}').classes('text-xs opacity-80')

        with ui.row().classes('w-full gap-4'):
            with ui.column().classes('card w-2/5 gap-3'):
                ui.label('Input').classes('text-lg font-semibold')
                sample_select = ui.select(
                    options={i: f'Sample #{i} (true={int(backend.y_test.iloc[i])})' for i in range(len(backend.x_test))},
                    value=0,
                    label='Choose breast cancer sample',
                ).classes('w-full')
                manual_toggle = ui.switch('Manual feature input', value=False)
                inputs: dict[str, Any] = {}
                with ui.expansion('Manual features', value=False).classes('w-full'):
                    with ui.grid(columns=2).classes('w-full gap-2'):
                        for name in backend.feature_names:
                            inputs[name] = ui.number(name, value=float(backend.x_test.iloc[0][name]), format='%.6f').classes('w-full')
                run_btn = ui.button('Evaluate case').classes('w-full')

            with ui.column().classes('card w-3/5 gap-3'):
                ui.label('Observer output').classes('text-lg font-semibold')
                result_md = ui.markdown('')
                context_table = ui.table(
                    columns=[
                        {'name': 'object', 'label': 'Object', 'field': 'object'},
                        {'name': 'RiskContext', 'label': 'RiskContext', 'field': 'RiskContext'},
                        {'name': 'AutoAccept', 'label': 'AutoAccept', 'field': 'AutoAccept'},
                    ],
                    rows=[],
                ).classes('w-full')

        def apply_sample(index: int) -> None:
            nonlocal current_vector
            idx = int(index)
            vec = backend.x_test.iloc[idx].to_numpy(dtype=float)
            current_vector = vec
            for name in backend.feature_names:
                inputs[name].value = float(backend.x_test.iloc[idx][name])

        def evaluate() -> None:
            nonlocal current_vector
            if manual_toggle.value:
                vec = np.array([float(inputs[name].value) for name in backend.feature_names], dtype=float)
                current_vector = vec
                sample_id = 'manual'
                true_y = '-'
            else:
                idx = int(sample_select.value)
                vec = backend.x_test.iloc[idx].to_numpy(dtype=float)
                current_vector = vec
                sample_id = f'sample_{idx}'
                true_y = int(backend.y_test.iloc[idx])
            out = evaluate_vector(backend, current_vector, sample_id=sample_id)
            result_md.content = (
                f"**Prediction:** `{out['prediction']}`  \n"
                f"**True y:** `{true_y}`  \n"
                f"**P(malignant):** `{out['prob_malignant']:.4f}`  \n"
                f"**I_pre:** `{out['I_pre']:.4f}`  \n"
                f"**rho:** `{out['rho']:.4f}`  \n"
                f"**Action:** `{out['action']}`  \n"
                f"**Rupture:** `{out['rupture']}`  \n"
                f"**Diagnostics:** `{out['diagnostics']}`  \n"
                f"**gamma(model->risk):** `{out['gamma_model_risk']}`  \n"
                f"**gamma(risk->action):** `{out['gamma_risk_action']}`"
            )
            context_table.rows = out['contexts']
            context_table.update()

        sample_select.on_value_change(lambda e: apply_sample(e.value))
        run_btn.on_click(evaluate)
        evaluate()
    ui.run(port=port, title='Chapter 5 Risk-Aware Demo')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8086)
    args = parser.parse_args()
    run_ui(port=args.port)


if __name__ == '__main__':
    main()
