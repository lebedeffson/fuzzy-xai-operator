from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose, interpretability_index, interpretability_loss
from fuzzyxai.demo.synthetic import build_demo_representation, default_candidates, metadata_for_demo, sample_dataframe
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from fuzzyxai.selection.profile_builder import build_profile
from fuzzyxai.selection.pareto_selector import select_minimal_sufficient
from fuzzyxai.visual.composition_graph import edge_report

from .decision_policy import RiskPolicy
from .uncertainty import confidence_from_uncertainty, entropy_uncertainty, margin_uncertainty


@dataclass(frozen=True)
class ObserverPipelineConfig:
    feature_cols: tuple[str, ...] = ('age', 'pressure', 'marker')
    target_col: str = 'target'
    case_index: int = 2
    positive_class: int = 1
    mode: str = 'audit'
    force_conflict: bool = False


class ObserverPipeline:
    """Executable risk-aware XAI observer over a probabilistic model.

    The observer does not modify the model. It consumes a prediction interface,
    builds E_M^ext, composes it with risk and decision explanations, then chooses
    a safe action through RiskPolicy.
    """

    def __init__(self, model, plan: ExplainPlan, policy: RiskPolicy | None = None, config: ObserverPipelineConfig | None = None) -> None:
        self.model = model
        self.plan = plan
        self.policy = policy or RiskPolicy(theta_mid=0.34, theta_high=0.62)
        self.config = config or ObserverPipelineConfig()
        self.operator = SystemOperator(plan)

    def explain_case(self, x_case: pd.DataFrame, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        proba = np.asarray(self.model.predict_proba(x_case), dtype=float)
        prediction = np.asarray(self.model.predict(x_case))[0]
        risk_col = min(int(self.config.positive_class), proba.shape[1] - 1)
        risk = float(proba[0, risk_col])
        ent = float(entropy_uncertainty(proba)[0])
        mar = float(margin_uncertainty(proba)[0])
        uncertainty = float(np.clip(0.70 * ent + 0.30 * mar, 0.0, 1.0))
        confidence = float(confidence_from_uncertainty([uncertainty])[0])

        metadata = dict(metadata or metadata_for_demo(intervals=True, experts=2, conflict=True, audit=self.config.mode == 'audit'))
        profile = build_profile(metadata)
        selected = select_minimal_sufficient(profile, default_candidates(), mode=self.config.mode)
        representation = build_demo_representation(risk, audit=self.config.mode == 'audit')
        reduction = MetaReducer(goal=self.config.mode).reduce(representation)

        e_model = self._model_explanation(risk, uncertainty)
        e_model.representation = representation
        e_model.reduction_loss = float(reduction.delta)

        provisional_i = _interpretability_for(e_model, self.plan, reduction_loss=float(reduction.delta))
        provisional_rho = self.policy.risk_score(risk, uncertainty, provisional_i, float(reduction.delta), [])
        e_risk = self._risk_explanation(provisional_rho)

        comp_mr = compose(e_model, e_risk, self.plan.beta, allow_missing_terms=True)
        diagnostics = []
        if hasattr(comp_mr, 'code'):
            diagnostics.append(getattr(comp_mr, 'code'))
            composed_for_index = e_model
        else:
            composed_for_index = comp_mr

        i_eg = _interpretability_for(composed_for_index, self.plan, reduction_loss=float(getattr(composed_for_index, 'reduction_loss', reduction.delta)))
        decision = self.policy.choose(risk, uncertainty, i_eg, float(reduction.delta), diagnostics)
        e_decision = self._decision_explanation(decision.action.value, risk)
        if self.config.force_conflict:
            e_decision = e_decision.copy_with(terms={'allow', 'deny'})

        comp_rd = compose(e_risk, e_decision, self.plan.beta, allow_missing_terms=not self.config.force_conflict)
        if hasattr(comp_rd, 'code'):
            diagnostics.append(getattr(comp_rd, 'code'))
            decision = self.policy.choose(risk, uncertainty, i_eg, float(reduction.delta), diagnostics)
            final_comp = comp_rd
        elif not hasattr(comp_mr, 'code'):
            final_comp = compose(comp_mr, e_decision, self.plan.beta, allow_missing_terms=True)
            if hasattr(final_comp, 'code'):
                diagnostics.append(getattr(final_comp, 'code'))
        else:
            final_comp = comp_rd

        if not diagnostics and not hasattr(final_comp, 'code'):
            i_eg = _interpretability_for(final_comp, self.plan, reduction_loss=float(getattr(final_comp, 'reduction_loss', reduction.delta)))
            decision = self.policy.choose(risk, uncertainty, i_eg, float(reduction.delta), diagnostics)

        rho = float(decision.risk_score)
        edges = [('Model', e_model, 'RiskModule', e_risk), ('RiskModule', e_risk, 'Decision', e_decision)]
        edge_rows = edge_report(edges, self.plan.beta)
        gamma_values = [float(row['gamma']) for row in edge_rows]

        return {
            'raw_prediction': int(prediction) if np.asarray(prediction).ndim == 0 else str(prediction),
            'raw_proba': [float(v) for v in proba[0]],
            'risk_score': risk,
            'uncertainty': uncertainty,
            'entropy_uncertainty': ent,
            'margin_uncertainty': mar,
            'confidence': confidence,
            'E_model_ext': {
                'terms': sorted(e_model.terms),
                'active_rules': sorted(e_model.active_rules),
                'uncertainty': float(e_model.uncertainty),
                'trace': e_model.trace.id,
            },
            'selected_representation': getattr(selected, 'name', str(selected)),
            'representation_class': getattr(representation, 'class_name', type(representation).__name__),
            'profile': sorted(profile),
            'Delta': float(reduction.delta),
            'reduction_policy': reduction.policy,
            'gamma': gamma_values,
            'I_EG': float(i_eg),
            'diagnostic_state': diagnostics,
            'risk_value': rho,
            'safe_action': decision.action.value,
            'human_readable_reason': decision.reason,
            'corrected_confidence': float(decision.corrected_confidence),
            'composition_edges': edge_rows,
        }

    def _model_explanation(self, risk: float, uncertainty: float):
        rules = [
            Rule('model_high_risk', {'risk': 'high'}, 'medical_review'),
            Rule('model_medium_risk', {'risk': 'medium'}, 'watch'),
        ]
        return self.operator.explain_scalar_risk(
            risk,
            rules,
            Trace('E_M_model', 'v1', 'runtime', source='predict_proba-model', checksum='observer-model'),
            model_uncertainty=uncertainty,
            trace_uncertainty=0.01,
        )

    def _risk_explanation(self, rho: float):
        rules = [
            Rule('risk_gate_high', {'risk': 'high'}, 'defer_to_human'),
            Rule('risk_gate_medium', {'risk': 'medium'}, 'request_or_lower_confidence'),
            Rule('risk_gate_low', {'risk': 'low'}, 'accept'),
        ]
        return self.operator.explain_scalar_risk(
            rho,
            rules,
            Trace('E_R_risk_module', 'v1', 'runtime', source='risk-policy', checksum='observer-risk'),
            model_uncertainty=0.05,
            trace_uncertainty=0.01,
        )

    def _decision_explanation(self, action: str, risk: float):
        rules = [Rule(f'decision_{action}', {'risk': 'high' if risk >= 0.65 else 'medium' if risk >= 0.35 else 'low'}, action)]
        return self.operator.explain_scalar_risk(
            risk,
            rules,
            Trace('E_D_decision', 'v1', 'runtime', source='observer-action', checksum=action),
            model_uncertainty=0.03,
            trace_uncertainty=0.01,
        )


def _make_default_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))


def _interpretability_for(obj, plan: ExplainPlan, reduction_loss: float) -> float:
    loss = interpretability_loss(0.30, 0.34, 0.18, 0.03, float(obj.uncertainty), plan.lambda_, float(reduction_loss), 0.10)
    return float(interpretability_index(loss))


def build_full_observer_pipeline_report(config: ObserverPipelineConfig | None = None) -> dict[str, Any]:
    config = config or ObserverPipelineConfig()
    df = sample_dataframe()
    feature_cols = list(config.feature_cols)
    x = df[feature_cols]
    y = df[config.target_col]
    model = _make_default_model()
    model.fit(x, y)
    plan_df = df.copy()
    plan = ExplainPlan.from_data(plan_df, target=config.target_col, mode=config.mode).with_reduction_weight(0.10)
    pipeline = ObserverPipeline(model, plan, config=config)
    case_index = max(0, min(int(config.case_index), len(df) - 1))
    case = x.iloc[[case_index]]
    result = pipeline.explain_case(case)
    baseline = {
        'raw_prediction': result['raw_prediction'],
        'risk_score': result['risk_score'],
        'available_information': ['raw_prediction', 'risk_score', 'raw_proba'],
        'limitation': 'без наблюдателя нет E_M^ext, Delta, gamma, I(E_G), D_ij и безопасного действия',
    }
    report = {
        'title': 'Risk-aware XAI observer as an active layer over chapters 2 and 3',
        'status': 'PASS',
        'thesis_layer': 'active risk-oriented observer over a predictive interface',
        'claim': 'observer does not modify model parameters; it determines the admissible mode of prediction use',
        'route': ['data', 'model prediction', 'E_M_ext', 'A_M^F', 'E_R', 'E_D', 'E_G', 'RiskPolicy', 'safe_action'],
        'data': {'rows': int(len(df)), 'features': feature_cols, 'target': config.target_col},
        'case': {'index': case_index, 'values': {k: float(v) for k, v in case.iloc[0].to_dict().items()}},
        'model': {'name': 'sklearn Pipeline(StandardScaler, LogisticRegression)', 'interface': 'predict_proba'},
        'without_observer': baseline,
        'with_observer': result,
        'math': {
            'model_interface': 'M(x)=p(x), r_M(x)=p_1(x)',
            'extended_explanation': 'E_M^ext=<L_M,A_M^F,R_M,alpha_M,u_M,tau_M,Delta_M>',
            'risk': 'rho=w_p*rho_p+w_u*u_M+w_I*(1-I(E_G))+w_Delta*Delta_M+w_D*1[D!=empty]',
            'composition': 'E_G=E_D o E_R o E_M^ext',
        },
    }
    return report


def write_full_observer_pipeline_report(report: dict[str, Any], out_dir: Path | str = 'reports/full_observer_pipeline') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'full_observer_pipeline.json'
    md_path = out / 'full_observer_pipeline.md'
    html_path = out / 'full_observer_pipeline.html'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path.write_text(render_observer_markdown(report), encoding='utf-8')
    html_path.write_text(render_observer_html(report), encoding='utf-8')
    return {'json': str(json_path), 'markdown': str(md_path), 'html': str(html_path)}


def render_observer_markdown(report: dict[str, Any]) -> str:
    obs = report['with_observer']
    lines = [
        '# Full observer pipeline',
        '',
        f"Status: **{report['status']}**",
        f"Layer: `{report['thesis_layer']}`",
        '',
        '## Core claim',
        '',
        report['claim'],
        '',
        '## Route',
        '',
        ' -> '.join(report['route']),
        '',
        '## Case result',
        '',
        f"- raw prediction: `{obs['raw_prediction']}`",
        f"- risk score r_M(x): `{obs['risk_score']:.6f}`",
        f"- uncertainty u_M(x): `{obs['uncertainty']:.6f}`",
        f"- selected A_M^F: `{obs['selected_representation']}` / `{obs['representation_class']}`",
        f"- Delta_M: `{obs['Delta']:.6f}`",
        f"- I(E_G): `{obs['I_EG']:.6f}`",
        f"- rho(x): `{obs['risk_value']:.6f}`",
        f"- safe action: `{obs['safe_action']}`",
        f"- reason: {obs['human_readable_reason']}",
        '',
        '## Without / with observer',
        '',
        '- Without observer: model probability only.',
        '- With observer: E_M^ext, uncertainty, Delta, gamma, I(E_G), diagnostics and safe action.',
        '',
        '## Composition edges',
        '',
    ]
    for edge in obs['composition_edges']:
        lines.append(f"- `{edge['source']} -> {edge['target']}`: gamma=`{edge['gamma']:.6f}`, severity=`{edge['severity']}`")
    lines.append('')
    return '\n'.join(lines)


def render_observer_html(report: dict[str, Any]) -> str:
    obs = report['with_observer']
    esc = html.escape
    edge_rows = ''.join(
        f"<tr><td>{esc(row['source'])}</td><td>{esc(row['target'])}</td><td>{row['gamma']:.4f}</td><td>{esc(row['severity'])}</td></tr>"
        for row in obs['composition_edges']
    )
    case_rows = ''.join(f"<tr><td>{esc(k)}</td><td>{v:.4f}</td></tr>" for k, v in report['case']['values'].items())
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Full observer pipeline</title>
<style>body{{font-family:Arial,sans-serif;max-width:1080px;margin:32px auto;color:#172033;line-height:1.45}}.card{{border:1px solid #d9e2ec;border-radius:14px;padding:18px;margin:16px 0}}.kpi{{display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;margin:6px;min-width:150px}}.kpi b{{display:block;font-size:22px;color:#0f766e}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d9e2ec;padding:8px}}th{{background:#f1f5f9}}</style></head>
<body><h1>Risk-aware XAI observer</h1><p>{esc(report['claim'])}</p>
<div class="card"><h2>Route</h2><p><code>{esc(' -> '.join(report['route']))}</code></p></div>
<div class="card"><h2>Case</h2><table><tr><th>feature</th><th>value</th></tr>{case_rows}</table></div>
<div class="card"><h2>Observer output</h2>
<div class="kpi">risk_score<b>{obs['risk_score']:.4f}</b></div><div class="kpi">u_M<b>{obs['uncertainty']:.4f}</b></div><div class="kpi">A_M^F<b>{esc(obs['selected_representation'])}</b></div><div class="kpi">Delta<b>{obs['Delta']:.4f}</b></div><div class="kpi">I(E_G)<b>{obs['I_EG']:.4f}</b></div><div class="kpi">rho<b>{obs['risk_value']:.4f}</b></div><div class="kpi">action<b>{esc(obs['safe_action'])}</b></div><p>{esc(obs['human_readable_reason'])}</p></div>
<div class="card"><h2>Composition</h2><table><tr><th>source</th><th>target</th><th>gamma</th><th>severity</th></tr>{edge_rows}</table></div>
</body></html>"""
