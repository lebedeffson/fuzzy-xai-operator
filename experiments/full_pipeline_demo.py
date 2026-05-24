from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.core.composition import compose
from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.data.breast_cancer_adapter import build_explanation_for_prediction
from fuzzyxai.hierarchy.f0 import F0
from fuzzyxai.risk import RiskAwareObserver
from fuzzyxai.trust import compute_interpretability_index


def _risk_module_explanation(sample_id: str, predicted_risk: float, uncertainty: float) -> ExplanationObject:
    p = max(0.0, min(1.0, float(predicted_risk)))
    if p >= 0.75:
        level = 'high_risk'
    elif p >= 0.35:
        level = 'medium_risk'
    else:
        level = 'low_risk'
    rules = [
        Rule('risk_low', {'predicted_risk': 'low'}, 'accept'),
        Rule('risk_medium', {'predicted_risk': 'medium'}, 'lower_confidence'),
        Rule('risk_high', {'predicted_risk': 'high'}, 'defer_to_human'),
    ]
    activations = {
        'risk_low': float(max(0.0, 1.0 - p * 1.4)),
        'risk_medium': float(max(0.0, 1.0 - abs(p - 0.5) * 4.0)),
        'risk_high': float(p),
    }
    return ExplanationObject(
        terms={'low_risk', 'medium_risk', 'high_risk'},
        representation=F0(lambda _x, val=p: val, label='risk_module'),
        rules=rules,
        activations=activations,
        uncertainty=float(uncertainty),
        trace=Trace(
            id=f'{sample_id}:risk',
            version='risk_module_v1',
            timestamp=datetime.now(timezone.utc).isoformat(),
            source='risk_module',
            checksum=f'{sample_id}:risk',
        ),
        metadata={'risk_level': level, 'activation_threshold': 0.05},
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


def run(seed: int = 42, test_size: float = 0.25, max_cases: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    risk_malignant = proba[:, 0]  # target 0 = malignant
    model_uncertainty = 1.0 - np.abs(proba[:, 1] - proba[:, 0])

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    beta = dict(plan.beta)
    beta['gamma_max'] = 0.45
    observer = RiskAwareObserver(plan=plan)

    limit = len(x_test) if max_cases is None else min(len(x_test), int(max_cases))
    rows: list[dict[str, Any]] = []
    for i in range(limit):
        sample_id = f'full_case_{i}'
        p = float(risk_malignant[i])
        unc = float(model_uncertainty[i])

        e_model = build_explanation_for_prediction(
            p,
            sample_id=sample_id,
            model_version='rf_breast_cancer_v1',
            dataset_id='sklearn_breast_cancer',
        )
        i_pre = compute_interpretability_index(e_model, lambda_weights=plan.lambda_, lambda_delta=0.10)
        e_risk = _risk_module_explanation(sample_id, p, unc)

        comp_mr = compose(e_model, e_risk, beta, allow_missing_terms=True)
        diagnostics: list[str] = []
        gamma_mr = None
        if isinstance(comp_mr, DiagnosticState):
            diagnostics.append(comp_mr.code)
            has_rupture = True
        else:
            gamma_mr = float(comp_mr.metadata.get('gamma', 0.0))
            has_rupture = False

        decision = observer.evaluate(
            predicted_risk=p,
            uncertainty=unc,
            pre_interpretability=float(i_pre),
            reduction_loss=float(e_model.reduction_loss),
            diagnostics=diagnostics,
        )
        e_action = _action_explanation(sample_id, decision.action, decision.rho)
        comp_ra = compose(e_risk, e_action, beta, allow_missing_terms=True)
        gamma_ra = None
        if isinstance(comp_ra, DiagnosticState):
            has_rupture = True
            diagnostics.append(comp_ra.code)
        else:
            gamma_ra = float(comp_ra.metadata.get('gamma', 0.0))

        rows.append(
            {
                'case_id': i,
                'true_y': int(y_test.iloc[i]),
                'pred_y': int(pred[i]),
                'prob_malignant': p,
                'uncertainty': unc,
                'I_pre': float(i_pre),
                'rho': float(decision.rho),
                'action': str(decision.action),
                'has_rupture': bool(has_rupture),
                'diagnostics': ';'.join(diagnostics),
                'gamma_model_risk': gamma_mr,
                'gamma_risk_action': gamma_ra,
            }
        )

    df = pd.DataFrame(rows)
    action_counts = Counter(df['action'].tolist())
    summary = {
        'dataset': 'sklearn_breast_cancer',
        'n_test': int(len(x_test)),
        'n_evaluated': int(len(df)),
        'model_accuracy': float(accuracy_score(y_test, pred)),
        'model_roc_auc': float(roc_auc_score(y_test, proba[:, 1])),
        'mean_I_pre': float(df['I_pre'].mean()),
        'mean_rho': float(df['rho'].mean()),
        'rupture_rate': float(df['has_rupture'].mean()),
        'actions': dict(action_counts),
        'route': 'model -> explanation -> I_pre -> risk_module -> composition -> observer',
    }
    return df, summary


def write_reports(df: pd.DataFrame, summary: dict[str, Any], out_dir: str | Path = 'reports/full_pipeline') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'predictions.csv'
    json_path = out / 'summary.json'
    md_path = out / 'summary.md'

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        '# Full pipeline demo (breast cancer)',
        '',
        f"- dataset: `{summary['dataset']}`",
        f"- n_test: `{summary['n_test']}`",
        f"- n_evaluated: `{summary['n_evaluated']}`",
        f"- model_accuracy: `{summary['model_accuracy']:.6f}`",
        f"- model_roc_auc: `{summary['model_roc_auc']:.6f}`",
        f"- mean_I_pre: `{summary['mean_I_pre']:.6f}`",
        f"- mean_rho: `{summary['mean_rho']:.6f}`",
        f"- rupture_rate: `{summary['rupture_rate']:.6f}`",
        '',
        '## Action distribution',
        '',
    ]
    for action, count in sorted(summary['actions'].items()):
        md.append(f"- `{action}`: `{count}`")
    md_path.write_text('\n'.join(md), encoding='utf-8')
    return {'csv': str(csv_path), 'json': str(json_path), 'markdown': str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--out-dir', default='reports/full_pipeline')
    args = parser.parse_args()

    df, summary = run(seed=args.seed, test_size=args.test_size, max_cases=args.max_cases)
    paths = write_reports(df, summary, args.out_dir)
    print(json.dumps({'status': 'PASS', 'summary': summary, 'paths': paths}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
