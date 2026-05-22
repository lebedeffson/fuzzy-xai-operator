from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fuzzyxai import FuzzyXAIPipeline, ExplainPlan, Rule, SystemOperator, Trace, compose
from fuzzyxai.core.trust_evaluator import interpretability_loss, interpretability_index

OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def _safe_auc(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def main():
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        import pandas as pd
    except Exception as exc:
        raise SystemExit('scikit-learn and pandas are required for this benchmark') from exc

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'random_forest': RandomForestClassifier(n_estimators=80, random_state=42),
    }

    plan = ExplainPlan.from_data(X_train, y_train, mode='audit').with_reduction_weight(0.10)
    op = SystemOperator(plan)
    rules = [Rule('r_high_check', {'risk': 'high'}, 'additional_check')]

    results = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = pred
        accuracy = float(accuracy_score(y_test, pred))
        auc = _safe_auc(y_test, scores)

        indices = []
        diagnostics = 0
        start = time.perf_counter()
        for idx, risk in enumerate(scores[:50]):
            e1 = op.explain_scalar_risk(float(risk), rules, Trace(f'{name}-{idx}-model', 'v1', 'demo', source=name, checksum='m'))
            e2 = op.explain_scalar_risk(float(min(1.0, max(0.0, risk + 0.02))), rules, Trace(f'{name}-{idx}-decision', 'v1', 'demo', source=name, checksum='d'))
            comp = compose(e1, e2, plan.beta)
            if hasattr(comp, 'code'):
                diagnostics += 1
                continue
            loss = interpretability_loss(0.34, 0.35, 0.15, 0.03, comp.uncertainty, plan.lambda_, comp.reduction_loss, 0.10)
            indices.append(interpretability_index(loss))
        explain_time = time.perf_counter() - start
        results.append({
            'model': name,
            'accuracy': round(accuracy, 6),
            'roc_auc': None if auc is None else round(auc, 6),
            'fit_time_sec': round(fit_time, 6),
            'mean_I_on_50_cases': round(float(np.mean(indices)), 6) if indices else None,
            'diagnostic_rate_on_50_cases': round(diagnostics / 50, 6),
            'explain_time_50_cases_sec': round(explain_time, 6),
        })

    report = {
        'benchmark': 'sklearn breast_cancer proof-of-concept',
        'purpose': 'minimal open medical-like benchmark for dissertation chapters 2 and 3',
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'results': results,
        'limitations': 'This is not a clinical validation. It is a reproducible technical benchmark before chapter 5 experiments.',
    }
    (OUT / 'breast_cancer_benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# Breast cancer benchmark', '', report['limitations'], '']
    for row in results:
        md.append(f"- {row['model']}: accuracy={row['accuracy']}, roc_auc={row['roc_auc']}, mean_I={row['mean_I_on_50_cases']}")
    (OUT / 'breast_cancer_benchmark.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
