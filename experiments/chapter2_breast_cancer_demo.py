from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.data.breast_cancer_adapter import build_explanation_for_prediction
from fuzzyxai.trust.trust_evaluator import compute_interpretability_index


def run(seed: int = 42, test_size: float = 0.25, max_cases: int | None = None) -> tuple[pd.DataFrame, dict]:
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

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    lambda_weights = plan.lambda_

    limit = len(x_test) if max_cases is None else min(len(x_test), int(max_cases))
    rows: list[dict] = []
    for i in range(limit):
        p = float(risk_malignant[i])
        exp_obj = build_explanation_for_prediction(
            p,
            sample_id=f'bc_case_{i}',
            model_version='rf_breast_cancer_v1',
            dataset_id='sklearn_breast_cancer',
        )
        i_pre = compute_interpretability_index(exp_obj, lambda_weights=lambda_weights, lambda_delta=0.10)
        rows.append(
            {
                'case_id': i,
                'prob_malignant': p,
                'i_pre': float(i_pre),
                'true_y': int(y_test.iloc[i]),
                'pred_y': int(pred[i]),
                'uncertainty': float(exp_obj.uncertainty),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        'dataset': 'sklearn_breast_cancer',
        'n_test': int(len(x_test)),
        'n_evaluated': int(len(df)),
        'model_accuracy': float(accuracy_score(y_test, pred)),
        'model_roc_auc': float(roc_auc_score(y_test, proba[:, 1])),
        'i_pre_mean': float(df['i_pre'].mean()),
        'i_pre_median': float(df['i_pre'].median()),
        'i_pre_q05': float(df['i_pre'].quantile(0.05)),
        'i_pre_q95': float(df['i_pre'].quantile(0.95)),
    }
    return df, summary


def write_reports(df: pd.DataFrame, summary: dict, out_dir: str | Path = 'reports/chapter2') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'chapter2_breast_cancer_i_pre.csv'
    json_path = out / 'chapter2_breast_cancer_summary.json'
    html_path = out / 'i_pre_distribution.html'
    md_path = out / 'chapter2_breast_cancer_summary.md'

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    fig = px.histogram(df, x='i_pre', nbins=20, title='I_pre distribution (breast cancer)')
    fig.write_html(html_path, include_plotlyjs='cdn')

    md = [
        '# Chapter 2 breast cancer demo',
        '',
        f"- dataset: `{summary['dataset']}`",
        f"- n_test: `{summary['n_test']}`",
        f"- n_evaluated: `{summary['n_evaluated']}`",
        f"- model_accuracy: `{summary['model_accuracy']:.6f}`",
        f"- model_roc_auc: `{summary['model_roc_auc']:.6f}`",
        f"- I_pre mean: `{summary['i_pre_mean']:.6f}`",
        f"- I_pre median: `{summary['i_pre_median']:.6f}`",
        f"- I_pre q05: `{summary['i_pre_q05']:.6f}`",
        f"- I_pre q95: `{summary['i_pre_q95']:.6f}`",
    ]
    md_path.write_text('\n'.join(md), encoding='utf-8')
    return {
        'csv': str(csv_path),
        'json': str(json_path),
        'html': str(html_path),
        'markdown': str(md_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--out-dir', default='reports/chapter2')
    args = parser.parse_args()

    df, summary = run(seed=args.seed, test_size=args.test_size, max_cases=args.max_cases)
    paths = write_reports(df, summary, args.out_dir)
    print(json.dumps({'status': 'PASS', 'summary': summary, 'paths': paths}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
