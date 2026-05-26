from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data import load_dataset_by_key
from fuzzyxai.data.dataset_record import DatasetRecord
from fuzzyxai.data.profile_inference import infer_dataset_profile
from fuzzyxai.data.dataset_loader import split_features_target
from fuzzyxai.datasets import get_dataset_mode, load_dataset_mode
from fuzzyxai.risk import RiskAwareModel, RiskPolicy
from fuzzyxai.risk.representation_selection import profile_from_dataset_profile


def _load_dataset(dataset: str) -> tuple[str, str, DatasetRecord | None, pd.DataFrame | None, str]:
    try:
        spec = get_dataset_mode(dataset)
        record, df = load_dataset_mode(dataset)
        return 'READY', spec.domain, record, df, ''
    except FileNotFoundError as exc:
        spec = get_dataset_mode(dataset)
        return 'MISSING', spec.domain, None, None, str(exc)
    except KeyError:
        pass

    try:
        card, record, df = load_dataset_by_key(dataset, allow_fallback=False)
        return 'READY', card.domain, record, df, ''
    except FileNotFoundError as exc:
        return 'MISSING', 'unknown', None, None, str(exc)


def _observer_expected_action(predicted_risk: float, uncertainty: float, chi_r_crit: int) -> str:
    if chi_r_crit == 1:
        return 'block'
    if predicted_risk >= 0.75:
        return 'defer_to_human'
    if predicted_risk >= 0.35 or uncertainty >= 0.45:
        return 'lower_confidence'
    return 'accept'


def _evaluate_ready_dataset(record: DatasetRecord, df: pd.DataFrame, domain: str) -> tuple[dict[str, Any], pd.DataFrame]:
    target_column = str(record.target_column)
    x_raw, y_raw = split_features_target(df, target_column)
    excluded_metadata_columns = [c for c in x_raw.columns if str(c).lower().startswith(('expert_', 'source_'))]
    x_model = pd.get_dummies(x_raw.drop(columns=excluded_metadata_columns), dummy_na=True)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw), index=y_raw.index, name=target_column)
    stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_model,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    model_accuracy = float(accuracy_score(y_test, pred))
    model_roc_auc = float(roc_auc_score(y_test, proba[:, 1])) if proba.shape[1] == 2 else None

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    profile = infer_dataset_profile(df, requires_audit=True)
    xai_profile = sorted(profile_from_dataset_profile(profile))
    observer = RiskAwareModel(
        model,
        plan=plan,
        policy=RiskPolicy(theta_mid=0.34, theta_high=0.62),
        positive_class=1 if len(encoder.classes_) > 1 else 0,
    )
    outputs = observer.predict_with_risk(x_test, metadata={'source': record.source, 'mode': 'audit', 'xai_profile': xai_profile})

    rows: list[dict[str, Any]] = []
    expected_actions: list[str] = []
    got_actions: list[str] = []
    for i, out in enumerate(outputs):
        chi_r = 1 if out.get('diagnostics') else 0
        chi_r_crit = 1 if chi_r else 0
        expected = _observer_expected_action(float(out['predicted_risk']), float(out['uncertainty']), chi_r_crit)
        got = str(out['action'])
        expected_actions.append(expected)
        got_actions.append(got)
        rows.append(
            {
                'row_id': int(i),
                'y_true': int(y_test.iloc[i]),
                'predicted_risk': float(out['predicted_risk']),
                'uncertainty': float(out['uncertainty']),
                'I_pre': float(out['pre_interpretability']),
                'rho': float(out['application_risk']),
                'chi_R': chi_r,
                'chi_R_crit': chi_r_crit,
                'action': got,
                'selected_representation': str(out['selected_representation']),
            }
        )

    pred_df = pd.DataFrame(rows)
    observer_action_accuracy = float(np.mean([a == b for a, b in zip(got_actions, expected_actions)]))

    summary = {
        'dataset': record.name,
        'status': 'READY',
        'n': int(len(df)),
        'domain': str(domain),
        'model_accuracy': model_accuracy,
        'model_roc_auc': model_roc_auc,
        'observer_action_accuracy': observer_action_accuracy,
        'mean_I_pre': float(pred_df['I_pre'].mean()) if not pred_df.empty else 0.0,
        'mean_rho': float(pred_df['rho'].mean()) if not pred_df.empty else 0.0,
        'rupture_rate': float(pred_df['chi_R'].mean()) if not pred_df.empty else 0.0,
        'action_distribution': dict(Counter(pred_df['action'].tolist())),
        'selected_representation_distribution': dict(Counter(pred_df['selected_representation'].tolist())),
        'notes': 'Prototype measurements per object; no I/O timing.',
    }
    return summary, pred_df


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Dataset summary: {summary['dataset']}",
        '',
        f"- status: `{summary['status']}`",
        f"- n: `{summary['n']}`",
        f"- domain: `{summary['domain']}`",
        f"- model_accuracy: `{summary['model_accuracy']}`",
        f"- model_roc_auc: `{summary['model_roc_auc']}`",
        f"- observer_action_accuracy: `{summary['observer_action_accuracy']}`",
        f"- mean_I_pre: `{summary['mean_I_pre']}`",
        f"- mean_rho: `{summary['mean_rho']}`",
        f"- rupture_rate: `{summary['rupture_rate']}`",
        f"- action_distribution: `{summary['action_distribution']}`",
        f"- selected_representation_distribution: `{summary['selected_representation_distribution']}`",
        f"- notes: {summary['notes']}",
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def run_benchmark(dataset: str, *, out_root: str | Path = 'reports/datasets') -> dict[str, Any]:
    status, domain, record, df, error = _load_dataset(dataset)
    out_dir = Path(out_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if status != 'READY' or record is None or df is None:
        summary = {
            'dataset': dataset,
            'status': status,
            'n': 0,
            'domain': domain,
            'model_accuracy': None,
            'model_roc_auc': None,
            'observer_action_accuracy': None,
            'mean_I_pre': None,
            'mean_rho': None,
            'rupture_rate': None,
            'action_distribution': {},
            'selected_representation_distribution': {},
            'notes': f'MISSING: {error}' if error else 'Dataset unavailable',
        }
        (out_dir / 'predictions.csv').write_text('row_id\n', encoding='utf-8')
    else:
        summary, pred_df = _evaluate_ready_dataset(record, df, domain)
        summary['dataset'] = dataset
        pred_df.to_csv(out_dir / 'predictions.csv', index=False)

    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_summary_md(out_dir / 'summary.md', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_cancer')
    parser.add_argument('--out-root', default='reports/datasets')
    args = parser.parse_args()
    summary = run_benchmark(args.dataset, out_root=args.out_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
