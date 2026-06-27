from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data import load_dataset_by_key
from fuzzyxai.data.dataset_record import DatasetRecord
from fuzzyxai.data.profile_inference import infer_dataset_profile
from fuzzyxai.data.dataset_loader import split_features_target
from fuzzyxai.datasets import get_dataset_mode, load_dataset_mode
from fuzzyxai.risk import RiskAwareModel, RiskPolicy, proxy_action_label
from fuzzyxai.risk.representation_selection import profile_from_dataset_profile


_ACTION_LABEL_COLUMNS = (
    'expert_action',
    'expected_action',
    'action_label',
    'gold_action',
    'target_action',
)
_VALID_ACTIONS = {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}


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


def _find_expert_action_labels(df: pd.DataFrame) -> pd.Series | None:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in _ACTION_LABEL_COLUMNS:
        if cand in cols:
            s = df[cols[cand]].astype(str).str.strip().str.lower()
            if s.isin(_VALID_ACTIONS).all():
                return s
    return None


def _rupture_proxy_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    any_r = pd.Series(False, index=df.index)
    crit_r = pd.Series(False, index=df.index)
    for col in ('chi_R', 'chi_r', 'rupture'):
        if col in df.columns:
            any_r = any_r | df[col].astype(bool)
    for col in ('chi_R_crit', 'chi_r_crit', 'critical_rupture', 'is_critical'):
        if col in df.columns:
            crit_r = crit_r | df[col].astype(bool)
    if {'expert_a', 'expert_b'}.issubset(df.columns):
        expert_disagreement = df['expert_a'].astype(str) != df['expert_b'].astype(str)
        any_r = any_r | expert_disagreement
    if {'source_model', 'source_expert'}.issubset(df.columns):
        source_conflict = df['source_model'].astype(str) != df['source_expert'].astype(str)
        any_r = any_r | source_conflict
        crit_r = crit_r | source_conflict
    # Critical implies rupture at the aggregate level.
    any_r = any_r | crit_r
    return any_r, crit_r


def _roc_auc_reason(y_test: pd.Series, model_roc_auc: float | None) -> str:
    if y_test.nunique() < 2:
        return 'roc_auc undefined: single class in test split'
    if model_roc_auc is None:
        return 'roc_auc unavailable'
    if np.isnan(model_roc_auc):
        return 'roc_auc is NaN'
    if abs(float(model_roc_auc) - 0.5) <= 0.02:
        return 'roc_auc near 0.5: ranking signal is weak or class imbalance dominates'
    return ''


def _series_stats(prefix: str, values: pd.Series) -> dict[str, float]:
    arr = values.astype(float).to_numpy()
    return {
        f'{prefix}_mean': float(np.mean(arr)),
        f'{prefix}_std': float(np.std(arr)),
        f'{prefix}_median': float(np.quantile(arr, 0.50)),
        f'{prefix}_p25': float(np.quantile(arr, 0.25)),
        f'{prefix}_p75': float(np.quantile(arr, 0.75)),
        f'{prefix}_p05': float(np.quantile(arr, 0.05)),
        f'{prefix}_p95': float(np.quantile(arr, 0.95)),
    }


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
    model_f1 = float(f1_score(y_test, pred, average='binary', zero_division=0)) if y_test.nunique() == 2 else float(
        f1_score(y_test, pred, average='macro', zero_division=0)
    )
    model_precision = float(
        precision_score(y_test, pred, average='binary', zero_division=0)
    ) if y_test.nunique() == 2 else float(precision_score(y_test, pred, average='macro', zero_division=0))
    model_recall = float(
        recall_score(y_test, pred, average='binary', zero_division=0)
    ) if y_test.nunique() == 2 else float(recall_score(y_test, pred, average='macro', zero_division=0))
    model_roc_auc = float(roc_auc_score(y_test, proba[:, 1])) if proba.shape[1] == 2 and y_test.nunique() == 2 else None

    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    profile = infer_dataset_profile(df, requires_audit=True)
    xai_profile = sorted(profile_from_dataset_profile(profile))
    observer = RiskAwareModel(
        model,
        plan=plan,
        policy=RiskPolicy(theta_mid=0.34, theta_high=0.62),
        positive_class=1 if len(encoder.classes_) > 1 else 0,
    )
    expert_action_labels = _find_expert_action_labels(df)
    proxy_rupture_any, proxy_rupture_crit = _rupture_proxy_flags(x_raw)
    test_idx = list(x_test.index)
    chi_r_flags = [int(bool(proxy_rupture_any.loc[idx])) for idx in test_idx]
    chi_r_crit_flags = [int(bool(proxy_rupture_crit.loc[idx])) for idx in test_idx]
    outputs = observer.predict_with_risk(
        x_test,
        metadata={
            'source': record.source,
            'mode': 'audit',
            'xai_profile': xai_profile,
            'chi_r_flags': chi_r_flags,
            'chi_r_crit_flags': chi_r_crit_flags,
        },
    )

    rows: list[dict[str, Any]] = []
    expected_proxy_actions: list[str] = []
    got_actions: list[str] = []
    expected_expert_actions: list[str] = []
    for i, out in enumerate(outputs):
        row_idx = test_idx[i]
        diag_rupture = bool(out.get('diagnostics'))
        chi_r = 1 if diag_rupture or bool(proxy_rupture_any.loc[row_idx]) else 0
        chi_r_crit = 1 if diag_rupture or bool(proxy_rupture_crit.loc[row_idx]) else 0
        expected = proxy_action_label(
            predicted_risk=float(out['predicted_risk']),
            uncertainty=float(out['uncertainty']),
            i_pre=float(out['pre_interpretability']),
            rupture=bool(chi_r),
            critical_rupture=bool(chi_r_crit),
        )
        got = str(out['action'])
        expected_proxy_actions.append(expected)
        got_actions.append(got)
        if expert_action_labels is not None:
            expected_expert_actions.append(str(expert_action_labels.loc[row_idx]))
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
                'expected_action_proxy': expected,
                'expert_action_label': str(expert_action_labels.loc[row_idx]) if expert_action_labels is not None else '',
                'selected_representation': str(out['selected_representation']),
                'reduction_loss': float(out.get('reduction_loss', 0.0)),
            }
        )

    pred_df = pd.DataFrame(rows)
    observer_action_proxy_accuracy = float(np.mean([a == b for a, b in zip(got_actions, expected_proxy_actions)]))
    observer_action_accuracy_applicable = expert_action_labels is not None
    observer_action_accuracy_reason = ''
    if observer_action_accuracy_applicable:
        observer_action_accuracy = float(np.mean([a == b for a, b in zip(got_actions, expected_expert_actions)]))
    else:
        observer_action_accuracy = None
        observer_action_accuracy_reason = 'no expert action labels'

    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    positive_rate = float(n_positive / len(y)) if len(y) else 0.0
    score_std = float(np.std(proba[:, 1])) if proba.shape[1] == 2 else None
    roc_reason = _roc_auc_reason(y_test, model_roc_auc)
    dataset_mode = str(record.metadata.get('dataset_mode', record.name))
    is_registry = dataset_mode.startswith('registry_')
    if is_registry:
        agreement_proxy = None
        agreement_proxy_applicable = False
        agreement_proxy_reason = 'no expert action labels'
    else:
        agreement_proxy = observer_action_proxy_accuracy
        agreement_proxy_applicable = True
        agreement_proxy_reason = 'simulated action rule, not clinical expert labels'

    notes = ['Prototype measurements per object; no I/O timing.']
    if is_registry:
        notes.append('Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A.')
    if dataset_mode == 'registry_programs':
        notes.append('No expert action labels: agreement_proxy is not applicable.')
    if dataset_mode == 'registry_steel_ir' and roc_reason:
        notes.append('ROC AUC must not be interpreted as quality here; use this mode for industrial contour portability.')
    if dataset_mode == 'registry_mosmed_doctor_analysis':
        notes.append('Small audit slice (n=10): not for statistical validation.')
    if dataset_mode == 'diabetes_binary':
        notes.append('Stress-test for borderline uncertainty; threshold calibration may be required.')
    if dataset_mode == 'synthetic_ruptures':
        notes.append('Rupture proxies are derived from expert/source disagreement fields.')
    use_tag = 'quantitative'
    if dataset_mode == 'diabetes_binary':
        use_tag = 'stress-test'
    elif dataset_mode == 'synthetic_ruptures':
        use_tag = 'control-diagnostics'
    elif dataset_mode.startswith('registry_'):
        use_tag = 'external-transfer'
    valid_for_quantitative_claims = dataset_mode in {'breast_cancer', 'wine_risk', 'synthetic_ruptures'}
    limitations: list[str] = []
    if not observer_action_accuracy_applicable:
        limitations.append('no expert action labels')
    if roc_reason:
        limitations.append(roc_reason)
    if dataset_mode == 'registry_mosmed_doctor_analysis':
        limitations.append('small sample size')
    metric_interpretation = (
        'Use accuracy/roc_auc with rupture rates for built-in datasets; '
        'for registry datasets prioritize pipeline readiness and transfer limitations.'
    )
    stat_i_pre = _series_stats('i_pre', pred_df['I_pre']) if not pred_df.empty else {}
    stat_rho = _series_stats('rho', pred_df['rho']) if not pred_df.empty else {}
    crit_total = int(pred_df['chi_R_crit'].sum()) if not pred_df.empty else 0
    missed_critical = int(((pred_df['chi_R_crit'] == 1) & (pred_df['action'] != 'block')).sum()) if not pred_df.empty else 0
    critical_recall = None if crit_total == 0 else float((crit_total - missed_critical) / crit_total)
    false_auto_accept = float(
        (
            (pred_df['action'].isin(['accept', 'lower_confidence']))
            & (pred_df['chi_R'] == 1)
        ).mean()
    ) if not pred_df.empty else 0.0
    false_block_rate = float(
        ((pred_df['action'] == 'block') & (pred_df['chi_R_crit'] == 0)).mean()
    ) if not pred_df.empty else 0.0
    auto_accept_coverage = float(
        pred_df['action'].isin(['accept', 'lower_confidence']).mean()
    ) if not pred_df.empty else 0.0
    mean_reduction_loss = float(pred_df['reduction_loss'].mean()) if not pred_df.empty else 0.0

    summary = {
        'dataset': record.name,
        'status': 'READY',
        'pipeline_completed': True,
        'n': int(len(df)),
        'domain': str(domain),
        'model_accuracy': model_accuracy,
        'model_roc_auc': model_roc_auc,
        'model_f1': model_f1,
        'model_precision': model_precision,
        'model_recall': model_recall,
        'reason_if_roc_auc_nan_or_05': roc_reason,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'positive_rate': positive_rate,
        'score_std': score_std,
        'observer_action_accuracy_applicable': observer_action_accuracy_applicable,
        'observer_action_accuracy_reason': observer_action_accuracy_reason,
        'observer_action_accuracy': observer_action_accuracy,
        'observer_action_proxy_accuracy': observer_action_proxy_accuracy,
        'agreement_proxy': agreement_proxy,
        'agreement_proxy_applicable': agreement_proxy_applicable,
        'agreement_proxy_reason': agreement_proxy_reason,
        'missed_critical_ruptures': missed_critical,
        'critical_rupture_recall': critical_recall,
        'false_auto_accept_rate': false_auto_accept,
        'false_block_rate': false_block_rate,
        'auto_accept_coverage': auto_accept_coverage,
        'mean_I_pre': float(pred_df['I_pre'].mean()) if not pred_df.empty else 0.0,
        'mean_rho': float(pred_df['rho'].mean()) if not pred_df.empty else 0.0,
        'mean_reduction_loss': mean_reduction_loss,
        'rupture_rate': float(pred_df['chi_R'].mean()) if not pred_df.empty else 0.0,
        'critical_rupture_rate': float(pred_df['chi_R_crit'].mean()) if not pred_df.empty else 0.0,
        'action_distribution': dict(Counter(pred_df['action'].tolist())),
        'selected_representation_distribution': dict(Counter(pred_df['selected_representation'].tolist())),
        'metric_interpretation': metric_interpretation,
        'valid_for_quantitative_claims': valid_for_quantitative_claims,
        'limitations': limitations,
        'recommended_use_in_dissertation': use_tag,
        'notes': ' '.join(notes),
        'model_metrics': {
            'accuracy': model_accuracy,
            'roc_auc': model_roc_auc,
            'f1': model_f1,
            'precision': model_precision,
            'recall': model_recall,
        },
        'observer_metrics': {
            'agreement_proxy': agreement_proxy,
            'missed_critical_ruptures': missed_critical,
            'critical_rupture_recall': critical_recall,
            'false_auto_accept_rate': false_auto_accept,
            'false_block_rate': false_block_rate,
            'auto_accept_coverage': auto_accept_coverage,
            'mean_reduction_loss': mean_reduction_loss,
        },
        **stat_i_pre,
        **stat_rho,
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
        f"- model_f1: `{summary.get('model_f1')}`",
        f"- model_precision: `{summary.get('model_precision')}`",
        f"- model_recall: `{summary.get('model_recall')}`",
        f"- reason_if_roc_auc_nan_or_05: `{summary.get('reason_if_roc_auc_nan_or_05')}`",
        f"- n_positive: `{summary.get('n_positive')}`",
        f"- n_negative: `{summary.get('n_negative')}`",
        f"- positive_rate: `{summary.get('positive_rate')}`",
        f"- score_std: `{summary.get('score_std')}`",
        f"- agreement_proxy: `{summary.get('agreement_proxy')}`",
        f"- agreement_proxy_applicable: `{summary.get('agreement_proxy_applicable')}`",
        f"- agreement_proxy_reason: `{summary.get('agreement_proxy_reason')}`",
        f"- missed_critical_ruptures: `{summary.get('missed_critical_ruptures')}`",
        f"- critical_rupture_recall: `{summary.get('critical_rupture_recall')}`",
        f"- false_auto_accept_rate: `{summary.get('false_auto_accept_rate')}`",
        f"- false_block_rate: `{summary.get('false_block_rate')}`",
        f"- auto_accept_coverage: `{summary.get('auto_accept_coverage')}`",
        f"- observer_action_accuracy (legacy): `{summary['observer_action_accuracy']}`",
        f"- observer_action_proxy_accuracy (legacy): `{summary.get('observer_action_proxy_accuracy')}`",
        f"- mean_I_pre: `{summary['mean_I_pre']}`",
        f"- mean_rho: `{summary['mean_rho']}`",
        f"- mean_reduction_loss: `{summary.get('mean_reduction_loss')}`",
        f"- i_pre_mean: `{summary.get('i_pre_mean')}`",
        f"- i_pre_std: `{summary.get('i_pre_std')}`",
        f"- i_pre_median: `{summary.get('i_pre_median')}`",
        f"- i_pre_p25: `{summary.get('i_pre_p25')}`",
        f"- i_pre_p75: `{summary.get('i_pre_p75')}`",
        f"- i_pre_p05: `{summary.get('i_pre_p05')}`",
        f"- i_pre_p95: `{summary.get('i_pre_p95')}`",
        f"- rho_mean: `{summary.get('rho_mean')}`",
        f"- rho_std: `{summary.get('rho_std')}`",
        f"- rho_median: `{summary.get('rho_median')}`",
        f"- rho_p25: `{summary.get('rho_p25')}`",
        f"- rho_p75: `{summary.get('rho_p75')}`",
        f"- rho_p05: `{summary.get('rho_p05')}`",
        f"- rho_p95: `{summary.get('rho_p95')}`",
        f"- rupture_rate: `{summary['rupture_rate']}`",
        f"- critical_rupture_rate: `{summary.get('critical_rupture_rate')}`",
        f"- metric_interpretation: `{summary.get('metric_interpretation')}`",
        f"- valid_for_quantitative_claims: `{summary.get('valid_for_quantitative_claims')}`",
        f"- limitations: `{summary.get('limitations')}`",
        f"- recommended_use_in_dissertation: `{summary.get('recommended_use_in_dissertation')}`",
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
            'pipeline_completed': False,
            'n': 0,
            'domain': domain,
            'model_accuracy': None,
            'model_roc_auc': None,
            'model_f1': None,
            'model_precision': None,
            'model_recall': None,
            'reason_if_roc_auc_nan_or_05': '',
            'n_positive': None,
            'n_negative': None,
            'positive_rate': None,
            'score_std': None,
            'observer_action_accuracy_applicable': False,
            'observer_action_accuracy_reason': 'no expert action labels',
            'observer_action_accuracy': None,
            'observer_action_proxy_accuracy': None,
            'agreement_proxy': None,
            'agreement_proxy_applicable': False,
            'agreement_proxy_reason': 'no expert action labels',
            'missed_critical_ruptures': None,
            'critical_rupture_recall': None,
            'false_auto_accept_rate': None,
            'false_block_rate': None,
            'auto_accept_coverage': None,
            'mean_I_pre': None,
            'mean_rho': None,
            'mean_reduction_loss': None,
            'i_pre_mean': None,
            'i_pre_std': None,
            'i_pre_median': None,
            'i_pre_p25': None,
            'i_pre_p75': None,
            'i_pre_p05': None,
            'i_pre_p95': None,
            'rho_mean': None,
            'rho_std': None,
            'rho_median': None,
            'rho_p25': None,
            'rho_p75': None,
            'rho_p05': None,
            'rho_p95': None,
            'rupture_rate': None,
            'critical_rupture_rate': None,
            'action_distribution': {},
            'selected_representation_distribution': {},
            'metric_interpretation': 'dataset unavailable',
            'valid_for_quantitative_claims': False,
            'limitations': ['dataset unavailable'],
            'recommended_use_in_dissertation': 'external-transfer',
            'notes': f'MISSING: {error}' if error else 'Dataset unavailable',
            'model_metrics': {
                'accuracy': None,
                'roc_auc': None,
                'f1': None,
                'precision': None,
                'recall': None,
            },
            'observer_metrics': {
                'agreement_proxy': None,
                'missed_critical_ruptures': None,
                'critical_rupture_recall': None,
                'false_auto_accept_rate': None,
                'false_block_rate': None,
                'auto_accept_coverage': None,
                'mean_reduction_loss': None,
            },
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
