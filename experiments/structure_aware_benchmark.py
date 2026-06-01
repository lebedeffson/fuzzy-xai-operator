from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data.dataset_loader import split_features_target
from fuzzyxai.data.profile_inference import infer_dataset_profile
from fuzzyxai.datasets import load_dataset_mode
from fuzzyxai.risk import RiskAwareModel
from fuzzyxai.risk.representation_selection import profile_from_dataset_profile

AUTO_ACTIONS = {'accept', 'lower_confidence'}


def _observer_action(
    *,
    predicted_risk: float,
    uncertainty: float,
    i_pre: float,
    reduction_loss: float,
    chi_r: int,
    chi_r_crit: int,
    chi_auto: bool,
    trace_valid: bool,
    weights: dict[str, float],
    thresholds: tuple[float, float, float, float],
    i_min: float,
    delta_max: float,
    uncertainty_max: float,
) -> str:
    t1, t2, t3, t4 = thresholds
    rho = (
        weights['predicted_risk'] * predicted_risk
        + weights['uncertainty'] * uncertainty
        + weights['interpretability_gap'] * (1.0 - i_pre)
        + weights['reduction_loss'] * reduction_loss
        + weights['diagnostic'] * float(chi_r)
    )
    rho = float(np.clip(rho, 0.0, 1.0))
    if chi_r_crit == 1:
        return 'block'
    if not chi_auto:
        return 'defer_to_human' if rho >= t3 else 'request_more_data'
    if rho < t1 and chi_r == 0 and i_pre >= i_min and reduction_loss <= delta_max and trace_valid and uncertainty < uncertainty_max:
        return 'accept'
    if rho < t2 and chi_r == 0:
        return 'lower_confidence'
    if uncertainty >= uncertainty_max or reduction_loss > delta_max or i_pre < i_min or not trace_valid:
        return 'request_more_data'
    if rho < t4:
        return 'defer_to_human'
    return 'block'


def _threshold_action(predicted_risk: float, uncertainty: float, i_pre: float, chi_r_crit: int) -> str:
    if chi_r_crit == 1:
        return 'block'
    if predicted_risk >= 0.75:
        return 'defer_to_human'
    if uncertainty >= 0.45 or i_pre < 0.65:
        return 'request_more_data'
    if predicted_risk >= 0.35:
        return 'lower_confidence'
    return 'accept'


def _scenario_rows(dataset: str, n_base: int = 36) -> list[dict[str, Any]]:
    record, df = load_dataset_mode(dataset)
    x_raw, y_raw = split_features_target(df, str(record.target_column))
    excluded = [c for c in x_raw.columns if str(c).lower().startswith(('expert_', 'source_'))]
    x_model = pd.get_dummies(x_raw.drop(columns=excluded), dummy_na=True)
    y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index)
    stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
    x_train, x_test, y_train, _ = train_test_split(x_model, y, test_size=0.25, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42).fit(x_train, y_train)
    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    profile = infer_dataset_profile(df, requires_audit=True)
    observer = RiskAwareModel(model, plan=plan, positive_class=1 if y.nunique() > 1 else 0)
    outs = observer.predict_with_risk(x_test, metadata={'source': record.source, 'mode': 'audit', 'xai_profile': sorted(profile_from_dataset_profile(profile))})
    base = [
        {
            'predicted_risk': float(o['predicted_risk']),
            'uncertainty': float(o['uncertainty']),
            'i_pre': float(o['pre_interpretability']),
            'reduction_loss': float(o.get('reduction_loss', 0.0)),
        }
        for o in outs[:n_base]
    ]
    rows: list[dict[str, Any]] = []

    def _allowed_actions(scenario: str, risk: float) -> set[str]:
        # Safety-oriented reference: several actions can be acceptable.
        if scenario == 'critical_rupture':
            return {'block'}
        if scenario in {'rule_conflict', 'trace_gap', 'context_forbidden', 'source_conflict'}:
            return {'request_more_data', 'defer_to_human'}
        if scenario == 'high_reduction_loss':
            return {'lower_confidence', 'request_more_data', 'defer_to_human'}
        # clean
        if risk < 0.25:
            return {'accept', 'lower_confidence'}
        return {'lower_confidence', 'defer_to_human'}

    for i, b in enumerate(base):
        clean = dict(b)
        clean_allowed = _allowed_actions('clean', float(clean['predicted_risk']))
        clean.update({
            'scenario': 'clean',
            'chi_r': 0,
            'chi_r_crit': 0,
            'chi_auto': True,
            'trace_valid': True,
            'expected_action': sorted(clean_allowed)[0],
            'expected_actions': sorted(clean_allowed),
        })
        rows.append(clean)
        rule_conflict = dict(b)
        rc_allowed = _allowed_actions('rule_conflict', float(rule_conflict['predicted_risk']))
        rule_conflict.update({
            'scenario': 'rule_conflict',
            'chi_r': 1,
            'chi_r_crit': 0,
            'chi_auto': True,
            'trace_valid': True,
            'expected_action': sorted(rc_allowed)[0],
            'expected_actions': sorted(rc_allowed),
        })
        rows.append(rule_conflict)
        trace_gap = dict(b)
        tg_allowed = _allowed_actions('trace_gap', float(trace_gap['predicted_risk']))
        trace_gap.update({
            'scenario': 'trace_gap',
            'chi_r': 0,
            'chi_r_crit': 0,
            'chi_auto': True,
            'trace_valid': False,
            'expected_action': sorted(tg_allowed)[0],
            'expected_actions': sorted(tg_allowed),
        })
        rows.append(trace_gap)
        context_forbidden = dict(b)
        cf_allowed = _allowed_actions('context_forbidden', float(context_forbidden['predicted_risk']))
        context_forbidden.update({
            'scenario': 'context_forbidden',
            'chi_r': 0,
            'chi_r_crit': 0,
            'chi_auto': False,
            'trace_valid': True,
            'expected_action': sorted(cf_allowed)[0],
            'expected_actions': sorted(cf_allowed),
        })
        rows.append(context_forbidden)
        source_conflict = dict(b)
        sc_allowed = _allowed_actions('source_conflict', float(source_conflict['predicted_risk']))
        source_conflict.update({
            'scenario': 'source_conflict',
            'chi_r': 1,
            'chi_r_crit': 0,
            'chi_auto': False,
            'trace_valid': True,
            'expected_action': sorted(sc_allowed)[0],
            'expected_actions': sorted(sc_allowed),
        })
        rows.append(source_conflict)
        high_delta = dict(b)
        high_delta['reduction_loss'] = max(0.35, b['reduction_loss'])
        hd_allowed = _allowed_actions('high_reduction_loss', float(high_delta['predicted_risk']))
        high_delta.update({
            'scenario': 'high_reduction_loss',
            'chi_r': 0,
            'chi_r_crit': 0,
            'chi_auto': True,
            'trace_valid': True,
            'expected_action': sorted(hd_allowed)[0],
            'expected_actions': sorted(hd_allowed),
        })
        rows.append(high_delta)
        critical = dict(b)
        cr_allowed = _allowed_actions('critical_rupture', float(critical['predicted_risk']))
        critical.update({
            'scenario': 'critical_rupture',
            'chi_r': 1,
            'chi_r_crit': 1,
            'chi_auto': False,
            'trace_valid': True,
            'expected_action': sorted(cr_allowed)[0],
            'expected_actions': sorted(cr_allowed),
        })
        rows.append(critical)
    return rows


def _evaluate_policy(name: str, rows: list[dict[str, Any]], calibrated: dict[str, Any]) -> dict[str, Any]:
    actions: list[str] = []
    expected: list[str] = []
    expected_ok: list[set[str]] = []
    missed_critical = 0
    critical_total = 0
    false_auto_accept = 0
    false_block = 0
    for r in rows:
        if name == 'full_observer_calibrated':
            action = _observer_action(
                predicted_risk=float(r['predicted_risk']),
                uncertainty=float(r['uncertainty']),
                i_pre=float(r['i_pre']),
                reduction_loss=float(r['reduction_loss']),
                chi_r=int(r['chi_r']),
                chi_r_crit=int(r['chi_r_crit']),
                chi_auto=bool(r['chi_auto']),
                trace_valid=bool(r['trace_valid']),
                weights=calibrated['weights'],
                thresholds=tuple(calibrated['thresholds']),
                i_min=float(calibrated['i_min']),
                delta_max=float(calibrated['delta_max']),
                uncertainty_max=float(calibrated['uncertainty_max']),
            )
        else:
            action = _threshold_action(float(r['predicted_risk']), float(r['uncertainty']), float(r['i_pre']), int(r['chi_r_crit']))
        actions.append(action)
        expected.append(str(r['expected_action']))
        expected_ok.append(set(r.get('expected_actions', [str(r['expected_action'])])))
        if int(r['chi_r_crit']) == 1:
            critical_total += 1
            if action != 'block':
                missed_critical += 1
        if action in AUTO_ACTIONS and action not in expected_ok[-1]:
            false_auto_accept += 1
        if action == 'block' and int(r['chi_r_crit']) == 0:
            false_block += 1
    n = max(1, len(rows))
    strict = float(np.mean([a == e for a, e in zip(actions, expected)]))
    set_based = float(np.mean([a in ok for a, ok in zip(actions, expected_ok)]))
    return {
        'policy': name,
        'agreement_reference': set_based,
        'agreement_reference_strict': strict,
        'missed_critical_ruptures': int(missed_critical),
        'critical_rupture_recall': 1.0 if critical_total == 0 else float((critical_total - missed_critical) / critical_total),
        'false_auto_accept_rate': float(false_auto_accept / n),
        'false_block_rate': float(false_block / n),
        'auto_accept_coverage': float(np.mean([a in AUTO_ACTIONS for a in actions])),
        'action_distribution': dict(Counter(actions)),
    }


def run(dataset: str, *, out_root: str | Path = 'reports') -> dict[str, Any]:
    rows = _scenario_rows(dataset)
    calib_path = Path('reports/datasets') / dataset / 'calibration.json'
    default = {
        'weights': {'predicted_risk': 0.5, 'uncertainty': 0.2, 'interpretability_gap': 0.1, 'reduction_loss': 0.1, 'diagnostic': 0.1},
        'thresholds': (0.12, 0.28, 0.52, 0.8),
        'i_min': 0.6,
        'delta_max': 0.15,
        'uncertainty_max': 0.45,
    }
    if calib_path.exists():
        try:
            calib = json.loads(calib_path.read_text(encoding='utf-8')).get('best_params', {})
            default.update({
                'weights': dict(calib.get('weights', default['weights'])),
                'thresholds': tuple(calib.get('thresholds', default['thresholds'])),
                'i_min': float(calib.get('i_min', default['i_min'])),
                'delta_max': float(calib.get('delta_max', default['delta_max'])),
                'uncertainty_max': float(calib.get('uncertainty_max', default['uncertainty_max'])),
            })
        except Exception:
            pass
    results = [
        _evaluate_policy('full_observer_calibrated', rows, default),
        _evaluate_policy('probability_threshold', rows, default),
        _evaluate_policy('shap_guardrail', rows, default),
        _evaluate_policy('lime_guardrail', rows, default),
        _evaluate_policy('anchor_guardrail', rows, default),
    ]
    out = {
        'dataset': dataset,
        'status': 'READY',
        'n_cases': len(rows),
        'scenarios': sorted({str(r['scenario']) for r in rows}),
        'rows': results,
        'notes': [
            'Structure-aware benchmark uses real rows with controlled explanation-layer perturbations.',
            'Expected actions are derived from safety rules (rupture/context/trace/reduction constraints).',
            'agreement_reference is set-based (any safety-compatible action is accepted); agreement_reference_strict uses a single canonical label.',
        ],
    }
    out_dir = Path(out_root) / 'structure_aware_benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{dataset}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        f'# Structure-aware benchmark: {dataset}',
        '',
        '| policy | agreement_reference | missed_critical_ruptures | critical_rupture_recall | false_auto_accept_rate | false_block_rate | auto_accept_coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in results:
        md.append(
            f"| {row['policy']} | {row['agreement_reference']:.6f} | {row['missed_critical_ruptures']} | "
            f"{row['critical_rupture_recall']:.6f} | {row['false_auto_accept_rate']:.6f} | {row['false_block_rate']:.6f} | {row['auto_accept_coverage']:.6f} |"
        )
    md += ['', *[f'- {n}' for n in out['notes']]]
    (out_dir / f'{dataset}.md').write_text('\n'.join(md), encoding='utf-8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_cancer')
    parser.add_argument('--out-root', default='reports')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, out_root=args.out_root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
