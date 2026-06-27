from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fuzzyxai import ExplainPlan
from fuzzyxai.data.dataset_loader import split_features_target
from fuzzyxai.datasets import get_dataset_mode, load_dataset_mode
from fuzzyxai.risk import RiskAwareModel, proxy_action_label, reference_action_label
from fuzzyxai.risk.representation_selection import profile_from_dataset_profile
from fuzzyxai.data.profile_inference import infer_dataset_profile


AUTO_ACTIONS = {'accept', 'lower_confidence'}


@dataclass
class EvalRow:
    predicted_risk: float
    uncertainty: float
    i_pre: float
    reduction_loss: float
    y_true: int
    y_pred: int
    chi_r_raw: int
    chi_r_crit_raw: int
    chi_auto: bool
    trace_valid: bool


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
        any_r = any_r | (df['expert_a'].astype(str) != df['expert_b'].astype(str))
    if {'source_model', 'source_expert'}.issubset(df.columns):
        source_conflict = df['source_model'].astype(str) != df['source_expert'].astype(str)
        any_r = any_r | source_conflict
        crit_r = crit_r | source_conflict
    any_r = any_r | crit_r
    return any_r, crit_r


def _load_eval_rows(dataset: str) -> tuple[list[EvalRow], list[EvalRow]]:
    _spec = get_dataset_mode(dataset)
    record, df = load_dataset_mode(dataset)
    target_column = str(record.target_column)
    x_raw, y_raw = split_features_target(df, target_column)
    excluded_metadata_columns = [c for c in x_raw.columns if str(c).lower().startswith(('expert_', 'source_'))]
    x_model = pd.get_dummies(x_raw.drop(columns=excluded_metadata_columns), dummy_na=True)
    y = pd.Series(LabelEncoder().fit_transform(y_raw), index=y_raw.index, name=target_column)
    stratify = y if y.value_counts().min() >= 2 and y.nunique() <= 20 else None
    x_train_full, x_test, y_train_full, _y_test = train_test_split(
        x_model,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    stratify_train = y_train_full if y_train_full.value_counts().min() >= 2 and y_train_full.nunique() <= 20 else None
    x_train, x_val, y_train, _y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=stratify_train,
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(x_train, y_train)
    plan = ExplainPlan.from_data(x_train, y_train, mode='audit').with_reduction_weight(0.10)
    profile = infer_dataset_profile(df, requires_audit=True)
    xai_profile = sorted(profile_from_dataset_profile(profile))
    observer = RiskAwareModel(model, plan=plan, positive_class=1 if y.nunique() > 1 else 0)
    proxy_any, proxy_crit = _rupture_proxy_flags(x_raw)

    def collect_rows(x_part: pd.DataFrame) -> list[EvalRow]:
        idxs = list(x_part.index)
        chi_r_flags = [int(bool(proxy_any.loc[idx])) for idx in idxs]
        chi_r_crit_flags = [int(bool(proxy_crit.loc[idx])) for idx in idxs]
        out_rows = observer.predict_with_risk(
            x_part,
            metadata={
                'source': record.source,
                'mode': 'audit',
                'xai_profile': xai_profile,
                'chi_r_flags': chi_r_flags,
                'chi_r_crit_flags': chi_r_crit_flags,
            },
        )
        rows: list[EvalRow] = []
        for i, out in enumerate(out_rows):
            row_idx = idxs[i]
            chi_r_raw = 1 if bool(out.get('diagnostics')) or bool(proxy_any.loc[row_idx]) else 0
            chi_r_crit_raw = 1 if bool(out.get('diagnostics')) or bool(proxy_crit.loc[row_idx]) else 0
            i_pre = float(out['pre_interpretability'])
            p = float(out['predicted_risk'])
            u = float(out['uncertainty'])
            chi_auto = (p < 0.35) and (u < 0.45) and (i_pre >= 0.65) and (chi_r_crit_raw == 0)
            rows.append(
                EvalRow(
                    predicted_risk=p,
                    uncertainty=u,
                    i_pre=i_pre,
                    reduction_loss=float(out.get('reduction_loss', 0.0)),
                    y_true=int(y.loc[row_idx]),
                    y_pred=int(out.get('raw_prediction', 0)),
                    chi_r_raw=chi_r_raw,
                    chi_r_crit_raw=chi_r_crit_raw,
                    chi_auto=chi_auto,
                    trace_valid=bool(
                        out.get('E_model_ext', {}).get('trace')
                        and out.get('E_model_ext', {}).get('representation_class')
                    ),
                )
            )
        return rows

    return collect_rows(x_val), collect_rows(x_test)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0.0:
        return {
            'predicted_risk': 0.40,
            'uncertainty': 0.20,
            'interpretability_gap': 0.15,
            'reduction_loss': 0.10,
            'diagnostic': 0.15,
        }
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def _choose_action(
    rho: float,
    uncertainty: float,
    chi_r: int,
    chi_r_crit: int,
    chi_auto: bool,
    thresholds: tuple[float, float, float, float],
    i_min: float,
    i_pre: float,
    reduction_loss: float,
    trace_valid: bool,
    delta_max: float,
    uncertainty_max: float,
) -> str:
    t1, t2, t3, t4 = thresholds
    if chi_r_crit == 1:
        return 'block'
    if not chi_auto:
        return 'defer_to_human' if rho >= t3 else 'request_more_data'
    if (
        rho < t1
        and chi_r == 0
        and i_pre >= i_min
        and reduction_loss <= delta_max
        and trace_valid
        and uncertainty < uncertainty_max
    ):
        return 'accept'
    if rho < t2 and chi_r == 0:
        return 'lower_confidence'
    if uncertainty >= uncertainty_max or reduction_loss > delta_max or i_pre < i_min or not trace_valid:
        return 'request_more_data'
    if rho < t4:
        return 'defer_to_human'
    return 'block'


def _evaluate(
    rows: list[EvalRow],
    *,
    weights: dict[str, float],
    thresholds: tuple[float, float, float, float],
    gamma_max: float,
    i_min: float,
    delta_max: float,
    uncertainty_max: float,
) -> dict[str, Any]:
    actions: list[str] = []
    expected_proxy: list[str] = []
    expected_reference: list[str] = []
    crit_total = 0
    missed_critical = 0
    false_auto_accept = 0
    false_block = 0
    auto_accept = 0
    for row in rows:
        gamma_proxy = abs(row.predicted_risk - row.i_pre)
        chi_r = 1 if row.chi_r_raw == 1 or gamma_proxy > gamma_max else 0
        chi_r_crit = 1 if row.chi_r_crit_raw == 1 or gamma_proxy > (gamma_max + 0.10) else 0
        rho = (
            weights['predicted_risk'] * row.predicted_risk
            + weights['uncertainty'] * row.uncertainty
            + weights['interpretability_gap'] * (1.0 - row.i_pre)
            + weights['reduction_loss'] * row.reduction_loss
            + weights['diagnostic'] * float(chi_r)
        )
        rho = max(0.0, min(1.0, rho))
        chi_auto = bool(row.chi_auto and chi_r_crit == 0 and row.trace_valid)
        action = _choose_action(
            rho,
            row.uncertainty,
            chi_r,
            chi_r_crit,
            chi_auto,
            thresholds,
            i_min,
            row.i_pre,
            row.reduction_loss,
            row.trace_valid,
            delta_max,
            uncertainty_max,
        )
        actions.append(action)
        expected_proxy.append(
            proxy_action_label(
                predicted_risk=row.predicted_risk,
                uncertainty=row.uncertainty,
                i_pre=row.i_pre,
                rupture=bool(chi_r),
                critical_rupture=bool(chi_r_crit),
            )
        )
        expected_reference.append(
            reference_action_label(
                predicted_risk=row.predicted_risk,
                uncertainty=row.uncertainty,
                i_pre=row.i_pre,
                rupture=bool(chi_r),
                critical_rupture=bool(chi_r_crit),
                y_true=row.y_true,
                y_pred=row.y_pred,
            )
        )
        if chi_r_crit == 1:
            crit_total += 1
            if action != 'block':
                missed_critical += 1
        if action in AUTO_ACTIONS:
            auto_accept += 1
            if not row.chi_auto:
                false_auto_accept += 1
        if action == 'block' and chi_r_crit == 0:
            false_block += 1
    n = max(1, len(rows))
    agreement_proxy = float(np.mean([a == b for a, b in zip(actions, expected_proxy)])) if rows else 0.0
    agreement_reference = float(np.mean([a == b for a, b in zip(actions, expected_reference)])) if rows else 0.0
    return {
        'agreement_proxy': agreement_proxy,
        'agreement_reference': agreement_reference,
        'missed_critical_ruptures': int(missed_critical),
        'critical_rupture_recall': 1.0 if crit_total == 0 else float((crit_total - missed_critical) / crit_total),
        'false_auto_accept_rate': float(false_auto_accept / n),
        'false_block_rate': float(false_block / n),
        'auto_accept_coverage': float(auto_accept / n),
    }


def calibrate(dataset: str, *, out_root: str | Path = 'reports/datasets') -> dict[str, Any]:
    val_rows, test_rows = _load_eval_rows(dataset)
    candidate_weights = [
        _normalize_weights({'predicted_risk': 0.45, 'uncertainty': 0.20, 'interpretability_gap': 0.15, 'reduction_loss': 0.10, 'diagnostic': 0.10}),
        _normalize_weights({'predicted_risk': 0.50, 'uncertainty': 0.20, 'interpretability_gap': 0.10, 'reduction_loss': 0.10, 'diagnostic': 0.10}),
        _normalize_weights({'predicted_risk': 0.40, 'uncertainty': 0.15, 'interpretability_gap': 0.15, 'reduction_loss': 0.10, 'diagnostic': 0.20}),
        _normalize_weights({'predicted_risk': 0.35, 'uncertainty': 0.20, 'interpretability_gap': 0.15, 'reduction_loss': 0.10, 'diagnostic': 0.20}),
    ]
    threshold_grid = [
        (0.10, 0.25, 0.50, 0.80),
        (0.12, 0.28, 0.52, 0.80),
        (0.15, 0.30, 0.55, 0.82),
    ]
    gamma_grid = [0.40, 0.45, 0.50]
    i_min_grid = [0.60, 0.65, 0.70]
    delta_max_grid = [0.12, 0.15, 0.20]
    uncertainty_max_grid = [0.40, 0.45, 0.50]

    default_weights = _normalize_weights({'predicted_risk': 0.40, 'uncertainty': 0.20, 'interpretability_gap': 0.15, 'reduction_loss': 0.10, 'diagnostic': 0.15})
    before = _evaluate(
        test_rows,
        weights=default_weights,
        thresholds=(0.10, 0.25, 0.50, 0.80),
        gamma_max=0.45,
        i_min=0.65,
        delta_max=0.15,
        uncertainty_max=0.45,
    )

    best_score = -1e9
    best_params: dict[str, Any] = {}
    for w in candidate_weights:
        for th in threshold_grid:
            for gm in gamma_grid:
                for i_min in i_min_grid:
                    for delta_max in delta_max_grid:
                        for uncertainty_max in uncertainty_max_grid:
                            res = _evaluate(
                                val_rows,
                                weights=w,
                                thresholds=th,
                                gamma_max=gm,
                                i_min=i_min,
                                delta_max=delta_max,
                                uncertainty_max=uncertainty_max,
                            )
                            constraint_penalty = 0.0
                            if int(res['missed_critical_ruptures']) > 0:
                                constraint_penalty += 100.0
                            if float(res['critical_rupture_recall']) < 1.0:
                                constraint_penalty += 100.0
                            if float(res['false_auto_accept_rate']) > 0.10:
                                constraint_penalty += 20.0 * (float(res['false_auto_accept_rate']) - 0.10)
                            score = (
                                float(res['agreement_reference'])
                                - 5.0 * float(res['missed_critical_ruptures'])
                                - 2.0 * float(res['false_auto_accept_rate'])
                                - 1.0 * float(res['false_block_rate'])
                                - constraint_penalty
                            )
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'weights': w,
                                    'thresholds': th,
                                    'gamma_max': gm,
                                    'i_min': i_min,
                                    'delta_max': delta_max,
                                    'uncertainty_max': uncertainty_max,
                                    'val_metrics': res,
                                    'objective': score,
                                }

    after = _evaluate(
        test_rows,
        weights=dict(best_params['weights']),
        thresholds=tuple(best_params['thresholds']),
        gamma_max=float(best_params['gamma_max']),
        i_min=float(best_params['i_min']),
        delta_max=float(best_params['delta_max']),
        uncertainty_max=float(best_params['uncertainty_max']),
    )
    out = {
        'dataset': dataset,
        'status': 'READY',
        'split': {'train_val_test': [0.5625, 0.1875, 0.25]},
        'before_calibration': before,
        'after_calibration': after,
        'best_params': best_params,
        'objective': 'maximize agreement_reference - 5*missed_critical_ruptures - 2*false_auto_accept_rate - false_block_rate with hard safety penalties',
        'constraints': {
            'missed_critical_ruptures': 0,
            'critical_rupture_recall': 1.0,
            'false_auto_accept_rate_max': 0.10,
        },
        'notes': 'agreement_proxy is proxy-rule based; agreement_reference is independent policy agreement, both are not clinical expert labels.',
    }
    out_dir = Path(out_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'calibration.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        f'# Observer calibration: {dataset}',
        '',
        '| Mode | agreement_proxy | missed_critical_ruptures | false_auto_accept_rate |',
        '|---|---:|---:|---:|',
        f"| before | {before['agreement_proxy']:.6f} | {before['missed_critical_ruptures']} | {before['false_auto_accept_rate']:.6f} |",
        f"| after | {after['agreement_proxy']:.6f} | {after['missed_critical_ruptures']} | {after['false_auto_accept_rate']:.6f} |",
        '',
        '| Mode | agreement_reference | missed_critical_ruptures | false_auto_accept_rate |',
        '|---|---:|---:|---:|',
        f"| before | {before['agreement_reference']:.6f} | {before['missed_critical_ruptures']} | {before['false_auto_accept_rate']:.6f} |",
        f"| after | {after['agreement_reference']:.6f} | {after['missed_critical_ruptures']} | {after['false_auto_accept_rate']:.6f} |",
        '',
        f"- thresholds: `{best_params['thresholds']}`",
        f"- gamma_max: `{best_params['gamma_max']}`",
        f"- i_min: `{best_params['i_min']}`",
        f"- delta_max: `{best_params['delta_max']}`",
        f"- uncertainty_max: `{best_params['uncertainty_max']}`",
        f"- weights: `{best_params['weights']}`",
        '',
        '- agreement_proxy is a proxy-rule metric, not clinical expert action accuracy.',
    ]
    (out_dir / 'calibration.md').write_text('\n'.join(md), encoding='utf-8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_cancer')
    parser.add_argument('--out-root', default='reports/datasets')
    args = parser.parse_args()
    result = calibrate(args.dataset, out_root=args.out_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
