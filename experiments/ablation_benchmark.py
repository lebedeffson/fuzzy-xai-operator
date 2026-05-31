from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.calibrate_observer import EvalRow, _load_eval_rows, _normalize_weights
from fuzzyxai.risk import proxy_action_label


AUTO_ACTIONS = {'accept', 'lower_confidence'}


def _policy_action(rho: float, uncertainty: float, chi_r: int, chi_r_crit: int, chi_auto: bool, use_critical: bool, use_topos: bool) -> str:
    if use_critical and chi_r_crit == 1:
        return 'block'
    if rho < 0.10:
        action = 'accept'
    elif rho < 0.25:
        action = 'lower_confidence'
    elif rho < 0.50:
        action = 'request_more_data'
    else:
        action = 'defer_to_human'
    if uncertainty >= 0.45 and action == 'accept':
        action = 'lower_confidence'
    if chi_r == 1 and action == 'accept':
        action = 'request_more_data'
    if use_topos and action in AUTO_ACTIONS and not chi_auto:
        action = 'request_more_data'
    return action


def _mode_action(row: EvalRow, mode: str) -> tuple[str, int, int, float]:
    weights = _normalize_weights({'predicted_risk': 0.40, 'uncertainty': 0.20, 'interpretability_gap': 0.15, 'reduction_loss': 0.10, 'diagnostic': 0.15})
    chi_r = row.chi_r_raw
    chi_r_crit = row.chi_r_crit_raw
    reduction_loss = row.reduction_loss
    use_topos = True
    use_critical = True
    if mode == 'no_trace':
        chi_r = 0
        chi_r_crit = 0
    elif mode == 'no_delta':
        reduction_loss = 0.0
    elif mode == 'no_critical_rupture':
        use_critical = False
    elif mode == 'f0_only':
        reduction_loss = max(0.25, reduction_loss)
    elif mode == 'no_topos':
        use_topos = False
    if mode == 'probability_threshold':
        if row.predicted_risk >= 0.80:
            action = 'defer_to_human'
        elif row.predicted_risk >= 0.35:
            action = 'lower_confidence'
        else:
            action = 'accept'
        return action, chi_r, chi_r_crit, reduction_loss

    rho = (
        weights['predicted_risk'] * row.predicted_risk
        + weights['uncertainty'] * row.uncertainty
        + weights['interpretability_gap'] * (1.0 - row.i_pre)
        + weights['reduction_loss'] * reduction_loss
        + weights['diagnostic'] * float(chi_r)
    )
    rho = max(0.0, min(1.0, rho))
    return _policy_action(rho, row.uncertainty, chi_r, chi_r_crit, row.chi_auto, use_critical, use_topos), chi_r, chi_r_crit, reduction_loss


def _metrics(rows: list[EvalRow], mode: str) -> dict[str, Any]:
    actions: list[str] = []
    expected: list[str] = []
    missed_critical = 0
    crit_total = 0
    false_auto_accept = 0
    false_block = 0
    auto_accept = 0
    reduction_vals: list[float] = []
    for row in rows:
        action, chi_r, chi_r_crit, reduction_loss = _mode_action(row, mode)
        actions.append(action)
        expected.append(
            proxy_action_label(
                predicted_risk=row.predicted_risk,
                uncertainty=row.uncertainty,
                i_pre=row.i_pre,
                rupture=bool(chi_r),
                critical_rupture=bool(chi_r_crit),
            )
        )
        reduction_vals.append(reduction_loss)
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
    return {
        'mode': mode,
        'agreement_proxy': float(np.mean([a == b for a, b in zip(actions, expected)])) if rows else 0.0,
        'missed_critical_ruptures': int(missed_critical),
        'critical_rupture_recall': 1.0 if crit_total == 0 else float((crit_total - missed_critical) / crit_total),
        'false_auto_accept_rate': float(false_auto_accept / n),
        'false_block_rate': float(false_block / n),
        'auto_accept_coverage': float(auto_accept / n),
        'mean_reduction_loss': float(np.mean(reduction_vals)) if reduction_vals else 0.0,
    }


def run_ablation(dataset: str, *, out_root: str | Path = 'reports/datasets') -> dict[str, Any]:
    _val_rows, test_rows = _load_eval_rows(dataset)
    modes = ['full', 'no_trace', 'no_delta', 'no_critical_rupture', 'f0_only', 'no_topos', 'probability_threshold']
    rows = [_metrics(test_rows, mode) for mode in modes]
    out = {
        'dataset': dataset,
        'status': 'READY',
        'rows': rows,
        'notes': 'Agreement uses proxy-rule labels, not expert clinical action labels.',
    }
    out_dir = Path(out_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ablation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        f'# Ablation benchmark: {dataset}',
        '',
        '| mode | agreement_proxy | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage | mean_reduction_loss |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        md.append(
            f"| {row['mode']} | {row['agreement_proxy']:.6f} | {row['missed_critical_ruptures']} | "
            f"{row['false_auto_accept_rate']:.6f} | {row['false_block_rate']:.6f} | "
            f"{row['auto_accept_coverage']:.6f} | {row['mean_reduction_loss']:.6f} |"
        )
    md.append('')
    md.append('- agreement_proxy is a proxy-rule metric, not clinical expert action accuracy.')
    (out_dir / 'ablation.md').write_text('\n'.join(md), encoding='utf-8')
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_cancer')
    parser.add_argument('--out-root', default='reports/datasets')
    args = parser.parse_args()
    result = run_ablation(args.dataset, out_root=args.out_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

