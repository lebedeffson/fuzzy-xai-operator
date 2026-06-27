from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from apps.chapter5_web_demo import evaluate_vector
from apps.services.layered_case import LayeredCaseService, build_case_state
from experiments.dataset_benchmark import run_benchmark
from fuzzyxai.risk import compute_application_risk


def _quantiles_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'i_pre_mean', 'i_pre_std', 'i_pre_median', 'i_pre_p25', 'i_pre_p75', 'i_pre_p05', 'i_pre_p95',
        'rho_mean', 'rho_std', 'rho_median', 'rho_p25', 'rho_p75', 'rho_p05', 'rho_p95',
    ]
    return {k: summary.get(k) for k in keys}


def _find_case_index(service: LayeredCaseService, predicate) -> int:
    for idx in range(len(service.backend.x_test)):
        vec = service.backend.x_test.iloc[idx].to_numpy(dtype=float)
        out = evaluate_vector(service.backend, vec, sample_id=f'sample_{idx}')
        if predicate(out):
            return int(idx)
    return 0


def _end_to_end_cases(service: LayeredCaseService) -> list[dict[str, Any]]:
    idx_accept = _find_case_index(
        service,
        lambda out: (str(out.get('action')) == 'accept') and (not bool(out.get('rupture', False))),
    )
    idx_audit = _find_case_index(
        service,
        lambda out: str(out.get('action')) in {'lower_confidence', 'request_more_data'},
    )
    if idx_audit == 0:
        idx_audit = _find_case_index(
            service,
            lambda out: str(out.get('action')) in {'lower_confidence', 'request_more_data', 'defer_to_human'},
        )
    idx_block = _find_case_index(service, lambda out: bool(out.get('rupture', False)) or str(out.get('action')) == 'block')

    rows: list[dict[str, Any]] = []
    configs = [
        ('accept', 'safe', idx_accept),
        ('audit', 'ambiguous', idx_audit),
        ('block', 'rupture', idx_block),
    ]
    for mode, scenario, idx in configs:
        state = build_case_state(service, scenario, sample_index=idx, dataset_mode='breast_cancer')
        action = str(state['risk']['action'])
        chi_auto = bool(state['contexts']['AutoAccept'].get('E_action'))
        rows.append(
            {
                'mode': mode,
                'object': str(state['input']['sample_id']),
                'P': float(state['model']['p_malignant']),
                'representation': str(state['uncertainty']['selected_class']),
                'delta': float(state['uncertainty']['delta']),
                'I_pre': float(state['explanation']['I_pre']),
                'rho': float(state['risk']['rho']),
                'chi_R': int(state['risk'].get('chi_R', 0)),
                'chi_R_crit': int(state['risk'].get('chi_R_crit', 0)),
                'chi_Auto': chi_auto,
                'action': action,
            }
        )
    return rows


def _proxy_expected_action(predicted_risk: float, uncertainty: float, chi_r_crit: int) -> str:
    if chi_r_crit == 1:
        return 'block'
    if predicted_risk >= 0.75:
        return 'defer_to_human'
    if predicted_risk >= 0.35 or uncertainty >= 0.45:
        return 'lower_confidence'
    return 'accept'


def _policy_action(
    rho: float,
    predicted_risk: float,
    uncertainty: float,
    chi_r: int,
    chi_r_crit: int,
    *,
    use_chi_r_crit: bool = True,
    use_chi_auto: bool = True,
    chi_auto: bool = True,
    thresholds: tuple[float, float, float, float] = (0.10, 0.25, 0.50, 0.80),
) -> str:
    t1, t2, t3, t4 = thresholds
    if use_chi_r_crit and chi_r_crit == 1:
        return 'block'
    if rho >= max(0.80, t4):
        return 'defer_to_human'
    if rho < t1:
        action = 'accept'
    elif rho < t2:
        action = 'lower_confidence'
    elif rho < t3:
        action = 'request_more_data'
    else:
        action = 'defer_to_human'
    if chi_r == 1 and chi_r_crit == 0 and rho < max(0.80, t4):
        action = 'request_more_data'
    if use_chi_auto and (action in {'accept', 'lower_confidence'}) and (not chi_auto):
        return 'request_more_data'
    if uncertainty >= 0.45 and predicted_risk >= 0.35 and action == 'accept':
        return 'lower_confidence'
    return action


def _collect_rows(service: LayeredCaseService) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(service.backend.x_test)):
        vec = service.backend.x_test.iloc[idx].to_numpy(dtype=float)
        out = evaluate_vector(service.backend, vec, sample_id=f'sample_{idx}')
        ctx = {row['object']: row for row in out.get('contexts', [])}
        auto_e_action = str(ctx.get('E_action', {}).get('AutoAccept', '')).strip()
        chi_auto = bool(auto_e_action)
        chi_r_crit = 1 if bool(out.get('rupture', False)) else 0
        chi_r = 1 if list(out.get('diagnostics', [])) else 0
        rb = dict(out.get('risk_breakdown', {}))
        rows.append(
            {
                'predicted_risk': float(out['prob_malignant']),
                'uncertainty': float(out['uncertainty']),
                'I_pre': float(out['I_pre']),
                'reduction_loss': float(rb.get('reduction_loss', 0.0)),
                'weights': dict(rb.get('weights', {})),
                'chi_R': chi_r,
                'chi_R_crit': chi_r_crit,
                'chi_Auto': chi_auto,
                'action_full': str(out['action']),
            }
        )
    return rows


def _ablation_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    actions: list[str] = []
    losses: list[float] = []
    critical: list[int] = []
    for row in rows:
        p = float(row['predicted_risk'])
        u = float(row['uncertainty'])
        i_pre = float(row['I_pre'])
        delta = float(row['reduction_loss'])
        chi_r = int(row['chi_R'])
        chi_r_crit = int(row['chi_R_crit'])
        chi_auto = bool(row['chi_Auto'])
        weights = dict(row['weights'])

        if variant == 'without_tau':
            chi_r = 0
            chi_r_crit = 0
        elif variant == 'without_delta':
            delta = 0.0
        elif variant == 'without_chi_r_crit':
            pass
        elif variant == 'only_f0':
            delta = max(delta, 0.25)
        elif variant == 'without_chi_auto':
            pass
        elif variant == 'probability_threshold_only':
            if p >= 0.80:
                actions.append('defer_to_human')
            elif p >= 0.35:
                actions.append('lower_confidence')
            else:
                actions.append('accept')
            losses.append(delta)
            critical.append(chi_r_crit)
            continue

        diagnostics = ['D_ij'] if chi_r == 1 else []
        rho = float(
            compute_application_risk(
                predicted_risk=p,
                uncertainty=u,
                pre_interpretability=i_pre,
                reduction_loss=delta,
                diagnostics=diagnostics,
                weights=weights,
            ).rho
        )
        action = _policy_action(
            rho=rho,
            predicted_risk=p,
            uncertainty=u,
            chi_r=chi_r,
            chi_r_crit=chi_r_crit,
            use_chi_r_crit=(variant != 'without_chi_r_crit'),
            use_chi_auto=(variant != 'without_chi_auto'),
            chi_auto=chi_auto,
        )
        actions.append(action)
        losses.append(delta)
        critical.append(chi_r_crit)

    p = np.array([float(r['predicted_risk']) for r in rows], dtype=float)
    u = np.array([float(r['uncertainty']) for r in rows], dtype=float)
    chi_crit = np.array([int(r['chi_R_crit']) for r in rows], dtype=int)
    chi_auto = np.array([1 if bool(r['chi_Auto']) else 0 for r in rows], dtype=int)
    a = np.array(actions, dtype=object)
    losses_arr = np.array(losses, dtype=float)

    crit_mask = chi_crit == 1
    missed = float(np.mean(a[crit_mask] != 'block')) if np.any(crit_mask) else 0.0
    false_auto = float(np.mean((np.isin(a, ['accept', 'lower_confidence'])) & (chi_auto == 0)))
    false_block = float(np.mean((a == 'block') & (chi_crit == 0)))
    auto_cov = float(np.mean(np.isin(a, ['accept', 'lower_confidence'])))
    proxy_expected = np.array([_proxy_expected_action(float(pp), float(uu), int(cc)) for pp, uu, cc in zip(p, u, chi_crit)], dtype=object)
    agreement_proxy = float(np.mean(a == proxy_expected))
    return {
        'variant': variant,
        'missed_critical_ruptures': missed,
        'false_automatic_accept': false_auto,
        'false_block_rate': false_block,
        'coverage_auto_accept': auto_cov,
        'mean_reduction_loss': float(np.mean(losses_arr)) if losses_arr.size else 0.0,
        'agreement_proxy': agreement_proxy,
    }


def _explainplan_parameters(service: LayeredCaseService) -> list[dict[str, Any]]:
    plan = service.backend.plan
    thresholds = tuple(float(v) for v in service.backend.observer.thresholds)
    weights = dict(service.backend.observer.weights)
    eta = dict(plan.eta)
    return [
        {'parameter': 'gamma_max', 'value': 0.45, 'where_used': 'certified alignment / composition', 'selection': 'fixed before experiments'},
        {'parameter': 'I_min', 'value': float(plan.i_min), 'where_used': 'interpretability floor', 'selection': 'ExplainPlan baseline'},
        {'parameter': 'theta_1...theta_4', 'value': list(thresholds), 'where_used': 'risk observer policy', 'selection': 'calibrated policy thresholds'},
        {'parameter': 'w_p,w_u,w_I,w_Delta,w_R', 'value': weights, 'where_used': 'rho risk aggregation', 'selection': 'observer calibration'},
        {'parameter': 'eta_1,eta_2,eta_3', 'value': {'model': eta.get('model'), 'rules': eta.get('rules'), 'trace': eta.get('trace')}, 'where_used': 'u_M decomposition', 'selection': 'ExplainPlan'},
        {'parameter': 'epsilon_int', 'value': float(plan.epsilon), 'where_used': 'interval tolerance / numerical stability', 'selection': 'fixed before experiments'},
        {'parameter': 'lambda_Delta', 'value': float(plan.beta.get('reduction', 0.10)), 'where_used': 'representation reduction penalty', 'selection': 'fixed before experiments'},
    ]


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(out)


def generate(*, out_dir: str | Path = 'reports/thesis_practice') -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = run_benchmark('breast_cancer', out_root='reports/datasets')
    quantiles = _quantiles_from_summary(summary)

    service = LayeredCaseService.create()
    cases = _end_to_end_cases(service)
    rows = _collect_rows(service)
    ablations = [
        _ablation_metrics(rows, 'without_tau'),
        _ablation_metrics(rows, 'without_delta'),
        _ablation_metrics(rows, 'without_chi_r_crit'),
        _ablation_metrics(rows, 'only_f0'),
        _ablation_metrics(rows, 'without_chi_auto'),
        _ablation_metrics(rows, 'probability_threshold_only'),
    ]
    params = _explainplan_parameters(service)

    payload = {
        'dataset': 'breast_cancer',
        'quantiles': quantiles,
        'end_to_end_cases': cases,
        'ablations': ablations,
        'explainplan_parameters': params,
    }
    (out / 'thesis_practice_tables.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# Thesis practice tables', '', '## I_pre / rho quantiles (breast_cancer)', '']
    md.append(
        _md_table(
            ['metric', 'mean', 'std', 'median', 'p25', 'p75', 'p05', 'p95'],
            [
                ['I_pre', quantiles['i_pre_mean'], quantiles['i_pre_std'], quantiles['i_pre_median'], quantiles['i_pre_p25'], quantiles['i_pre_p75'], quantiles['i_pre_p05'], quantiles['i_pre_p95']],
                ['rho', quantiles['rho_mean'], quantiles['rho_std'], quantiles['rho_median'], quantiles['rho_p25'], quantiles['rho_p75'], quantiles['rho_p05'], quantiles['rho_p95']],
            ],
        )
    )
    md += ['', '## End-to-end cases', '']
    md.append(
        _md_table(
            ['mode', 'object', 'P', 'representation', 'Delta', 'I_pre', 'rho', 'chi_R', 'chi_R_crit', 'chi_Auto', 'action'],
            [
                [
                    c['mode'],
                    c['object'],
                    round(float(c['P']), 6),
                    c['representation'],
                    round(float(c['delta']), 6),
                    round(float(c['I_pre']), 6),
                    round(float(c['rho']), 6),
                    c['chi_R'],
                    c['chi_R_crit'],
                    c['chi_Auto'],
                    c['action'],
                ]
                for c in cases
            ],
        )
    )
    md += ['', '## Ablations', '']
    md.append(
        _md_table(
            ['variant', 'missed_critical_ruptures', 'false_automatic_accept', 'false_block_rate', 'coverage_auto_accept', 'mean_reduction_loss', 'agreement_proxy'],
            [
                [
                    a['variant'],
                    round(float(a['missed_critical_ruptures']), 6),
                    round(float(a['false_automatic_accept']), 6),
                    round(float(a['false_block_rate']), 6),
                    round(float(a['coverage_auto_accept']), 6),
                    round(float(a['mean_reduction_loss']), 6),
                    round(float(a['agreement_proxy']), 6),
                ]
                for a in ablations
            ],
        )
    )
    md += ['', '## ExplainPlan parameters', '']
    md.append(
        _md_table(
            ['parameter', 'value', 'where_used', 'selection'],
            [[p['parameter'], p['value'], p['where_used'], p['selection']] for p in params],
        )
    )
    (out / 'thesis_practice_tables.md').write_text('\n'.join(md), encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/thesis_practice')
    args = parser.parse_args()
    payload = generate(out_dir=args.out_dir)
    print(json.dumps({'status': 'ok', 'cases': len(payload['end_to_end_cases']), 'ablations': len(payload['ablations'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
