from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apps.services.layered_case import LayeredCaseService, build_case_state


def _case_row(state: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        'case': label,
        'object': state['input']['sample_id'],
        'P': float(state['model']['p_malignant']),
        'representation': str(state['uncertainty']['selected_class']),
        'delta': float(state['uncertainty']['delta']),
        'I_pre': float(state['explanation']['I_pre']),
        'rho': float(state['risk']['rho']),
        'chi_R': int(state['risk'].get('chi_R', 0)),
        'chi_R_crit': int(state['risk'].get('chi_R_crit', 0)),
        'chi_Auto': bool(state['contexts']['AutoAccept'].get('E_action')),
        'action': str(state['risk']['action']),
    }


def _pick_case(service: LayeredCaseService, target: str) -> dict[str, Any]:
    if target == 'audit':
        for idx in range(len(service.backend.x_test)):
            st = build_case_state(service, 'safe', sample_index=idx, dataset_mode='breast_cancer')
            act = str(st['risk']['action'])
            if act in {'lower_confidence', 'request_more_data'} and int(st['risk'].get('chi_R_crit', 0)) == 0:
                return st
    for idx in range(len(service.backend.x_test)):
        st = build_case_state(service, 'safe', sample_index=idx, dataset_mode='breast_cancer')
        act = str(st['risk']['action'])
        if target == 'accept' and act == 'accept' and int(st['risk'].get('chi_R_crit', 0)) == 0:
            return st
        if target == 'audit' and act in {'defer_to_human'} and int(st['risk'].get('chi_R_crit', 0)) == 0:
            return st
    # deterministic fallback
    if target == 'accept':
        return build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    if target == 'audit':
        return build_case_state(service, 'ambiguous', dataset_mode='breast_cancer')
    return build_case_state(service, 'rupture', sample_index=113, dataset_mode='breast_cancer')


def generate_defense_cases(*, out_dir: str | Path = 'reports/defense_cases') -> dict[str, Any]:
    service = LayeredCaseService.create()
    accept_state = _pick_case(service, 'accept')
    audit_state = _pick_case(service, 'audit')
    block_state = build_case_state(service, 'rupture', sample_index=113, dataset_mode='breast_cancer')

    accept = _case_row(accept_state, 'safe')
    audit = _case_row(audit_state, 'audit')
    block = _case_row(block_state, 'block')
    summary = {'dataset': 'breast_cancer', 'cases': [accept, audit, block]}

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'accept_case.json').write_text(json.dumps(accept, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'audit_case.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'block_case.json').write_text(json.dumps(block, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    md = [
        '# Defense cases summary',
        '',
        '| case | object | P | representation | Delta | I_pre | rho | chi_R | chi_R_crit | chi_Auto | action |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in summary['cases']:
        md.append(
            f"| {row['case']} | {row['object']} | {row['P']:.6f} | {row['representation']} | {row['delta']:.6f} | "
            f"{row['I_pre']:.6f} | {row['rho']:.6f} | {row['chi_R']} | {row['chi_R_crit']} | {row['chi_Auto']} | {row['action']} |"
        )
    (root / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/defense_cases')
    args = parser.parse_args()
    res = generate_defense_cases(out_dir=args.out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
