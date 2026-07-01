from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ['probability_threshold', 'shap_guardrail', 'lime_guardrail', 'anchor_guardrail', 'fuzzyxai_observer']
MODES = ['native', 'equal_guardrail', 'equal_raw_structure']


def _cases() -> list[dict[str, Any]]:
    rows = []
    for i in range(120):
        crit = i % 6 == 0
        rupture = crit or i % 5 == 0
        context_forbidden = i % 7 == 0
        delta_high = i % 8 == 0
        p = 0.12 + (i % 17) / 20
        rows.append({'case_id': i, 'p': min(0.98, p), 'chi_R': int(rupture), 'chi_R_crit': int(crit), 'chi_Auto': not context_forbidden, 'Delta': 0.28 if delta_high else 0.07})
    return rows


def _decide(policy: str, mode: str, row: dict[str, Any]) -> str:
    risk_action = 'accept' if row['p'] < 0.55 else 'defer_to_human'
    if policy == 'fuzzyxai_observer':
        if row['chi_R_crit']:
            return 'block'
        if not row['chi_Auto'] or row['Delta'] > 0.15 or row['chi_R']:
            return 'request_more_data'
        return risk_action
    if mode == 'equal_guardrail' and row['chi_R_crit']:
        return 'block'
    if mode == 'equal_raw_structure':
        if row['chi_R_crit'] and policy in {'shap_guardrail', 'lime_guardrail'}:
            return 'defer_to_human'
        if row['chi_R'] and policy == 'anchor_guardrail':
            return 'request_more_data'
    return risk_action


def run(out_dir: str | Path = 'reports/chapter2') -> dict[str, Any]:
    out = ROOT / out_dir
    fig_dir = ROOT / 'figures/chapter2'
    out.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    rows = []
    for mode in MODES:
        for policy in POLICIES:
            actions = [_decide(policy, mode, r) for r in cases]
            missed = sum(1 for r, a in zip(cases, actions) if r['chi_R_crit'] and a != 'block')
            false_auto = sum(1 for r, a in zip(cases, actions) if (r['chi_R'] or not r['chi_Auto'] or r['Delta'] > 0.15) and a == 'accept') / len(cases)
            agreement = sum(1 for r, a in zip(cases, actions) if (r['chi_R_crit'] and a == 'block') or (not r['chi_R_crit'] and a != 'block')) / len(cases)
            rows.append({
                'mode': mode,
                'policy': policy,
                'agreement': agreement,
                'missed_critical': missed,
                'false_auto_accept': false_auto,
                'trace_access': mode in {'equal_raw_structure'} or policy == 'fuzzyxai_observer',
                'rule_access': mode in {'equal_raw_structure'} or policy == 'fuzzyxai_observer',
                'delta_access': mode in {'equal_raw_structure'} or policy == 'fuzzyxai_observer',
                'certified_path_access': policy == 'fuzzyxai_observer',
            })
    fields = list(rows[0].keys())
    with (out / 'equal_raw_structure_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    payload = {'status': 'ok', 'n_cases': len(cases), 'modes': MODES, 'rows': rows, 'note': 'equal_raw_structure gives raw fields to baselines but not certified path semantics'}
    (out / 'equal_raw_structure_report.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    fx = [r for r in rows if r['policy'] == 'fuzzyxai_observer']
    base = [r for r in rows if r['policy'] == 'probability_threshold']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(MODES))
    ax.bar([i - 0.18 for i in x], [r['agreement'] for r in base], width=0.35, label='threshold', color='#f97316')
    ax.bar([i + 0.18 for i in x], [r['agreement'] for r in fx], width=0.35, label='FuzzyXAI', color='#0f766e')
    ax.set_xticks(list(x), MODES); ax.set_ylim(0, 1.05); ax.set_title('Equal raw structure benchmark')
    ax.legend(); ax.grid(axis='y', alpha=0.25); fig.tight_layout(); fig.savefig(fig_dir / 'equal_raw_structure_comparison.png', dpi=180); plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--out-dir', default='reports/chapter2')
    args = parser.parse_args(); print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
