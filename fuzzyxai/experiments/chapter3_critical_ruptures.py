from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 314159
N_OBJECTS = 1000
RUPTURE_TYPES = [
    'trace_gap',
    'rule_conflict',
    'term_inversion',
    'source_conflict',
    'missing_certified_transition',
]
POLICIES = [
    'probability_threshold',
    'shap_rule',
    'lime_rule',
    'anchors_rule',
    'fuzzyxai_full_contour',
]
COMMAND = 'make reproduce-critical-ruptures'


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _explain_plan_version() -> dict[str, str]:
    candidates = [
        ROOT / 'configs' / 'explain_plan_chapter2.yaml',
        ROOT / 'reports' / 'reproducibility_artifacts' / 'explain_plan.json',
    ]
    for path in candidates:
        if path.exists():
            return {'path': str(path.relative_to(ROOT)), 'sha256': _sha_file(path)}
    return {'path': 'not_found', 'sha256': ''}


def _generate_cases(seed: int = SEED, n_objects: int = N_OBJECTS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n_objects):
        is_critical = i % 4 == 0
        rupture_type = RUPTURE_TYPES[(i // 4) % len(RUPTURE_TYPES)] if is_critical else 'clean'
        p = float(np.clip(rng.beta(2.2, 2.8), 0.01, 0.99))

        trace_valid = rupture_type != 'trace_gap'
        certified_transition = rupture_type != 'missing_certified_transition'
        rule_conflict = rupture_type == 'rule_conflict'
        term_inversion = rupture_type == 'term_inversion'
        source_conflict = rupture_type == 'source_conflict'

        # Native XAI baselines receive noisy local signals, not chi_R_crit.
        shap_instability = float(np.clip(rng.normal(0.25, 0.12), 0, 1))
        lime_instability = float(np.clip(rng.normal(0.22, 0.12), 0, 1))
        anchor_conflict = float(np.clip(rng.normal(0.18, 0.10), 0, 1))
        if rule_conflict:
            shap_instability = float(np.clip(rng.normal(0.78, 0.08), 0, 1))
            lime_instability = float(np.clip(rng.normal(0.66, 0.10), 0, 1))
        if term_inversion:
            lime_instability = float(np.clip(rng.normal(0.80, 0.08), 0, 1))
        if source_conflict:
            shap_instability = float(np.clip(rng.normal(0.58, 0.12), 0, 1))
        if rupture_type == 'missing_certified_transition':
            anchor_conflict = float(np.clip(rng.normal(0.70, 0.10), 0, 1))

        rows.append(
            {
                'object_id': f'cr_{i:04d}',
                'predicted_risk': round(p, 6),
                'rupture_type': rupture_type,
                'chi_R': int(is_critical),
                'chi_R_crit': int(is_critical),
                'trace_valid': trace_valid,
                'certified_transition': certified_transition,
                'rule_conflict': rule_conflict,
                'term_inversion': term_inversion,
                'source_conflict': source_conflict,
                'shap_instability': round(shap_instability, 6),
                'lime_instability': round(lime_instability, 6),
                'anchor_conflict': round(anchor_conflict, 6),
            }
        )
    return rows


def _reference_action(row: dict[str, Any]) -> str:
    if int(row['chi_R_crit']):
        return 'block'
    p = float(row['predicted_risk'])
    if p < 0.55:
        return 'accept'
    if p < 0.78:
        return 'lower_confidence'
    return 'defer_to_human'


def _policy_action(policy: str, row: dict[str, Any]) -> str:
    p = float(row['predicted_risk'])
    if policy == 'probability_threshold':
        return 'block' if p >= 0.92 else ('accept' if p < 0.55 else 'defer_to_human')
    if policy == 'shap_rule':
        if float(row['shap_instability']) >= 0.72 or p >= 0.94:
            return 'block'
        return 'accept' if p < 0.55 else 'defer_to_human'
    if policy == 'lime_rule':
        if float(row['lime_instability']) >= 0.72 or p >= 0.94:
            return 'block'
        return 'accept' if p < 0.55 else 'defer_to_human'
    if policy == 'anchors_rule':
        if float(row['anchor_conflict']) >= 0.72 or p >= 0.94:
            return 'block'
        return 'accept' if p < 0.55 else 'defer_to_human'
    if policy == 'fuzzyxai_full_contour':
        if int(row['chi_R_crit']):
            return 'block'
        if not bool(row['trace_valid']) or not bool(row['certified_transition']):
            return 'request_more_data'
        return 'accept' if p < 0.55 else ('lower_confidence' if p < 0.78 else 'defer_to_human')
    raise ValueError(f'Unknown policy: {policy}')


def _evaluate_policy(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    actions = [_policy_action(policy, row) for row in rows]
    elapsed = time.perf_counter() - t0
    refs = [_reference_action(row) for row in rows]
    total_critical = sum(int(row['chi_R_crit']) for row in rows)
    missed = sum(1 for row, action in zip(rows, actions) if int(row['chi_R_crit']) and action != 'block')
    blocked_critical = total_critical - missed
    noncritical = len(rows) - total_critical
    false_blocks = sum(1 for row, action in zip(rows, actions) if not int(row['chi_R_crit']) and action == 'block')
    agreement = sum(1 for action, ref in zip(actions, refs) if action == ref) / len(rows)
    return {
        'policy': policy,
        'n_objects': len(rows),
        'critical_objects': total_critical,
        'missed_critical_ruptures': missed,
        'critical_rupture_recall': blocked_critical / total_critical if total_critical else 'N/A',
        'false_block_rate': false_blocks / noncritical if noncritical else 'N/A',
        'agreement_reference': agreement,
        'mean_processing_time_ms': round((elapsed / len(rows)) * 1000.0, 4),
    }


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        'policy',
        'n_objects',
        'critical_objects',
        'missed_critical_ruptures',
        'critical_rupture_recall',
        'false_block_rate',
        'agreement_reference',
        'mean_processing_time_ms',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})


def _write_figure(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {
        'probability_threshold': 'порог\nвероятности',
        'shap_rule': 'SHAP\nправило',
        'lime_rule': 'LIME\nправило',
        'anchors_rule': 'Anchors\nправило',
        'fuzzyxai_full_contour': 'полный\nFuzzyXAI',
    }
    labels = [label_map.get(str(r['policy']), str(r['policy']).replace('_', '\n')) for r in results]
    missed = [float(r['missed_critical_ruptures']) for r in results]
    recall = [float(r['critical_rupture_recall']) for r in results]
    x = np.arange(len(results))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, missed, width=0.36, color='#dc2626', label='пропущенные критические разрывы')
    ax2.bar(x + 0.18, recall, width=0.36, color='#0f766e', label='полнота обнаружения')
    ax1.set_xticks(x, labels)
    ax1.set_ylabel('число пропущенных критических разрывов')
    ax2.set_ylabel('полнота обнаружения критических разрывов')
    ax2.set_ylim(0, 1.08)
    ax1.set_title('Контролируемые критические разрывы: сравнение политик')
    ax1.grid(axis='y', alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(out_dir: str | Path = 'reports/chapter3', seed: int = SEED, n_objects: int = N_OBJECTS) -> dict[str, Any]:
    out = ROOT / out_dir
    fig_path = ROOT / 'figures' / 'chapter3' / 'critical_ruptures_comparison.png'
    manifest_path = ROOT / 'evidence' / 'critical_ruptures_manifest_sha256.json'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _generate_cases(seed=seed, n_objects=n_objects)
    canonical_input = '\n'.join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows).encode('utf-8')
    input_checksum = _sha_bytes(canonical_input)
    results = [_evaluate_policy(policy, rows) for policy in POLICIES]
    results_path = out / 'synthetic_ruptures_results.csv'
    _write_results_csv(results_path, results)
    _write_figure(fig_path, results)

    results_checksum = _sha_file(results_path)
    plan = _explain_plan_version()
    summary = {
        'status': 'ok',
        'experiment': 'controlled_critical_ruptures',
        'seed': seed,
        'n_objects': n_objects,
        'rupture_types': RUPTURE_TYPES,
        'policies': POLICIES,
        'explain_plan_version': plan,
        'input_checksum_sha256': input_checksum,
        'results_checksum_sha256': results_checksum,
        'command': COMMAND,
        'results_csv': str(results_path.relative_to(ROOT)),
        'figure': str(fig_path.relative_to(ROOT)),
        'rows': results,
    }
    summary_path = out / 'synthetic_ruptures_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest = {
        'status': 'ok',
        'command': COMMAND,
        'seed': seed,
        'n_objects': n_objects,
        'input_checksum_sha256': input_checksum,
        'results_checksum_sha256': results_checksum,
        'files': {
            str(summary_path.relative_to(ROOT)): _sha_file(summary_path),
            str(results_path.relative_to(ROOT)): _sha_file(results_path),
            str(fig_path.relative_to(ROOT)): _sha_file(fig_path),
        },
        'explain_plan_version': plan,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/chapter3')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-objects', type=int, default=N_OBJECTS)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir, seed=args.seed, n_objects=args.n_objects), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
