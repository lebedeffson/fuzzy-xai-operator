from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from fuzzyxai.rules import (
    binary_predictions_from_logits,
    bootstrap_lofo_f1_importance,
    budget_prune_importance,
    jaccard_similarity,
    lofo_f1_importance,
    logits_from_rules,
    select_top_rules_by_lofo_f1,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)


def _top_by_heuristic(scores: np.ndarray, budget: int) -> list[int]:
    return [int(idx) for idx in np.argsort(scores)[::-1][:budget]]


def _f1_for_subset(h: np.ndarray, theta: np.ndarray, y: np.ndarray, selected: list[int], bias: float) -> float:
    z = np.full(h.shape[0], bias, dtype=float)
    if selected:
        z += h[:, selected] @ theta[selected]
    return float(f1_score(y, binary_predictions_from_logits(z), zero_division=0))


def build_lofo_f1_report(*, write: bool = True) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_samples = 900
    n_rules = 96
    budget = 12
    rule_names = [f'r_{idx:03d}' for idx in range(n_rules)]

    h = rng.beta(1.8, 4.0, size=(n_samples, n_rules))
    true_theta = np.zeros(n_rules)
    true_theta[:6] = [3.5, -2.8, 3.0, 2.4, -2.2, 2.0]

    bias = -0.40
    y_logits = bias + h @ true_theta + rng.normal(0, 0.20, n_samples)
    y = binary_predictions_from_logits(y_logits)

    theta = true_theta + rng.normal(0, 0.05, n_rules)

    z_full = logits_from_rules(h, theta, bias=bias)
    f1_full = float(f1_score(y, binary_predictions_from_logits(z_full), zero_division=0))

    lofo = lofo_f1_importance(h, theta, y, bias=bias, rule_names=rule_names)
    lofo_top = select_top_rules_by_lofo_f1(lofo, budget)
    heuristic_scores = budget_prune_importance(h, theta)
    heuristic_top = _top_by_heuristic(heuristic_scores, budget)
    boot = bootstrap_lofo_f1_importance(h, theta, y, bias=bias, rule_names=rule_names, n_bootstraps=40, top_k=budget, random_state=7)
    boot_top = [row.rule_index for row in boot[:budget]]
    true_rules = set(range(6))

    report = {
        'method': 'LOFO-F1 / Leave-One-Rule-Out F1 importance',
        'formula': 'z_without_r = z_full - H_val[:, r] * theta[r]; importance_r = F1_full - F1_without_r',
        'dataset': {'type': 'synthetic KAFN activation matrix', 'samples': n_samples, 'rules': n_rules, 'budget': budget},
        'baseline': {'f1_full': round(f1_full, 6)},
        'top_rules': {
            'lofo_f1': [rule_names[idx] for idx in lofo_top],
            'budget_prune': [rule_names[idx] for idx in heuristic_top],
            'bootstrap_lofo_f1': [rule_names[idx] for idx in boot_top],
        },
        'subset_quality': {
            'lofo_f1_top_budget_f1': round(_f1_for_subset(h, theta, y, lofo_top, bias), 6),
            'budget_prune_top_budget_f1': round(_f1_for_subset(h, theta, y, heuristic_top, bias), 6),
            'bootstrap_lofo_top_budget_f1': round(_f1_for_subset(h, theta, y, boot_top, bias), 6),
        },
        'stability_proxy': {
            'jaccard_lofo_vs_bootstrap_lofo': round(jaccard_similarity(lofo_top, boot_top), 6),
            'jaccard_lofo_vs_budget_prune': round(jaccard_similarity(lofo_top, heuristic_top), 6),
        },
        'recovery': {
            'true_useful_rules': [rule_names[idx] for idx in sorted(true_rules)],
            'lofo_hits': len(true_rules & set(lofo_top)),
            'budget_prune_hits': len(true_rules & set(heuristic_top)),
            'bootstrap_lofo_hits': len(true_rules & set(boot_top)),
        },
        'lofo_table': [
            {
                'rank': rank,
                'rule': row.rule_name,
                'delta_f1': round(row.delta_f1, 6),
                'f1_without': round(row.f1_without, 6),
                'theta': round(row.coefficient, 6),
                'mean_abs_activation': round(row.mean_abs_activation, 6),
            }
            for rank, row in enumerate(lofo[:15], start=1)
        ],
        'bootstrap_table': [
            {
                'rank': rank,
                'rule': row.rule_name,
                'mean_delta_f1': round(row.mean_delta_f1, 6),
                'std_delta_f1': round(row.std_delta_f1, 6),
                'top_frequency': round(row.top_frequency, 6),
                'mean_rank': round(row.mean_rank, 3),
            }
            for rank, row in enumerate(boot[:15], start=1)
        ],
        'conclusion': 'LOFO-F1 ranks rules by real validation F1 drop without retraining; bootstrap aggregation turns it into a stable top-B selector.',
    }
    if write:
        (OUT / 'lofo_f1_rule_pruning.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / 'lofo_f1_rule_pruning.md').write_text(render_markdown(report), encoding='utf-8')
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# LOFO-F1 rule pruning demo',
        '',
        f"Method: `{report['method']}`",
        f"Formula: `{report['formula']}`",
        '',
        '## Quality',
        '',
        f"- Full F1: `{report['baseline']['f1_full']}`",
        f"- LOFO-F1 top-B F1: `{report['subset_quality']['lofo_f1_top_budget_f1']}`",
        f"- Budget-Prune top-B F1: `{report['subset_quality']['budget_prune_top_budget_f1']}`",
        f"- Bootstrap LOFO-F1 top-B F1: `{report['subset_quality']['bootstrap_lofo_top_budget_f1']}`",
        '',
        '## Stability proxy',
        '',
        f"- Jaccard LOFO vs bootstrap LOFO: `{report['stability_proxy']['jaccard_lofo_vs_bootstrap_lofo']}`",
        f"- Jaccard LOFO vs Budget-Prune: `{report['stability_proxy']['jaccard_lofo_vs_budget_prune']}`",
        '',
        '## Top LOFO-F1 rules',
        '',
    ]
    for row in report['lofo_table'][:10]:
        lines.append(f"- {row['rank']}. `{row['rule']}`: delta_f1=`{row['delta_f1']}`, f1_without=`{row['f1_without']}`")
    lines.extend(['', '## Conclusion', '', report['conclusion'], ''])
    return '\n'.join(lines)


def main() -> None:
    report = build_lofo_f1_report(write=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
