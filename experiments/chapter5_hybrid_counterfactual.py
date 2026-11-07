"""Run a structural HYBRID-XIRIS quality/confidence grid through the operator.

Synthetic controlled inputs are not biometric recognition ground truth.
Unlike the legacy 1000-row batch, actions are actually returned by FuzzyXAI.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'framework' / 'fuzzyxai'
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))


def controlled_cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for img, seg, score, accept_level in product(
        (0.2, 0.4, 0.6, 0.8),
        (0.2, 0.4, 0.6, 0.8),
        (0.55, 0.85),
        (0.60, 0.80),
    ):
        result.append({
            'case_id': f'hybrid_grid_{len(result):03d}',
            'image_quality': img,
            'segmentation_quality': seg,
            'model_match_signal': score,
            'alpha_accept': accept_level,
            'alpha_block': 0.85,
        })
    return result


def predeclared_conflict(case: dict[str, Any]) -> bool:
    """Input-driven reference for testing the specific operator guardrail."""
    return (
        case['model_match_signal'] >= case['alpha_accept']
        and case['image_quality'] < 0.5
        and case['segmentation_quality'] < 0.5
    )


def run_controlled_grid(out_dir: Path) -> dict[str, Any]:
    from fuzzyxai.core.types import AdaptedInput
    from fuzzyxai.core.route import build_route
    from fuzzyxai.proof.trace import build_proof_trace
    from fuzzyxai.proof.verifier import verify_proof_trace
    from fuzzyxai.examples import load_example

    prototype = load_example('hybrid_xiris')
    rows: list[dict[str, Any]] = []
    for spec in controlled_cases():
        payload = {**prototype.values, **{k: v for k, v in spec.items() if k != 'case_id'}}
        route = build_route(AdaptedInput(scenario_id='hybrid_xiris', values=payload, source='controlled_grid'))
        trace = build_proof_trace(route)
        verification = verify_proof_trace(trace)
        if not verification.valid:
            raise RuntimeError(f"{spec['case_id']}: invalid proof: {verification.errors}")
        critical_reference = predeclared_conflict(spec)
        if critical_reference and route.final_action != 'block':
            raise AssertionError(f"{spec['case_id']}: critical guardrail violation")
        baseline = 'accept' if spec['model_match_signal'] >= 0.75 else 'lower_confidence'
        rows.append({
            **spec,
            'critical_reference': int(critical_reference),
            'baseline_action': baseline,
            'fuzzyxai_action': route.final_action,
            'chi_crit': int(route.computed_result.get('chi_crit', 0)),
            'baseline_missed_guardrail': int(critical_reference and baseline != 'block'),
            'fuzzyxai_missed_guardrail': int(critical_reference and route.final_action != 'block'),
            'proof_valid': True,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'cases.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'protocol': 'chapter5_hybrid_controlled_grid_v1',
        'n_cases': len(rows),
        'n_critical_reference': sum(row['critical_reference'] for row in rows),
        'baseline_missed_guardrail': sum(row['baseline_missed_guardrail'] for row in rows),
        'fuzzyxai_missed_guardrail': sum(row['fuzzyxai_missed_guardrail'] for row in rows),
        'proofs_verified': len(rows),
        'actions_computed_by_operator': True,
        'data_role': 'synthetic_structural_guardrail_test',
        'clinical_or_independent_biometric_validation': False,
        'reference_uses_same_explicit_guardrail_as_operator': True,
        'note': 'This is a correctness/consistency test of a declared guardrail, not an independent effectiveness estimate.',
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/chapter5/hybrid_counterfactual')
    args = parser.parse_args()
    print(json.dumps(run_controlled_grid(args.out), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
