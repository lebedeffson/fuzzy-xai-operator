from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'formal_theorems'
OUT.mkdir(parents=True, exist_ok=True)

CHECKS = [
    ('2.1 partial composition', 'tests/test_composition_well_defined.py'),
    ('2.2 d_E pseudometric', 'tests/test_semantic_disagreement_pseudometric.py'),
    ('2.3 chain loss bound', 'tests/test_chain_loss_bound.py'),
    ('3.1 coverage/minimality', 'tests/test_operational_coverage_minimality.py'),
    ('3.2 reduction approximation', 'tests/test_reduction_approximation_theorem.py'),
    ('3.3 FML termination', 'tests/test_multilevel_reduction_termination.py'),
    ('4.1 no cyclic risk', 'tests/test_observer_no_cyclic_dependency.py'),
    ('4.2 expected cost', 'tests/test_expected_cost_optimality.py'),
    ('4.3 dataset pipeline', 'tests/test_dataset_observer_pipeline_correctness.py'),
]


def main() -> int:
    rows = []
    overall = 0
    for theorem, test_file in CHECKS:
        cmd = [sys.executable, '-m', 'pytest', '-q', test_file]
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT)
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
        status = 'PASS' if proc.returncode == 0 else 'FAIL'
        if proc.returncode != 0:
            overall = proc.returncode
        row = {
            'theorem': theorem,
            'code_check': test_file,
            'status': status,
        }
        if proc.returncode != 0:
            row['stdout'] = proc.stdout.strip()
            row['stderr'] = proc.stderr.strip()
        rows.append(row)
        print(f'{theorem}: {status}')

    report = {'status': 'PASS' if overall == 0 else 'FAIL', 'checks': rows}
    json_path = OUT / 'formal_theorem_checks.json'
    md_path = OUT / 'formal_theorem_checks.md'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# Formal theorem checks', '', f"Status: **{report['status']}**", '', '| Theorem | Code check | Status |', '|---|---|---|']
    md += [f"| {r['theorem']} | `{r['code_check']}` | {r['status']} |" for r in rows]
    md_path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'json': str(json_path), 'markdown': str(md_path)}, ensure_ascii=False, indent=2))
    return overall


if __name__ == '__main__':
    raise SystemExit(main())
