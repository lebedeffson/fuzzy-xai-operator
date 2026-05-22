from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)

scripts = [
    ROOT / 'proofs' / 'chapter2_operator_proof.py',
    ROOT / 'proofs' / 'chapter2_calibration_proof.py',
    ROOT / 'proofs' / 'chapter3_hierarchy_proof.py',
    ROOT / 'examples' / 'chapter3_end_to_end.py',
    ROOT / 'proofs' / 'validate_thesis_examples.py',
    ROOT / 'examples' / 'thesis_demo.py',
]

ran = []
for script in scripts:
    print(f"\n=== running {script.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    ran.append(str(script.relative_to(ROOT)))

# Benchmark is included in the proof package, but it may be skipped in minimal environments.
bench = ROOT / 'benchmarks' / 'breast_cancer_benchmark.py'
try:
    print(f"\n=== running {bench.relative_to(ROOT)} ===")
    subprocess.run([sys.executable, str(bench)], cwd=ROOT, check=True)
    ran.append(str(bench.relative_to(ROOT)))
except Exception as exc:
    print(f"benchmark skipped or failed: {exc}")

summary = {
    'package': 'fuzzyxai_doctoral_core_v8_gui_fixed',
    'ran': ran,
    'reports': sorted(p.name for p in OUT.glob('*.json')),
}
(OUT / 'proof_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
md = ['# FuzzyXAI proof summary', '', 'Executed scripts:']
md += [f'- `{s}`' for s in ran]
md += ['', 'Generated JSON reports:']
md += [f'- `reports/{name}`' for name in summary['reports']]
(OUT / 'proof_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
