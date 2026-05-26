from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.check_dataset_modes import main as dataset_modes_main
from experiments.dataset_benchmark import run_benchmark
from experiments.real_reduction_example import generate_real_reduction_example
from fuzzyxai.datasets import list_dataset_modes, load_dataset_mode


def _dataset_mode_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in list_dataset_modes():
        status = 'READY'
        n = None
        notes = ''
        try:
            _record, df = load_dataset_mode(spec.key)
            n = int(len(df))
        except FileNotFoundError as exc:
            status = 'MISSING'
            notes = str(exc)
        except Exception as exc:
            status = 'ERROR'
            notes = f'{type(exc).__name__}: {exc}'
        rows.append({'dataset_mode': spec.key, 'status': status, 'rows': n, 'domain': spec.domain, 'notes': notes})
    return rows


def generate_summary(*, out_dir: str | Path = 'reports') -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [
        'breast_cancer',
        'diabetes_binary',
        'wine_risk',
        'synthetic_ruptures',
        'registry_programs',
        'registry_mosmed_doctor_analysis',
        'registry_steel_ir',
    ]
    benchmark = {ds: run_benchmark(ds, out_root=out_root / 'datasets') for ds in datasets}
    real_reduction = generate_real_reduction_example(out_dir=out_root / 'real_reduction_example')
    mode_rows = _dataset_mode_status()

    category_checks_path = out_root / 'category_hott/category_hott_checks.json'
    category_checks = None
    if category_checks_path.exists():
        category_checks = json.loads(category_checks_path.read_text(encoding='utf-8'))

    summary = {
        'dataset_modes': mode_rows,
        'benchmark_summaries': benchmark,
        'real_reduction_example': real_reduction,
        'category_topos_checks': category_checks,
        'gui_smoke': {
            'layered_demo_script': str((Path('apps/layered_demo.py')).as_posix()),
            'unified_demo_script': str((Path('apps/unified_demo.py')).as_posix()),
            'status': 'available',
        },
        'test_status': 'run `PYTHONPATH=. pytest` to refresh in current environment',
    }

    json_path = out_root / 'dissertation_demo_summary.json'
    md_path = out_root / 'dissertation_demo_summary.md'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    md_lines = [
        '# Dissertation demo summary',
        '',
        '## Dataset modes',
        '',
        '| dataset_mode | status | rows | domain |',
        '|---|---:|---:|---|',
    ]
    for row in mode_rows:
        md_lines.append(f"| {row['dataset_mode']} | {row['status']} | {row['rows'] if row['rows'] is not None else '-'} | {row['domain']} |")

    md_lines += [
        '',
        '## Real reduction example',
        '',
        f"- object: `{real_reduction['object']}`",
        f"- selected_class: `{real_reduction['selected_class']}`",
        f"- Delta: `{real_reduction['Delta']}`",
        f"- action: `{real_reduction['action']}`",
        '',
        '## Notes',
        '',
        '- Registry modes may remain `MISSING` until local files are connected.',
        '- Benchmark timing is prototype-level per object and excludes I/O.',
    ]
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports')
    args = parser.parse_args()
    # keep console compatibility with dataset-modes table call
    dataset_modes_main()
    summary = generate_summary(out_dir=args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
