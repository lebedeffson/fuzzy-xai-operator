from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.ecosystem import build_evidence_matrix, load_ecosystem_registry, write_evidence_pack


def run(*, evidence_dir: str | Path = 'evidence', report_dir: str | Path = 'reports/chapter4') -> dict[str, str]:
    paths = write_evidence_pack(evidence_dir)
    modules = load_ecosystem_registry()
    matrix = build_evidence_matrix(modules)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'ecosystem_evidence.json'
    md_path = out / 'ecosystem_evidence.md'

    payload = {
        'status': 'ok',
        'module_count': len(modules),
        'runnable_count': sum(1 for row in matrix if row['run_allowed']),
        'quantitative_claim_count': sum(1 for row in matrix if row['quantitative_claim_allowed']),
        'evidence_paths': paths,
        'rows': matrix,
        'notes': [
            'External repositories are integrated through lightweight registry metadata and fixture-compatible adapters.',
            'Planned modules are visible in the registry but excluded from quantitative claims.',
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    md = [
        '# Chapter 4 Ecosystem Evidence',
        '',
        f"- modules: `{payload['module_count']}`",
        f"- runnable: `{payload['runnable_count']}`",
        f"- quantitative_claim_allowed: `{payload['quantitative_claim_count']}`",
        '',
        '| module | status | evidence_level | role | claim_scope |',
        '| --- | --- | --- | --- | --- |',
    ]
    for row in matrix:
        md.append(
            f"| `{row['registry_id']}` | `{row['status']}` | `{row['evidence_level']}` | "
            f"{row['role']} | {row['claim_scope']} |"
        )
    md_path.write_text('\n'.join(md), encoding='utf-8')

    return {**paths, 'json': str(json_path), 'md': str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-dir', default='evidence')
    parser.add_argument('--report-dir', default='reports/chapter4')
    args = parser.parse_args()
    print(json.dumps(run(evidence_dir=args.evidence_dir, report_dir=args.report_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

