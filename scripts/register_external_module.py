from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'registry_id', 'title', 'role', 'source_repo', 'source_artifact', 'fixture_path',
    'adapter', 'evidence_level', 'status', 'chapter_role', 'claim_scope'
]


def _load_entry(module_id: str, entry_path: str | None) -> dict[str, Any]:
    if entry_path:
        return json.loads(Path(entry_path).read_text(encoding='utf-8'))
    template = json.loads((ROOT / 'templates/module_registry_entry.json').read_text(encoding='utf-8'))
    template['registry_id'] = module_id
    template['title'] = module_id.replace('_', ' ').title()
    template['fixture_path'] = f'data/fixtures/{module_id}_output.json'
    return template


def validate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED:
        if not entry.get(key):
            errors.append(f'missing required field: {key}')
    rid = str(entry.get('registry_id', ''))
    if rid and not re.fullmatch(r'[a-z0-9_\-]+', rid):
        errors.append('registry_id must use lowercase letters, digits, underscore or dash')
    source = str(entry.get('source_repo', ''))
    if source and not (source.startswith('https://github.com/') or source.startswith('source-not-provided:')):
        warnings.append('source_repo is not a GitHub URL or source-not-provided marker')
    fixture = ROOT / str(entry.get('fixture_path', ''))
    if entry.get('status') in {'fixture-certified', 'real-output-compatible'} and not fixture.exists():
        warnings.append(f'fixture not found locally: {entry.get("fixture_path")}')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings}


def write_stub_report(entry: dict[str, Any], validation: dict[str, Any], out_dir: str | Path = 'reports/registry_registration') -> Path:
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{entry.get('registry_id', 'module')}_registration_report.json"
    payload = {
        'entry': entry,
        'validation': validation,
        'adapter_stub': 'templates/adapter_stub.py',
        'notes': 'Registration report validates metadata contract only; it does not certify domain quality.',
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-id', required=True)
    parser.add_argument('--entry-json')
    parser.add_argument('--out-dir', default='reports/registry_registration')
    args = parser.parse_args()
    entry = _load_entry(args.module_id, args.entry_json)
    validation = validate_entry(entry)
    report = write_stub_report(entry, validation, args.out_dir)
    print(json.dumps({'status': 'ok' if validation['ok'] else 'error', 'report': str(report), **validation}, ensure_ascii=False, indent=2))
    if not validation['ok']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
