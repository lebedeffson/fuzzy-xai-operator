from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).with_name('registry.json')


@dataclass(frozen=True)
class EcosystemModule:
    registry_id: str
    title: str
    role: str
    source_repo: str
    source_commit: str
    source_artifact: str
    fixture_path: str
    adapter: str
    evidence_level: str
    status: str
    chapter_role: str
    claim_scope: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EcosystemModule':
        return cls(**{field: str(data.get(field, '')) for field in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, str]:
        return {
            'registry_id': self.registry_id,
            'title': self.title,
            'role': self.role,
            'source_repo': self.source_repo,
            'source_commit': self.source_commit,
            'source_artifact': self.source_artifact,
            'fixture_path': self.fixture_path,
            'adapter': self.adapter,
            'evidence_level': self.evidence_level,
            'status': self.status,
            'chapter_role': self.chapter_role,
            'claim_scope': self.claim_scope,
        }


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_ecosystem_registry(path: str | Path = REGISTRY_PATH) -> list[EcosystemModule]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    modules = payload.get('modules', [])
    if not isinstance(modules, list):
        raise ValueError('ecosystem registry must contain modules list')
    return [EcosystemModule.from_dict(m) for m in modules]


def build_evidence_matrix(modules: list[EcosystemModule] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module in modules or load_ecosystem_registry():
        fixture = ROOT / module.fixture_path if module.fixture_path else None
        rows.append(
            {
                **module.as_dict(),
                'fixture_exists': bool(fixture and fixture.exists()),
                'fixture_sha256': _sha256(fixture) if fixture else '',
                'run_allowed': module.status in {'real-output-compatible', 'fixture-certified'},
                'quantitative_claim_allowed': module.status == 'real-output-compatible',
            }
        )
    return rows


def write_evidence_pack(out_dir: str | Path = 'evidence') -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules = load_ecosystem_registry()
    matrix = build_evidence_matrix(modules)

    snapshot_path = out / 'registry_snapshot.json'
    matrix_path = out / 'evidence_matrix.csv'
    index_path = out / 'reproduction_index.md'

    snapshot_path.write_text(
        json.dumps({'status': 'ok', 'modules': [m.as_dict() for m in modules], 'matrix': matrix}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    with matrix_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0].keys()) if matrix else ['registry_id'])
        writer.writeheader()
        writer.writerows(matrix)

    md = [
        '# Ecosystem Evidence Matrix',
        '',
        '| module | status | evidence_level | adapter | run_allowed | quantitative_claim_allowed |',
        '| --- | --- | --- | --- | ---: | ---: |',
    ]
    for row in matrix:
        md.append(
            f"| `{row['registry_id']}` | `{row['status']}` | `{row['evidence_level']}` | "
            f"`{row['adapter']}` | `{row['run_allowed']}` | `{row['quantitative_claim_allowed']}` |"
        )
    md += [
        '',
        'Planned modules are registered for traceability and excluded from quantitative claims.',
    ]
    index_path.write_text('\n'.join(md), encoding='utf-8')
    return {'snapshot': str(snapshot_path), 'matrix': str(matrix_path), 'index': str(index_path)}

