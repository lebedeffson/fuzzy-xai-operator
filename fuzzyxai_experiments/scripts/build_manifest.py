from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ['data', 'registry', 'reports/chapter4', 'reports/chapter5', 'reports/gui_screenshots', 'src']
INCLUDE_FILES = [
    'README.md', 'EVIDENCE_HANDOFF.md', 'Dockerfile', 'requirements.txt', 'run_chapter4_5.sh',
    'start_gui.sh', 'export_gui_screenshots.sh', 'smoke_gui.sh', 'gui_app.py',
    'compare_reports.py', 'tables/generated_tables.tex', 'tables/scenario_summary.md',
    'tables/hybrid_xiris_baseline_comparison.md', 'tables/beacon_xai_summary.md',
    'tables/gis_integro_metrics.md', 'tables/gd_anfis_shap_metrics.md',
]
EXCLUDE_NAMES = {'manifest_sha256.json', 'checksums.sha256'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_files() -> list[Path]:
    files: set[Path] = set()
    for rel in INCLUDE_DIRS:
        base = ROOT / rel
        if base.exists():
            for path in base.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
                    files.add(path)
    for rel in INCLUDE_FILES:
        path = ROOT / rel
        if path.exists() and path.name not in EXCLUDE_NAMES:
            files.add(path)
    return sorted(files)


def main() -> None:
    rows = []
    for path in iter_files():
        rel = path.relative_to(ROOT).as_posix()
        rows.append({'path': rel, 'sha256': sha(path), 'bytes': path.stat().st_size})
    (ROOT / 'manifest_sha256.json').write_text(json.dumps({'files': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'checksums.sha256').write_text('\n'.join(f"{row['sha256']}  {row['path']}" for row in rows) + '\n', encoding='utf-8')
    print(f'manifest files={len(rows)}')

if __name__ == '__main__':
    main()
