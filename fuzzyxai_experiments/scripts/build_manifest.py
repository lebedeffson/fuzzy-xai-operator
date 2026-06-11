from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'manifest_sha256.json', 'checksums.sha256'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in EXCLUDE or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        files.append({'path': rel, 'sha256': sha(path), 'bytes': path.stat().st_size})
    (ROOT / 'manifest_sha256.json').write_text(json.dumps({'files': files}, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'checksums.sha256').write_text('\n'.join(f"{row['sha256']}  {row['path']}" for row in files) + '\n', encoding='utf-8')
    print(f'manifest files={len(files)}')

if __name__ == '__main__':
    main()
