from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _numeric_only(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _numeric_only(v) for k, v in sorted(obj.items()) if k not in {'generated_at', 'runtime_seconds'}}
    if isinstance(obj, list):
        return [_numeric_only(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def report_hash(path: Path) -> str:
    """Hash stable numeric/report fields with timestamp/runtime ignored."""
    payload = _numeric_only(json.loads(path.read_text(encoding='utf-8')))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def main() -> None:
    """Compare hashes of generated reports passed on CLI."""
    for arg in sys.argv[1:]:
        path = Path(arg)
        print(f'{path}: {report_hash(path)}')


if __name__ == '__main__':
    main()
