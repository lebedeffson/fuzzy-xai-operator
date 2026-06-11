from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
REPORTS_DIR = PACKAGE_ROOT / 'reports'
LOGGER = logging.getLogger('fuzzyxai_experiments')


def resolve_path(path: str | Path) -> Path:
    """Resolve a path inside the evidence package, falling back to repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    package_path = PACKAGE_ROOT / p
    if package_path.exists():
        return package_path
    repo_path = REPO_ROOT / p
    if repo_path.exists():
        return repo_path
    return package_path


def utc_stamp() -> str:
    """Return an ISO-like UTC stamp safe for filenames."""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from package path, repo path, or absolute path."""
    return json.loads(resolve_path(path).read_text(encoding='utf-8'))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows from package path, repo path, or absolute path."""
    with resolve_path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_yaml_or_json(path: str | Path) -> dict[str, Any]:
    """Read a YAML/JSON mapping."""
    p = resolve_path(path)
    text = p.read_text(encoding='utf-8')
    return json.loads(text) if p.suffix.lower() == '.json' else yaml.safe_load(text)


def sha256_text(text: str) -> str:
    """Return SHA256 for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return SHA256 for a file or an empty string when absent."""
    p = resolve_path(path)
    if not p.exists():
        return ''
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON report with stable UTF-8 formatting."""
    p = Path(path)
    if not p.is_absolute():
        p = PACKAGE_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> Path:
    """Write CSV rows with fixed columns."""
    p = Path(path)
    if not p.is_absolute():
        p = PACKAGE_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
    return p


def _plan_file(explain_plan_path: str) -> Path:
    p = resolve_path(explain_plan_path)
    if p.exists():
        return p
    return resolve_path('explain_plans/bc_plan.yaml')


def timed_report(name: str, build: Callable[[], dict[str, Any]], *, explain_plan_path: str = 'configs/explain_plan_chapter2.yaml') -> dict[str, Any]:
    """Run a report builder and save deterministic plus timestamped JSON outputs."""
    started = time.perf_counter()
    payload = build()
    elapsed = time.perf_counter() - started
    plan_file = _plan_file(explain_plan_path)
    report = {
        'experiment': name,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'runtime_seconds': round(elapsed, 6),
        'explain_plan': {
            'path': str(plan_file.relative_to(PACKAGE_ROOT)) if str(plan_file).startswith(str(PACKAGE_ROOT)) else str(plan_file),
            'sha256': sha256_file(plan_file),
        },
        **payload,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / f'{name}.json', report)
    write_json(REPORTS_DIR / f'{name}_{utc_stamp()}.json', report)
    return report
