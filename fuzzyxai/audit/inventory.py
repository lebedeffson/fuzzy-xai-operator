from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import AUDIT_DIR, ROOT


PATTERNS = [
    "configs/studio_scenarios/*.json",
    "reports/studio/*.json",
    "reports/studio_batch/*.json",
    "reports/studio_batch/*.csv",
    "reports/chapter5/studio_tables/*.csv",
    "visual_artifacts_latest.zip",
    "patches/chapter4_5_doctoral_strengthening.md",
    "apps/fuzzyxai_studio.py",
    "fuzzyxai/core/*.py",
]


def _json_schema_ok(path: Path, payload: dict[str, Any]) -> bool:
    if "proof_package" in path.name:
        return all(key in payload for key in ["computed_result", "operator_values", "diagnostics", "code_version"])
    if path.parts[-2:] and "studio_scenarios" in path.parts:
        return all(key in payload for key in ["scenario_id", "pipeline", "summary"])
    if path.suffix == ".json":
        return bool(payload)
    return True


def build_inventory() -> list[dict[str, Any]]:
    artifacts: list[Path] = []
    for pattern in PATTERNS:
        matches = sorted(ROOT.glob(pattern))
        artifacts.extend(matches or [ROOT / pattern])
    rows: list[dict[str, Any]] = []
    for path in sorted(set(artifacts)):
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        schema_ok = True
        code_version = ""
        notes = ""
        if exists and path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                schema_ok = _json_schema_ok(path, payload)
                code_version = str(payload.get("code_version", ""))
            except Exception as exc:
                schema_ok = False
                notes = str(exc)
        empty_ok = path.name == "__init__.py"
        rows.append(
            {
                "artifact_path": str(path.relative_to(ROOT)) if path.is_absolute() and path.exists() else str(path),
                "exists": exists,
                "size_bytes": size,
                "schema_ok": schema_ok,
                "code_version": code_version,
                "status": "ok" if exists and (size > 0 or empty_ok) and schema_ok else "issue",
                "notes": notes,
            }
        )
    return rows


def main() -> None:
    rows = build_inventory()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / "artifact_inventory.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
