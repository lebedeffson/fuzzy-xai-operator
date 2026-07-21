#!/usr/bin/env python3
"""Build SHA256 manifest for Q1 evidence and generated dissertation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/q1_remediation/manifest_sha256.json"
ROOTS = (
    ROOT / "release_evidence/q1_remediation",
    ROOT / "reports/q1",
    ROOT / "dissertation_artifacts/q1",
    ROOT / "research/preregistration",
    ROOT / "study/comprehension",
    ROOT / "study/domain_review",
    ROOT / "study/expert_action_review",
)


def main() -> None:
    rows = []
    for directory in ROOTS:
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file() and item != OUTPUT):
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    payload = {"schema_version": "1.0", "algorithm": "sha256", "file_count": len(rows), "files": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: q1_manifest files={len(rows)}")


if __name__ == "__main__":
    main()
