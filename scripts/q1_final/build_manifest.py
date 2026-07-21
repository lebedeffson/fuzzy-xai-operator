#!/usr/bin/env python3
"""Build the final SHA256 manifest with one commit identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"
OUTPUT = EVIDENCE / "manifest_sha256.json"
ROOTS = (EVIDENCE, ROOT / "reports/q1_final", ROOT / "dissertation_artifacts/q1_final", ROOT / "research/q1_final", ROOT / "study/q1_final")


def main() -> None:
    identity = json.loads((EVIDENCE / "run_identity.json").read_text(encoding="utf-8"))
    rows = []
    for directory in ROOTS:
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file() and item != OUTPUT):
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    payload = {
        "schema_version": "2.0",
        "algorithm": "sha256",
        "final_commit": identity["final_commit"],
        "file_count": len(rows),
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: q1_final_manifest files={len(rows)} commit={identity['final_commit']}")


if __name__ == "__main__":
    main()
