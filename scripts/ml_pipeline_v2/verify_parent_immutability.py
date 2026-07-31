#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocol/ml_pipeline_v2/PARENT_FILES_SHA256"
OUTPUT = ROOT / "results/ml_pipeline_v2/PARENT_IMMUTABILITY.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    checked = 0
    changed: list[str] = []
    missing: list[str] = []
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = ROOT / relative
        checked += 1
        if not path.is_file():
            missing.append(relative)
        elif file_sha256(path) != expected:
            changed.append(relative)
    payload = {
        "status": "PASS" if not changed and not missing else "FAIL",
        "checked_files": checked,
        "changed_files": changed,
        "missing_files": missing,
        "parent_manifest_sha256": file_sha256(LOCK),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
