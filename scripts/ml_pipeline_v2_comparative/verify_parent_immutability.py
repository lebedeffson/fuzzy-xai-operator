#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "protocol/ml_pipeline_v2_comparative/PARENT_FILES_SHA256"


def main() -> int:
    failures: list[dict[str, str]] = []
    checked = 0
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        checked += 1
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checked_files": checked,
        "failures": failures,
    }
    output = ROOT / "results/ml_pipeline_v2_comparative/PARENT_IMMUTABILITY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
