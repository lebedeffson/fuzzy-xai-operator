#!/usr/bin/env python3
"""Hash the two-stage protocol without pretending that the draft is preregistered."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/selective_observer"
OUTPUT = STUDY / "protocol_manifest.json"


def main() -> None:
    inputs = sorted(path for path in STUDY.iterdir() if path.is_file() and path.name != OUTPUT.name)
    files = [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs]
    cycle = json.loads((STUDY / "research_cycle.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "1.0",
        "phase": cycle["current_phase"],
        "frozen_predecessor_commit": cycle["frozen_predecessor"]["commit"],
        "confirmatory_protocol_locked": False,
        "confirmatory_test_opened": False,
        "external_records_present": False,
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: selective_observer_protocol_manifest files={len(files)} phase={payload['phase']}")


if __name__ == "__main__":
    main()
