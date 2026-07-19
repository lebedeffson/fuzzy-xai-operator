"""Verify the aggregate Chapter 4 candidate and its manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_evidence/chapter4_final_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((OUTPUT / "manifest_sha256.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = OUTPUT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"FAIL: manifest mismatch {relative}")
    status = json.loads((OUTPUT / "release_gate_status.json").read_text(encoding="utf-8"))
    computed = status["computed_gates"]
    if computed["prediction_parity_rate"] != 1.0 or computed["conformance_rate"] != 1.0:
        raise SystemExit("FAIL: universal model contract rates")
    if computed["model_universality_verified_configurations"] < 27:
        raise SystemExit("FAIL: insufficient model benchmark coverage")
    print("PASS: chapter4_candidate_manifest")
    print("PASS: chapter4_computed_gates")
    print(f"{status['release_gate']}: chapter4_release_gate")


if __name__ == "__main__":
    main()
