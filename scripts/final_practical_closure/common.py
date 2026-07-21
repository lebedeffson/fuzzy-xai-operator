"""Shared paths and integrity helpers for final practical closure scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/final_practical_closure"
FORMATIVE = ROOT / "release_evidence/final_practical_closure/formative"
CONFIRMATORY = ROOT / "release_evidence/final_practical_closure/confirmatory"
LOCK = STUDY / "confirmatory_protocol_lock.json"
EXPERIMENTS = (
    "H3_practical",
    "H5_A_route_validity",
    "H6_A_detectability",
    "H7_canonical_projection",
    "H8_grid",
    "H9_scaling",
)
IMMUTABLE_RESULTS = {
    "H1": "supported",
    "H2": "supported",
    "H3-original": "not_supported",
    "H4": "supported",
    "H5-P-original": "not_supported",
    "H5-S": "supported",
    "H6-general": "not_supported",
}


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"FAIL: missing {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"FAIL: expected JSON object in {path.relative_to(ROOT)}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_immutable_results() -> None:
    frozen = load_json(STUDY / "frozen_results.json")
    if frozen.get("immutable") != IMMUTABLE_RESULTS:
        raise SystemExit("FAIL: immutable predecessor result statuses changed")


def verify_sha256s(directory: Path) -> int:
    checksum_file = directory / "SHA256SUMS"
    if not checksum_file.is_file():
        raise SystemExit(f"FAIL: missing {checksum_file.relative_to(ROOT)}")
    count = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        path = directory / relative
        if not path.is_file() or sha256(path) != digest:
            raise SystemExit(f"FAIL: checksum mismatch {path.relative_to(ROOT)}")
        count += 1
    return count

