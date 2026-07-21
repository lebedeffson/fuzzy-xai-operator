#!/usr/bin/env python3
"""Verify formative evidence and reject confirmatory wording or opened tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "study/strong_confirmatory"
EVIDENCE = ROOT / "release_evidence/strong_confirmatory/formative"
EXPECTED = (
    "H3_v2_selective",
    "H5_A_route_validity",
    "H6_A_planted_rules",
    "H7_stability",
    "H8_grid_sensitivity",
    "H9_scalability",
)


def main() -> None:
    _require_protocol()
    summary = _load(EVIDENCE / "formative_summary.json")
    if summary.get("confirmatory_claim_allowed") is not False:
        raise SystemExit("FAIL: formative evidence permits a confirmatory claim")
    if summary.get("confirmatory_test_opened") is not False:
        raise SystemExit("FAIL: formative evidence reports an opened confirmatory test")
    files = []
    for experiment in EXPECTED:
        path = EVIDENCE / f"{experiment}.json"
        payload = _load(path)
        if payload.get("confirmatory_claim_allowed") is not False:
            raise SystemExit(f"FAIL: {experiment} permits a confirmatory claim")
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)})
    summary_path = EVIDENCE / "formative_summary.json"
    files.append({"path": summary_path.relative_to(ROOT).as_posix(), "sha256": _sha(summary_path)})
    report_path = ROOT / "reports/strong_confirmatory/FORMATIVE_HANDOFF.md"
    if report_path.is_file():
        files.append({"path": report_path.relative_to(ROOT).as_posix(), "sha256": _sha(report_path)})
    manifest = {
        "schema_version": "1.0",
        "phase": "formative_development",
        "confirmatory_test_opened": False,
        "confirmatory_claim_allowed": False,
        "external_human_gates_completed": False,
        "files": files,
    }
    path = EVIDENCE / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: strong_formative_evidence files={len(files)} confirmatory_opened=false")


def _require_protocol() -> None:
    protocol = _load(PROTOCOL / "protocol_v1.json")
    frozen = _load(PROTOCOL / "frozen_claims.json")
    if protocol.get("confirmatory_test_opened") is not False:
        raise SystemExit("FAIL: protocol has opened confirmatory data")
    expected = {"H3-original", "H5-P-original", "H6-general"}
    statuses = frozen.get("immutable_original_statuses", {})
    if set(statuses) != expected or set(statuses.values()) != {"not_supported"}:
        raise SystemExit("FAIL: immutable negative results changed")


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"FAIL: missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
