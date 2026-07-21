#!/usr/bin/env python3
"""Create the confirmatory lock only after all frozen prerequisites exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "study/strong_confirmatory"
EVIDENCE = ROOT / "release_evidence/strong_confirmatory/formative"
LOCK = STUDY / "confirmatory_protocol_lock.json"


def main() -> None:
    blockers = _blockers()
    if blockers:
        LOCK.unlink(missing_ok=True)
        print("BLOCKED: strong_confirmatory_protocol_lock")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(2)
    inputs = _load(STUDY / "confirmatory_inputs.json")
    review = _load(STUDY / "formative_review_gate.json")
    protocol = STUDY / "protocol_v1.json"
    lock = {
        "schema_version": "1.0",
        "status": "locked",
        "confirmatory_test_opened": False,
        "protocol_sha256": _sha(protocol),
        "confirmatory_inputs_sha256": _sha(STUDY / "confirmatory_inputs.json"),
        "formative_review_gate_sha256": _sha(STUDY / "formative_review_gate.json"),
        "dataset_count": len(inputs["datasets"]),
        "review_gate_status": review["status"],
        "post_lock_changes_forbidden": True,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: strong_confirmatory_protocol_locked datasets={lock['dataset_count']} test_opened=false")


def _blockers() -> list[str]:
    blockers: list[str] = []
    for required in (
        STUDY / "protocol_v1.json",
        STUDY / "power_analysis.json",
        EVIDENCE / "manifest.json",
        STUDY / "confirmatory_inputs.json",
        STUDY / "formative_review_gate.json",
    ):
        if not required.is_file():
            blockers.append(f"missing {required.relative_to(ROOT)}")
    if blockers:
        return blockers
    inputs = _load(STUDY / "confirmatory_inputs.json")
    datasets = inputs.get("datasets", [])
    counts: dict[str, int] = {}
    for dataset in datasets if isinstance(datasets, list) else []:
        if not isinstance(dataset, dict):
            continue
        modality = str(dataset.get("modality", ""))
        counts[modality] = counts.get(modality, 0) + 1
        source = ROOT / str(dataset.get("sealed_path", ""))
        digest = dataset.get("sha256")
        if not source.is_file() or not isinstance(digest, str) or _sha(source) != digest:
            blockers.append(f"invalid sealed dataset entry {dataset.get('dataset_id', '<unknown>')}")
    required_counts = {"tabular": 2, "image": 1, "text": 1, "timeseries": 1}
    for modality, minimum in required_counts.items():
        if counts.get(modality, 0) < minimum:
            blockers.append(f"confirmatory {modality} datasets {counts.get(modality, 0)}/{minimum}")
    review = _load(STUDY / "formative_review_gate.json")
    if review.get("status") != "pass" or review.get("critical_unsupported_claims") != 0:
        blockers.append("formative AI-review acceptance is not complete")
    if review.get("ai_review_is_external_validation") is not False:
        blockers.append("AI pre-review must not be labeled external validation")
    return blockers


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
