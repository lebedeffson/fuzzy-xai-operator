#!/usr/bin/env python3
"""Fail closed unless genuine anonymized external-study evidence is present."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/fuzzyxai"))

from fuzzyxai.q1_final.contracts import ExternalGateRecord  # noqa: E402
from scripts.q1_final.score_comprehension import score as score_comprehension  # noqa: E402
from scripts.q1_final.score_domain_review import score as score_domain_review  # noqa: E402
from scripts.q1_final.score_expert_action_review import score as score_expert_action  # noqa: E402


STUDY = ROOT / "study/q1_final"
EVIDENCE = ROOT / "release_evidence/q1_final/external"
SCORERS = {
    "domain_language_review": ROOT / "scripts/q1_final/score_domain_review.py",
    "comprehension": ROOT / "scripts/q1_final/score_comprehension.py",
    "expert_action_review": ROOT / "scripts/q1_final/score_expert_action_review.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen_protocol() -> dict[str, object]:
    manifest_path = STUDY / "stimuli_manifest/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("human_responses_generated") is not False:
        raise RuntimeError("study manifest must state that no human responses were generated")
    for row in manifest.get("files", []):
        path = ROOT / str(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen study file mismatch: {row['path']}")
    for gate_id, scorer in SCORERS.items():
        expected = manifest.get("scorer_sha256", {}).get(gate_id)
        if expected != sha256(scorer):
            raise RuntimeError(f"scorer changed after protocol freeze: {gate_id}")
    return manifest


def _verify_ethics(payload: dict[str, object]) -> str:
    status = str(payload.get("status", ""))
    allowed = {"not_started", "submitted", "approved", "exempt", "rejected"}
    if status not in allowed:
        raise RuntimeError(f"invalid ethics status: {status}")
    if status in {"not_started", "submitted", "rejected"} and payload.get("recruitment_allowed") is not False:
        raise RuntimeError(f"recruitment must be disabled while ethics status is {status}")
    if status in {"approved", "exempt"}:
        record = payload.get("public_record") or payload.get("full_document")
        if not record:
            raise RuntimeError("approved or exempt ethics status requires a public metadata record or document")
        record_path = ROOT / str(record)
        if not record_path.is_file():
            raise RuntimeError(f"ethics record does not exist: {record}")
        if payload.get("record_sha256") != sha256(record_path):
            raise RuntimeError("ethics record checksum mismatch")
    return status


def _raw_file_with_hash(paths: tuple[str, ...], digest: str | None) -> bool:
    return bool(digest) and any(sha256(ROOT / path) == digest for path in paths)


def _path_with_hash(paths: tuple[str, ...], digest: str) -> Path:
    matches = [ROOT / path for path in paths if sha256(ROOT / path) == digest]
    if len(matches) != 1:
        raise RuntimeError(f"expected one raw record with SHA256 {digest}, got {len(matches)}")
    return matches[0]


def _recompute(gate_id: str, payload: dict[str, object], raw: tuple[str, ...]) -> dict[str, object]:
    if gate_id == "comprehension":
        response = _path_with_hash(raw, str(payload["response_sha256"]))
        assignments = STUDY / "comprehension/assignments.json"
        if payload.get("assignments_sha256") != sha256(assignments):
            raise RuntimeError("comprehension assignment hash mismatch")
        return score_comprehension(response, assignments)
    if gate_id == "expert_action_review":
        response = _path_with_hash(raw, str(payload["response_sha256"]))
        return score_expert_action(response)
    reviewers = _path_with_hash(raw, str(payload["reviewers_sha256"]))
    findings = _path_with_hash(raw, str(payload["findings_sha256"]))
    dictionary = STUDY / "domain_language_review/dictionary_after.json"
    cards = STUDY / "domain_language_review/cards.json"
    if payload.get("final_dictionary_sha256") != sha256(dictionary) or payload.get("final_cards_sha256") != sha256(cards):
        raise RuntimeError("domain dictionary or cards hash mismatch")
    return score_domain_review(reviewers, findings, dictionary, cards)


def gate(gate_id: str, required: int, ethics_status: str) -> ExternalGateRecord:
    result_path = EVIDENCE / gate_id / "scoring.json"
    raw_dir = STUDY / gate_id / "raw_anonymized"
    signed_dir = STUDY / gate_id / "signed_records"
    raw = tuple(sorted(path.relative_to(ROOT).as_posix() for path in raw_dir.glob("*") if path.is_file()))
    signed = tuple(sorted(path.relative_to(ROOT).as_posix() for path in signed_dir.glob("*") if path.is_file()))
    if not result_path.is_file():
        return ExternalGateRecord(gate_id, "open", required, 0, ethics_status, raw, signed, None, False)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("records_generated_by_scorer", payload.get("participant_records_generated_by_scorer")) is not False:
        raise RuntimeError(f"scorer origin declaration is missing for {gate_id}")
    if payload.get("scorer_sha256") != sha256(SCORERS[gate_id]):
        raise RuntimeError(f"scorer checksum mismatch for {gate_id}")
    response_digest = payload.get("response_sha256") or payload.get("findings_sha256")
    if not _raw_file_with_hash(raw, str(response_digest) if response_digest else None):
        raise RuntimeError(f"scorer output is not linked to a raw anonymized record for {gate_id}")
    if _recompute(gate_id, payload, raw) != payload:
        raise RuntimeError(f"external result differs from deterministic scorer output: {gate_id}")
    observed = int(payload.get("valid_participants", payload.get("reviewer_count", 0)))
    status = str(payload.get("status", "inconclusive"))
    if status == "pass":
        status = "supported"
    return ExternalGateRecord(
        gate_id,
        status,
        required,
        observed,
        ethics_status,
        raw,
        signed,
        result_path.relative_to(ROOT).as_posix(),
        status in {"not_supported", "inconclusive"},
    )


def main(allow_open: bool) -> None:
    _verify_frozen_protocol()
    ethics = json.loads((STUDY / "ethics/status.json").read_text(encoding="utf-8"))
    ethics_status = _verify_ethics(ethics)
    records = (
        gate("domain_language_review", 3, ethics_status),
        gate("comprehension", 24, ethics_status),
        gate("expert_action_review", 3, ethics_status),
    )
    statuses = {record.gate_id: record.status for record in records}
    output = {
        "schema_version": "2.0",
        "ethics_status": ethics_status,
        "study_manifest_sha256": sha256(STUDY / "stimuli_manifest/manifest.json"),
        "gates": statuses,
        "stable_release_allowed": all(value != "open" and value != "failed" for value in statuses.values()),
        "human_records_generated_by_code": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "status.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    open_gates = [key for key, value in statuses.items() if value == "open"]
    if open_gates and not allow_open:
        raise RuntimeError(f"external gates remain open: {open_gates}")
    print(f"PASS: external_gate_integrity gates={statuses} stable={output['stable_release_allowed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-open", action="store_true")
    args = parser.parse_args()
    main(args.allow_open)
