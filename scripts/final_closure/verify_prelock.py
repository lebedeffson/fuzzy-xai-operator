#!/usr/bin/env python3
"""Verify the technical prelock package without implying confirmation."""

from __future__ import annotations

from fuzzyxai.final_closure import compositional_faults, fault_library

from common import EVIDENCE, ROOT, STUDY, load, sha256, write


def main() -> None:
    required = (
        STUDY / "protocol.json",
        STUDY / "formative_iteration_log.json",
        STUDY / "confirmatory_dataset_registry.json",
        STUDY / "ai_formative_run2/protocol.json",
        STUDY / "ai_formative_run2/reviewer_cases.jsonl",
        STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip",
        EVIDENCE / "fault_library.json",
        EVIDENCE / "shadow_replay_summary.json",
        EVIDENCE / "shadow_replay_events.jsonl",
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    blockers: list[str] = []
    if missing:
        blockers.extend(f"MISSING:{item}" for item in missing)
    else:
        protocol = load(required[0])
        registry = load(required[2])
        ai_protocol = load(required[3])
        faults = load(required[6])
        replay = load(required[7])
        if protocol.get("confirmatory_test_opened") is not False:
            blockers.append("CONFIRMATORY_TEST_NOT_CLOSED")
        if registry.get("status") not in {"blocked_pending_download_and_sealing", "blocked_formative_overlap"}:
            blockers.append("DATASET_REGISTRY_PRELOCK_STATUS")
        if ai_protocol.get("status") != "input_ready_scores_not_run":
            blockers.append("AI_SCORES_SHOULD_NOT_EXIST")
        if len(fault_library()) < 40 or len(compositional_faults()) < 10:
            blockers.append("FAULT_LIBRARY_INCOMPLETE")
        if len(faults.get("controlled_fault_templates", [])) < 40:
            blockers.append("FAULT_ARTIFACT_INCOMPLETE")
        if replay.get("event_count") != 100_000 or replay.get("confirmatory_claim_allowed") is not False:
            blockers.append("SHADOW_REPLAY_BOUNDARY")
        if sum(1 for _ in required[4].open(encoding="utf-8")) != 720:
            blockers.append("AI_VARIANT_COUNT")
        if sum(1 for _ in required[8].open(encoding="utf-8")) != 100_000:
            blockers.append("SHADOW_EVENT_COUNT")
    status = "pass" if not blockers else "fail"
    manifest = {
        "status": status,
        "scope": "prelock_technical_formative",
        "confirmatory_claim_allowed": False,
        "stable_release_allowed": False,
        "blockers": blockers,
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for path in required
            if path.is_file()
        ],
    }
    write(EVIDENCE / "prelock_manifest.json", manifest)
    if blockers:
        raise SystemExit(f"FAIL: final prelock verification: {blockers}")
    print(f"PASS: final_prelock artifacts={len(required)} confirmatory=false stable=false")


if __name__ == "__main__":
    main()
