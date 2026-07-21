#!/usr/bin/env python3
"""One-way final confirmatory lock; absent external inputs remain blockers."""

from __future__ import annotations

from common import STUDY, load, sha256, write


def main() -> None:
    lock = STUDY / "confirmatory_protocol_lock.json"
    if lock.is_file():
        payload = load(lock)
        if payload.get("status") != "locked":
            raise SystemExit("FAIL: existing final lock is invalid")
        print("PASS: final_confirmatory_already_locked test_opened=false")
        return
    required = (
        STUDY / "protocol.json",
        STUDY / "confirmatory_dataset_manifest.json",
        STUDY / "confirmatory_split_manifest.json",
        STUDY / "dataset_leakage_audit.json",
        STUDY / "ai_formative_run2_acceptance.json",
    )
    blockers = [f"missing {path.name}" for path in required if not path.is_file()]
    if not blockers:
        audit, review = load(required[3]), load(required[4])
        if audit.get("status") != "pass":
            blockers.append("dataset leakage audit is not PASS")
        if review.get("status") != "pass" or review.get("ai_review_is_external_validation") is not False:
            blockers.append("AI formative run 2 is not accepted")
    if blockers:
        print("BLOCKED: final_confirmatory_protocol_lock")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(2)
    write(
        lock,
        {
            "status": "locked",
            "confirmatory_test_opened": False,
            "protocol_sha256": sha256(required[0]),
            "dataset_manifest_sha256": sha256(required[1]),
            "split_manifest_sha256": sha256(required[2]),
            "leakage_audit_sha256": sha256(required[3]),
            "ai_run2_sha256": sha256(required[4]),
            "post_lock_changes_forbidden": True,
        },
    )
    print("PASS: final_confirmatory_protocol_locked test_opened=false")


if __name__ == "__main__":
    main()
