#!/usr/bin/env python3
"""One-way final confirmatory lock; absent external inputs remain blockers."""

from __future__ import annotations

import subprocess

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
        STUDY / "final_leakage_audit.json",
        STUDY / "confirmatory_feature_manifest.json",
        STUDY / "oof_feature_audit.json",
        STUDY / "ai_formative_run2_acceptance.json",
    )
    blockers = [f"missing {path.name}" for path in required if not path.is_file()]
    if required[3].is_file():
        audit = load(required[3])
        if audit.get("status") != "pass":
            blockers.append("dataset leakage audit is not PASS")
    if required[4].is_file():
        features = load(required[4])
        if features.get("lock_status") != "ready_for_lock":
            blockers.append("real route/explanation features are not ready for lock")
        if features.get("P0_status") != "pass_predictive_oof" or features.get("P1_status") != "pass_route_oof":
            blockers.append("P0/P1 OOF feature evidence is incomplete")
    if required[5].is_file():
        feature_audit = load(required[5])
        if feature_audit.get("status") != "pass":
            blockers.append("OOF feature audit is not PASS")
    if required[6].is_file():
        review = load(required[6])
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
            "source_commit": _git_head(),
            "confirmatory_test_opened": False,
            "protocol_sha256": sha256(required[0]),
            "dataset_manifest_sha256": sha256(required[1]),
            "split_manifest_sha256": sha256(required[2]),
            "leakage_audit_sha256": sha256(required[3]),
            "feature_manifest_sha256": sha256(required[4]),
            "feature_audit_sha256": sha256(required[5]),
            "ai_run2_sha256": sha256(required[6]),
            "post_lock_changes_forbidden": True,
        },
    )
    print("PASS: final_confirmatory_protocol_locked test_opened=false")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
