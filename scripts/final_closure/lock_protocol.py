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
    required = {
        "protocol": STUDY / "protocol.json",
        "dataset_manifest": STUDY / "confirmatory_dataset_manifest.json",
        "split_manifest": STUDY / "confirmatory_split_manifest.json",
        "leakage_audit": STUDY / "final_leakage_audit.json",
        "near_duplicate_audit": STUDY / "near_duplicate_audit.json",
        "feature_manifest": STUDY / "confirmatory_feature_manifest.json",
        "feature_audit": STUDY / "oof_feature_audit.json",
        "p0_p1_audit": STUDY / "p0_p1_feature_audit.json",
        "formative_summary": STUDY / "formative_real/summary.json",
        "formative_model_manifest": STUDY / "formative_real/p0_p1_model_manifest.json",
        "comparator_taxonomy": STUDY / "comparator_taxonomy.json",
        "comparator_resolution": STUDY / "comparator_resolution.json",
        "comparator_summary": STUDY / "comparator_formative/summary.json",
        "h7_summary": STUDY / "h7_formative/summary.json",
        "method_registry": STUDY / "prelock_method_registry.json",
        "ai_scope": STUDY / "ai_text_review_scope.json",
    }
    blockers = [f"missing {name}: {path.name}" for name, path in required.items() if not path.is_file()]
    if required["leakage_audit"].is_file():
        audit = load(required["leakage_audit"])
        if audit.get("status") != "pass":
            blockers.append("dataset leakage audit is not PASS")
    if required["near_duplicate_audit"].is_file() and load(required["near_duplicate_audit"]).get("status") != "pass":
        blockers.append("near-duplicate audit is not PASS")
    if required["feature_manifest"].is_file():
        features = load(required["feature_manifest"])
        if features.get("lock_status") != "ready_for_lock":
            blockers.append("real route/explanation features are not ready for lock")
        if features.get("P0_status") != "pass_predictive_oof" or features.get("P1_status") != "pass_route_oof":
            blockers.append("P0/P1 OOF feature evidence is incomplete")
    for audit_name in ("feature_audit", "p0_p1_audit"):
        if not required[audit_name].is_file():
            continue
        feature_audit = load(required[audit_name])
        if feature_audit.get("status") != "pass":
            blockers.append(f"{audit_name} is not PASS")
    if required["h7_summary"].is_file():
        h7 = load(required["h7_summary"])
        if h7.get("H7_A", {}).get("canonical_hash_preservation_rate") != 1.0:
            blockers.append("H7-A canonical preservation is not exact")
    if required["method_registry"].is_file():
        methods = load(required["method_registry"])
        if methods.get("lock_readiness") != "ready_with_confirmatory_only_experiments_pending":
            blockers.append("method registry is not ready for lock")
    if required["ai_scope"].is_file():
        scope = load(required["ai_scope"])
        accepted = scope.get("status") == "not_run_not_blocking_technical_release"
        accepted &= scope.get("technical_release_may_proceed") is True
        accepted &= len(scope.get("disabled_claims", [])) == 4
        if not accepted or scope.get("ai_review_is_external_validation") is not False:
            blockers.append("AI text-review scope does not disable human claims")
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
            "artifacts": {name: {"path": path.name, "sha256": sha256(path)} for name, path in required.items()},
            "ai_text_review_status": load(required["ai_scope"])["status"],
            "post_lock_changes_forbidden": True,
        },
    )
    print("PASS: final_confirmatory_protocol_locked test_opened=false")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
