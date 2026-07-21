#!/usr/bin/env python3
"""Verify the technical prelock package without implying confirmation."""

from __future__ import annotations

from fuzzyxai.final_closure import compositional_faults, fault_library

from common import EVIDENCE, ROOT, STUDY, load, sha256, write


def main() -> None:
    required = {
        "protocol": STUDY / "protocol.json",
        "iteration_log": STUDY / "formative_iteration_log.json",
        "registry": STUDY / "confirmatory_dataset_registry.json",
        "dataset_manifest": STUDY / "confirmatory_dataset_manifest.json",
        "split_manifest": STUDY / "confirmatory_split_manifest.json",
        "leakage_audit": STUDY / "final_leakage_audit.json",
        "near_duplicate_audit": STUDY / "near_duplicate_audit.json",
        "feature_manifest": STUDY / "confirmatory_feature_manifest.json",
        "feature_audit": STUDY / "oof_feature_audit.json",
        "p0_p1_audit": STUDY / "p0_p1_feature_audit.json",
        "formative_summary": STUDY / "formative_real/summary.json",
        "comparator_taxonomy": STUDY / "comparator_taxonomy.json",
        "comparator_summary": STUDY / "comparator_formative/summary.json",
        "h7_summary": STUDY / "h7_formative/summary.json",
        "method_registry": STUDY / "prelock_method_registry.json",
        "ai_protocol": STUDY / "ai_formative_run2/protocol.json",
        "ai_cases": STUDY / "ai_formative_run2/reviewer_cases.jsonl",
        "ai_bundle": STUDY / "ai_formative_run2/fuzzyxai-ai-formative-run2-input.zip",
        "ai_scope": STUDY / "ai_text_review_scope.json",
        "claims": EVIDENCE / "claim_status_prelock.json",
        "faults": EVIDENCE / "fault_library.json",
        "replay_summary": EVIDENCE / "shadow_replay_summary.json",
        "replay_events": EVIDENCE / "shadow_replay_events.jsonl",
    }
    missing = [path.relative_to(ROOT).as_posix() for path in required.values() if not path.is_file()]
    blockers: list[str] = []
    if missing:
        blockers.extend(f"MISSING:{item}" for item in missing)
    else:
        protocol = load(required["protocol"])
        registry = load(required["registry"])
        dataset_audit = load(required["leakage_audit"])
        near_duplicates = load(required["near_duplicate_audit"])
        feature_manifest = load(required["feature_manifest"])
        feature_audit = load(required["feature_audit"])
        p0_p1_audit = load(required["p0_p1_audit"])
        formative = load(required["formative_summary"])
        comparator = load(required["comparator_summary"])
        h7 = load(required["h7_summary"])
        methods = load(required["method_registry"])
        ai_protocol = load(required["ai_protocol"])
        ai_scope = load(required["ai_scope"])
        claims = load(required["claims"])
        faults = load(required["faults"])
        replay = load(required["replay_summary"])
        if protocol.get("confirmatory_test_opened") is not False:
            blockers.append("CONFIRMATORY_TEST_NOT_CLOSED")
        if registry.get("status") not in {
            "blocked_pending_download_and_sealing",
            "blocked_formative_overlap",
            "prepared_pending_sealing",
            "sealed_pass",
        }:
            blockers.append("DATASET_REGISTRY_PRELOCK_STATUS")
        if (
            dataset_audit.get("status") != "pass"
            or dataset_audit.get("tuning_runner_can_read_test_labels") is not False
        ):
            blockers.append("DATASET_LEAKAGE_AUDIT")
        if near_duplicates.get("status") != "pass" or near_duplicates.get("near_duplicate_violations") != 0:
            blockers.append("NEAR_DUPLICATE_AUDIT")
        if feature_audit.get("status") != "pass" or feature_audit.get("oof_test_overlap") != 0:
            blockers.append("OOF_FEATURE_AUDIT")
        if p0_p1_audit.get("status") != "pass":
            blockers.append("P0_P1_FEATURE_AUDIT")
        if feature_manifest.get("sealed_test_loaded") is not False:
            blockers.append("SEALED_TEST_WAS_LOADED")
        if feature_manifest.get("lock_status") != "ready_for_lock":
            blockers.append("FEATURE_PRELOCK_STATUS")
        if feature_manifest.get("P0_status") != "pass_predictive_oof" or feature_manifest.get("P1_status") != "pass_route_oof":
            blockers.append("P0_P1_STATUS")
        if formative.get("phase") != "formative_real_oof" or formative.get("sealed_test_opened") is not False:
            blockers.append("FORMATIVE_REAL_BOUNDARY")
        if comparator.get("phase") != "formative_train_development_only" or comparator.get("sealed_test_opened") is not False:
            blockers.append("COMPARATOR_BOUNDARY")
        if h7.get("H7_A", {}).get("canonical_hash_preservation_rate") != 1.0:
            blockers.append("H7_A_PRESERVATION")
        if methods.get("lock_readiness") != "ready_with_confirmatory_only_experiments_pending":
            blockers.append("METHOD_REGISTRY")
        if ai_protocol.get("status") != "input_ready_scores_not_run":
            blockers.append("AI_SCORES_SHOULD_NOT_EXIST")
        if (
            ai_scope.get("status") != "not_run_not_blocking_technical_release"
            or ai_scope.get("technical_release_may_proceed") is not True
            or ai_scope.get("ai_review_is_external_validation") is not False
        ):
            blockers.append("AI_SCOPE_BOUNDARY")
        if set(claims.get("frozen_claims", {}).values()) != {"supported", "not_supported"}:
            blockers.append("FROZEN_CLAIM_STATUS")
        if set(claims.get("new_claims", {}).values()) != {"blocked_pending_sealed_confirmation"}:
            blockers.append("NEW_CLAIM_STATUS")
        if len(fault_library()) < 40 or len(compositional_faults()) < 10:
            blockers.append("FAULT_LIBRARY_INCOMPLETE")
        if len(faults.get("controlled_fault_templates", [])) < 40:
            blockers.append("FAULT_ARTIFACT_INCOMPLETE")
        if replay.get("event_count") != 100_000 or replay.get("confirmatory_claim_allowed") is not False:
            blockers.append("SHADOW_REPLAY_BOUNDARY")
        if sum(1 for _ in required["ai_cases"].open(encoding="utf-8")) != 720:
            blockers.append("AI_VARIANT_COUNT")
        if sum(1 for _ in required["replay_events"].open(encoding="utf-8")) != 100_000:
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
            for path in required.values()
            if path.is_file()
        ],
    }
    write(EVIDENCE / "prelock_manifest.json", manifest)
    if blockers:
        raise SystemExit(f"FAIL: final prelock verification: {blockers}")
    print(f"PASS: final_prelock artifacts={len(required)} confirmatory=false stable=false")


if __name__ == "__main__":
    main()
