#!/usr/bin/env python3
"""Score immutable pre-score artifacts after the declared vault-envelope adapter failure."""

from __future__ import annotations

import subprocess

from common import ROOT, STUDY, load, sha256, write
from run_sealed_confirmatory import _open_all_vaults, _score_preserved_actions


ORIGINAL_LOCK = STUDY / "confirmatory_protocol_lock.json"
OPENING = STUDY / "confirmatory_opening_record.json"
INVALID = STUDY / "confirmatory_invalid_marker.json"
PRESCORE = STUDY / "confirmatory/prescore_manifest.json"
RECOVERY_LOCK = STUDY / "confirmatory_scoring_recovery_lock.json"
RECOVERY_INVALID = STUDY / "confirmatory_scoring_recovery_invalid_marker.json"
COMPLETION = STUDY / "confirmatory_completion_marker.json"
ORIGINAL_SOURCE_COMMIT = "3a7ad13ae33f2d84f0384c34c0a417c0f1a34def"
ALLOWED_RECOVERY_FILES = {
    "Makefile",
    "scripts/final_closure/build_chapter4.py",
    "scripts/final_closure/build_final_statistics.py",
    "scripts/final_closure/build_one_zip.py",
    "scripts/final_closure/run_scoring_recovery.py",
    "scripts/final_closure/run_sealed_confirmatory.py",
    "tests/final_closure/test_final_closure.py",
}


def lock_recovery() -> None:
    if RECOVERY_LOCK.is_file():
        raise SystemExit("BLOCKED: scoring recovery is already locked")
    _require_original_failure()
    _validate_prescore()
    changed = set(_git("diff", "--name-only", f"{ORIGINAL_SOURCE_COMMIT}..HEAD").splitlines())
    unexpected = sorted(changed - ALLOWED_RECOVERY_FILES)
    if unexpected:
        raise SystemExit(f"BLOCKED: scoring-recovery commit changes non-adapter files: {unexpected}")
    write(
        RECOVERY_LOCK,
        {
            "status": "locked_scoring_only_recovery",
            "source_commit": _git("rev-parse", "HEAD"),
            "original_source_commit": ORIGINAL_SOURCE_COMMIT,
            "original_protocol_lock_sha256": sha256(ORIGINAL_LOCK),
            "original_opening_record_sha256": sha256(OPENING),
            "original_invalid_marker_sha256": sha256(INVALID),
            "prescore_manifest_sha256": sha256(PRESCORE),
            "failure_class": "vault_envelope_adapter_schema_error",
            "outcomes_observed_before_recovery_lock": False,
            "models_features_actions_thresholds_changed": False,
            "recovery_scope": "unwrap_the_preexisting_labels_envelope_and_score_immutable_actions",
            "original_confirmatory_retry": False,
            "protocol_deviation_must_be_reported": True,
        },
    )
    print("PASS: confirmatory_scoring_recovery_locked prescore_immutable=true original_retry=false")


def run_recovery() -> None:
    if not RECOVERY_LOCK.is_file() or load(RECOVERY_LOCK).get("status") != "locked_scoring_only_recovery":
        raise SystemExit("BLOCKED: scoring recovery is not locked")
    if load(RECOVERY_LOCK).get("source_commit") != _git("rev-parse", "HEAD"):
        raise SystemExit("BLOCKED: HEAD differs from scoring-recovery lock")
    if COMPLETION.is_file() or RECOVERY_INVALID.is_file():
        raise SystemExit("BLOCKED: scoring recovery has already completed or failed")
    _require_original_failure()
    _validate_prescore()
    try:
        labels = _open_all_vaults()
        result = _score_preserved_actions(labels, load(PRESCORE))
        result["protocol_deviation"] = {
            "type": "declared_scoring_only_recovery_after_vault_envelope_adapter_error",
            "prescore_recomputed": False,
            "models_features_actions_thresholds_changed": False,
            "original_invalid_marker_preserved": True,
            "original_confirmatory_retry": False,
        }
        write(STUDY / "confirmatory/h3_h7_summary.json", result)
        write(
            COMPLETION,
            {
                "status": "completed_via_declared_scoring_recovery",
                "original_protocol_lock_sha256": sha256(ORIGINAL_LOCK),
                "recovery_lock_sha256": sha256(RECOVERY_LOCK),
                "result_sha256": sha256(STUDY / "confirmatory/h3_h7_summary.json"),
                "prescore_recomputed": False,
                "post_open_tuning": False,
                "protocol_deviation_reported": True,
            },
        )
    except BaseException as error:
        write(
            RECOVERY_INVALID,
            {
                "status": "scoring_recovery_failed",
                "error_type": type(error).__name__,
                "retry_forbidden": True,
            },
        )
        raise
    print(f"PASS: confirmatory_scoring_recovery objects={result['objects']} prescore_recomputed=false")


def _require_original_failure() -> None:
    required = (ORIGINAL_LOCK, OPENING, INVALID, PRESCORE)
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"BLOCKED: original scoring failure evidence missing: {missing}")
    if load(INVALID).get("status") != "invalid_after_label_opening":
        raise SystemExit("BLOCKED: original invalid marker has unexpected status")
    if load(OPENING).get("prescore_manifest_sha256") != sha256(PRESCORE):
        raise SystemExit("BLOCKED: immutable pre-score manifest differs from opening record")


def _validate_prescore() -> None:
    manifest = load(PRESCORE)
    artifacts = [
        *manifest["feature_artifacts"],
        *manifest["canonical_evidence_artifacts"],
        *manifest["predictive_model_artifacts"],
        manifest["controller_models"],
        manifest["policy_actions"],
        *manifest["label_free_experiments"].values(),
    ]
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise SystemExit(f"BLOCKED: immutable pre-score artifact changed: {artifact['path']}")
    forbidden = (
        STUDY / "confirmatory/scored_policy_results.parquet",
        STUDY / "confirmatory/policy_summary.parquet",
        STUDY / "confirmatory/H6_B.json",
    )
    if any(path.exists() for path in forbidden):
        raise SystemExit("BLOCKED: scoring output exists before recovery lock")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock", "run"))
    arguments = parser.parse_args()
    lock_recovery() if arguments.command == "lock" else run_recovery()
