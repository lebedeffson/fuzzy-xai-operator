from __future__ import annotations

import subprocess
from pathlib import Path

from ..adjudication.adjudication_gate import adjudication_status
from ..audit import audit_baselines, audit_oracle, run_leakage_audit
from ..hashing import file_sha256, read_json, tree_sha256, write_json
from ..paths import ARTIFACT_ROOT, PACKAGE_ROOT, PROTOCOL_DIR, REPO_ROOT


BASE_COMMIT = "a8f150b1ef3b5c6041c28098a5cc90d0e8e20ae5"


def initialize_design_approval() -> Path:
    path = ARTIFACT_ROOT / "lock" / "design_approval_template.json"
    write_json(
        path,
        {
            "approved": False,
            "approved_by": "",
            "signature": "",
            "recommended_design_sha256": "",
            "note": "Must be completed by the protocol owner after reviewing power_report.md.",
        },
    )
    return path


def freeze_protocol(approval_path: Path) -> dict:
    approval = read_json(approval_path)
    design = ARTIFACT_ROOT / "power" / "recommended_design.json"
    if approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("signature"):
        raise PermissionError("BLOCKED_PROTOCOL: protocol-owner design approval is required")
    if approval.get("recommended_design_sha256") != file_sha256(design):
        raise PermissionError("BLOCKED_PROTOCOL: design approval hash mismatch")
    protocol_files = sorted(PROTOCOL_DIR.glob("*")) + sorted((PACKAGE_ROOT / "configs").glob("*.yaml"))
    method_files = sorted((PACKAGE_ROOT / "src" / "h10_c2" / "methods").glob("*.py"))
    method_files += sorted((PACKAGE_ROOT / "src" / "h10_c2" / "baselines").glob("*.py"))
    value = {
        "experiment_id": "FXAI-H10-C2-PRECONFIRMATORY",
        "base_commit": BASE_COMMIT,
        "protocol_sha256": tree_sha256(protocol_files, PACKAGE_ROOT),
        "method_tree_sha256": tree_sha256(method_files, PACKAGE_ROOT),
        "recommended_design_sha256": file_sha256(design),
        "design_approval_sha256": file_sha256(approval_path),
        "opening_count": 0,
    }
    write_json(ARTIFACT_ROOT / "lock" / "protocol.lock.json", value)
    return value


def _git_base_unchanged() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", BASE_COMMIT, "--", "framework/fuzzyxai"],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def preconfirmatory_gate() -> dict:
    blockers = []
    if not _git_base_unchanged():
        blockers.append("BLOCKED_CODE:v21_core_changed")
    power_path = ARTIFACT_ROOT / "power" / "recommended_design.json"
    if not power_path.exists():
        blockers.append("BLOCKED_POWER:missing_power_design")
    elif read_json(power_path).get("status") != "power_target_reached":
        blockers.append("BLOCKED_POWER:candidate_grid_does_not_reach_target")
    required = [
        ARTIFACT_ROOT / "power" / "recommended_design.json",
        ARTIFACT_ROOT / "data" / "development" / "manifest.json",
        ARTIFACT_ROOT / "data" / "protocol_validation" / "manifest.json",
        ARTIFACT_ROOT / "sealed" / "opening_record.json",
    ]
    if any(not path.exists() for path in required):
        blockers.append("BLOCKED_CODE:missing_generated_artifacts")
    try:
        audit_baselines()
        audit_oracle()
        run_leakage_audit()
    except RuntimeError as exc:
        blockers.append(str(exc))
    lock = ARTIFACT_ROOT / "lock" / "protocol.lock.json"
    if not lock.exists():
        blockers.append("BLOCKED_PROTOCOL:design_not_approved_or_protocol_not_locked")
    if adjudication_status() != "PASS":
        blockers.append("BLOCKED_HUMAN_ADJUDICATION")
    if not blockers:
        sealed_required = [
            ARTIFACT_ROOT / "data" / "sealed" / "manifest.json",
            ARTIFACT_ROOT / "sealed" / "sealed_manifest.json",
        ]
        if any(not path.exists() for path in sealed_required):
            blockers.append("BLOCKED_GOLD:sealed_inputs_not_created")
    opening = ARTIFACT_ROOT / "sealed" / "opening_record.json"
    opening_count = read_json(opening)["opening_count"] if opening.exists() else None
    if opening_count != 0:
        blockers.append("BLOCKED_LEAKAGE:sealed_opening_count_not_zero")
    status = "READY_FOR_SEALED_SCORING" if not blockers else blockers[0].split(":", 1)[0]
    report = {
        "status": status,
        "blockers": blockers,
        "v21_integrity": _git_base_unchanged(),
        "manual_adjudication": adjudication_status(),
        "sealed_opening_count": opening_count,
        "h10_c2a": "NOT_EVALUATED",
        "h10_c2b": "NOT_EVALUATED",
    }
    write_json(ARTIFACT_ROOT / "audit" / "preconfirmatory_gate.json", report)
    return report
