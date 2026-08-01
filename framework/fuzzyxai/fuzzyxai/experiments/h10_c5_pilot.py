from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

LOCK = Path(
    "results/h10_c5_pilot/H10_C5_PILOT_PROTOCOL_LOCK.json"
)
PARENT_FILES = {
    "h10_c5_incident_manifest_sha256": Path(
        "results/h10_c5/INCIDENT_MANIFEST.csv"
    ),
    "h10_c5_per_incident_sha256": Path(
        "results/h10_c5/PER_INCIDENT_RESULTS.csv"
    ),
    "h10_c5b_per_incident_sha256": Path(
        "results/h10_c5b/PER_INCIDENT_RESULTS.csv"
    ),
    "h10_c5b_final_status_sha256": Path(
        "results/h10_c5b/H10_C5B_FINAL_STATUS.json"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock(root: Path) -> dict[str, object]:
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_SELECTION":
        raise ValueError("H10-C5-PILOT protocol is not locked")
    expected = lock["parent_inputs"]
    for key, relative in PARENT_FILES.items():
        if _sha256(root / relative) != expected[key]:
            raise ValueError(f"parent result changed: {relative}")
    return lock


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _h10_c5_candidates(root: Path) -> list[dict[str, object]]:
    manifests = {
        row["incident_id"]: row
        for row in _read(
            root / "results/h10_c5/INCIDENT_MANIFEST.csv"
        )
    }
    rows = _read(root / "results/h10_c5/PER_INCIDENT_RESULTS.csv")
    values = []
    for row in rows:
        if row["method"] != "O_ROUTE":
            continue
        manifest = manifests[row["incident_id"]]
        reasons = []
        if not manifest["buggy_commit"]:
            reasons.append("BUGGY_COMMIT_MISSING")
        # A boolean metadata flag is not an executable command.
        reasons.append("LITERAL_FAIL_TO_PASS_COMMAND_MISSING")
        if int(float(row["repair_action_count"])) < 1:
            reasons.append("NONEMPTY_REPAIR_PLAN_MISSING")
        if not row["predicted_repair_operation"]:
            reasons.append("REPAIR_OPERATION_MISSING")
        reasons.extend(
            [
                "REPAIR_OPERATION_ARGUMENTS_MISSING",
                "DIGEST_PINNED_CONTAINER_MISSING",
            ]
        )
        values.append(
            {
                "source": "H10-C5",
                "incident_id": row["incident_id"],
                "repository": manifest["repository_id"],
                "buggy_commit": manifest["buggy_commit"],
                "selection_rank": hashlib.sha256(
                    (
                        manifest["repository_id"]
                        + row["incident_id"]
                        + manifest["buggy_commit"]
                    ).encode()
                ).hexdigest(),
                "eligible": not reasons,
                "rejection_reasons": sorted(set(reasons)),
            }
        )
    return values


def _h10_c5b_candidates(root: Path) -> list[dict[str, object]]:
    values = []
    for row in _read(
        root / "results/h10_c5b/PER_INCIDENT_RESULTS.csv"
    ):
        if row["method"] != "O_ROUTE":
            continue
        cut = json.loads(row["selected_cut"])
        reasons = []
        if not row["buggy_revision"]:
            reasons.append("BUGGY_COMMIT_MISSING")
        if not cut:
            reasons.append("NONEMPTY_DIAGNOSTIC_CUT_MISSING")
        reasons.extend(
            [
                "LITERAL_FAIL_TO_PASS_COMMAND_MISSING",
                "EXECUTABLE_REPAIR_PLAN_MISSING",
                "REPAIR_OPERATION_ARGUMENTS_MISSING",
                "DIGEST_PINNED_CONTAINER_MISSING",
            ]
        )
        values.append(
            {
                "source": "H10-C5b",
                "incident_id": row["incident_id"],
                "repository": row["repository"],
                "buggy_commit": row["buggy_revision"],
                "selection_rank": hashlib.sha256(
                    (
                        row["repository"]
                        + row["incident_id"]
                        + row["buggy_revision"]
                    ).encode()
                ).hexdigest(),
                "eligible": not reasons,
                "rejection_reasons": sorted(set(reasons)),
            }
        )
    return values


def run_pilot_selection(root: Path) -> dict[str, object]:
    lock = _load_lock(root)
    candidates = [
        *_h10_c5_candidates(root),
        *_h10_c5b_candidates(root),
    ]
    deduplicated = {}
    for row in candidates:
        key = (row["repository"], row["incident_id"])
        current = deduplicated.get(key)
        if current is None or (
            row["eligible"],
            row["source"],
        ) > (
            current["eligible"],
            current["source"],
        ):
            deduplicated[key] = row
    ordered = sorted(
        deduplicated.values(),
        key=lambda row: row["selection_rank"],
    )
    selected = [
        row for row in ordered if row["eligible"]
    ][: int(lock["selection_size"])]
    output = root / "results/h10_c5_pilot"
    selection = {
        "pilot_id": lock["pilot_id"],
        "selection_rule": lock["selection_rank"],
        "target_N": lock["selection_size"],
        "eligible_N": len(
            [row for row in ordered if row["eligible"]]
        ),
        "selected_N": len(selected),
        "selected": selected,
        "candidate_audit": ordered,
        "gold_accessed_during_selection": False,
    }
    (output / "H10_C5_PILOT_SELECTION.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runs = output / "H10_C5_PILOT_RUNS.jsonl"
    runs.write_text("", encoding="utf-8")
    summary = pd.DataFrame(
        [
            {
                "selected_incidents": len(selected),
                "bug_reproduced": 0,
                "plans_executed": 0,
                "fail_to_pass_passed": 0,
                "regression_passed": 0,
                "recertification_passed": 0,
                "restoration_confirmed": 0,
            }
        ]
    )
    summary.to_csv(
        output / "H10_C5_PILOT_SUMMARY.csv",
        index=False,
    )
    status = (
        "H10_C5_PILOT_BLOCKED_NO_ELIGIBLE_INCIDENTS"
        if not selected
        else "H10_C5_PILOT_INFRASTRUCTURE_BLOCKED"
    )
    final = {
        "pilot_id": lock["pilot_id"],
        "status": status,
        "selected_incidents": len(selected),
        "bug_reproduced": 0,
        "plans_executed": 0,
        "fail_to_pass_passed": 0,
        "regression_passed": 0,
        "recertification_passed": 0,
        "restoration_confirmed": 0,
        "gold_accessed_before_execution": False,
        "official_h10_c5_or_h10_c5b_rescored": False,
        "universal_practical_claim_permitted": False,
        "reason": (
            "Published parent evidence contains no incident with all "
            "preregistered executable-command, operation-argument, and "
            "digest-pinned-container fields."
        ),
        "selection_sha256": _sha256(
            output / "H10_C5_PILOT_SELECTION.json"
        ),
    }
    (output / "H10_C5_PILOT_FINAL_STATUS.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_reports(root, final)
    return final


def _write_reports(
    root: Path,
    final: dict[str, object],
) -> None:
    report_root = root / "reports/h10_c5_pilot"
    report_root.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        [
            (
                "| Incident | Repository | Reproduced | Plan | "
                "FAIL_TO_PASS | Regression | Recertification | Result |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|",
            (
                "| No eligible incident | - | - | - | - | - | - | "
                f"`{final['status']}` |"
            ),
        ]
    )
    report = "\n".join(
        [
            "# H10-C5-PILOT Executable Recovery",
            "",
            f"- Status: `{final['status']}`",
            f"- Selected incidents: `{final['selected_incidents']}`",
            "- Official H10-C5/H10-C5b metrics were not rescored.",
            "- Gold was not used for selection or execution.",
            (
                "- No published incident satisfied all prospective "
                "execution prerequisites; manual substitution was forbidden."
            ),
            (
                "- This blocked descriptive pilot provides no universal "
                "practical-utility claim."
            ),
            "",
            table,
            "",
        ]
    )
    (report_root / "H10_C5_PILOT_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    (
        report_root / "H10_C5_PILOT_TABLE_FOR_CHAPTER.md"
    ).write_text(table + "\n", encoding="utf-8")
