#!/usr/bin/env python3
"""Fail-closed operational controls for H10-C5b runtime collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_COMMIT = "7aa72a19a70bdb5eedea520742f269bc6c26aeea"
PROTOCOL_ID = "h10-c5b-repository-grounded-v1"
TRACE_COMPLETE = "BUG_REPRODUCED_WITH_TRACE"
REPLACEMENT_STATUSES = frozenset(
    {
        "BUG_NOT_REPRODUCED",
        "BUG_REPRODUCED_WITHOUT_TRACE",
        "RUNTIME_COMMAND_NOT_REGISTERED",
        "RUNTIME_EXECUTION_ERROR",
        "RUNTIME_INFRASTRUCTURE_ERROR",
        "RUNTIME_TIMEOUT",
        "UNEXPECTED_FAILURE_RETURNCODE",
    }
)
FORBIDDEN_METHOD_FIELDS = frozenset(
    {
        "patch",
        "patch_path",
        "fix_commit",
        "after_sources_path",
        "changed_files",
        "changed_symbols",
        "gold_contracts",
        "gold_repair_atoms",
    }
)
RUNTIME_FIELDS = (
    "incident_id",
    "repository",
    "repository_root",
    "buggy_revision",
    "failing_tests",
    "selection_rank_sha256",
    "split",
)
SAFE_INCIDENT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SWEBENCH_TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git_bytes(commit: str, relative_path: str, root: Path) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"locked Git object is unavailable: {relative_path}")
    return completed.stdout


def _git_object_available(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{METHOD_COMMIT}^{{commit}}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def verify_method_lock(lock_path: Path, root: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("method_commit") != METHOD_COMMIT:
        raise ValueError("unexpected H10-C5b method commit")
    if lock.get("method_code_status") != "FROZEN_FOR_RUNTIME_COLLECTION":
        raise ValueError("method is not frozen for runtime collection")
    protected = lock.get("protected_files")
    if not isinstance(protected, dict) or not protected:
        raise ValueError("method lock has no protected files")

    git_object_verified = _git_object_available(root)
    actual: dict[str, str] = {}
    for relative_path, expected in sorted(protected.items()):
        current_path = root / relative_path
        if not current_path.is_file():
            raise ValueError(f"protected file is missing: {relative_path}")
        current_sha = sha256_path(current_path)
        if current_sha != expected:
            raise ValueError(f"protected file changed: {relative_path}")
        if git_object_verified:
            committed = _git_bytes(METHOD_COMMIT, relative_path, root)
            committed_sha = hashlib.sha256(committed).hexdigest()
            if committed_sha != expected:
                raise ValueError(f"frozen Git blob changed: {relative_path}")
        actual[relative_path] = current_sha

    aggregate = canonical_sha256(actual)
    if aggregate != lock.get("method_code_sha256"):
        raise ValueError("method aggregate SHA256 mismatch")
    protocol_path = root / "protocol/h10_c5b_repository_grounded/H10_C5B_PROTOCOL_LOCK.json"
    if sha256_path(protocol_path) != lock.get("protocol_lock_sha256"):
        raise ValueError("protocol lock SHA256 mismatch")
    return {
        "status": "PASS",
        "method_commit": METHOD_COMMIT,
        "method_code_sha256": aggregate,
        "protected_file_count": len(actual),
        "scientific_implementation_diff": 0,
        "git_object_verified": git_object_verified,
    }


def build_runtime_inputs(
    manifest_path: Path,
    output: Path,
    *,
    python_command: str = "python",
    container_images_path: Path | None = None,
    source_dataset_path: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("runtime manifest is empty")
    output.mkdir(parents=True, exist_ok=True)
    runtime_rows: list[dict[str, Any]] = []
    commands: dict[str, dict[str, Any]] = {}
    container_images = (
        json.loads(container_images_path.read_text(encoding="utf-8"))
        if container_images_path is not None
        else {}
    )
    repository_counts: dict[str, int] = {}
    for row in rows:
        repository = str(row["repository"])
        repository_counts[repository] = repository_counts.get(repository, 0) + 1
    source_rows: dict[str, dict[str, Any]] = {}
    if container_images_path is not None:
        if source_dataset_path is None:
            raise ValueError("container runtime requires the registered source dataset")
        source_frame = pd.read_parquet(source_dataset_path)
        source_rows = {
            str(row["instance_id"]): row
            for row in source_frame.to_dict(orient="records")
        }
        test_patch_root = output / "runtime-test-patches"
        test_patch_root.mkdir()
    splits = {str(row.get("split")) for row in rows}
    if len(splits) != 1 or splits.pop() not in {"development", "held_out"}:
        raise ValueError("runtime input must contain exactly one registered split")

    for row in rows:
        incident_id = str(row["incident_id"])
        if not SAFE_INCIDENT_ID.fullmatch(incident_id):
            raise ValueError(f"unsafe incident ID: {incident_id}")
        failing_tests = tuple(str(item) for item in row.get("failing_tests", ()))
        if not failing_tests:
            raise ValueError(f"no failing test registered for {incident_id}")
        runtime_row = {field: row[field] for field in RUNTIME_FIELDS if field in row}
        if set(runtime_row) & FORBIDDEN_METHOD_FIELDS:
            raise ValueError("Gold field entered runtime collection channel")
        runtime_rows.append(runtime_row)
        commands[incident_id] = {
            "command": [
                (
                    SWEBENCH_TESTBED_PYTHON
                    if container_images_path is not None
                    else python_command
                ),
                "-m",
                "pytest",
                *failing_tests,
                "-x",
                "-vv",
            ],
            "expected_failure_returncodes": [1],
            "execution_backend": (
                "container" if container_images_path is not None else "host"
            ),
            "timeout_seconds": 900,
        }
        if container_images_path is not None:
            repository = str(row["repository"])
            image = container_images.get(incident_id)
            if image is None:
                image = container_images.get(repository)
                if image is not None and repository_counts[repository] > 1:
                    raise ValueError(
                        "repository-level container image is ambiguous for "
                        f"{repository}; register each incident ID"
                    )
            if image is None:
                raise ValueError(
                    f"no container image registered for incident {incident_id}"
                )
            source_row = source_rows.get(incident_id)
            if source_row is None:
                raise ValueError(f"incident missing from registered source: {incident_id}")
            test_patch = str(source_row.get("test_patch", ""))
            if not test_patch.strip():
                raise ValueError(f"runtime test patch missing for {incident_id}")
            test_patch_path = test_patch_root / f"{incident_id}.patch"
            test_patch_path.write_text(test_patch, encoding="utf-8")
            commands[incident_id]["container_image"] = str(image)
            commands[incident_id]["runtime_test_patch_path"] = str(
                test_patch_path.resolve()
            )
            commands[incident_id]["runtime_test_patch_sha256"] = sha256_path(
                test_patch_path
            )

    runtime_manifest = output / "H10_C5B_RUNTIME_COLLECTION_MANIFEST.jsonl"
    command_path = output / "H10_C5B_RUNTIME_COMMANDS.json"
    write_jsonl(runtime_manifest, runtime_rows)
    command_path.write_text(
        json.dumps(commands, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "incident_count": len(runtime_rows),
        "source_manifest_sha256": sha256_path(manifest_path),
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256_path(runtime_manifest),
        "command_registry": str(command_path),
        "command_registry_sha256": sha256_path(command_path),
        "gold_fields_in_runtime_channel": [],
        "execution_backend": (
            "container" if container_images_path is not None else "host"
        ),
        "container_image_registry_sha256": (
            sha256_path(container_images_path)
            if container_images_path is not None
            else None
        ),
        "source_dataset_sha256": (
            sha256_path(source_dataset_path) if source_dataset_path else None
        ),
        "runtime_test_patch_count": (
            len(runtime_rows) if container_images_path is not None else 0
        ),
    }
    (output / "RUNTIME_INPUT_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def merge_runtime_evidence(
    original_manifest: Path,
    runtime_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    original_rows = read_jsonl(original_manifest)
    runtime_rows = read_jsonl(runtime_manifest)
    runtime_by_id = {str(row["incident_id"]): row for row in runtime_rows}
    original_ids = {str(row["incident_id"]) for row in original_rows}
    if original_ids != set(runtime_by_id):
        raise ValueError("runtime and original manifest incident sets differ")
    copied_fields = (
        "stdout_path",
        "stderr_path",
        "traceback_path",
        "runtime_evidence_status",
    )
    merged = [
        {
            **row,
            **{
                field: runtime_by_id[str(row["incident_id"])][field]
                for field in copied_fields
                if field in runtime_by_id[str(row["incident_id"])]
            },
        }
        for row in original_rows
    ]
    write_jsonl(output, merged)
    return {
        "status": "PASS",
        "incident_count": len(merged),
        "output": str(output),
        "sha256": sha256_path(output),
    }


def _selection_rank(incident_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_ID}\0{incident_id}".encode()).hexdigest()


def plan_replacements(
    candidate_parquet: Path,
    selected_manifest: Path,
    runtime_report: Path,
    output: Path,
) -> dict[str, Any]:
    selected = read_jsonl(selected_manifest)
    selected_by_id = {str(row["incident_id"]): row for row in selected}
    report = json.loads(runtime_report.read_text(encoding="utf-8"))
    evidence = {str(row["incident_id"]): row for row in report.get("evidence", ())}
    if set(evidence) != set(selected_by_id):
        raise ValueError("runtime report and selected manifest incident sets differ")

    frame = pd.read_parquet(candidate_parquet, columns=["repo", "instance_id"])
    candidate_rows = [
        {"repository": str(row["repo"]), "incident_id": str(row["instance_id"])}
        for row in frame.to_dict(orient="records")
    ]
    used = set(selected_by_id)
    ledger: list[dict[str, Any]] = []
    for incident_id, selected_row in sorted(selected_by_id.items()):
        status = str(evidence[incident_id]["status"])
        if status == TRACE_COMPLETE:
            continue
        if status not in REPLACEMENT_STATUSES:
            raise ValueError(f"status is not eligible for replacement: {status}")
        repository = str(selected_row["repository"])
        candidates = sorted(
            (
                row
                for row in candidate_rows
                if row["repository"] == repository and row["incident_id"] not in used
            ),
            key=lambda row: (_selection_rank(row["incident_id"]), row["incident_id"]),
        )
        replacement = candidates[0] if candidates else None
        if replacement is not None:
            used.add(replacement["incident_id"])
        ledger.append(
            {
                "candidate_incident": incident_id,
                "repository": repository,
                "selection_rank": str(selected_row["selection_rank_sha256"]),
                "runtime_command_sha256": evidence[incident_id].get(
                    "runtime_command_sha256"
                ),
                "return_code": evidence[incident_id].get("returncode"),
                "reproduction_status": status,
                "replacement_reason": status,
                "replacement_incident": (
                    replacement["incident_id"] if replacement is not None else None
                ),
                "replacement_selection_rank": (
                    _selection_rank(replacement["incident_id"])
                    if replacement is not None
                    else None
                ),
                "gold_or_prediction_viewed": False,
            }
        )
    result = {
        "status": "PASS" if all(row["replacement_incident"] for row in ledger) else "BLOCKED",
        "protocol_id": PROTOCOL_ID,
        "replacement_rule": "next_same_repository_candidate_by_registered_sha256_rank",
        "replacement_count": len(ledger),
        "ledger": ledger,
    }
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def verify_runtime_readiness(
    manifest_path: Path,
    report_path: Path,
    split: str,
    *,
    development_lock: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = list(report.get("evidence", ()))
    incident_ids = {str(row["incident_id"]) for row in rows}
    evidence_ids = {str(row["incident_id"]) for row in evidence}
    if incident_ids != evidence_ids:
        raise ValueError("runtime report does not cover the manifest exactly")
    if {str(row.get("split")) for row in rows} != {split}:
        raise ValueError("manifest split does not match requested split")
    complete = (
        report.get("status") == "PASS"
        and report.get("incident_count") == len(rows)
        and report.get("trace_complete_count") == len(rows)
        and all(row.get("status") == TRACE_COMPLETE for row in evidence)
    )
    if not complete:
        raise ValueError("runtime evidence is incomplete")

    repository_count = len({str(row["repository"]) for row in rows})
    if split == "held_out":
        if development_lock is None or not development_lock.is_file():
            raise ValueError("held-out collection requires a development runtime lock")
        lock = json.loads(development_lock.read_text(encoding="utf-8"))
        if (
            lock.get("status") != "DEVELOPMENT_RUNTIME_FROZEN"
            or lock.get("method_commit") != METHOD_COMMIT
        ):
            raise ValueError("development runtime lock is invalid")
        if len(rows) < 24 or repository_count < 8:
            raise ValueError("held-out minimum design is not met")

    return {
        "status": "PASS",
        "split": split,
        "incident_count": len(rows),
        "repository_count": repository_count,
        "trace_complete_count": len(rows),
        "runtime_evidence_sha256": sha256_path(report_path),
    }


def freeze_development_runtime(
    manifest_path: Path,
    report_path: Path,
    development_results: Path,
    method_lock_path: Path,
    output: Path,
    root: Path,
) -> dict[str, Any]:
    readiness = verify_runtime_readiness(
        manifest_path,
        report_path,
        "development",
    )
    method = verify_method_lock(method_lock_path, root)
    results = json.loads(development_results.read_text(encoding="utf-8"))
    if results.get("gold_leakage_audit") != "PASS":
        raise ValueError("development Gold leakage audit did not pass")
    if int(results.get("development_incidents", 0)) != readiness["incident_count"]:
        raise ValueError("development result and runtime incident counts differ")
    lock = {
        "status": "DEVELOPMENT_RUNTIME_FROZEN",
        "method_commit": METHOD_COMMIT,
        "method_code_sha256": method["method_code_sha256"],
        "development_manifest_sha256": sha256_path(manifest_path),
        "runtime_evidence_report_sha256": sha256_path(report_path),
        "development_results_sha256": sha256_path(development_results),
        "incident_count": readiness["incident_count"],
        "repository_count": readiness["repository_count"],
        "trace_complete_count": readiness["trace_complete_count"],
        "gold_leakage_audit": "PASS",
        "held_out_scored": False,
    }
    output.write_text(
        json.dumps(lock, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-method-lock")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--root", type=Path, default=Path("."))

    prepare = subparsers.add_parser("prepare-runtime-inputs")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--python-command", default="python")
    prepare.add_argument("--container-images", type=Path)
    prepare.add_argument("--source-dataset", type=Path)

    merge = subparsers.add_parser("merge-runtime")
    merge.add_argument("--original-manifest", type=Path, required=True)
    merge.add_argument("--runtime-manifest", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    replace = subparsers.add_parser("plan-replacements")
    replace.add_argument("--candidates", type=Path, required=True)
    replace.add_argument("--manifest", type=Path, required=True)
    replace.add_argument("--runtime-report", type=Path, required=True)
    replace.add_argument("--output", type=Path, required=True)

    readiness = subparsers.add_parser("verify-runtime-readiness")
    readiness.add_argument("--manifest", type=Path, required=True)
    readiness.add_argument("--runtime-report", type=Path, required=True)
    readiness.add_argument("--split", choices=("development", "held_out"), required=True)
    readiness.add_argument("--development-lock", type=Path)

    freeze = subparsers.add_parser("freeze-development")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--runtime-report", type=Path, required=True)
    freeze.add_argument("--development-results", type=Path, required=True)
    freeze.add_argument("--method-lock", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--root", type=Path, default=Path("."))

    args = parser.parse_args()
    if args.command == "verify-method-lock":
        result = verify_method_lock(args.lock.resolve(), args.root.resolve())
    elif args.command == "prepare-runtime-inputs":
        result = build_runtime_inputs(
            args.manifest.resolve(),
            args.output.resolve(),
            python_command=args.python_command,
            container_images_path=(
                args.container_images.resolve() if args.container_images else None
            ),
            source_dataset_path=(
                args.source_dataset.resolve() if args.source_dataset else None
            ),
        )
    elif args.command == "merge-runtime":
        result = merge_runtime_evidence(
            args.original_manifest.resolve(),
            args.runtime_manifest.resolve(),
            args.output.resolve(),
        )
    elif args.command == "plan-replacements":
        result = plan_replacements(
            args.candidates.resolve(),
            args.manifest.resolve(),
            args.runtime_report.resolve(),
            args.output.resolve(),
        )
    elif args.command == "verify-runtime-readiness":
        result = verify_runtime_readiness(
            args.manifest.resolve(),
            args.runtime_report.resolve(),
            args.split,
            development_lock=(
                args.development_lock.resolve() if args.development_lock else None
            ),
        )
    else:
        result = freeze_development_runtime(
            args.manifest.resolve(),
            args.runtime_report.resolve(),
            args.development_results.resolve(),
            args.method_lock.resolve(),
            args.output.resolve(),
            args.root.resolve(),
        )
    _json_print(result)


if __name__ == "__main__":
    main()
