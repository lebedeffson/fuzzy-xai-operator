from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_data import (
    COLLECTION_LOCK_PATH,
    PROTOCOL_LOCK_PATH,
    canonical_repository,
)
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

DATA_COLLECTION_AMENDMENT_PATH = Path(
    "protocol/h10_c5c_evidence_retrieval/H10_C5C_DATA_COLLECTION_AMENDMENT_001.json"
)

MIN_INCIDENTS = 30
MIN_REPOSITORIES = 8
REQUIRED_RUNTIME_STATUS = "BUG_REPRODUCED_WITH_TRACE"
REQUIRED_EVENT_KINDS = frozenset({"coverage", "traceback_frame"})
FORBIDDEN_EVENT_KEYS = frozenset(
    {
        "patch",
        "patch_path",
        "fixed_commit",
        "fixed_revision",
        "gold",
        "gold_patch",
        "changed_files",
        "before_sources",
        "before_sources_path",
        "after_sources",
        "after_sources_path",
    }
)


@dataclass(frozen=True)
class DevelopmentReadinessResult:
    report_path: Path
    status: str
    incident_count: int
    repository_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(base: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (base / path).resolve()


def _event_key_leakage(path: Path) -> list[str]:
    leaked: set[str] = set()
    for row in _load_jsonl(path):
        for key in row:
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_EVENT_KEYS:
                leaked.add(normalized)
    return sorted(leaked)


def _load_locks(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    protocol = json.loads((root / PROTOCOL_LOCK_PATH).read_text(encoding="utf-8"))
    collection = json.loads((root / COLLECTION_LOCK_PATH).read_text(encoding="utf-8"))
    amendment = json.loads(
        (root / DATA_COLLECTION_AMENDMENT_PATH).read_text(encoding="utf-8")
    )
    if protocol.get("status") != "LOCKED_BEFORE_IMPLEMENTATION":
        raise ValueError("H10-C5c protocol lock is invalid")
    if collection.get("status") != "LOCKED_BEFORE_DEVELOPMENT_COLLECTION":
        raise ValueError("H10-C5c collection lock is invalid")
    if amendment.get("status") != "LOCKED_BEFORE_DEVELOPMENT_COLLECTION":
        raise ValueError("H10-C5c collection amendment is invalid")
    if amendment.get("applies_to") != collection.get("collection_id"):
        raise ValueError("H10-C5c collection amendment target is invalid")
    return protocol, collection, amendment


def verify_h10_c5c_development_readiness(
    manifest_path: Path,
    command_registry_path: Path,
    source_registry_path: Path,
    runtime_report_path: Path,
    output_path: Path,
    root: Path,
) -> DevelopmentReadinessResult:
    protocol, collection, amendment = _load_locks(root)
    rows = _load_jsonl(manifest_path)
    commands = json.loads(command_registry_path.read_text(encoding="utf-8"))
    sources = json.loads(source_registry_path.read_text(encoding="utf-8"))
    runtime_report = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    runtime_evidence = runtime_report.get("evidence", [])
    runtime_by_incident = (
        {
            str(item.get("incident_id", "")): item
            for item in runtime_evidence
            if isinstance(item, dict)
        }
        if isinstance(runtime_evidence, list)
        else {}
    )
    if not isinstance(commands, dict):
        raise TypeError("H10-C5c command registry must be an object")
    if not isinstance(sources, dict):
        raise TypeError("H10-C5c source registry must be an object")
    source_rows = sources.get("incidents", [])
    if not isinstance(source_rows, list):
        raise TypeError("H10-C5c source registry incidents must be a list")

    identifiers = [str(row.get("incident_id", "")) for row in rows]
    repositories = [canonical_repository(str(row.get("repository", ""))) for row in rows]
    repository_counts = Counter(repositories)
    source_by_incident = {
        str(row.get("incident_id", "")): row
        for row in source_rows
        if isinstance(row, dict)
    }
    excluded = {
        canonical_repository(str(value))
        for value in protocol["registered_h10_c5b_held_out_repositories"]
    }
    base = manifest_path.parent.resolve()
    command_base = command_registry_path.parent.resolve()
    per_incident: list[dict[str, object]] = []
    all_runtime_complete = True
    all_event_streams_valid = True
    all_test_ids_covered = True
    all_gold_boundaries_clean = True
    all_source_records_present = True
    all_command_records_present = True
    all_python_runtimes_exact = True

    for row, repository in zip(rows, repositories, strict=True):
        incident_id = str(row.get("incident_id", ""))
        failing_tests = {str(item) for item in row.get("failing_tests", [])}
        runtime_path = _resolve(base, row.get("runtime_events_path", ""))
        runtime_status = str(row.get("runtime_evidence_status", ""))
        runtime_complete = runtime_status == REQUIRED_RUNTIME_STATUS
        all_runtime_complete &= runtime_complete
        event_kinds: set[str] = set()
        event_test_ids: set[str] = set()
        event_error = ""
        leaked_keys: list[str] = []
        if runtime_path.is_file():
            try:
                events = load_runtime_events(runtime_path)
                event_kinds = {event.kind for event in events}
                event_test_ids = {event.test_id for event in events}
                leaked_keys = _event_key_leakage(runtime_path)
            except (ValueError, json.JSONDecodeError) as error:
                event_error = str(error)
        else:
            event_error = f"missing runtime event stream: {runtime_path}"
        event_stream_valid = not event_error and REQUIRED_EVENT_KINDS <= event_kinds
        tests_covered = bool(failing_tests) and failing_tests <= event_test_ids
        gold_clean = not leaked_keys
        source_present = incident_id in source_by_incident
        command = commands.get(incident_id)
        command_present = isinstance(command, dict)
        registered_tests = (
            {
                str(item.get("test_id", ""))
                for item in command.get("commands", [])
                if isinstance(item, dict)
            }
            if command_present
            else set()
        )
        command_tests_match = command_present and failing_tests == registered_tests
        runtime_record = runtime_by_incident.get(incident_id, {})
        python_runtime_exact = bool(runtime_record.get("python_runtime_exact"))
        source_hashes_valid = False
        exposing_tests_valid = False
        if source_present:
            source = source_by_incident[incident_id]
            patch_path = _resolve(base, row.get("patch_path", ""))
            registered_patch_hash = str(source.get("patch_sha256", ""))
            source_hashes_valid = (
                patch_path.is_file()
                and bool(registered_patch_hash)
                and _sha256(patch_path) == registered_patch_hash
            )
            for path_field, hash_field in (
                ("setup_script", "setup_script_materialized_sha256"),
                ("requirements_path", "requirements_materialized_sha256"),
            ):
                expected_hash = str(source.get(hash_field, ""))
                registered_path = (
                    str(command.get(path_field, "")).strip()
                    if command_present
                    else ""
                )
                if expected_hash or registered_path:
                    materialized_path = _resolve(command_base, registered_path)
                    source_hashes_valid &= (
                        materialized_path.is_file()
                        and bool(expected_hash)
                        and _sha256(materialized_path) == expected_hash
                    )
            repository_root = _resolve(base, row.get("repository_root", ""))
            overlays = source.get("exposing_test_overlays", [])
            if isinstance(overlays, list) and overlays:
                overlay_checks = []
                for overlay in overlays:
                    if not isinstance(overlay, dict):
                        overlay_checks.append(False)
                        continue
                    relative = Path(str(overlay.get("path", "")))
                    materialized = repository_root / relative
                    expected = str(overlay.get("materialized_test_sha256", ""))
                    overlay_checks.append(
                        materialized.is_file()
                        and bool(expected)
                        and _sha256(materialized) == expected
                    )
                exposing_tests_valid = all(overlay_checks)
        all_event_streams_valid &= event_stream_valid
        all_test_ids_covered &= tests_covered and command_tests_match
        all_gold_boundaries_clean &= gold_clean
        all_source_records_present &= (
            source_present and source_hashes_valid and exposing_tests_valid
        )
        all_command_records_present &= command_present and command_tests_match
        all_python_runtimes_exact &= python_runtime_exact
        per_incident.append(
            {
                "incident_id": incident_id,
                "repository": repository,
                "runtime_status": runtime_status,
                "runtime_complete": runtime_complete,
                "runtime_events_path": str(runtime_path),
                "event_kinds": sorted(event_kinds),
                "failing_tests": sorted(failing_tests),
                "event_test_ids": sorted(event_test_ids),
                "registered_test_ids": sorted(registered_tests),
                "event_stream_valid": event_stream_valid,
                "test_ids_covered": tests_covered,
                "command_tests_match": command_tests_match,
                "python_runtime_exact": python_runtime_exact,
                "gold_event_key_leakage": leaked_keys,
                "source_record_present": source_present,
                "source_hashes_valid": source_hashes_valid,
                "exposing_tests_valid": exposing_tests_valid,
                "error": event_error,
            }
        )

    selection = collection["selection"]
    if not isinstance(selection, dict):
        raise TypeError("H10-C5c collection selection lock is invalid")
    benchmark = collection["benchmark"]
    if not isinstance(benchmark, dict):
        raise TypeError("H10-C5c benchmark lock is invalid")
    locked_bugsinpy_commit = str(benchmark.get("commit", ""))
    checks = {
        "collection_id_matches": sources.get("collection_id") == collection.get("collection_id"),
        "unique_incident_ids": bool(identifiers) and len(identifiers) == len(set(identifiers)),
        "development_split_only": all(str(row.get("split", "")) == "development" for row in rows),
        "minimum_incidents": len(rows) >= max(MIN_INCIDENTS, int(selection["target_incidents"])),
        "minimum_repositories": len(set(repositories)) >= max(MIN_REPOSITORIES, int(selection["minimum_repositories"])),
        "maximum_incidents_per_repository": max(repository_counts.values(), default=0) <= int(selection["maximum_incidents_per_repository"]),
        "h10_c5b_held_out_repositories_excluded": not (set(repositories) & excluded),
        "all_runtime_complete": all_runtime_complete,
        "all_event_streams_valid": all_event_streams_valid,
        "all_failing_test_ids_covered": all_test_ids_covered,
        "all_command_records_present": all_command_records_present,
        "all_source_records_patch_and_exposing_tests_valid": all_source_records_present,
        "runtime_report_complete": runtime_report.get("status") == "DEVELOPMENT_RUNTIME_COMPLETE",
        "runtime_report_manifest_hash_matches": (
            runtime_report.get("enriched_manifest_sha256") == _sha256(manifest_path)
        ),
        "runtime_report_command_registry_hash_matches": (
            runtime_report.get("command_registry_sha256")
            == _sha256(command_registry_path)
        ),
        "runtime_report_incident_count_matches": (
            runtime_report.get("total_incidents") == len(rows)
            and runtime_report.get("complete_incidents") == len(rows)
        ),
        "runtime_report_incident_ids_match": (
            set(runtime_by_incident) == set(identifiers)
        ),
        "command_registry_incident_ids_match": set(commands) == set(identifiers),
        "source_registry_incident_ids_match": set(source_by_incident) == set(identifiers),
        "all_python_runtimes_exact": (
            all_python_runtimes_exact
            and runtime_report.get("all_python_runtimes_exact") is True
        ),
        "runtime_report_scientific_result_not_evaluated": runtime_report.get("scientific_result") == "NOT_EVALUATED",
        "gold_event_key_leakage_zero": all_gold_boundaries_clean,
        "bugsinpy_commit_matches_lock": (
            bool(locked_bugsinpy_commit)
            and sources.get("bugsinpy_commit") == locked_bugsinpy_commit
        ),
    }
    status = (
        "H10_C5C_DEVELOPMENT_READINESS_PASS"
        if all(checks.values())
        else "H10_C5C_DEVELOPMENT_READINESS_FAIL"
    )
    report = {
        "collection_id": collection["collection_id"],
        "data_collection_amendment_id": amendment["amendment_id"],
        "protocol_id": "h10-c5c-evidence-retrieval-v1",
        "status": status,
        "scientific_result": "NOT_EVALUATED",
        "development_scored": False,
        "held_out_created": False,
        "held_out_scored": False,
        "incident_count": len(rows),
        "repository_count": len(set(repositories)),
        "repository_counts": dict(sorted(repository_counts.items())),
        "checks": checks,
        "input_hashes": {
            "manifest_sha256": _sha256(manifest_path),
            "command_registry_sha256": _sha256(command_registry_path),
            "source_registry_sha256": _sha256(source_registry_path),
            "runtime_report_sha256": _sha256(runtime_report_path),
        },
        "per_incident": per_incident,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DevelopmentReadinessResult(
        output_path,
        status,
        len(rows),
        len(set(repositories)),
    )
