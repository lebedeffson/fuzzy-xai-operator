#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import _graph, _reject_gold
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

METHOD_COMMIT = "358ed40a0fb7f5adc1291695ff15affa39cae485"
SOURCE_RELEASE_SHA256 = (
    "cd33dd6689623c043dfb74485d1618d10b2922f6a9934047db5ce43293f56a4b"
)
METHOD_FILES = (
    "framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_retrieval.py",
    "framework/fuzzyxai/fuzzyxai/repository_diagnostics/guided_diagnosis.py",
    "framework/fuzzyxai/fuzzyxai/repository_diagnostics/runtime_events.py",
    "framework/fuzzyxai/fuzzyxai/experiments/h10_c7a.py",
)
LOCK_NAMES = (
    "H10_C7R_PROTOCOL_LOCK.json",
    "H10_C7R_METHOD_LOCK.json",
    "H10_C7R_BUDGET_LOCK.json",
    "H10_C7R_BASELINE_LOCK.json",
    "H10_C7R_EXCLUSION_LOCK.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _git_blob(root: Path, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{METHOD_COMMIT}:{path}"],
        cwd=root,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _reference_signatures(path: Path) -> dict[str, tuple[str, ...]]:
    values = {}
    for row in _read_jsonl(path):
        values[str(row["incident_id"])] = tuple(
            json.loads(str(row["top_20_signature"]))
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--r5-reference", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.development_manifest.resolve()
    protocol_dir = args.protocol_dir.resolve()
    output = args.output.resolve()
    source_release = args.source_release.resolve()
    if _sha256(source_release) != SOURCE_RELEASE_SHA256:
        raise ValueError("H10-C7R source release SHA256 mismatch")

    file_hashes = {}
    for relative in METHOD_FILES:
        expected = _sha256_bytes(_git_blob(root, relative))
        actual = _sha256(root / relative)
        if actual != expected:
            raise ValueError(f"frozen method file changed: {relative}")
        file_hashes[relative] = expected

    records = _read_jsonl(manifest)
    if len(records) != 40:
        raise ValueError("H10-C7R lock requires exactly 40 development incidents")
    repositories = {str(row["repository"]) for row in records}
    if len(repositories) < 10:
        raise ValueError("H10-C7R lock requires at least 10 repositories")
    references = _reference_signatures(args.r5_reference.resolve())
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    signatures = []
    mismatches = []
    for index, row in enumerate(records):
        _reject_gold(row, f"$[{index}]")
        query_value = row["query"]
        if not isinstance(query_value, dict):
            raise TypeError("query must be a mapping")
        graph = _graph(
            json.loads(
                _resolve(manifest.parent, row["graph_path"]).read_text(
                    encoding="utf-8"
                )
            )
        )
        runtime_events = load_runtime_events(
            _resolve(manifest.parent, row["runtime_events_path"])
        )
        diagnosis = engine.diagnose(
            graph,
            IncidentQuery(
                str(row["incident_id"]),
                str(query_value.get("issue", "")),
                tuple(
                    str(item)
                    for item in query_value.get("failing_tests", ())
                ),
                str(query_value.get("traceback", "")),
                str(query_value.get("assertion", "")),
            ),
            "R5",
            runtime_events,
        )
        top_20 = tuple(item.node_id for item in diagnosis.candidates[:20])
        identifier = str(row["incident_id"])
        reference = references.get(identifier)
        if reference is not None and top_20 != reference:
            mismatches.append(identifier)
        signatures.append(
            {
                "incident_id": identifier,
                "repository": str(row["repository"]),
                "top_20": list(top_20),
                "top_20_sha256": _sha256_bytes(
                    json.dumps(
                        top_20,
                        separators=(",", ":"),
                    ).encode()
                ),
            }
        )
    if mismatches:
        raise ValueError(
            f"frozen R5 top-20 mismatch: {sorted(mismatches)}"
        )
    signatures.sort(key=lambda row: str(row["incident_id"]))
    signatures_path = output / "DEVELOPMENT_TOP20_SIGNATURES.jsonl"
    _write_jsonl(signatures_path, signatures)
    method_lock = {
        "development_incidents": len(records),
        "development_repositories": len(repositories),
        "development_top20_signatures_sha256": _sha256(signatures_path),
        "frozen_file_sha256": file_hashes,
        "method": "R5",
        "method_commit": METHOD_COMMIT,
        "r5_budget": 20,
        "reference_prefix_incidents": len(references),
        "reference_prefix_mismatches": 0,
        "retrieval_changed": False,
        "status": "LOCKED_BEFORE_HELD_OUT",
    }
    method_lock_path = protocol_dir / "H10_C7R_METHOD_LOCK.json"
    _write_json(method_lock_path, method_lock)
    lock_hashes = {
        name: _sha256(protocol_dir / name)
        for name in LOCK_NAMES
    }
    lock_manifest = {
        "held_out_manifest_created": False,
        "lock_sha256": lock_hashes,
        "method_commit": METHOD_COMMIT,
        "protocol_id": "H10-C7R-v1",
        "source_release_sha256": SOURCE_RELEASE_SHA256,
        "status": "H10_C7R_LOCK_PACKAGE_COMPLETE",
    }
    _write_json(
        protocol_dir / "H10_C7R_LOCK_MANIFEST.json",
        lock_manifest,
    )
    print(
        json.dumps(
            {
                "status": lock_manifest["status"],
                "development_incidents": len(records),
                "development_repositories": len(repositories),
                "signature_sha256": method_lock[
                    "development_top20_signatures_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
