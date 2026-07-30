#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_REVISION = "a637bd46829f3132e12938c8a0ca93173a977b8e"
SOURCE_SHA256 = "1202acd70b011211ab552087ecc69d3c85fccccbfabeb19895a7f20c72c6ca4f"
SELECTION_FIELDS = ("repo", "instance_id", "base_commit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


def write_jsonl(
    path: Path,
    values: list[dict[str, object]],
    *,
    mode: int = 0o644,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    path.chmod(mode)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def selection_digest(row: dict[str, Any]) -> str:
    payload = "\0".join(str(row[field]) for field in SELECTION_FIELDS)
    return hashlib.sha256(payload.encode()).hexdigest()


def image_tag(instance_id: str) -> str:
    normalized = instance_id.replace("__", "_1776_").lower()
    return f"starryzhang/sweb.eval.x86_64.{normalized}:latest"


def prepare(
    source: Path,
    exclusion_lock: Path,
    output: Path,
    *,
    primary_count: int = 40,
    minimum_repositories: int = 12,
) -> dict[str, object]:
    observed_sha = sha256(source)
    if observed_sha != SOURCE_SHA256:
        raise ValueError(f"unexpected source SHA256: {observed_sha}")
    excluded = set(
        json.loads(exclusion_lock.read_text(encoding="utf-8"))[
            "excluded_repositories"
        ]
    )
    frame = pd.read_parquet(source)
    required = {
        "repo",
        "instance_id",
        "base_commit",
        "problem_statement",
        "test_patch",
        "patch",
        "test_cmds",
        "FAIL_TO_PASS",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source fields missing: {sorted(missing)}")

    candidates = []
    for source_index, value in enumerate(frame.to_dict("records")):
        repository = str(value["repo"])
        if repository in excluded:
            continue
        ranked = dict(value)
        ranked["selection_sha256"] = selection_digest(ranked)
        ranked["source_index"] = source_index
        candidates.append(ranked)
    candidates.sort(
        key=lambda row: (
            str(row["selection_sha256"]),
            str(row["instance_id"]),
        )
    )
    if len(candidates) < primary_count:
        raise ValueError("insufficient repository-disjoint source incidents")
    primary = candidates[:primary_count]
    repositories = {str(row["repo"]) for row in primary}
    if len(repositories) < minimum_repositories:
        raise ValueError("deterministic primary set has too few repositories")

    public_rows = []
    runtime_rows = []
    sealed_rows = []
    ledger_rows = []
    for rank, row in enumerate(candidates, start=1):
        identifier = str(row["instance_id"])
        role = "PRIMARY" if rank <= primary_count else "RESERVE"
        common = {
            "incident_id": identifier,
            "repository": str(row["repo"]),
            "buggy_revision": str(row["base_commit"]),
            "selection_rank": rank,
            "selection_role": role,
            "selection_sha256": str(row["selection_sha256"]),
        }
        ledger_rows.append(common)
        public_rows.append(
            {
                **common,
                "split": "held_out",
                "runtime_evidence_status": "PENDING_RUNTIME_COLLECTION",
                "query": {
                    "issue": str(row["problem_statement"]),
                    "failing_tests": as_list(row["FAIL_TO_PASS"]),
                    "traceback": "",
                    "assertion": "",
                },
            }
        )
        runtime_rows.append(
            {
                **common,
                "container_image_tag": image_tag(identifier),
                "test_commands": as_list(row["test_cmds"]),
                "fail_to_pass": as_list(row["FAIL_TO_PASS"]),
                "test_patch": str(row["test_patch"]),
                "timeout_seconds": 900,
            }
        )
        sealed_rows.append(
            {
                "incident_id": identifier,
                "repository": str(row["repo"]),
                "buggy_revision": str(row["base_commit"]),
                "patch": str(row["patch"]),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "HELD_OUT_SELECTION.jsonl", public_rows)
    write_jsonl(output / "RUNTIME_REGISTRY.jsonl", runtime_rows)
    write_jsonl(output / "CANDIDATE_LEDGER.jsonl", ledger_rows)
    sealed = output / "SEALED_GOLD_SOURCE.jsonl"
    write_jsonl(sealed, sealed_rows, mode=0o600)
    source_lock = {
        "dataset": "SWE-bench-Live/SWE-bench-Live",
        "dataset_revision": SOURCE_REVISION,
        "dataset_split": "202506",
        "source_sha256": observed_sha,
        "selection_fields": list(SELECTION_FIELDS),
        "selection_formula": "SHA256(repo + NUL + instance_id + NUL + base_commit)",
        "primary_incidents": primary_count,
        "primary_repositories": len(repositories),
        "reserve_incidents": len(candidates) - primary_count,
        "excluded_repository_count": len(excluded),
        "gold_used_for_selection": False,
        "status": "H10_C7R_HELD_OUT_SELECTION_LOCKED",
    }
    write_json(output / "H10_C7R_SOURCE_SELECTION_LOCK.json", source_lock)
    write_json(
        output / "H10_C7R_SELECTION_HASHES.json",
        {
            "candidate_ledger_sha256": sha256(
                output / "CANDIDATE_LEDGER.jsonl"
            ),
            "held_out_selection_sha256": sha256(
                output / "HELD_OUT_SELECTION.jsonl"
            ),
            "runtime_registry_sha256": sha256(
                output / "RUNTIME_REGISTRY.jsonl"
            ),
            "sealed_gold_source_sha256": sha256(sealed),
        },
    )
    os.sync()
    return source_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclusion-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source.resolve(),
                args.exclusion_lock.resolve(),
                args.output.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
