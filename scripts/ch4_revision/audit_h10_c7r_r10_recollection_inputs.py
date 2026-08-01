#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "changed_files",
        "fix_commit",
        "gold",
        "gold_contract",
        "gold_file",
        "gold_patch",
        "gold_symbol",
        "patch_path",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_forbidden(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in FORBIDDEN_OBSERVABLE_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit locked R10 technical-barrier inputs"
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--image-lock", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    selection = _read_jsonl(args.selection)
    registry = _read_jsonl(args.runtime_registry)
    image_payload = json.loads(args.image_lock.read_text(encoding="utf-8"))
    images = list(image_payload["entries"])

    selection_ids = [str(row["incident_id"]) for row in selection]
    registry_by_id = {str(row["incident_id"]): row for row in registry}
    images_by_id = {str(row["incident_id"]): row for row in images}
    expected_ids = [str(value) for value in lock["incident_ids"]]
    forbidden = [
        location
        for index, row in enumerate(selection)
        for location in _find_forbidden(row, f"$[{index}]")
    ]
    hash_checks = {
        "selection": (
            _sha256(args.selection) == lock["sha256"]["selection"]
        ),
        "runtime_registry": (
            _sha256(args.runtime_registry)
            == lock["sha256"]["runtime_registry"]
        ),
        "image_lock": (
            _sha256(args.image_lock) == lock["sha256"]["image_lock"]
        ),
    }
    if args.source_evidence is not None:
        hash_checks["source_evidence"] = (
            _sha256(args.source_evidence)
            == lock["source_evidence_zip_sha256"]
        )

    checks = {
        "hashes": all(hash_checks.values()),
        "incident_order": selection_ids == expected_ids,
        "selection_registry_ids": set(selection_ids) == set(registry_by_id),
        "selection_image_ids": set(selection_ids) == set(images_by_id),
        "incident_count": len(selection_ids) == int(lock["incident_count"]),
        "unique_incidents": len(selection_ids) == len(set(selection_ids)),
        "gold_leakage_zero": not forbidden,
        "image_tags_match": all(
            registry_by_id[identifier]["container_image_tag"]
            == images_by_id[identifier]["container_image_tag"]
            for identifier in selection_ids
        ),
        "image_digests_locked": all(
            str(images_by_id[identifier].get("manifest_digest", "")).startswith(
                "sha256:"
            )
            for identifier in selection_ids
        ),
        "images_registered_available": all(
            images_by_id[identifier]["availability_status"]
            == "AVAILABLE_WITHIN_RUNNER_IMAGE_BUDGET"
            for identifier in selection_ids
        ),
    }
    passed = all(checks.values())
    payload = {
        "status": (
            "R10_RECOLLECTION_INPUT_AUDIT_PASS"
            if passed
            else "R10_RECOLLECTION_INPUT_AUDIT_FAIL"
        ),
        "checks": checks,
        "hash_checks": hash_checks,
        "incident_count": len(selection_ids),
        "repository_count": len(
            {str(row["repository"]) for row in selection}
        ),
        "forbidden_observable_paths": forbidden,
        "scientific_result": "NOT_EVALUATED",
        "development_scored": False,
        "held_out_created": False,
        "held_out_scored": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
