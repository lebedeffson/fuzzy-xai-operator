#!/usr/bin/env python3
"""Verify final archive checksums, internal manifests and artifact identity."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts/q1_final"
ABSOLUTE_PATH = re.compile(rb"/(?:home|tmp|Users)/[^\s\"']+")
SECRET = re.compile(rb"(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s\"']+", re.IGNORECASE)


def main() -> None:
    index = json.loads((OUTPUT / "archive_index.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if index["final_commit"] != commit:
        raise RuntimeError("archive index final_commit differs from HEAD")
    for row in index["archives"]:
        path = OUTPUT / row["archive"]
        if commit[:12] not in path.name:
            raise RuntimeError(f"archive filename does not identify HEAD: {path.name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"archive checksum mismatch: {path.name}")
        sidecar = path.with_suffix(".zip.sha256").read_text(encoding="ascii").split()
        if sidecar != [row["sha256"], path.name]:
            raise RuntimeError(f"archive SHA256 sidecar mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("fuzzy-xai-operator/_archive_manifest.json"))
            identity = json.loads(archive.read("fuzzy-xai-operator/run_identity.json"))
            if manifest["final_commit"] != commit or identity["final_commit"] != commit:
                raise RuntimeError(f"embedded identity mismatch: {path.name}")
            if manifest["file_count"] != row["file_count"]:
                raise RuntimeError(f"embedded file count mismatch: {path.name}")
            required_metadata = {
                "_environment_manifest.json",
                "_known_limitations.json",
                "_hypothesis_statuses.json",
                "_data_license_manifest.json",
            }
            if not required_metadata.issubset({entry["path"] for entry in manifest["files"]}):
                raise RuntimeError(f"archive support metadata is incomplete: {path.name}")
            for entry in manifest["files"]:
                content = archive.read(f"fuzzy-xai-operator/{entry['path']}")
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"embedded checksum mismatch: {entry['path']}")
                if ABSOLUTE_PATH.search(content) or SECRET.search(content):
                    raise RuntimeError(f"private path or secret-like content in archive: {entry['path']}")
                if entry["path"].endswith(".json"):
                    _verify_json_identity(content, commit, entry["path"])
            if identity["stable_release_allowed"] and any(
                value not in {"supported", "not_supported", "inconclusive"}
                for value in identity["external_gate_status"].values()
            ):
                raise RuntimeError("stable archive contains an open external gate")
    report = {
        "schema_version": "2.0",
        "status": "PASS",
        "final_commit": commit,
        "archive_count": len(index["archives"]),
        "absolute_private_paths": 0,
        "secret_like_values": 0,
        "internal_hash_failures": 0,
    }
    (OUTPUT / "archive_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: q1_final_archive_verification archives={len(index['archives'])} commit={commit}")


def _verify_json_identity(content: bytes, commit: str, path: str) -> None:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "final_commit" in value and value["final_commit"] != commit:
                raise RuntimeError(f"stale final_commit in {path}: {value['final_commit']}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)


if __name__ == "__main__":
    main()
