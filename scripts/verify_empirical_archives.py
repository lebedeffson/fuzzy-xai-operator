#!/usr/bin/env python3
"""Verify empirical release archives and their detached SHA256 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_artifacts/empirical_archive_verification.json"
ARCHIVE_REQUIREMENTS = {
    "fuzzyxai-full-empirical-evidence-*.zip": (
        "fuzzy-xai-operator/archive_manifest.json",
        "fuzzy-xai-operator/release_evidence/full_empirical_validation/run_manifest.json",
        "fuzzy-xai-operator/release_evidence/full_empirical_validation/dod_90.json",
    ),
    "fuzzyxai-dissertation-artifacts-*.zip": (
        "fuzzy-xai-operator/archive_manifest.json",
        "fuzzy-xai-operator/dissertation_artifacts/chapter4/table_4_multimodal_models.csv",
        "fuzzy-xai-operator/reports/empirical_validation/full_empirical_validation.md",
    ),
    "fuzzyxai-reproducibility-bundle-*.zip": (
        "fuzzy-xai-operator/archive_manifest.json",
        "fuzzy-xai-operator/Dockerfile",
        "fuzzy-xai-operator/scripts/reproduce_all.py",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for pattern, required in ARCHIVE_REQUIREMENTS.items():
        matches = sorted((ROOT / "release_artifacts").glob(pattern), key=lambda path: path.stat().st_mtime)
        if not matches:
            raise RuntimeError(f"archive is missing: {pattern}")
        archive = matches[-1]
        checksum = archive.with_suffix(".zip.sha256")
        if not checksum.is_file():
            raise RuntimeError(f"checksum is missing: {checksum}")
        declared = checksum.read_text(encoding="ascii").split()[0]
        measured = sha256(archive)
        if declared != measured:
            raise RuntimeError(f"checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as handle:
            corrupt = handle.testzip()
            if corrupt is not None:
                raise RuntimeError(f"corrupt member in {archive}: {corrupt}")
            names = set(handle.namelist())
        missing = sorted(set(required) - names)
        if missing:
            raise RuntimeError(f"required archive members are missing from {archive}: {missing}")
        rows.append(
            {
                "archive": archive.name,
                "sha256": measured,
                "file_count": len(names),
                "status": "PASS",
            }
        )
    payload = {"schema_version": "1.0", "status": "PASS", "archives": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: empirical_archives count={len(rows)}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    verify(args.output)
