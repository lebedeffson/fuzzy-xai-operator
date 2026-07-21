#!/usr/bin/env python3
"""Verify Q1 archive checksums, embedded manifests and commit anchors."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts"


def main() -> None:
    report_path = OUTPUT / "q1_archive_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if report["commit"] != commit:
        raise RuntimeError("archive verification commit differs from HEAD")
    for row in report["archives"]:
        path = OUTPUT / row["archive"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"archive checksum mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("fuzzy-xai-operator/_archive_manifest.json"))
            if manifest["commit"] != commit or manifest["file_count"] != row["file_count"]:
                raise RuntimeError(f"archive manifest mismatch: {path}")
            for entry in manifest["files"]:
                content = archive.read(f"fuzzy-xai-operator/{entry['path']}")
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"embedded checksum mismatch: {entry['path']}")
    print(f"PASS: q1_archive_verification archives={len(report['archives'])}")


if __name__ == "__main__":
    main()
