#!/usr/bin/env python3
"""Rebuild SHA256 after separately executed empirical runtime jobs merge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "release_evidence/full_empirical_validation"


def rebuild(root: Path) -> dict[str, object]:
    if not (root / "run_manifest.json").is_file():
        raise RuntimeError("run_manifest.json must exist before checksums are frozen")
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest_sha256.json"
    }
    payload = {"algorithm": "sha256", "files": files}
    (root / "manifest_sha256.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: empirical_manifest files={len(files)}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    rebuild(arguments.root)
