#!/usr/bin/env python3
"""Freeze the prior empirical package from Git objects, not the worktree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "cafe403c7d60e36b08f56a5325ba380718a5be35"
PREFIX = "release_evidence/full_empirical_validation/"
OUTPUT = ROOT / "release_evidence/q1_remediation/baseline_snapshot"
PACKAGED_SNAPSHOT = ROOT / "research/preregistration/q1_baseline_snapshot.json"


def git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary)


def build_from_git() -> dict[str, object]:
    tracked = str(git("ls-tree", "-r", "--name-only", BASE_COMMIT, PREFIX)).splitlines()
    if not tracked:
        raise RuntimeError(f"no baseline files found at {BASE_COMMIT}:{PREFIX}")
    rows: list[dict[str, object]] = []
    for path in tracked:
        content = bytes(git("show", f"{BASE_COMMIT}:{path}", binary=True))
        rows.append({"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return {
        "schema_version": "1.0",
        "base_commit": BASE_COMMIT,
        "source_prefix": PREFIX,
        "source_mode": "git_objects",
        "immutable": True,
        "file_count": len(rows),
        "files": rows,
        "preserved_negative_results": {
            "rule_ablation_general_effect": "not_confirmed",
            "critical_rupture_incremental_auprc": -0.12556024709557384,
            "critical_rupture_interpretation": "structural_diagnostic_only",
            "external_studies": "open",
        },
    }


def main() -> None:
    packaged = json.loads(PACKAGED_SNAPSHOT.read_text(encoding="utf-8"))
    try:
        manifest = build_from_git()
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest = packaged
        source_mode = "packaged_preregistered_snapshot"
    else:
        if manifest != packaged:
            raise RuntimeError("packaged Q1 baseline differs from the frozen Git objects")
        source_mode = "git_objects_verified_against_packaged_snapshot"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in manifest["files"]),
        encoding="utf-8",
    )
    print(
        f"PASS: q1_baseline_snapshot files={manifest['file_count']} "
        f"commit={BASE_COMMIT} source={source_mode}"
    )


if __name__ == "__main__":
    main()
