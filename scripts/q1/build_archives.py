#!/usr/bin/env python3
"""Build deterministic Q1 archives exclusively from committed HEAD blobs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts"


ARCHIVES: dict[str, tuple[str, ...]] = {
    "fuzzyxai-q1-validation-evidence": ("release_evidence/q1_remediation/", "reports/q1/", "research/preregistration/"),
    "fuzzyxai-q1-dissertation-artifacts": ("dissertation_artifacts/q1/", "reports/q1/", "release_evidence/q1_remediation/claim_registry"),
    "fuzzyxai-q1-reproducibility-bundle": (
        "framework/fuzzyxai/fuzzyxai/q1_validation/",
        "scripts/q1/",
        "configs/q1/",
        "tests/test_q1_validation_contracts.py",
        "research/preregistration/",
        "Dockerfile.q1",
        "docker-compose.q1.yml",
        "requirements.lock",
        "uv.lock",
        "Makefile",
    ),
    "fuzzyxai-q1-external-study-pack": (
        "study/comprehension/",
        "study/domain_review/",
        "study/expert_action_review/",
        "release_evidence/q1_remediation/external_studies/",
    ),
}


def git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary)


def tracked_paths(commit: str, prefixes: Sequence[str]) -> list[str]:
    all_paths = str(git("ls-tree", "-r", "--name-only", commit)).splitlines()
    return sorted(path for path in all_paths if any(path == prefix or path.startswith(prefix) for prefix in prefixes))


def main() -> None:
    commit = str(git("rev-parse", "HEAD")).strip()
    branch = str(git("branch", "--show-current")).strip()
    short = commit[:12]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, prefixes in ARCHIVES.items():
        paths = tracked_paths(commit, prefixes)
        if not paths:
            raise RuntimeError(f"archive {name} has no committed input files")
        entries = []
        archive_path = OUTPUT / f"{name}-{short}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                content = bytes(git("show", f"{commit}:{path}", binary=True))
                info = zipfile.ZipInfo(f"fuzzy-xai-operator/{path}", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
                entries.append({"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            manifest = {
                "schema_version": "1.0",
                "archive_type": name,
                "commit": commit,
                "branch": branch,
                "file_count": len(entries),
                "files": entries,
                "external_gates": ["comprehension", "expert_action_review", "domain_language_review"],
                "stable_release_allowed": False,
            }
            info = zipfile.ZipInfo("fuzzy-xai-operator/_archive_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        archive_path.with_suffix(".zip.sha256").write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
        records.append({"archive": archive_path.name, "sha256": digest, "file_count": len(entries), "commit": commit})
    verification = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": branch,
        "status": "PASS",
        "archives": records,
        "stable_release_allowed": False,
    }
    (OUTPUT / "q1_archive_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: q1_archives count={len(records)} commit={commit}")


if __name__ == "__main__":
    main()
