#!/usr/bin/env python3
"""Build evidence, dissertation, and reproducibility archives with manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_artifacts"


def git(*args: str) -> str:
    if args == ("rev-parse", "HEAD") and os.environ.get("FUZZYXAI_COMMIT"):
        return os.environ["FUZZYXAI_COMMIT"]
    if args == ("branch", "--show-current") and os.environ.get("FUZZYXAI_BRANCH"):
        return os.environ["FUZZYXAI_BRANCH"]
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_archive(name: str, files: Iterable[Path], metadata: dict[str, object]) -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / name
    paths = sorted({path.resolve() for path in files if path.is_file()})
    manifest = {**metadata, "archive": archive.name, "files": [str(path.relative_to(ROOT)) for path in paths]}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, Path("fuzzy-xai-operator") / path.relative_to(ROOT))
        handle.writestr("fuzzy-xai-operator/archive_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    digest = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return {"archive": str(archive), "sha256": digest, "file_count": len(paths) + 1}


def build() -> dict[str, object]:
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    evidence = ROOT / "release_evidence/full_empirical_validation"
    run_manifest = json.loads((evidence / "run_manifest.json").read_text(encoding="utf-8"))
    common = {
        "schema_version": "1.0",
        "commit": commit,
        "branch": branch,
        "profile": run_manifest["profile"],
        "technical_status": {row["experiment_id"]: row["status"] for row in run_manifest["experiments"]},
        "external_gates": run_manifest["external_gates"],
        "tag_allowed": run_manifest["tag_allowed"],
    }
    short = commit[:12]
    empirical = build_archive(
        f"fuzzyxai-full-empirical-evidence-{short}.zip",
        evidence.rglob("*"),
        common,
    )
    dissertation_files = [
        *list((ROOT / "dissertation_artifacts/chapter3").rglob("*")),
        *list((ROOT / "dissertation_artifacts/chapter4").rglob("*")),
        *list((ROOT / "dissertation_artifacts/claims").rglob("*")),
        *list((ROOT / "reports/empirical_validation").rglob("*")),
    ]
    dissertation = build_archive(f"fuzzyxai-dissertation-artifacts-{short}.zip", dissertation_files, common)
    reproduction_files = [
        ROOT / "Dockerfile",
        ROOT / "docker-compose.yml",
        ROOT / "requirements.lock",
        ROOT / "pyproject.toml",
        ROOT / "Makefile",
        ROOT / "configs/full_empirical_validation.json",
        ROOT / "scripts/reproduce_all.py",
        ROOT / "scripts/verify_reproduction.py",
        ROOT / "scripts/run_full_empirical_validation.py",
        ROOT / "scripts/run_optional_multimodal_models.py",
        ROOT / "scripts/verify_full_empirical_validation.py",
        evidence / "run_manifest.json",
        evidence / "manifest_sha256.json",
    ]
    reproduction = build_archive(f"fuzzyxai-reproducibility-bundle-{short}.zip", reproduction_files, common)
    payload = {"empirical": empirical, "dissertation": dissertation, "reproducibility": reproduction}
    (OUTPUT / "empirical_archive_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in payload.values():
        print(f"PASS: archive {item['archive']} files={item['file_count']}")
    return payload


if __name__ == "__main__":
    build()
