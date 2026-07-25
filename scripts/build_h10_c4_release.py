#!/usr/bin/env python3
"""Build one auditable H10-C4 ZIP from the committed Git tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "release_artifacts"
PREFIXES = (
    "artifacts/h10_c3_r4/",
    "docs/chapters/glava_4_FuzzyXAI_corrected_final.docx",
    "docs/chapters/glava_4_FuzzyXAI_h10_c4_revision.docx",
    "docs/supplementary/",
    "framework/fuzzyxai/",
    "protocol/h10_c4/",
    "reports/h10_c4/",
    "results/h10_c4/",
    "scripts/build_h10_c4_chapter4.py",
    "scripts/build_h10_c4_release.py",
    "scripts/manuscript_claim_lint.py",
    "scripts/run_h10_c4.py",
    "tests/h10_c4/",
    "tests/operators/",
)
EXACT = {
    "AGENTS.md",
    "Makefile",
    "PROJECT_MEMORY.md",
    "RELEASE_STATUS.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "uv.lock",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _selected(path: str) -> bool:
    return path in EXACT or any(path.startswith(prefix) for prefix in PREFIXES)


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git_entries(commit: str) -> list[tuple[str, int]]:
    entries = []
    for line in _git("ls-tree", "-r", commit).splitlines():
        metadata, path = line.split("\t", 1)
        git_mode = metadata.split(" ", 1)[0]
        entries.append((path, int(git_mode[-3:], 8)))
    return entries


def _write_member(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: Path,
    mode: int,
) -> None:
    source.chmod(mode)
    archive.write(source, archive_name)


def _zip_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0o777


def build() -> tuple[Path, str]:
    if _git("status", "--porcelain"):
        raise RuntimeError("release build requires a clean committed worktree")
    commit = _git("rev-parse", "HEAD")
    entries = [
        (path, mode)
        for path, mode in _git_entries(commit)
        if _selected(path)
    ]
    paths = [path for path, _mode in entries]
    if not paths:
        raise RuntimeError("H10-C4 release allowlist selected no files")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"fuzzyxai-h10-c4-operational-utility-{commit[:12]}-one-zip.zip"
    checksums = []
    with tempfile.TemporaryDirectory(prefix="h10-c4-release-") as temp:
        staging = Path(temp) / "fuzzyxai-h10-c4"
        for path, mode in entries:
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = _git_bytes(commit, path)
            target.write_bytes(data)
            target.chmod(mode)
            checksums.append(f"{hashlib.sha256(data).hexdigest()}  {path}")
        manifest = {
            "study_id": "FXAI-H10-C4-OPERATIONAL-UTILITY",
            "source_commit": commit,
            "source_branch": _git("branch", "--show-current"),
            "built_at_utc": datetime.now(UTC).isoformat(),
            "source": "committed_git_tree",
            "file_count": len(paths),
            "executable_files": sorted(
                path for path, mode in entries if mode & 0o111
            ),
            "h10_c3_parent_modified": False,
            "h10_c4_status": json.loads(
                _git_bytes(
                    commit,
                    "results/h10_c4/H10_C4_FINAL_STATUS.json",
                )
            )["status"],
        }
        (staging / "BUILD_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(staging)
                    mode = dict(entries).get(relative.as_posix(), 0o644)
                    _write_member(
                        archive,
                        path,
                        path.relative_to(staging.parent),
                        mode,
                    )

    archive_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_sha256}  {output.name}\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        expected = {f"fuzzyxai-h10-c4/{path}" for path in paths}
        if not expected.issubset(names):
            raise AssertionError("release ZIP is missing committed allowlist files")
        mode_by_path = dict(entries)
        for path, expected_mode in mode_by_path.items():
            info = archive.getinfo(f"fuzzyxai-h10-c4/{path}")
            if _zip_mode(info) != expected_mode:
                raise AssertionError(
                    f"release mode mismatch: {path}: "
                    f"{_zip_mode(info):#o} != {expected_mode:#o}"
                )
        for line in archive.read("fuzzyxai-h10-c4/SHA256SUMS").decode().splitlines():
            digest, path = line.split("  ", 1)
            actual = hashlib.sha256(
                archive.read(f"fuzzyxai-h10-c4/{path}")
            ).hexdigest()
            if actual != digest:
                raise AssertionError(f"release checksum mismatch: {path}")
    return output, archive_sha256


if __name__ == "__main__":
    path, digest = build()
    print(path)
    print(digest)
