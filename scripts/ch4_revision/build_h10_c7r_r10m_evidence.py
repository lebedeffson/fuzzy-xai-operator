#!/usr/bin/env python3
"""Build and verify the final H10-C7R-R10M evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "release_artifacts"
PACKAGE_ROOT = "fuzzyxai-h10-c7r-r10m-evidence"
RECOLLECTION = OUTPUT_DIR / "h10-c7r-r10-recollection-40-run-30589146636.zip"
INCLUDED_PATHS = (
    Path("PROJECT_MEMORY.md"),
    Path("protocol/h10_c7r_r10m"),
    Path("results/h10_c7r_r10m"),
    Path("reports/h10_c7r_r10m"),
)


def _run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_committed_path(relative: Path, destination: Path) -> None:
    tracked = _run("git", "ls-files", str(relative)).splitlines()
    if not tracked:
        raise RuntimeError(f"no committed files found for {relative}")
    for name in tracked:
        source = ROOT / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_internal_manifest(package: Path) -> int:
    files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(package).as_posix()}"
        for path in files
    ]
    (package / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def _zip_tree(package: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                relative = Path(PACKAGE_ROOT) / path.relative_to(package)
                info = zipfile.ZipInfo.from_file(path, relative.as_posix())
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.external_attr = 0o100644 << 16
                with path.open("rb") as stream:
                    archive.writestr(
                        info,
                        stream.read(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )


def _verify_archive(output: Path, expected_files: int) -> None:
    with tempfile.TemporaryDirectory(prefix="h10-c7r-r10m-verify-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(output) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"ZIP CRC failure: {bad_member}")
            archive.extractall(destination)
        package = destination / PACKAGE_ROOT
        manifest = package / "SHA256SUMS"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if len(lines) != expected_files:
            raise RuntimeError(
                f"internal manifest count {len(lines)} != {expected_files}"
            )
        for line in lines:
            expected, relative = line.split(maxsplit=1)
            path = package / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"internal SHA256 mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci-run", default="NOT_RUN_FOR_RELEASE_COMMIT")
    args = parser.parse_args()

    if _run("git", "status", "--porcelain"):
        raise RuntimeError("evidence packaging requires a clean worktree")

    commit = _run("git", "rev-parse", "HEAD")
    short_commit = commit[:12]
    branch = _run("git", "branch", "--show-current")
    remote_line = _run(
        "git",
        "ls-remote",
        "--heads",
        "origin",
        branch,
    )
    remote_head = remote_line.split()[0] if remote_line else "NOT_PUSHED"
    source_archive = OUTPUT_DIR / f"fuzzyxai-source-release-{short_commit}.zip"
    required = (source_archive, RECOLLECTION)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"fuzzyxai-h10-c7r-r10m-evidence-{short_commit}.zip"
    with tempfile.TemporaryDirectory(prefix="h10-c7r-r10m-build-") as temporary:
        package = Path(temporary) / PACKAGE_ROOT
        package.mkdir()
        for relative in INCLUDED_PATHS:
            _copy_committed_path(relative, package)

        artifacts = package / "release_artifacts"
        artifacts.mkdir()
        shutil.copy2(source_archive, artifacts / source_archive.name)
        shutil.copy2(RECOLLECTION, artifacts / RECOLLECTION.name)
        identity = {
            "branch": branch,
            "commit": commit,
            "remote_head": remote_head,
            "worktree_clean": True,
            "ci_run": args.ci_run,
            "development_status": "H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED",
            "scientific_result": "NOT_EVALUATED",
            "held_out_created": False,
            "held_out_scored": False,
            "source_archive": source_archive.name,
            "source_archive_sha256": _sha256(source_archive),
            "recollection_archive": RECOLLECTION.name,
            "recollection_archive_sha256": _sha256(RECOLLECTION),
            "model_weights_embedded": False,
            "model_weight_verification": (
                "Pinned revisions and snapshot/weight SHA256 values are recorded "
                "in protocol/h10_c7r_r10m/R10M_MODEL_LOCK.json."
            ),
        }
        (package / "PACKAGE_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checked_files = _write_internal_manifest(package)
        _zip_tree(package, output)

    _verify_archive(output, checked_files)
    digest = _sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"PASS: evidence_archive {output}")
    print(f"PASS: evidence_sha256 {digest}")
    print(f"PASS: internal_sha256 files={checked_files}")


if __name__ == "__main__":
    main()
