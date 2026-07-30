#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "fuzzyxai-h10-c7r-evidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"required evidence directory is missing: {source}")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _verify_manifest(package: Path) -> int:
    lines = (package / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        path = package / relative.lstrip("*")
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"evidence checksum mismatch: {relative}")
    return len(lines)


def _write_zip(package: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(
                path.relative_to(package.parent).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.external_attr = 0o100644 << 16
            output.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation-root", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--clean-source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release_artifacts")
    args = parser.parse_args()

    operation_root = args.operation_root.resolve()
    source_release = args.source_release.resolve()
    clean_source_report = args.clean_source_report.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    commit = _git("rev-parse", "HEAD")
    if _git("write-tree") != _git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("release index differs from HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree changes must be committed before packaging")
    if not source_release.is_file() or not clean_source_report.is_file():
        raise RuntimeError("source release or clean-source report is missing")
    source_sidecar = source_release.with_suffix(".zip.sha256")
    if not source_sidecar.is_file():
        raise RuntimeError("source release SHA256 sidecar is missing")
    expected_source_sha = source_sidecar.read_text(encoding="ascii").split()[0]
    if _sha256(source_release) != expected_source_sha:
        raise RuntimeError("source release SHA256 sidecar mismatch")

    short = commit[:12]
    archive = output / f"fuzzyxai-h10-c7r-evidence-{short}.zip"
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / PACKAGE_NAME
        package.mkdir()

        _copy_tree(ROOT / "protocol/h10_c7r", package / "protocol/h10_c7r")
        _copy_tree(ROOT / "results/h10_c7r", package / "results/h10_c7r")
        _copy_tree(ROOT / "reports/h10_c7r", package / "reports/h10_c7r")
        _copy_tree(operation_root / "selection", package / "operation/selection")
        _copy_tree(operation_root / "runtime", package / "operation/runtime")
        _copy_tree(operation_root / "sealed-gold", package / "operation/sealed-gold")
        _copy_tree(operation_root / "source", package / "operation/source")

        shutil.copy2(ROOT / "PROJECT_MEMORY.md", package / "PROJECT_MEMORY.md")
        shutil.copy2(
            ROOT / "reports/h10_c7r/RELEASE_STATUS.md",
            package / "RELEASE_STATUS.md",
        )
        source_target = package / "source_release"
        source_target.mkdir()
        shutil.copy2(source_release, source_target / source_release.name)
        shutil.copy2(source_sidecar, source_target / source_sidecar.name)
        shutil.copy2(clean_source_report, source_target / clean_source_report.name)

        identity = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "commit": commit,
            "branch": _git("branch", "--show-current"),
            "official_status": "H10_C7R_NOT_SUPPORTED",
            "scientific_result": "NOT_SUPPORTED",
            "opening_count": 1,
            "source_release": source_release.name,
            "source_release_sha256": expected_source_sha,
            "docx_modified": False,
            "pdf_included": False,
            "parent_results_recalculated": False,
            "runtime_recollected_during_packaging": False,
            "scoring_repeated_during_packaging": False,
        }
        (package / "PACKAGE_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = sorted(path for path in package.rglob("*") if path.is_file())
        (package / "SHA256SUMS").write_text(
            "\n".join(
                f"{_sha256(path)}  {path.relative_to(package).as_posix()}"
                for path in files
            )
            + "\n",
            encoding="ascii",
        )
        _write_zip(package, archive)

        with tempfile.TemporaryDirectory() as extracted:
            with zipfile.ZipFile(archive) as zipped:
                bad_member = zipped.testzip()
                if bad_member:
                    raise RuntimeError(f"ZIP CRC failure: {bad_member}")
                zipped.extractall(extracted)
            checked = _verify_manifest(Path(extracted) / PACKAGE_NAME)

    archive_sha = _sha256(archive)
    sidecar = archive.with_suffix(".zip.sha256")
    sidecar.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    manifest = {
        "archive": archive.name,
        "sha256": archive_sha,
        "commit": commit,
        "files_verified": checked,
        "source_release": source_release.name,
        "source_release_sha256": expected_source_sha,
        "status": "PASS",
    }
    (output / "h10_c7r_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
