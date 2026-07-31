#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts"
PREFIXES = (
    "examples/ml_vertical_v1/",
    "protocol/ml_vertical_v1/",
    "reports/ml_vertical_v1/",
    "results/ml_vertical_v1/",
    "tests/ml_vertical_v1/",
)
EXACT = {
    "AGENTS.md",
    "Dockerfile.ml-vertical",
    "PROJECT_MEMORY.md",
    "RELEASE_STATUS.md",
    "TEST_REPORT.txt",
    "docker-compose.yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tracked_inputs() -> list[Path]:
    names = git("ls-files").splitlines()
    selected = [ROOT / name for name in names if name in EXACT or name.startswith(PREFIXES)]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise RuntimeError(f"tracked evidence is missing: {missing}")
    return sorted(selected)


def verify_internal_checksums(package: Path) -> int:
    lines = (package / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for line in lines:
        expected, name = line.split(maxsplit=1)
        if sha256(package / name.lstrip("*")) != expected:
            raise RuntimeError(f"evidence checksum mismatch: {name}")
    return len(lines)


def main() -> None:
    commit = git("rev-parse", "HEAD")
    if git("write-tree") != git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("release index differs from HEAD")
    short = commit[:12]
    source = OUTPUT / f"fuzzyxai-source-release-{short}.zip"
    source_sidecar = source.with_suffix(".zip.sha256")
    clean_report = OUTPUT / "ML_VERTICAL_CLEAN_SOURCE_TESTS.json"
    for required in (source, source_sidecar, clean_report):
        if not required.is_file():
            raise RuntimeError(f"required release input is missing: {required}")
    expected_source = source_sidecar.read_text(encoding="ascii").split()[0]
    if sha256(source) != expected_source:
        raise RuntimeError("source release sidecar mismatch")

    archive = OUTPUT / f"fuzzyxai-ml-vertical-v1-evidence-{short}.zip"
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / "fuzzyxai-ml-vertical-v1-evidence"
        package.mkdir()
        for source_path in tracked_inputs():
            relative = source_path.relative_to(ROOT)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        source_root = package / "source_release"
        source_root.mkdir()
        shutil.copy2(source, source_root / source.name)
        shutil.copy2(source_sidecar, source_root / source_sidecar.name)
        shutil.copy2(clean_report, source_root / clean_report.name)
        identity = {
            "schema_version": "1.0",
            "commit": commit,
            "branch": git("branch", "--show-current"),
            "source_release": source.name,
            "source_release_sha256": expected_source,
            "docx_modified": False,
            "parent_results_recalculated": False,
            "status": "FUZZYXAI_ML_VERTICAL_V1_IMPLEMENTED",
        }
        (package / "PACKAGE_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = sorted(path for path in package.rglob("*") if path.is_file())
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(package).as_posix()}" for path in files
        ]
        (package / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
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
        with tempfile.TemporaryDirectory() as extracted:
            with zipfile.ZipFile(archive) as packaged:
                packaged.testzip()
                packaged.extractall(extracted)
            verified = verify_internal_checksums(Path(extracted) / package.name)

    archive_sha = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(
        f"{archive_sha}  {archive.name}\n", encoding="ascii"
    )
    manifest = {
        "archive": archive.name,
        "commit": commit,
        "files_verified": verified,
        "sha256": archive_sha,
        "source_release_sha256": expected_source,
    }
    (OUTPUT / "ml_vertical_v1_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
